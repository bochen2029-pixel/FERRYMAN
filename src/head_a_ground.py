r"""Head A — source grounding core (P9). The NotebookLM upstream: a document in, a
citation-anchored, span-traceable grounded JSON out, with a non-model faithfulness check.

Stack (all on-box, zero new pip deps — stdlib over llama-server HTTP):
  parse   : split the doc into stable-id spans (C0..Cn)   [chunker organ for pdf/docx; text here]
  embed   : qwen3-embedding-0.6b (GGUF) via llama-server --embedding  (last-token pooling + query instruct)
  retrieve: cosine top-k (pure-python; LanceDB is the scale/persistence upgrade, P9b)
  ground  : Qwen3-8B (GGUF) answers ONLY from the retrieved spans, emits {answer, claims:[{text,cites}]}
  oracle  : per-claim entailment back-check (LLM yes/no) -> % of claims span-attributable  [FALSIFIER]

Run (system python):  python src\head_a_ground.py <doc> "<question>" [out.json]
Sequential VRAM: embed server up->embed->down, then chat server up->ground+verify->down.
"""
import json
import math
import re
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA = os.environ.get("LLAMA_SERVER", "llama-server.exe")
EMBED_MODEL = os.environ.get("FERRYMAN_EMBED_GGUF", "qwen3-embedding-0.6b-q8_0.gguf")
CHAT_MODEL = os.environ.get("FERRYMAN_LLM_GGUF", "Qwen3-8B-Q4_K_M.gguf")
ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
LOGS = ROOT / "logs"
TOP_K = 5


def _log(m): print(f"[head-a] {m}", flush=True)


def start_server(model, port, extra, tag):
    LOGS.mkdir(parents=True, exist_ok=True)
    logf = open(LOGS / f"llama_{tag}_{port}.log", "w", encoding="utf-8")
    p = subprocess.Popen([LLAMA, "-m", model, "--host", "127.0.0.1", "--port", str(port),
                          "-ngl", "99", "-c", "8192", *extra], stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(180):
        if p.poll() is not None:
            raise RuntimeError(f"llama-server ({tag}) exited early — see {logf.name}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    _log(f"{tag} server ready on :{port}")
                    return p
        except Exception:  # noqa: BLE001
            time.sleep(1)
    p.terminate()
    raise RuntimeError(f"llama-server ({tag}) health timeout")


def _post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(port, text):
    d = _post(f"http://127.0.0.1:{port}/v1/embeddings", {"input": text, "model": "q"})
    return d["data"][0]["embedding"]


def chat(port, prompt, temp=0.1, n=900):
    d = _post(f"http://127.0.0.1:{port}/v1/chat/completions",
              {"messages": [{"role": "user", "content": prompt}], "temperature": temp,
               "max_tokens": n, "cache_prompt": True})
    return d["choices"][0]["message"]["content"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def split_spans(doc):
    """Sentence-ish spans with stable ids; keeps CJK sentences whole, merges tiny fragments."""
    doc = re.sub(r"^#.*$", "", doc, flags=re.M)  # drop markdown headers
    parts, buf = [], ""
    for ch in doc.replace("\n", ""):
        buf += ch
        if ch in "。！？!?…":
            if len(buf.strip()) >= 6:
                parts.append(buf.strip()); buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return [{"id": f"C{i}", "text": t} for i, t in enumerate(parts)]


def extract_json(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)  # strip Qwen3 thinking if any
    m = re.search(r"\{.*\}", s, flags=re.S)
    return json.loads(m.group(0)) if m else None


def ground(doc_path, question, out_path=None):
    t0 = time.time()
    doc = Path(doc_path).read_text(encoding="utf-8")
    spans = split_spans(doc)
    _log(f"doc={Path(doc_path).name} spans={len(spans)} | Q: {question}")

    # 1 · embed spans + query (query gets the Qwen3-embedding retrieval instruct)
    srv = start_server(EMBED_MODEL, 8085, ["--embedding", "--pooling", "last"], "embed")
    try:
        svecs = [embed(8085, s["text"]) for s in spans]
        qvec = embed(8085, f"Instruct: Given a question, retrieve passages that answer it\nQuery: {question}")
    finally:
        srv.terminate(); time.sleep(2)

    # 2 · retrieve top-k
    ranked = sorted(range(len(spans)), key=lambda i: cosine(qvec, svecs[i]), reverse=True)[:TOP_K]
    top = [spans[i] for i in ranked]
    _log(f"retrieved: {[s['id'] for s in top]}")

    # 3 · grounded answer (only from the retrieved spans) + 4 · faithfulness oracle
    srv = start_server(CHAT_MODEL, 8080, ["--jinja"], "chat")
    try:
        src_block = "\n".join(f'{s["id"]}: {s["text"]}' for s in top)
        gp = ("你是一个严格的『引用式问答』引擎。只能依据下面编号的【资料片段】回答，"
              "每一个论断都必须给出它所依据的片段编号。若资料中没有答案，answerable 设为 false。"
              "只输出合法 JSON，不要解释：\n"
              '{"answer":"...","claims":[{"text":"一句话论断","cites":["C3"]}],"answerable":true}\n/no_think\n\n'
              f"【资料片段】\n{src_block}\n\n【问题】{question}\n\nJSON:")
        raw = chat(8080, gp)
        g = extract_json(raw) or {"answer": raw, "claims": [], "answerable": None}

        by_id = {s["id"]: s["text"] for s in top}
        verified = []
        for c in g.get("claims", []):
            cites = [cid for cid in c.get("cites", []) if cid in by_id]
            supported = False
            for cid in cites:
                vp = (f"片段：{by_id[cid]}\n论断：{c['text']}\n"
                      "该片段是否支持这个论断？只回答 YES 或 NO。/no_think")
                if re.search(r"\byes\b", chat(8080, vp, n=8), flags=re.I):
                    supported = True; break
            verified.append({**c, "cites": cites, "attributable": bool(cites), "supported": supported})
    finally:
        srv.terminate(); time.sleep(1)

    n = len(verified)
    attributable = sum(1 for c in verified if c["attributable"])
    supported = sum(1 for c in verified if c["supported"])
    result = {
        "doc": str(doc_path), "question": question,
        "answer": g.get("answer"), "answerable": g.get("answerable"),
        "claims": verified, "retrieved_spans": [s["id"] for s in top],
        "faithfulness": {"claims": n,
                         "attributable_pct": round(100 * attributable / n, 1) if n else None,
                         "supported_pct": round(100 * supported / n, 1) if n else None,
                         "pass": bool(n and supported / n >= 0.95)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(out_path) if out_path else (ROOT / "work" / "grounded.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"grounded -> {out} | claims {n} | attributable {result['faithfulness']['attributable_pct']}% | "
         f"supported {result['faithfulness']['supported_pct']}% | {result['elapsed_s']}s")
    return result


if __name__ == "__main__":
    ground(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
