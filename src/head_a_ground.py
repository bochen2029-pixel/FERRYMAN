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
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# QC C-01/C-02: tree paths resolve through FERRYMAN_HOME; the LLM substrate (llama.cpp +
# GGUFs) lives OUTSIDE the tree and gets env overrides for foreign boxes. Defaults = the
# dev box's substrate layout (manifests/substrate.json documents how to recreate it).
ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
LLAMA_DIR = Path(os.environ.get("FERRYMAN_LLAMA_DIR") or r"C:\llama.cpp")
GGUF_DIR = Path(os.environ.get("FERRYMAN_GGUF_DIR") or r"C:\models")
LLAMA = str(LLAMA_DIR / "llama-server.exe")
EMBED_MODEL = str(GGUF_DIR / "qwen3-embedding-0.6b-q8_0.gguf")
CHAT_MODEL = str(GGUF_DIR / "Qwen3-8B-Q4_K_M.gguf")
LOGS = ROOT / "logs"
TOP_K = 5
NGL = os.environ.get("FERRYMAN_NGL", "99")   # QC C-21: full GPU offload default; smaller GPUs override
CTX = int(os.environ.get("FERRYMAN_LLAMA_CTX", "8192"))   # REM-4.12 (R-12): overridable server ctx


def _log(m): print(f"[head-a] {m}", flush=True)


def stop_server(p):
    """Terminate and WAIT until the process is really gone — an orphaned llama-server
    holds VRAM and breaks the next one-model-at-a-time stage (QC C-09)."""
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _log("WARNING: llama-server did not die after kill() — VRAM may still be held")
    time.sleep(1)  # driver settle before the next heavy load


def start_server(model, port, extra, tag):
    if not Path(LLAMA).exists():
        raise RuntimeError(f"llama-server not found: {LLAMA} — set FERRYMAN_LLAMA_DIR "
                           "(substrate; see manifests/substrate.json)")
    if not Path(model).exists():
        raise RuntimeError(f"GGUF model not found: {model} — set FERRYMAN_GGUF_DIR / fetch substrate")
    # QC C-08: if SOMETHING already serves this port, /health would answer 200 and we'd
    # silently query an UNKNOWN model. Refuse loudly instead of attaching.
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            if r.status == 200:
                raise RuntimeError(f"port {port} is already serving (a sibling llama-server?) — "
                                   f"refusing to attach to an unknown model; stop it first (QC C-08)")
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001 — nothing listening: the port is ours
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    logf = open(LOGS / f"llama_{tag}_{port}.log", "w", encoding="utf-8")
    p = subprocess.Popen([LLAMA, "-m", model, "--host", "127.0.0.1", "--port", str(port),
                          "-ngl", NGL, "-c", str(CTX), *extra], stdout=logf, stderr=subprocess.STDOUT)
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
    stop_server(p)
    raise RuntimeError(f"llama-server ({tag}) health timeout")


def _post(url, payload, timeout=180, tries=3):
    """QC C-18: a single transient hiccup (timeout on a cold cache, a slow box) used to
    abort minutes of grounding work — retry with backoff before giving up."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — URLError / socket.timeout / HTTPError
            last = e
            if k < tries - 1:
                _log(f"llama HTTP retry {k+1}/{tries-1} after: {e}")
                time.sleep(2 * (k + 1))
    raise RuntimeError(f"llama HTTP failed after {tries} tries: {last} (QC C-18)")


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
    if not spans:
        # QC C-06: fail HERE (before a needless VRAM spin-up) instead of "grounding" nothing
        raise SystemExit(f"no content spans extracted from {doc_path} — empty doc or headers only")
    _log(f"doc={Path(doc_path).name} spans={len(spans)} | Q: {question}")

    # 1 · embed spans + query (query gets the Qwen3-embedding retrieval instruct)
    srv = start_server(EMBED_MODEL, 8085, ["--embedding", "--pooling", "last"], "embed")
    try:
        svecs = [embed(8085, s["text"]) for s in spans]
        qvec = embed(8085, f"Instruct: Given a question, retrieve passages that answer it\nQuery: {question}")
    finally:
        stop_server(srv)

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
        g = extract_json(raw)
        parse_error = g is None   # QC C-05: a parse failure must be distinguishable from "no claims"
        if parse_error:
            _log("WARNING: grounder output was not parseable JSON — recording parse_error=true")
            g = {"answer": raw, "claims": [], "answerable": None}

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
        stop_server(srv)

    n = len(verified)
    attributable = sum(1 for c in verified if c["attributable"])
    supported = sum(1 for c in verified if c["supported"])
    # QC C-04: no vacuous certification — a 1-claim answer can't earn a 100% faithfulness
    # PASS off a single self-judged entailment. pass=None means "not certifiable", distinct
    # from False (failed). parse_error also forces None.
    MIN_CLAIMS = 2
    if parse_error or n < MIN_CLAIMS:
        faith_pass = None
        _log(f"faithfulness NOT CERTIFIABLE (parse_error={parse_error}, claims={n} < {MIN_CLAIMS})")
    else:
        faith_pass = bool(supported / n >= 0.95)
    result = {
        "doc": str(doc_path), "question": question,
        "answer": g.get("answer"), "answerable": g.get("answerable"),
        "parse_error": parse_error,
        "claims": verified, "retrieved_spans": [s["id"] for s in top],
        "faithfulness": {"claims": n,
                         "attributable_pct": round(100 * attributable / n, 1) if n else None,
                         "supported_pct": round(100 * supported / n, 1) if n else None,
                         "pass": faith_pass},
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
