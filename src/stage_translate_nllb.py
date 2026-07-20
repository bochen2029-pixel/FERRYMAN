"""Translation stage (Head D · cross-lingual dub) — runs INSIDE **venv-tts** (torch 2.8:
transformers 4.51 refuses pytorch_model.bin under torch<2.6, CVE-2025-32434 — so NOT
venv-oracle, whose torch is 2.4.1; QC E-07). Loads NLLB-200 once, translates every item.

Task JSON: {"model": "<nllb dir>", "src_lang": "zho_Hans", "tgt_lang": "eng_Latn",
            "items": [{"idx": 0, "text": "..."}, ...]}
Writes <task>.result: {"items": [{"idx": 0, "text": "<translated>"}, ...]}

NLLB uses FLORES-200 language codes (e.g. zho_Hans, zho_Hant, eng_Latn, jpn_Jpan, spa_Latn,
fra_Latn, deu_Latn, kor_Hang, rus_Cyrl). The naive dub path (operator-approved): ASR -> here
-> TTS in the target language with the cloned voice = "same voice, new language."
"""
import json
import os
import sys
import time

task = json.loads(open(sys.argv[1], encoding="utf-8").read())
mdir = task["model"]
src_lang = task["src_lang"]
tgt_lang = task["tgt_lang"]
items = task["items"]

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

t0 = time.time()
tok = AutoTokenizer.from_pretrained(mdir, src_lang=src_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(mdir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
# resolve the target-language forced BOS token across transformers versions
try:
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    if tgt_id is None or tgt_id == tok.unk_token_id:
        tgt_id = tok.lang_code_to_id[tgt_lang]  # older API
except KeyError:
    # QC B-17: a typo'd/unsupported FLORES code used to die as a raw KeyError
    sys.exit(f"[translate] FATAL: unknown FLORES-200 target code {tgt_lang!r} "
             "(expected e.g. zho_Hans, eng_Latn, jpn_Jpan — case-sensitive)")
except Exception:  # noqa: BLE001
    try:
        tgt_id = tok.lang_code_to_id[tgt_lang]
    except KeyError:
        sys.exit(f"[translate] FATAL: unknown FLORES-200 target code {tgt_lang!r}")
print(f"[translate] NLLB loaded in {time.time()-t0:.1f}s on {device} | {src_lang}->{tgt_lang} | "
      f"{len(items)} items | tgt_bos={tgt_id}")

out = []
for i, it in enumerate(items):
    # QC B-08: detect input truncation instead of silently dropping the tail of a long
    # segment; size the OUTPUT budget to the input instead of a hard 512 ceiling.
    n_in = len(tok(it["text"]).input_ids)
    if n_in > 512:
        sys.exit(f"[translate] FATAL: item {it['idx']} is {n_in} tokens (>512) — split the "
                 "segment upstream; refusing to silently drop text")
    enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        gen = model.generate(**enc, forced_bos_token_id=int(tgt_id),
                             max_new_tokens=min(1024, n_in * 2 + 50),
                             num_beams=4, no_repeat_ngram_size=3)
    txt = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    out.append({"idx": it["idx"], "text": txt})
    print(f"[translate] {i+1}/{len(items)} -> {txt[:70]}")

json.dump({"items": out}, open(sys.argv[1] + ".result", "w", encoding="utf-8"), ensure_ascii=False)
print("[translate] ALL DONE")
