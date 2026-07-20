"""Oracle stage runner — executes INSIDE venv-oracle. Per-segment CER (FireRedASR2-AED,
C5-normalized) + speaker similarity (WeSpeaker cnceleb cosine vs the enrolled ref).

Task JSON in argv[1]: {"ref_wav": "...", "items": [{"wav": "...", "text": "..."}]}
Result JSON written to argv[1] + ".result": {"ref_emb_ok": bool,
  "items": [{"wav","cer","sim","hyp_norm","ref_norm"}]}
"""
import json
import os
import re
import sys
from pathlib import Path

# QC B-01 (P8a completion): resolve everything through FERRYMAN_HOME. The orchestrator
# exports the resolved root (QC B-11); this file self-derives from its own location as
# the fallback, so the tree runs from ANY path with zero edits.
ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT / "vendor" / "FireRedASR2S"))

ASR_DIR = str(ROOT / "models" / "fireredasr2-aed")
SPK_DIR = str(ROOT / "models" / "wespeaker-cnceleb-resnet34")

task = json.loads(open(sys.argv[1], encoding="utf-8").read())

# ---- text normalization (C5): opencc fold + number normalization + strip non-content
import opencc  # noqa: E402
import cn2an  # noqa: E402
_cc = opencc.OpenCC("t2s")


LEXICON: list[list[str]] = task.get("lexicon") or []   # homophone groups; member -> group[0]

# QC B-04: fold the LONGEST variant first across ALL groups — a short variant nested inside
# a longer one otherwise makes the fold order-dependent and non-idempotent between ref and hyp.
# QC B-05 (kept BY DESIGN): pronoun groups like 他/它/她 are all "tā" — a mismatch there is
# always an ASR transcription artifact, never an audible TTS error, so folding is CORRECT
# for an audio oracle.
_FOLDS = sorted(((v, g[0]) for g in LEXICON for v in g[1:]), key=lambda p: -len(p[0]))


def norm(s: str) -> str:
    s = _cc.convert(s)
    try:
        s = cn2an.transform(s, "an2cn")     # 26岁 → 二十六岁 (match spoken form)
        # (QC B-06 noted: an2cn direction kept as calibrated 2026-07-11 — changing the
        # numeral normalization would invalidate the CER threshold calibration.)
    except Exception:
        pass
    s = re.sub(r"[^0-9A-Za-z一-鿿]+", "", s)
    s = s.lower()
    for variant, canon in _FOLDS:           # proper-noun homophones fold to canonical
        s = s.replace(variant, canon)
    return s


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[len(h)] / len(r)


# ---- preflight the CHEAP dependency first (REM-4.6/R-17: the WeSpeaker existence check
# used to run AFTER the FireRedASR GPU load — a common bring-up miss cost a heavy load
# before the actionable error). QC B-16: actionable message, not a cryptic ORT error.
_onnx_path = os.path.join(SPK_DIR, "cnceleb_resnet34.onnx")
if not os.path.exists(_onnx_path):
    sys.exit(f"[oracle] FATAL: WeSpeaker ONNX missing: {_onnx_path} — "
             "run bin/hydrate_weights.ps1 (weights manifest) first")

# ---- load models once
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config  # noqa: E402

cfg = FireRedAsr2Config()
cfg.use_gpu = True
asr = FireRedAsr2.from_pretrained("aed", ASR_DIR, cfg)
print(f"[oracle] FireRedASR2-AED loaded from {ASR_DIR}")

# Speaker embeddings via the WeSpeaker ONNX export directly — the pip package is a
# dependency tarpit (hard-imports s3prl); onnxruntime + kaldi fbank is all it needs.
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import kaldi_native_fbank as knf  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

_sess = ort.InferenceSession(_onnx_path, providers=["CPUExecutionProvider"])
_in_name = _sess.get_inputs()[0].name
_emb_cache: dict = {}


def _embed(path: str):
    if path in _emb_cache:
        return _emb_cache[path]
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_poly(wav, 16000, sr).astype("float32")
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, (wav * 32768.0).tolist())
    fb.input_finished()
    feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype("float32")
    feats = feats - feats.mean(axis=0, keepdims=True)          # per-utterance CMN
    emb = _sess.run(None, {_in_name: feats[None]})[0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    _emb_cache[path] = emb
    return emb


def spk_similarity(a: str, b: str) -> float:
    return float(np.dot(_embed(a), _embed(b)))


# QC B-14 + REM-4.6: tri-state — None = sim never attempted (no ref_wav), True = the REF
# embedding succeeded, False = the REF embedding itself failed (the gate cannot run).
# Per-ITEM failures no longer latch False — they yield sim=None for that item only, so one
# transient odd wav can't make the orchestrator report a total sim-gate failure.
print("[oracle] wespeaker cnceleb-resnet34 ONNX loaded")
ref_wav = task.get("ref_wav")
ref_emb_ok = None
if ref_wav:
    try:
        _embed(ref_wav)
        ref_emb_ok = True
    except Exception as e:  # noqa: BLE001
        ref_emb_ok = False
        print(f"[oracle] REF embedding FAILED ({e}) — the speaker gate cannot run (ref_emb_ok=false)")

results = []
for i, it in enumerate(task["items"]):
    out = asr.transcribe([f"seg{i}"], [it["wav"]])
    hyp = out[0]["text"] if out else ""
    if not hyp.strip():
        # QC B-19: distinguish "empty/corrupt audio" from a genuine mis-transcription
        print(f"[oracle] WARNING: empty ASR for {it['wav']} "
              f"(size={os.path.getsize(it['wav'])}B) — check the wav, not just the CER")
    rn, hn = norm(it["text"]), norm(hyp)
    if not rn:
        # QC B-12: a segment that normalizes to nothing (punctuation-only) is ungradeable —
        # record 0.0 explicitly instead of letting the floor arithmetic pass garbage.
        print(f"[oracle] WARNING: item {i} ref normalizes to empty — CER not gradeable, recording 0.0")
        c = 0.0
    else:
        c = round(cer(rn, hn), 4)
    s = None
    if ref_wav and ref_emb_ok:
        try:
            s = round(spk_similarity(ref_wav, it["wav"]), 4)
        except Exception as e:  # noqa: BLE001 — item-level only; ref_emb_ok unchanged (REM-4.6)
            print(f"[oracle] sim failed for {it['wav']}: {e} (this item graded sim=None)")
    results.append({"idx": it.get("idx"), "wav": it["wav"], "cer": c, "sim": s,
                    "hyp_norm": hn, "ref_norm": rn})
    print(f"[oracle] {i+1}/{len(task['items'])} cer={c} sim={s}")

# ref_sr recorded for calibration provenance (QC B-10): thresholds were calibrated on a
# specific ref sample-rate path; a re-enrolled ref at a different SR is now detectable.
_ref_sr = None
if ref_wav:
    try:
        _ref_sr = int(sf.info(ref_wav).samplerate)
    except Exception:  # noqa: BLE001
        pass
with open(sys.argv[1] + ".result", "w", encoding="utf-8") as f:
    json.dump({"ref_emb_ok": ref_emb_ok, "ref_sr": _ref_sr, "items": results},
              f, ensure_ascii=False, indent=1)
print("[oracle] ALL DONE")
