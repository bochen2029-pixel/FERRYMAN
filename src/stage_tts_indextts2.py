"""TTS stage runner — executes INSIDE venv-tts. Loads IndexTTS2 once, renders every
item in the task file. Task JSON: {"ref": "<ref.wav>", "items": [{"text","out"}...]}.
Speaker-agnostic: the ref path defines the voice."""
import json
import os
import sys
import time
from pathlib import Path

# QC B-02 (P8a completion): model dir resolves through FERRYMAN_HOME (exported by the
# orchestrator, QC B-11) with a self-derived fallback — portable to any unpack path.
ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
MODEL_DIR = str(ROOT / "models" / "indextts2")

task = json.loads(open(sys.argv[1], encoding="utf-8").read())
ref = task["ref"]
items = task["items"]

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from indextts.infer_v2 import IndexTTS2  # noqa: E402


def _seed_all(x: int) -> None:
    """QC B-13: IndexTTS2 sampling is stochastic — seed every RNG per item so a cache key
    (which embeds the take/seed) maps to ONE deterministic waveform, re-derivable forever."""
    random.seed(x)
    np.random.seed(x % (2 ** 32))
    torch.manual_seed(x)


t0 = time.time()
tts = IndexTTS2(cfg_path=os.path.join(MODEL_DIR, "config.yaml"), model_dir=MODEL_DIR,
                use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
print(f"[tts-stage] model loaded in {time.time()-t0:.1f}s | {len(items)} items | ref={ref}")

for i, it in enumerate(items):
    os.makedirs(os.path.dirname(it["out"]), exist_ok=True)
    if it.get("seed") is not None:
        _seed_all(int(it["seed"]))
    t1 = time.time()
    tts.infer(spk_audio_prompt=ref, text=it["text"], output_path=it["out"])
    # QC B-13: an engine-internal failure that "returns normally" must not count as done —
    # a missing or header-only wav is a loud failure here, not a downstream mystery.
    if not os.path.exists(it["out"]) or os.path.getsize(it["out"]) <= 44:
        sys.exit(f"[tts-stage] FATAL: empty/missing output for item {i}: {it['out']}")
    print(f"[tts-stage] {i+1}/{len(items)} done in {time.time()-t1:.1f}s -> {it['out']}")

print("[tts-stage] ALL DONE")
