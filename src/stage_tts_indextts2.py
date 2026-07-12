"""TTS stage runner — executes INSIDE venv-tts. Loads IndexTTS2 once, renders every
item in the task file. Task JSON: {"ref": "<ref.wav>", "items": [{"text","out"}...]}.
Speaker-agnostic: the ref path defines the voice."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
MODEL_DIR = str(ROOT / "models" / "indextts2")

task = json.loads(open(sys.argv[1], encoding="utf-8").read())
ref = task["ref"]
items = task["items"]

from indextts.infer_v2 import IndexTTS2  # noqa: E402

t0 = time.time()
tts = IndexTTS2(cfg_path=os.path.join(MODEL_DIR, "config.yaml"), model_dir=MODEL_DIR,
                use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
print(f"[tts-stage] model loaded in {time.time()-t0:.1f}s | {len(items)} items | ref={ref}")

for i, it in enumerate(items):
    os.makedirs(os.path.dirname(it["out"]), exist_ok=True)
    t1 = time.time()
    tts.infer(spk_audio_prompt=ref, text=it["text"], output_path=it["out"])
    print(f"[tts-stage] {i+1}/{len(items)} done in {time.time()-t1:.1f}s -> {it['out']}")

print("[tts-stage] ALL DONE")
