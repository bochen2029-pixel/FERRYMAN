r"""P0 gate: prove torch CUDA kernels + a real diffusion render on this GPU.
Writes out\p0_smoke.png. Exit 0 + 'P0 SMOKE PASS' line = gate passed."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)

print(f"torch {torch.__version__} | cuda {torch.version.cuda} | available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("P0 SMOKE FAIL: CUDA not available")
    sys.exit(1)
print(f"device: {torch.cuda.get_device_name(0)} | compute cap {torch.cuda.get_device_capability(0)}")

from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    str(ROOT / "models" / "sd-turbo"),
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
)
pipe = pipe.to("cuda")
img = pipe(
    prompt="a wooden ferry boat crossing a calm river at dawn, oil painting",
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]
out = str(ROOT / "out" / "p0_smoke.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out)
print(f"P0 SMOKE PASS: wrote {out} | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
