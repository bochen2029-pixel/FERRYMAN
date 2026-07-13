r"""Image head I5 — the vision oracle (the image-quality falsifier; FERRYMAN's oracle doctrine
applied to pixels). Asks a local Qwen-vision model whether a generated image matches its intent
and is clean (no garbled text/watermark). Returns {ok: bool|None, reason}.

IMPLEMENTATION NOTE: uses **llama-mtmd-cli.exe** (the dedicated multimodal runner). llama-server's
OpenAI `/v1/chat/completions` image_url path returns EMPTY content with these mmproj files on this
build — the mtmd CLI is the working path. One model-load per call (no persistent server), which is
fine for a per-candidate gate; ComfyUI is stopped by the caller (imagegen) so there's no VRAM clash.
"""
import re
import os
import subprocess
import sys
from pathlib import Path

MTMD = os.environ.get("LLAMA_MTMD_CLI", "llama-mtmd-cli.exe")
VMODEL = os.environ.get("FERRYMAN_VISION_GGUF", "Qwen3.5-9B-Q5_K_M.gguf")
MMPROJ = os.environ.get("FERRYMAN_MMPROJ", "mmproj-F16.gguf")


def verify(image_path, intent: str) -> dict:
    prompt = ('Judge this AI-generated illustration. Intended theme: "' + intent[:260] + '". '
              "Is it a clear, high-quality, on-theme image with NO garbled text, letters, or watermark? "
              "Answer YES or NO on the first line, then one short sentence why.")
    try:
        cp = subprocess.run([MTMD, "-m", VMODEL, "--mmproj", MMPROJ, "--image", str(image_path),
                             "-p", prompt, "-n", "110", "--temp", "0"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=200)
    except Exception as e:  # noqa: BLE001
        return {"ok": None, "reason": f"mtmd-cli failed: {e}"}
    out = re.sub(r"<think>.*?</think>", "", cp.stdout or "", flags=re.S)
    lines = [ln.strip() for ln in out.splitlines()
             if ln.strip() and not re.match(r"^[\d.]+\s", ln) and "find_slot" not in ln]
    text = " ".join(lines).strip()
    m = re.search(r"\b(yes|no)\b", text, re.I)
    ok = (m.group(1).lower() == "yes") if m else None
    return {"ok": ok, "reason": text[:220]}


if __name__ == "__main__":
    import json
    print(json.dumps(verify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "an evocative illustration"),
                     ensure_ascii=False, indent=2))
