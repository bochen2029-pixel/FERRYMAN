r"""Image head — I2: document/script content -> image prompt -> ComfyUI -> PNG.

The NET-NEW piece over FERRYMAN: FERRYMAN's image prompt is hand-typed; here FERRYMAN's own
Qwen (Head A's llama-server) reads the source CONTENT and art-directs a vivid image prompt, which
is then handed to the reused FERRYMAN `comfy_client` (SDXL on :8188). Output PNG feeds either
HyperFrames (`<img>`/background, I3) or an ffmpeg overlay on the video (via compose_graphics, I4).

Run (system python; ComfyUI must be up on :8188):
  python imagehead\image_from_doc.py <doc-or-text> [out_dir] [n]
Sequential VRAM: Qwen server up->prompt->down, then ComfyUI generates.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_a_ground as H   # start_server, chat, CHAT_MODEL, _log
import comfy_client as CC    # ComfyClient, run_single, unwrap_workflow

WF = str(Path(__file__).resolve().parent / "workflows" / "sdxl_txt2img.json")
NEG = "text, watermark, letters, words, signature, blurry, low quality, deformed, ugly, oversaturated"


def art_direct(content: str, n: int = 1) -> list[str]:
    """FERRYMAN's Qwen reads the content and writes N vivid English image prompts."""
    srv = H.start_server(H.CHAT_MODEL, 8080, ["--jinja"], "chat")
    try:
        prompts = []
        for i in range(n):
            sp = ("You are an art director for a philosophy program. Read the CONTENT and write ONE "
                  "vivid, concrete ENGLISH image-generation prompt that visually illustrates its core "
                  "theme as an evocative, cinematic SCENE. Rules: describe subject + setting + light + "
                  "mood + colour palette + art style; absolutely NO text/words/letters in the image; "
                  "avoid recognizable human faces; 40-70 words; be specific and beautiful."
                  + (f" Make variant #{i+1} visually distinct from prior ones." if i else "")
                  + " Output ONLY the prompt. /no_think\n\nCONTENT:\n" + content[:2200] + "\n\nPROMPT:")
            raw = H.chat(8080, sp, temp=0.8, n=220)
            p = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip().strip('"').strip()
            p = p.splitlines()[0] if p else p
            prompts.append(p)
            H._log(f"prompt {i+1}/{n}: {p}")
        return prompts
    finally:
        srv.terminate(); time.sleep(2)   # free VRAM before ComfyUI loads SDXL


def generate(content: str, out_dir: str, n: int = 1, seed: int = -1, steps: int = 26,
             wh=(1024, 1024)) -> list[dict]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = art_direct(content, n)
    wf = CC.unwrap_workflow(json.loads(Path(WF).read_text(encoding="utf-8")))
    client = CC.ComfyClient("http://127.0.0.1:8188")
    if not client.server_up():
        raise SystemExit("ComfyUI not reachable on :8188 — launch it first (ComfyUI.exe).")
    results = []
    for i, p in enumerate(prompts):
        args = {"prompt": p, "negative_prompt": NEG,
                "seed": (seed if seed >= 0 else CC.coerce_seed(None)), "steps": steps,
                "width": wh[0], "height": wh[1]}
        res, code = CC.run_single(client, wf, args, out / f"img_{i}")
        if code != 0:
            print(json.dumps(res, ensure_ascii=False, indent=2)); raise SystemExit(f"gen {i} failed")
        img = res["outputs"][0]["file"]
        (out / f"img_{i}" / "prompt.txt").write_text(p, encoding="utf-8")
        H._log(f"IMAGE {i}: {img}")
        results.append({"image": img, "prompt": p, "seed": args["seed"]})
    (out / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    arg = sys.argv[1]
    content = Path(arg).read_text(encoding="utf-8") if Path(arg).exists() else arg
    generate(content, sys.argv[2] if len(sys.argv) > 2 else r"C:\FERRYMAN\work\imagehead_doc",
             int(sys.argv[3]) if len(sys.argv) > 3 else 1)
