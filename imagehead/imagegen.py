r"""Image head I5 — oracle-gated illustration (the productionized callable a job's `image`/
`illustrate` field invokes). content -> FERRYMAN-Qwen art-direction -> ComfyUI SDXL (N candidates)
-> Qwen-vision oracle -> pick the best-passing -> provenance. Reuses I2 (image_from_doc.art_direct)
+ the reused comfy_client + the I5 vision oracle.

VRAM discipline: Qwen(prompt) -> down; ComfyUI(generate N) -> stopped; Qwen-vision(verify) -> down.
Never two heavy models resident at once (the one-model-at-a-time rule, across process boundaries).

Job-field pattern (the declarative hook a future render()/`ferryman illustrate` uses):
  "image": {"from": "<content or @doc>", "n": 2}   ->  imagegen.illustrate(content, out, n) -> chosen PNG
  then feed chosen PNG to graphics (HyperFrames <img>/bg, I3) or compose_graphics (video overlay, I4).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_a_ground as H
import comfy_client as CC
import image_from_doc as IFD
import vision_verify as VV
import ferryman as F

WF = str(Path(__file__).resolve().parent / "workflows" / "sdxl_txt2img.json")
START_COMFY = str(Path(__file__).resolve().parent / "start_comfyui.cmd")


def _stop_comfy():
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue | "
                    "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
                   capture_output=True)
    time.sleep(3)


def illustrate(content: str, out_dir: str, n: int = 2) -> tuple[dict, list]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = IFD.art_direct(content, n)                       # Qwen up -> down (in art_direct)

    client = CC.ComfyClient("http://127.0.0.1:8188")
    if not client.server_up():
        raise SystemExit(f"ComfyUI not up on :8188 — start it: {START_COMFY}")
    wf = CC.unwrap_workflow(json.loads(Path(WF).read_text(encoding="utf-8")))
    cands = []
    for i, p in enumerate(prompts):
        args = {"prompt": p, "negative_prompt": IFD.NEG, "seed": CC.coerce_seed(None),
                "steps": 26, "width": 1024, "height": 1024}
        res, code = CC.run_single(client, wf, args, out / f"cand_{i}")
        if code == 0:
            cands.append({"image": res["outputs"][0]["file"], "prompt": p, "seed": args["seed"]})
    if not cands:
        raise SystemExit("no candidates generated")

    _stop_comfy()                                              # free ComfyUI VRAM before the vision oracle
    for c in cands:
        v = VV.verify(c["image"], c["prompt"])
        c["ok"], c["reason"] = v.get("ok"), v.get("reason")
        H._log(f"oracle {Path(c['image']).name}: ok={c['ok']} — {c['reason']}")

    passing = [c for c in cands if c["ok"]]
    best = (passing or cands)[0]
    best["sha256"] = F.sha256(Path(best["image"]))
    best["gate"] = "vision" if best.get("ok") else "fallback(no candidate passed)"
    (out / "chosen.json").write_text(json.dumps({"chosen": best, "candidates": cands},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    H._log(f"CHOSEN {best['image']} | ok={best.get('ok')} | {len(passing)}/{len(cands)} passed")
    return best, cands


if __name__ == "__main__":
    arg = sys.argv[1]
    content = Path(arg).read_text(encoding="utf-8") if Path(arg).exists() else arg
    illustrate(content, sys.argv[2] if len(sys.argv) > 2 else r"C:\FERRYMAN\work\imagehead_i5",
               int(sys.argv[3]) if len(sys.argv) > 3 else 2)
