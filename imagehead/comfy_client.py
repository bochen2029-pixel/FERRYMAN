#!/usr/bin/env python3
r"""
comfy_client.py — self-contained ComfyUI runner for FERRYMAN cover art.

A pure-stdlib (urllib / json / argparse / time / pathlib) drop-in replacement
for the hermes ComfyUI runner skill (run_workflow.py + run_batch.py). It exists
so the cover-art pipeline GROWS on any machine: nothing here depends on the
reference box's hermes install. The ComfyUI daemon itself + a GPU + the SDXL
checkpoint are the only external pieces (checkpoint self-downloads via
`python _tools/fetch_weights.py sdxl`).

WHY THIS FILE MATCHES cover_gen.py EXACTLY
------------------------------------------
cover_gen.py shells to two scripts and parses their stdout JSON. The contract
it depends on (extracted from cover_gen.py, do not drift):

  run_workflow.py  (cover_gen.run_single):
    argv: --workflow <path> --args <json> --output-dir <dir> --host <url>
          [--timeout <s>]
    --args JSON keys: {"prompt","negative_prompt","seed","steps"}
    stdout JSON it reads (cover_gen._extract_png / _parse_runner_json):
        {"status":"success",
         "outputs":[{"file":"<abs path>","type":"image", ...}],
         "prompt_id":"<id>"}
    -> cover_gen scans outputs[] for type=="image" then any .png/.jpg/.jpeg/.webp

  run_batch.py  (cover_gen.run_batch_shell):
    argv: --workflow <path> --args <json> --count <N> --randomize-seed
          --output-dir <dir> --host <url> [--timeout <s>]
    --args JSON keys: {"prompt","negative_prompt","steps"}   (seed per-run)
    stdout JSON it reads (cover_gen._collect_batch_pngs):
        {"status":"success"|"partial",
         "results":[{"outputs":[{"file":...}], ...}, ...]}
    -> cover_gen harvests results[].outputs[].file (also tolerates top-level
       "outputs" and keys "runs"/"batch")

PARAMETER INJECTION — by class_type, never hardcoded ids
--------------------------------------------------------
The hermes runner names its params by tracing sampler connections
(KSampler.positive/.negative -> the source CLIPTextEncode). We replicate that
so `prompt` lands on the POSITIVE encoder and `negative_prompt` on the NEGATIVE
one, regardless of node ids. Other params map by (class_type, field):
  seed   -> KSampler.seed / KSamplerAdvanced.noise_seed / RandomNoise.noise_seed
           / SamplerCustom.noise_seed
  steps  -> KSampler / KSamplerAdvanced / BasicScheduler / *Scheduler .steps
  width  -> Empty*LatentImage.width       height -> .height
  checkpoint -> CheckpointLoaderSimple.ckpt_name  (also patched by cover_gen)

MODES (single file, both entry styles)
--------------------------------------
  * Batch when: --count is present, OR argv[0] basename contains "batch",
    OR --mode batch. Otherwise single. (--mode {single,batch} forces it.)
  A sibling run_batch.py re-execs this file in batch mode, so kit_env can point
  cover_gen.run_workflow at THIS file and let cover_gen derive run_batch beside
  it — both resolve here.

Exit codes: 0 success (single ok / batch all-ok), 1 any failure/timeout.
All result JSON is printed to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 600  # generous: SDXL on modest GPUs + model load on first run
USER_AGENT = "ferryman-comfy-client/1.0"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ComfyUI seed range; -1 / None => randomize.
SEED_MAX = 2 ** 63 - 1
SEED_MIN = 0

# Sampler nodes whose positive/negative connections we trace to find the
# prompt / negative-prompt text encoders (mirrors hermes extract_schema).
SAMPLER_FAMILY = {
    "KSampler", "KSamplerAdvanced",
    "SamplerCustom", "SamplerCustomAdvanced",
    "BasicGuider", "CFGGuider", "DualCFGGuider",
}

# (class_type, field) sites for the scalar params cover_gen may pass.
SEED_SITES = [
    ("KSampler", "seed"),
    ("KSamplerAdvanced", "noise_seed"),
    ("RandomNoise", "noise_seed"),
    ("SamplerCustom", "noise_seed"),
    ("HunyuanVideoSampler", "seed"),
    ("WanVideoSampler", "seed"),
    ("Seed (rgthree)", "seed"),
    ("easy seed", "seed"),
]
STEPS_SITES = [
    ("KSampler", "steps"),
    ("KSamplerAdvanced", "steps"),
    ("BasicScheduler", "steps"),
    ("SDTurboScheduler", "steps"),
    ("HunyuanVideoSampler", "steps"),
    ("WanVideoSampler", "steps"),
]
WIDTH_SITES = [
    ("EmptyLatentImage", "width"),
    ("EmptySD3LatentImage", "width"),
    ("EmptyHunyuanLatentVideo", "width"),
]
HEIGHT_SITES = [
    ("EmptyLatentImage", "height"),
    ("EmptySD3LatentImage", "height"),
    ("EmptyHunyuanLatentVideo", "height"),
]
CHECKPOINT_SITES = [
    ("CheckpointLoaderSimple", "ckpt_name"),
    ("CheckpointLoader", "ckpt_name"),
    ("ImageOnlyCheckpointLoader", "ckpt_name"),
]

# Output nodes that carry saved images.
OUTPUT_IMAGE_KEYS = ("images", "gifs", "videos", "video", "files")


# ============================================================
# small utilities
# ============================================================
def log(msg: str) -> None:
    print(f"[comfy-client] {msg}", file=sys.stderr)


def emit(obj: dict) -> None:
    """Print the result JSON to stdout (what cover_gen parses)."""
    print(json.dumps(obj, indent=2, default=str))


def coerce_seed(value) -> int:
    """-1 / None / '-1' -> fresh random seed; else int(value)."""
    if value is None:
        return random.randint(SEED_MIN, SEED_MAX)
    if isinstance(value, str) and value.strip() == "-1":
        return random.randint(SEED_MIN, SEED_MAX)
    if value == -1:
        return random.randint(SEED_MIN, SEED_MAX)
    return int(value)


def is_link(value) -> bool:
    """True if value is a [node_id, slot] connection."""
    return (
        isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and isinstance(value[1], int)
    )


def is_api_format(wf) -> bool:
    if not isinstance(wf, dict):
        return False
    if "nodes" in wf and "links" in wf:
        return False
    return any(isinstance(v, dict) and "class_type" in v for v in wf.values())


def unwrap_workflow(payload) -> dict:
    """Return an API-format graph or raise ValueError (also drops _comment)."""
    if isinstance(payload, dict) and "prompt" in payload and is_api_format(payload.get("prompt")):
        payload = payload["prompt"]
    if not is_api_format(payload):
        if isinstance(payload, dict) and "nodes" in payload and "links" in payload:
            raise ValueError(
                "workflow is in editor format (top-level 'nodes'/'links'); "
                "re-export via 'Save (API Format)' in ComfyUI"
            )
        raise ValueError(
            "workflow is not API format — each top-level entry needs a 'class_type'"
        )
    # Keep only real nodes (drops _comment and any stray non-node key so the
    # /prompt endpoint never sees 'a node is missing class_type').
    return {k: v for k, v in payload.items()
            if isinstance(v, dict) and "class_type" in v}


def iter_nodes(wf: dict):
    for nid, node in wf.items():
        if isinstance(node, dict) and "class_type" in node:
            yield nid, node


# ============================================================
# HTTP (pure stdlib, small retry)
# ============================================================
def _request(method: str, url: str, *, data: bytes | None = None,
             headers: dict | None = None, timeout: float = 60.0):
    """One HTTP round-trip. Returns (status, body_bytes). Raises on transport error."""
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"")


def http_get_json(url: str, *, timeout: float = 30.0):
    status, body = _request("GET", url, timeout=timeout)
    if status != 200:
        return status, None
    try:
        return status, json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return status, None


def http_post_json(url: str, payload: dict, *, timeout: float = 120.0):
    body = json.dumps(payload).encode("utf-8")
    status, resp = _request("POST", url, data=body,
                            headers={"Content-Type": "application/json"},
                            timeout=timeout)
    try:
        parsed = json.loads(resp.decode("utf-8", "replace")) if resp else {}
    except json.JSONDecodeError:
        parsed = {"raw": resp.decode("utf-8", "replace")[:500]}
    return status, parsed


# ============================================================
# ComfyUI client
# ============================================================
class ComfyClient:
    def __init__(self, server: str, timeout: int = DEFAULT_TIMEOUT):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.client_id = "%032x" % random.getrandbits(128)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.server + path

    def server_up(self) -> bool:
        try:
            status, _ = _request("GET", self._url("/system_stats"), timeout=4.0)
            return status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def submit(self, workflow: dict):
        """POST /prompt. Returns (ok, prompt_id_or_None, error_dict_or_None)."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            status, body = http_post_json(self._url("/prompt"), payload, timeout=120.0)
        except (urllib.error.URLError, OSError) as e:
            return False, None, {"error": f"submit failed: {e}"}
        if status != 200:
            return False, None, {
                "error": f"/prompt returned HTTP {status}",
                "body": body,
            }
        # ComfyUI returns node_errors on a validation failure (still HTTP 200
        # in some builds, non-200 in others — handle both).
        node_errors = body.get("node_errors") if isinstance(body, dict) else None
        if node_errors:
            return False, None, {"error": "workflow validation failed",
                                 "node_errors": node_errors}
        pid = body.get("prompt_id") if isinstance(body, dict) else None
        if not pid:
            return False, None, {"error": "no prompt_id in /prompt response",
                                 "response": body}
        return True, pid, None

    def poll(self, prompt_id: str):
        """Poll /history/<id> until complete. Returns (status, outputs_or_data).

        status in {"success","error","timeout"}. On success, outputs is the raw
        ComfyUI outputs dict.
        """
        deadline = time.time() + self.timeout
        interval = 1.0
        while time.time() < deadline:
            status, data = http_get_json(self._url(f"/history/{prompt_id}"),
                                         timeout=30.0)
            if status == 200 and isinstance(data, dict):
                entry = data.get(prompt_id)
                if isinstance(entry, dict):
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        return "error", entry
                    if st.get("completed", False):
                        return "success", entry.get("outputs", {}) or {}
            time.sleep(interval)
            interval = min(6.0, interval * 1.4)
        return "timeout", {"elapsed": self.timeout}

    def download(self, filename: str, subfolder: str, ftype: str,
                 output_dir: Path) -> Path | None:
        """GET /view and stream the image into output_dir (flat). None on failure."""
        params = {"filename": filename, "subfolder": subfolder or "", "type": ftype or "output"}
        url = self._url("/view") + "?" + urlencode(params)
        # Keep a flat filename inside output_dir; de-dupe if it already exists.
        target = output_dir / Path(filename).name
        stem, suffix = target.stem, target.suffix
        i = 1
        while target.exists():
            target = output_dir / f"{stem}_{i}{suffix}"
            i += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as r:
                if r.status != 200:
                    log(f"download {filename}: HTTP {r.status}")
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
            return target
        except (urllib.error.URLError, OSError) as e:
            log(f"download {filename} failed: {e}")
            return None

    def collect_outputs(self, outputs: dict, output_dir: Path) -> list[dict]:
        """Walk the ComfyUI outputs dict, download every image, return the
        cover_gen-shaped list: [{"file","type","node_id","filename","subfolder"}].
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        found: list[dict] = []
        for node_id, node_out in (outputs or {}).items():
            if not isinstance(node_out, dict):
                continue
            for key in OUTPUT_IMAGE_KEYS:
                entries = node_out.get(key)
                if not entries:
                    continue
                if not isinstance(entries, list):
                    entries = [entries]
                for fi in entries:
                    if not isinstance(fi, dict):
                        continue
                    filename = fi.get("filename") or ""
                    if not filename:
                        continue
                    saved = self.download(
                        filename, fi.get("subfolder") or "",
                        fi.get("type") or "output", output_dir,
                    )
                    if saved is None:
                        continue
                    ftype = "image" if saved.suffix.lower() in IMAGE_EXTS else "file"
                    found.append({
                        "file": str(saved.resolve()),
                        "type": ftype,
                        "node_id": node_id,
                        "filename": filename,
                        "subfolder": fi.get("subfolder") or "",
                    })
        return found


# ============================================================
# workflow parameter injection (by class_type, connection-traced)
# ============================================================
def _trace_to_encoder(wf: dict, link, max_hops: int = 8) -> str | None:
    """Follow a [node_id, slot] link through passthroughs to the CLIPTextEncode."""
    if not is_link(link):
        return None
    nid = link[0]
    seen: set[str] = set()
    for _ in range(max_hops):
        if nid is None or nid in seen:
            return nid
        seen.add(nid)
        node = wf.get(nid)
        if not isinstance(node, dict):
            return None
        cls = node.get("class_type", "")
        if cls in {"Reroute", "PrimitiveNode", "Note"}:
            inputs = node.get("inputs", {}) or {}
            nxt = next((v for v in inputs.values() if is_link(v)), None)
            if nxt is None:
                return nid
            nid = nxt[0]
            continue
        return nid
    return nid


def _find_prompt_nodes(wf: dict) -> tuple[str | None, str | None]:
    """Return (positive_encoder_id, negative_encoder_id) by tracing a sampler's
    positive/negative (or a Guider's conditioning) back to CLIPTextEncode nodes."""
    pos = neg = None
    for _, node in iter_nodes(wf):
        if node.get("class_type") not in SAMPLER_FAMILY:
            continue
        inputs = node.get("inputs", {}) or {}
        if pos is None:
            src = _trace_to_encoder(wf, inputs.get("positive") or inputs.get("conditioning"))
            if src and _is_encoder(wf.get(src)):
                pos = src
        if neg is None:
            src = _trace_to_encoder(wf, inputs.get("negative"))
            if src and _is_encoder(wf.get(src)):
                neg = src
    return pos, neg


def _is_encoder(node) -> bool:
    if not isinstance(node, dict):
        return False
    cls = node.get("class_type", "")
    return cls.startswith("CLIPTextEncode") or cls in {
        "smZ CLIPTextEncode", "BNK_CLIPTextEncodeAdvanced",
    }


def _set_first_site(wf: dict, sites: list[tuple[str, str]], value) -> bool:
    """Set value on the first node matching any (class_type, field) in sites,
    but never overwrite a link. Returns True if something was set."""
    changed = False
    for nid, node in iter_nodes(wf):
        cls = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        for sc, sf in sites:
            if cls == sc and sf in inputs and not is_link(inputs.get(sf)):
                inputs[sf] = value
                changed = True
    return changed


def apply_overrides(workflow: dict, args: dict) -> tuple[dict, list[str]]:
    """Patch prompt / negative_prompt / seed / steps / width / height / checkpoint
    into the workflow by class_type. Returns (patched_copy, notes).

    `args` uses cover_gen's names: prompt, negative_prompt, seed, steps, width,
    height, checkpoint / ckpt_name. Extra keys are ignored (with a note).
    """
    import copy
    wf = copy.deepcopy(workflow)
    notes: list[str] = []

    pos_id, neg_id = _find_prompt_nodes(wf)

    # ---- prompt / negative_prompt onto the traced encoders ----
    if "prompt" in args and args["prompt"] is not None:
        if pos_id and isinstance(wf.get(pos_id), dict):
            wf[pos_id].setdefault("inputs", {})["text"] = args["prompt"]
        else:
            # Fallback: first CLIPTextEncode that is NOT the negative one.
            for nid, node in iter_nodes(wf):
                if _is_encoder(node) and nid != neg_id:
                    node.setdefault("inputs", {})["text"] = args["prompt"]
                    notes.append(f"prompt set on {nid} (untraced fallback)")
                    break
            else:
                notes.append("no positive CLIPTextEncode found for 'prompt'")

    if "negative_prompt" in args and args["negative_prompt"] is not None:
        if neg_id and isinstance(wf.get(neg_id), dict):
            wf[neg_id].setdefault("inputs", {})["text"] = args["negative_prompt"]
        else:
            notes.append("no negative CLIPTextEncode found for 'negative_prompt'")

    # ---- seed (expand -1 to a real value so the run is reproducible/logged) ----
    if "seed" in args and args["seed"] is not None:
        seed_val = coerce_seed(args["seed"])
        if _set_first_site(wf, SEED_SITES, seed_val):
            notes.append(f"seed={seed_val}")
        else:
            notes.append("no seed field found to set")

    # ---- steps ----
    if args.get("steps") is not None:
        if not _set_first_site(wf, STEPS_SITES, int(args["steps"])):
            notes.append("no steps field found to set")

    # ---- width / height ----
    if args.get("width") is not None:
        _set_first_site(wf, WIDTH_SITES, int(args["width"]))
    if args.get("height") is not None:
        _set_first_site(wf, HEIGHT_SITES, int(args["height"]))

    # ---- checkpoint (cover_gen also patches this itself; harmless to repeat) ----
    ckpt = args.get("checkpoint") or args.get("ckpt_name")
    if ckpt:
        _set_first_site(wf, CHECKPOINT_SITES, ckpt)

    # ---- note any unknown keys (parity with hermes 'unknown parameter' warns) ----
    known = {"prompt", "negative_prompt", "seed", "steps", "width", "height",
             "checkpoint", "ckpt_name", "cfg", "sampler_name", "scheduler"}
    for k in args:
        if k not in known:
            notes.append(f"unknown parameter {k!r} ignored")

    return wf, notes


# ============================================================
# run: single + batch
# ============================================================
def _load_workflow(path: str) -> dict:
    wf_path = Path(path).expanduser()
    if not wf_path.exists():
        raise FileNotFoundError(f"workflow file not found: {path}")
    with wf_path.open(encoding="utf-8") as f:
        return unwrap_workflow(json.load(f))


def _unreachable_error(server: str) -> dict:
    return {
        "status": "error",
        "error": f"ComfyUI server not reachable at {server}",
        "hint": "launch ComfyUI first (e.g. `comfy launch --background`, "
                "or start the ComfyUI app so it serves :8188)",
    }


def run_single(client: ComfyClient, workflow: dict, args_obj: dict,
               output_dir: Path) -> tuple[dict, int]:
    """One generation. Returns (result_json, exit_code)."""
    patched, notes = apply_overrides(workflow, args_obj)
    for n in notes:
        log(n)
    ok, pid, err = client.submit(patched)
    if not ok:
        result = {"status": "error", **(err or {})}
        _augment_submit_hint(result)
        return result, 1

    st, data = client.poll(pid)
    if st == "timeout":
        return {"status": "timeout", "prompt_id": pid,
                "elapsed": data.get("elapsed"),
                "hint": "raise --timeout or check the ComfyUI console for a stall"}, 1
    if st == "error":
        return {"status": "error", "prompt_id": pid,
                "error": "workflow execution failed", "details": data}, 1

    downloaded = client.collect_outputs(data, output_dir)
    if not downloaded:
        return {"status": "error", "prompt_id": pid,
                "error": "run completed but produced no downloadable images",
                "raw_outputs": data}, 1

    return {
        "status": "success",
        "prompt_id": pid,
        "outputs": downloaded,
        "warnings": notes,
    }, 0


def _augment_submit_hint(result: dict) -> None:
    """Add a checkpoint hint if the failure looks like a missing model."""
    blob = json.dumps(result).lower()
    if "ckpt" in blob or "checkpoint" in blob or "value not in list" in blob:
        result.setdefault(
            "hint",
            "the requested checkpoint may be missing — run "
            "`python _tools/fetch_weights.py sdxl` to download it into "
            "checkpoints_dir",
        )


def run_batch(client: ComfyClient, workflow: dict, base_args: dict,
              output_dir: Path, count: int, randomize_seed: bool) -> tuple[dict, int]:
    """N sequential runs with (optionally) a fresh seed each. Returns (json, code).

    The result shape matches what cover_gen._collect_batch_pngs harvests:
    a top-level 'results' list, each entry mirroring a single-run result.
    """
    results: list[dict] = []
    failures = 0
    for i in range(count):
        run_args = dict(base_args)
        if randomize_seed or "seed" not in run_args:
            run_args["seed"] = coerce_seed(None)
        run_dir = output_dir / f"run_{i:04d}"
        single, code = run_single(client, workflow, run_args, run_dir)
        single["index"] = i
        results.append(single)
        if code != 0:
            failures += 1
            log(f"run {i} -> {single.get('status')}: {single.get('error', '?')}")
        else:
            log(f"run {i} -> success: {len(single.get('outputs', []))} file(s)")

    status = "success" if failures == 0 else ("partial" if failures < count else "error")
    out = {
        "status": status,
        "total": count,
        "completed": count - failures,
        "failed": failures,
        "output_dir": str(output_dir.resolve()),
        "results": results,
    }
    return out, (0 if failures == 0 else 1)


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Self-contained ComfyUI runner (FERRYMAN). Drop-in for the "
                    "hermes run_workflow.py / run_batch.py contract.",
    )
    p.add_argument("--workflow", required=True, help="Path to an API-format workflow JSON.")
    p.add_argument("--args", default="{}",
                   help='JSON of overrides: {"prompt","negative_prompt","seed","steps",'
                        '"width","height","checkpoint"}. `@file.json` also accepted.')
    p.add_argument("--output-dir", default="./outputs", help="Where to save images.")
    p.add_argument("--host", "--server", dest="server", default=None,
                   help=f"ComfyUI server URL (default: kit_env or {DEFAULT_SERVER}).")
    p.add_argument("--timeout", type=int, default=0,
                   help=f"Max seconds to wait per run (0 => {DEFAULT_TIMEOUT}).")
    p.add_argument("--kit-env", default=None,
                   help="kit_env.json to read cover_gen.comfyui_server from.")
    p.add_argument("--mode", choices=["single", "batch"], default=None,
                   help="Force mode. Default: batch if --count present or argv0 has 'batch'.")
    # batch-only
    p.add_argument("--count", type=int, default=None, help="Batch: number of runs.")
    p.add_argument("--randomize-seed", action="store_true",
                   help="Batch: fresh random seed each run.")
    return p


def _resolve_server(args) -> str:
    if args.server:
        return args.server
    if args.kit_env:
        try:
            env = json.loads(Path(args.kit_env).expanduser().read_text(encoding="utf-8"))
            srv = ((env.get("cover_gen") or {}).get("comfyui_server"))
            if srv:
                return srv
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_SERVER


def _parse_args_json(raw: str) -> dict:
    if raw.startswith("@"):
        raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
    obj = json.loads(raw) if raw.strip() else {}
    if not isinstance(obj, dict):
        raise ValueError("--args must be a JSON object")
    return obj


def _detect_batch(args, argv0: str) -> bool:
    if args.mode == "batch":
        return True
    if args.mode == "single":
        return False
    if args.count is not None:
        return True
    return "batch" in Path(argv0).name.lower()


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    server = _resolve_server(args)
    timeout = args.timeout if args.timeout and args.timeout > 0 else DEFAULT_TIMEOUT
    output_dir = Path(args.output_dir).expanduser()

    # ---- parse overrides ----
    try:
        args_obj = _parse_args_json(args.args)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        emit({"status": "error", "error": f"bad --args JSON: {e}"})
        return 1

    # ---- load workflow ----
    try:
        workflow = _load_workflow(args.workflow)
    except (FileNotFoundError, ValueError) as e:
        emit({"status": "error", "error": str(e)})
        return 1

    # ---- server reachability (structured error + hint) ----
    client = ComfyClient(server, timeout=timeout)
    if not client.server_up():
        emit(_unreachable_error(server))
        return 1

    # ---- dispatch ----
    is_batch = _detect_batch(args, sys.argv[0] if argv is None else "comfy_client")
    if is_batch:
        count = args.count if args.count and args.count > 0 else 1
        result, code = run_batch(client, workflow, args_obj, output_dir,
                                 count, args.randomize_seed or True)
    else:
        result, code = run_single(client, workflow, args_obj, output_dir)

    emit(result)
    return code


if __name__ == "__main__":
    sys.exit(main())
