#!/usr/bin/env python3
"""FERRYMAN runtime — speaker-parametric talking-head factory (SPEC v0.1.0 + review deltas).

Any {script, enrolled voice, enrolled face} within spec → finished video. Nothing is
hardcoded to a speaker: jobs reference speakers/<id>/, created by the enroll verbs.

Orchestrator runs on system Python (stdlib + ffmpeg on PATH). Heavy models run
sequentially in their own venvs via subprocess (VRAM discipline §9: the process
boundary IS the teardown). Every stage is idempotent; TTS is content-addressed (B7).

CLI:
  ferryman render <job.json>            run stages 1..6 for one job
  ferryman batch                        drain jobs/inbox/ (oldest first)
  ferryman enroll-voice <spk> <audio>   one-time voice enrollment
  ferryman make-idle <spk> <src> [--driving <mp4>]   footage OR still image
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# P8a — de-hardcoded for the portable appliance. ROOT defaults to the repo root
# (parent of src/), overridable by FERRYMAN_HOME; VENVS defaults to ROOT/venvs
# (a junction to the NVMe on the dev box), overridable by FERRYMAN_VENVS. Unset,
# these resolve to C:\FERRYMAN and C:\FERRYMAN\venvs (-> C:\FERRYMAN_DATA\venvs)
# exactly as before, so existing renders stay byte-identical; set FERRYMAN_HOME to
# run the same tree from any path (USB, prod box) with zero edits.
ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
VENVS = Path(os.environ.get("FERRYMAN_VENVS") or (ROOT / "venvs"))
# QC B-11: export the RESOLVED roots so every stage subprocess inherits them explicitly.
# Stage scripts also self-derive from __file__ as a fallback — belt and suspenders.
os.environ.setdefault("FERRYMAN_HOME", str(ROOT))
os.environ.setdefault("FERRYMAN_VENVS", str(VENVS))
SPEAKERS = ROOT / "speakers"
JOBS = ROOT / "jobs"
WORK = ROOT / "work"
OUT = ROOT / "out"
LEDGER = ROOT / "ledger" / "runs.jsonl"
VENV_TTS = VENVS / "venv-tts" / "Scripts" / "python.exe"
VENV_ORACLE = VENVS / "venv-oracle" / "Scripts" / "python.exe"
VENV_MUSETALK = VENVS / "venv-musetalk" / "Scripts" / "python.exe"
VENV_INPAINT = VENVS / "venv-inpaint" / "Scripts" / "python.exe"
MUSETALK_REPO = ROOT / "vendor" / "MuseTalk"
LIVEPORTRAIT_REPO = ROOT / "vendor" / "LivePortrait"
MODELS = ROOT / "models"
# Head D — dubbing
VENV_SEEDVC = VENVS / "venv-seedvc" / "Scripts" / "python.exe"
SEEDVC_REPO = ROOT / "vendor" / "seed-vc"
NLLB_DIR = MODELS / "nllb200"
# QC A-02: earshot ships IN-TREE (earshot/ — byte-identical copy of the C:\earshot organ)
# so the dub head travels with the zip. EARSHOT_PY still overrides for a custom install.
# Its whisper.cpp engine + ggml models are substrate (manifests/substrate.json).
EARSHOT = Path(os.environ.get("EARSHOT_PY") or (ROOT / "earshot" / "earshot.py"))
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
TTS_CACHE = WORK / "_cache" / "tts"
CJK_FONT_NAME = "Microsoft YaHei"
CJK_FONT_FILE = os.environ.get("FERRYMAN_CJK_FONT") or r"C:/Windows/Fonts/msyh.ttc"

SENT_END = "。！？!?…"
UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_XET": "1"}

import tomllib  # noqa: E402  (py3.11+; orchestrator runs on system python)

def _load_cfg() -> dict:
    p = ROOT / "config" / "pipeline.toml"
    try:
        return tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception as e:  # noqa: BLE001
        print(f"[ferryman] WARNING: pipeline.toml unreadable ({e}); using defaults")
        return {}

CFG = _load_cfg()
ORACLE_CFG = CFG.get("oracles", {})


def _load_cloud() -> dict:
    p = ROOT / "config" / "cloud.toml"
    try:
        return tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception as e:  # noqa: BLE001
        print(f"[ferryman] WARNING: cloud.toml unreadable ({e}); cloud tier unavailable")
        return {}


CLOUD_CFG = _load_cloud()

# QC 2026-07-13 (A-25): single source of truth for oracle thresholds — render and dub
# must degrade identically when pipeline.toml is absent. Fail-closed posture (A-01/A-09/
# B-09): an oracle that CANNOT RUN refuses to certify; it never silently passes. The one
# escape hatch is FERRYMAN_ALLOW_UNVERIFIED=1, and every use of it is logged loudly.
ORACLE_DEFAULTS = {"audio_cer_max": 0.05, "speaker_sim_min": 0.75,
                   "speaker_sim_enforce": True, "cer_error_floor_chars": 1,
                   "av_len_delta_ms_max": 100}


def oracle_cfg(key: str):
    return ORACLE_CFG.get(key, ORACLE_DEFAULTS[key])


ALLOW_UNVERIFIED = os.environ.get("FERRYMAN_ALLOW_UNVERIFIED") == "1"


def log(msg: str) -> None:
    print(f"[ferryman] {msg}", flush=True)


_NVENC_OK: bool | None = None


def video_codec(job: "Job") -> tuple[str, list[str]]:
    """Resolve the video encoder + quality args. REM-4.1 (R-09): doctor's 'CPU x264
    fallback' text is now TRUE — if the job wants h264_nvenc and this ffmpeg lacks it,
    fall back to libx264 loudly instead of dying at encode time."""
    global _NVENC_OK
    want = job.output.get("codec", "h264_nvenc")
    if want != "h264_nvenc":
        return want, (["-cq", "23"] if "nvenc" in want else ["-crf", "20"])
    if _NVENC_OK is None:
        cp = run(["ffmpeg", "-hide_banner", "-encoders"])
        _NVENC_OK = cp.returncode == 0 and "h264_nvenc" in (cp.stdout or "")
    if _NVENC_OK:
        return "h264_nvenc", ["-cq", "23"]
    log("encode: h264_nvenc unavailable on this box — falling back to libx264 -crf 20 (CPU, slow)")
    return "libx264", ["-crf", "20"]


def run(cmd: list[str], cwd: Path | None = None, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **UTF8_ENV, **(extra_env or {})}
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def must(cp: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    if cp.returncode != 0:
        tail = (cp.stderr or cp.stdout or "")[-2000:]
        raise RuntimeError(f"{what} failed (exit {cp.returncode}):\n{tail}")
    return cp


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_duration(path: Path) -> float:
    cp = must(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", path]), f"ffprobe {path.name}")
    return float(cp.stdout.strip())


def probe_stream(path: Path, kind: str, entry: str) -> str:
    cp = must(run(["ffprobe", "-v", "error", "-select_streams", kind[0] + ":0",
                   "-show_entries", f"stream={entry}", "-of", "csv=p=0", path]),
              f"ffprobe {path.name}")
    return cp.stdout.strip().split("\n")[0].strip()


def vram_guard(min_free_gb: float | None = None) -> None:
    if min_free_gb is None:   # QC E-10: threshold from config, hardcoded fallback identical
        min_free_gb = float(CFG.get("vram", {}).get("min_free_gb_before_load", 1.5))
    cp = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    if cp.returncode != 0:
        # QC A-27: never guard silently — a missing/failed nvidia-smi means the next heavy
        # load runs blind and OOMs deep in a stage instead of failing here with context.
        log("vram_guard: nvidia-smi unavailable — proceeding UNGUARDED (heavy load may OOM)")
        return
    free_gb = int(cp.stdout.strip().split("\n")[0]) / 1024   # GPU0 by design (single-GPU boxes)
    if free_gb < min_free_gb:
        raise RuntimeError(f"vram_guard: only {free_gb:.1f} GB free (< {min_free_gb})")
    log(f"vram_guard: {free_gb:.1f} GB free")


# ---------------------------------------------------------------------- job spec (§6)
@dataclass
class Job:
    job_id: str
    speaker: str
    script: str = ""               # video: the text to speak; dub jobs have none
    lang: str = "zh"
    tier: str = "T1"               # provenance-only today (QC E-13): T2/T3/hero change NOTHING —
    #                                all tiers render via MuseTalk; hero tier is not wired

    voice_engine: str = "indextts2"
    face_model: str = "musetalk"
    idle: str = "auto"              # auto | hi | src | <path>
    captions: bool = True
    label: bool = True              # C6 — AIGC mark; False only for private outputs
    pinyin_overrides: dict = field(default_factory=dict)   # C3 — INERT (QC E-18): only cosyvoice3
    #                                would consume it (deferred, Q4); IndexTTS2 ignores it. For zh
    #                                proper-noun scoring use speakers/<id>/lexicon.json instead.
    seed: int = 1234
    graphics: dict = field(default_factory=dict)           # P7 Head G — OFF unless {"enabled":true,"cues":[...]}
    compute: str = "auto"          # v0.2 contract §9: auto (tier decides) | local | cloud (gated by cloud_preflight)
    target: str = "video"          # video (default; render()) | dub (Head D). audio|infographic|motion = future heads
    dub: dict = field(default_factory=dict)                # Head D: {"mode":"same-lang"|"cross-lingual","source",
    #                                                        "source_lang","target_lang","diffusion_steps","sim_min","cer_max"}
    output: dict = field(default_factory=lambda: {
        "codec": "h264_nvenc", "fps": 25, "res": "source", "audio": "aac_48k"})

    @staticmethod
    def load(path: Path) -> "Job":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "script_path" in d:
            sp = Path(d.pop("script_path"))
            if not sp.is_absolute():
                # QC A-23: relative script_path resolves against the JOB FILE's dir, then
                # ROOT — never the caller's CWD (portable jobs on any box).
                cand = Path(path).parent / sp
                sp = cand if cand.exists() else (ROOT / sp)
            d["script"] = sp.read_text(encoding="utf-8")
        known = {f.name for f in Job.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            # WP-5: a typo'd near-miss ("graphic" for "graphics") silently disables a
            # feature and still renders GREEN — warn loudly when the intent is guessable.
            import difflib
            plain = []
            for k in sorted(extra):
                m = difflib.get_close_matches(k, known, n=1, cutoff=0.7)
                if m:
                    log(f"job: WARNING unknown field {k!r} IGNORED — did you mean {m[0]!r}?")
                else:
                    plain.append(k)
            if plain:
                log(f"job: ignoring unknown fields {plain}")
        return Job(**{k: v for k, v in d.items() if k in known})

    def spk_dir(self) -> Path:
        d = SPEAKERS / self.speaker
        if not d.exists():
            raise FileNotFoundError(f"speaker not enrolled: {self.speaker} ({d})")
        return d

    def idle_base(self) -> Path:
        d = self.spk_dir()
        if self.idle not in ("auto", "hi", "src"):
            p = Path(self.idle)
            if p.exists():
                return p
            raise FileNotFoundError(f"idle base not found: {p}")
        order = {"auto": ["idle_hi.mp4", "idle_src.mp4", "idle_loop.mp4"],
                 "hi": ["idle_hi.mp4"], "src": ["idle_src.mp4"]}[self.idle]
        for name in order:
            if (d / name).exists():
                return d / name
        raise FileNotFoundError(f"no idle base for {self.speaker} (looked for {order}; run make-idle)")


# ------------------------------------------------ compute router preflight (v0.2 contract §9, pre-wired)
def cloud_preflight(job: "Job") -> tuple[bool, list[str]]:
    """The §9 fail-closed ladder, pre-wired so the operator gates are PLUG-AND-PLAY: every
    unmet gate is named with its socket, so when the values land it's flip-and-go. Order:
    toggle -> secure tier -> charter v1.3 -> the SPEAKER'S OWN consent -> budget caps -> key."""
    unmet: list[str] = []
    cc = CLOUD_CFG.get("cloud", {})
    if not cc.get("enabled"):
        unmet.append("config/cloud.toml [cloud].enabled=false — flip LAST, after provider DPA + key + caps review")
    if str(cc.get("tier", "")) != "secure-cloud":
        unmet.append("cloud.tier must be 'secure-cloud' — single-tenant only for biometric payloads")
    gov = CFG.get("governance", {})
    if not gov.get("charter_v13_signed"):
        unmet.append("pipeline.toml [governance].charter_v13_signed=false — needs the operator's "
                     "explicit sign-off of the project charter's cloud carve-out")
    try:
        prof = json.loads((SPEAKERS / job.speaker / "profile.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prof = {}
    if not prof.get("cloud_consent"):
        unmet.append(f"speakers/{job.speaker}/profile.json cloud_consent=false — the SPEAKER'S own "
                     "consent (not the operator's); fill cloud_consent_utc/_note when given")
    b = cc.get("budget", {})
    if not (b.get("max_usd_per_job") and b.get("max_usd_per_month")):
        unmet.append("cloud.budget caps unset in config/cloud.toml")
    key_env = str(cc.get("api_key_env") or "RUNPOD_API_KEY")
    if not os.environ.get(key_env):
        unmet.append(f"env var {key_env} is unset — set the RunPod API key there (never in a file)")
    return (not unmet, unmet)


def route_compute(job: "Job") -> str:
    """Resolve the executor for the face stage. T3/cloud jobs run the FULL §9 ladder now —
    a refusal names every remaining gate; once all gates are green only the D4 executor
    remains. T2 refuses until D2's face_render lands. 'hero' logs + renders T1 (back-compat)."""
    wants_cloud = job.compute == "cloud" or (job.compute == "auto" and job.tier == "T3")
    if wants_cloud:
        ok, unmet = cloud_preflight(job)
        if not ok:
            raise RuntimeError("cloud render REFUSED (fail-closed, v0.2 contract §9). Unmet gates:\n  - "
                               + "\n  - ".join(unmet)
                               + "\n  (live checklist any time: `ferryman preflight-cloud <speaker>`)")
        raise NotImplementedError(
            "ALL §9 GATES GREEN — only the cloud executor remains (D4: face_render adapter + "
            "RunPod Secure Cloud backend per manifests/cloud-image). Nothing else blocks T3.")
    if job.tier == "T2":
        raise NotImplementedError("tier T2 (LatentSync 512² quality) lands with D2's face_render() "
                                  "adapter — render T1 today, or wait for V2-P2")
    if job.tier not in ("T1", "T3", "hero"):
        log(f"tier {job.tier!r} unknown — rendering T1 (recognized: T1|T2|T3|hero)")
    elif job.tier == "hero":
        log("tier 'hero' is not a distinct recipe yet (v0.2 contract §4: generator inserts only) — rendering T1")
    return "local"


# ---------------------------------------------------------------------- 1 · segment (§7.1)
def segment_script(script: str, max_chars: int = 120) -> list[str]:
    """CJK-aware sentence segmentation; merges to <=max_chars chunks (validated in the
    TalkingHead skeleton). Strips markdown-ish headers and stage directions in （）brackets
    only when they are a full line."""
    lines = []
    for raw in script.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("**") and s.endswith("**"):
            continue
        if (s.startswith("（") and s.endswith("）")) or (s.startswith("(") and s.endswith(")")):
            continue
        s = s.replace("\\-", "-").replace("\\[", "[").replace("\\]", "]")
        lines.append(s)
    text = "".join(lines)
    import re
    text = re.sub(r"[\[【]\d+[\]】]\s*", "", text)   # strip citation markers like [5] / 【5】 (QC A-18)
    sents, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            sents.append(buf)
            buf = ""
    if buf.strip():
        sents.append(buf)
    segs, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > max_chars:
            segs.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        segs.append(cur)
    out = [s for s in (x.strip() for x in segs) if s]
    if not out:
        # QC A-18: fail HERE with a clear message, not deep in ffmpeg with an empty concat
        raise RuntimeError("segment_script: no speakable text after filtering "
                           "(script is empty or only headers/stage directions)")
    return out


# ---------------------------------------------------------------------- 2 · tts (§7.2, B7 cache)
def tts_segments(job: Job, segs: list[str], workdir: Path,
                 take_overrides: dict[int, int] | None = None) -> list[Path]:
    # 'take' feeds only the cache key: IndexTTS2 sampling is stochastic, so a new
    # take = a fresh roll. The oracle retry path bumps takes for failing segments.
    take_overrides = take_overrides or {}
    # REVENUE TRIPWIRE (licenses.md, now CODE): the whole voice line runs on IndexTTS2's
    # NON-COMMERCIAL license under the nonprofit stance. The moment the operator's ruling
    # sets governance.revenue=true, this chokepoint (all heads TTS through here) refuses
    # with the pre-scoped Lane-B escape — no silent license drift possible.
    if CFG.get("governance", {}).get("revenue") and job.voice_engine == "indextts2":
        raise RuntimeError(
            "LICENSE TRIPWIRE: [governance].revenue=true but voice_engine=indextts2 (Bilibili "
            "NON-COMMERCIAL). Lane B (v0.2 contract §5-lanes): re-bake CosyVoice3 (Apache) / GPT-SoVITS "
            "(MIT) — the swap seam is this one TTS stage + a sim re-calibration — or obtain the "
            "Bilibili commercial license (indexspeech@bilibili.com), or correct revenue=false.")
    if job.voice_engine != "indextts2":
        raise NotImplementedError(f"voice_engine {job.voice_engine} not wired (Q4: cosyvoice3 deferred)")
    ref = job.spk_dir() / "ref.wav"
    ref_sha = sha256(ref)[:16]
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    # QC A-06 + REM-4.10 honesty: the key carries lang + a model-rev token. NOTE the real
    # scope — IndexTTS2 never RECEIVES `lang` (it infers language from the text itself), so
    # the lang component is provenance, not behavior; text identity is what keys the wave.
    # A rev bump (weight update) invalidates the cache by design.
    rev = str(CFG.get("models", {}).get("indextts2", {}).get("rev", "local"))
    outs, todo = [], []
    for i, text in enumerate(segs):
        take = take_overrides.get(i, job.seed)
        key = hashlib.sha256(
            f"indextts2@{rev}|{job.lang}|{job.speaker}|{ref_sha}|{take}|{text}"
            .encode("utf-8")).hexdigest()[:24]
        cached = TTS_CACHE / f"{key}.wav"
        outs.append(cached)
        if not cached.exists():
            todo.append({"text": text, "out": str(cached), "seed": take})
    log(f"tts: {len(segs)} segments, {len(todo)} to render ({len(segs)-len(todo)} cached)")
    if todo:
        vram_guard()
        task = workdir / "tts_task.json"
        task.write_text(json.dumps({"ref": str(ref), "items": todo}, ensure_ascii=False), encoding="utf-8")
        must(run([VENV_TTS, ROOT / "src" / "stage_tts_indextts2.py", task]), "tts stage")
    for p in outs:
        if not p.exists():
            raise RuntimeError(f"tts oracle: missing segment wav {p}")
    return outs


# -------------------------------------------------- 2a · segment oracles (§10, C5+B2) + retry
def oracle_segments(job: Job, segs: list[str], seg_wavs: list[Path], workdir: Path) -> dict:
    """FireRedASR2 CER + WeSpeaker similarity per segment; failing segments get ONE
    re-render (fresh take) and keep the better result. Returns aggregate metrics."""
    if not VENV_ORACLE.exists():
        if ALLOW_UNVERIFIED:
            log("oracle: venv-oracle NOT PRESENT — proceeding UNVERIFIED "
                "(FERRYMAN_ALLOW_UNVERIFIED=1; metrics null; final status will NOT say GREEN)")
            return {"cer_max": None, "cer_mean": None, "sim_min": None, "sim_mean": None,
                    "items": None, "verified": False}
        raise RuntimeError(
            f"oracle: venv-oracle not present ({VENV_ORACLE}) — refusing to render unverified "
            "speech (fail-closed; QC A-01). Install venv-oracle, or set "
            "FERRYMAN_ALLOW_UNVERIFIED=1 to accept unverified output.")
    ref = job.spk_dir() / "ref.wav"

    lex_path = job.spk_dir() / "lexicon.json"
    lexicon = json.loads(lex_path.read_text(encoding="utf-8")) if lex_path.exists() else []

    cer_max = float(oracle_cfg("audio_cer_max"))
    sim_min = float(oracle_cfg("speaker_sim_min"))
    sim_enforce = bool(oracle_cfg("speaker_sim_enforce"))
    floor_chars = float(oracle_cfg("cer_error_floor_chars"))

    def grade(pairs: list[tuple[int, Path]], tag: str) -> list[dict]:
        task = workdir / f"oracle_{tag}.json"
        task.write_text(json.dumps(
            {"ref_wav": str(ref), "lexicon": lexicon,
             "items": [{"idx": i, "wav": str(w), "text": segs[i]} for i, w in pairs]},
            ensure_ascii=False), encoding="utf-8")
        vram_guard()
        must(run([VENV_ORACLE, ROOT / "src" / "stage_oracle.py", task]), f"oracle stage ({tag})")
        result = json.loads(Path(str(task) + ".result").read_text(encoding="utf-8"))
        # QC B-09/B-14: ref_emb_ok is tri-state (None=never attempted, True, False). False
        # while sim is enforced = the speaker gate cannot run; refuse, don't silently skip.
        if sim_enforce and result.get("ref_emb_ok") is False and not ALLOW_UNVERIFIED:
            raise RuntimeError("oracle: speaker embedding failed (ref_emb_ok=false) while "
                               "speaker_sim_enforce=true — refusing to certify (QC B-09)")
        items = result["items"]
        # QC A-24: defend the positional contract — the stage echoes idx; verify it.
        for k, it in enumerate(items):
            if it.get("idx") is not None and it["idx"] != pairs[k][0]:
                raise RuntimeError(f"oracle: result order mismatch at {k} (idx {it['idx']} != "
                                   f"{pairs[k][0]}) — refusing misaligned grading")
        return items

    res = grade(list(enumerate(seg_wavs)), "pass1")
    if sim_enforce and res and all(r.get("sim") is None for r in res):
        if not ALLOW_UNVERIFIED:
            raise RuntimeError("oracle: speaker-sim unavailable for EVERY segment while "
                               "speaker_sim_enforce=true — refusing to certify (QC B-09)")
        log("oracle: speaker-sim unavailable for every segment — proceeding UNVERIFIED (flag set)")

    def seg_pass(r: dict) -> bool:
        # short segments get an absolute floor: one char of ASR ambiguity must not
        # fail a 16-char sign-off (a real garble produces multiple char errors)
        ref_len = max(len(r.get("ref_norm") or ""), 1)
        thresh = max(cer_max, floor_chars / ref_len)
        if r["cer"] > thresh:
            return False
        return not (sim_enforce and r["sim"] is not None and r["sim"] < sim_min)

    def bad_idx() -> list[int]:
        return [i for i, r in enumerate(res) if not seg_pass(r)]

    bad = bad_idx()
    if bad:
        log(f"oracle: segments {bad} failed (cer>{cer_max}"
            + (f" or sim<{sim_min}" if sim_enforce else "") + ") — re-rendering fresh takes")
        # QC A-07: takes are a DETERMINISTIC ladder (seed+1000) — a re-run of the same job
        # reproduces the same takes (ledger/reproducibility doctrine). Bump job.seed for a
        # genuinely new roll.
        wavs2 = tts_segments(job, segs, workdir, take_overrides={i: job.seed + 1000 for i in bad})
        res2 = grade([(i, wavs2[i]) for i in bad], "pass2")
        for (i, r2) in zip(bad, res2):
            # QC B-15: keep the take that PASSES the whole gate (CER and sim); between two
            # takes on the same side of the gate, keep the lower CER. Never discard a
            # sim-fixing retake because its CER is marginally worse.
            r1 = res[i]
            keep2 = ((seg_pass(r2) and not seg_pass(r1))
                     or (seg_pass(r2) == seg_pass(r1) and r2["cer"] <= r1["cer"]))
            if keep2:
                res[i] = r2
                seg_wavs[i] = wavs2[i]
        still = bad_idx()
        if still:
            worst = {i: {"cer": res[i]["cer"], "sim": res[i]["sim"]} for i in still}
            raise RuntimeError(f"oracle: segments still failing after retry {worst} — flag to "
                               "human (§10). Takes are deterministic per seed; bump job.seed "
                               "for a fresh roll.")
    cers = [r["cer"] for r in res]
    sims = [r["sim"] for r in res if r["sim"] is not None]
    agg = {"cer_max": max(cers), "cer_mean": round(sum(cers) / len(cers), 4),
           "sim_min": min(sims) if sims else None,
           "sim_mean": round(sum(sims) / len(sims), 4) if sims else None,
           "items": [{"cer": r["cer"], "sim": r["sim"]} for r in res],
           "verified": True}
    log(f"oracle: cer max/mean {agg['cer_max']}/{agg['cer_mean']} | sim min/mean {agg['sim_min']}/{agg['sim_mean']}")
    return agg


# ------------------------------------------------------- 2b · concat + mastering (§7.2b, B4)
def concat_audio(seg_wavs: list[Path], workdir: Path, pause_ms: int | None = None,
                 lufs: float | None = None, trim_silence: bool = True
                 ) -> tuple[Path, list[tuple[float, float]]]:
    """Silence-trim each segment, insert deliberate pauses, one soxr resample to 48 kHz,
    one loudnorm pass. Returns (master.wav, [(start_s, end_s) per segment]) — the exact
    offsets that make script-sourced captions (B1) free.
    trim_silence=False preserves the source's own pauses (dub/VC path, QC A-08).
    pause_ms / lufs default from pipeline.toml [audio] (QC E-02)."""
    acfg = CFG.get("audio", {})
    pause_ms = int(acfg.get("pause_ms_sentence", 300)) if pause_ms is None else pause_ms
    lufs = float(acfg.get("loudnorm_lufs", -16.0)) if lufs is None else lufs
    audio_dir = workdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trimmed, spans, t = [], [], 0.0
    for i, w in enumerate(seg_wavs):
        tw = audio_dir / f"trim_{i:03d}.wav"
        if trim_silence:
            must(run(["ffmpeg", "-v", "error", "-i", w, "-af",
                      "silenceremove=start_periods=1:start_threshold=-45dB,"
                      "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse",
                      "-ar", "48000", "-ac", "1", "-y", tw]), f"trim seg {i}")
            # QC A-08: a very quiet segment can trim to (near-)zero — fall back to untrimmed.
            # A fully-trimmed wav probes as 'N/A' (no audio stream duration) → treat as 0.
            try:
                trimmed_d = probe_duration(tw)
            except ValueError:
                trimmed_d = 0.0
            if trimmed_d < 0.05:
                log(f"concat: segment {i} trimmed to <0.05s — using untrimmed original (QC A-08)")
                must(run(["ffmpeg", "-v", "error", "-i", w, "-ar", "48000", "-ac", "1",
                          "-y", tw]), f"resample seg {i}")
        else:
            must(run(["ffmpeg", "-v", "error", "-i", w, "-ar", "48000", "-ac", "1",
                      "-y", tw]), f"resample seg {i}")
        d = probe_duration(tw)
        spans.append((t, t + d))
        t += d + pause_ms / 1000.0
        trimmed.append(tw)
    silence = audio_dir / "pause.wav"
    must(run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
              "anullsrc=r=48000:cl=mono", "-t", f"{pause_ms/1000:.3f}", "-y", silence]), "pause gen")
    lst = audio_dir / "concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for i, tw in enumerate(trimmed):
            f.write(f"file '{tw.as_posix()}'\n")
            if i < len(trimmed) - 1:
                f.write(f"file '{silence.as_posix()}'\n")
    raw = audio_dir / "master_raw.wav"
    must(run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
              "-c", "copy", "-y", raw]), "concat")
    master = workdir / "master.wav"
    must(run(["ffmpeg", "-v", "error", "-i", raw, "-af",
              f"loudnorm=I={lufs}:TP=-1.5:LRA=11", "-ar", "48000", "-y", master]), "loudnorm")
    log(f"master: {probe_duration(master):.1f}s, {len(seg_wavs)} segments, pauses {pause_ms}ms, {lufs} LUFS")
    return master, spans


# ---------------------------------------------------------------- 3a · lipsync (§7.3a + tail fix)
def lipsync_inpaint(job: Job, master: Path, workdir: Path) -> Path:
    if job.face_model != "musetalk":
        raise NotImplementedError(f"face_model {job.face_model} not wired yet (latentsync = P2 fallback slot)")
    base = job.idle_base()
    audio_d = probe_duration(master)
    base_d = probe_duration(base)
    looped = base
    if base_d < audio_d + 2:
        loops = int((audio_d + 4) // base_d) + 1
        looped = workdir / "base_looped.mp4"
        must(run(["ffmpeg", "-v", "error", "-stream_loop", str(loops), "-i", base,
                  "-c", "copy", "-t", f"{audio_d + 3:.2f}", "-y", looped]), "base loop")
    res_dir = workdir / "lipsync"
    # REM-1.1 (R-01): the resume sentinel binds IDENTITY, not just completion (QC A-19).
    # A re-drop of the same job_id with an edited script — or another speaker — must never
    # reuse the old mouth track (that shipped a desynced video ALL-GREEN). Legacy "ok"
    # sentinels fail the check and re-render once.
    master_sha = sha256(master)
    idle_sha = sha256(base)

    def _sentinel_ok(cand: Path) -> bool:
        try:
            d = json.loads(Path(str(cand) + ".done").read_text(encoding="utf-8"))
            return d.get("master_sha256") == master_sha and d.get("idle_sha256") == idle_sha
        except (OSError, ValueError, AttributeError):
            return False   # missing, torn, or legacy "ok" sentinel → not identity-verified

    existing = sorted((res_dir / "v15").glob("*.mp4"), key=lambda p: p.stat().st_mtime) \
        if (res_dir / "v15").exists() else []
    if existing:
        cand = existing[-1]
        if _sentinel_ok(cand) and probe_duration(cand) >= audio_d - 0.05:
            log(f"lipsync: reusing {cand.name} (identity-verified resume; covers {audio_d:.1f}s)")
            return cand
        log("lipsync: existing render unusable (sentinel missing/legacy/identity-mismatch, or "
            "shorter than the new master) — re-rendering")
    yaml = workdir / "task.yaml"
    yaml.write_text(
        f'task_0:\n video_path: "{looped.as_posix()}"\n audio_path: "{master.as_posix()}"\n',
        encoding="utf-8")
    vram_guard()
    log(f"lipsync: musetalk v1.5 on {base.name} ({probe_stream(base,'video','width')}x"
        f"{probe_stream(base,'video','height')}), audio {audio_d:.1f}s")
    must(run([VENV_MUSETALK, "-m", "scripts.inference",
              "--inference_config", yaml, "--result_dir", res_dir,
              "--unet_model_path", r".\models\musetalkV15\unet.pth",
              "--unet_config", r".\models\musetalkV15\musetalk.json",
              "--whisper_dir", r".\models\whisper",
              "--version", "v15", "--use_float16"], cwd=MUSETALK_REPO), "musetalk")
    vids = sorted((res_dir / "v15").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        raise RuntimeError("lipsync oracle: no output mp4 (exit codes lie; artifacts don't)")
    Path(str(vids[-1]) + ".done").write_text(   # QC A-19 completion + REM-1.1 identity sentinel
        json.dumps({"master_sha256": master_sha, "idle_sha256": idle_sha}), encoding="utf-8")
    return vids[-1]


# -------------------------------------------- 3a' · lip-sync confidence oracle (V2-P1, REM-2.1)
SYNCNET_STAGE = ROOT / "src" / "stage_syncnet.py"


def syncnet_score(video: Path, audio: Path, workdir: Path) -> dict | None:
    """REPORT-ONLY sync-confidence (SyncNet on stable_syncnet.pt). Never gates, never raises —
    returns {"mean","p10","min","windows"} or None (unavailable/failed, logged). Runs in
    venv-musetalk (librosa+einops live there). Builder falsifier 2026-07-19: real clip 0.401
    vs ±400ms-desynced 0.10-0.21 with a 0ms-remux control at 0.399 — the score measures sync.
    Enforcement waits for multi-clip calibration (provisional flag: mean<0.25 or p10<0.08)."""
    if not (SYNCNET_STAGE.exists() and VENV_MUSETALK.exists()):
        return None
    task = workdir / "syncnet_task.json"
    task.write_text(json.dumps({"video": str(video), "audio": str(audio), "windows": 24}),
                    encoding="utf-8")
    cp = run([VENV_MUSETALK, SYNCNET_STAGE, task])
    if cp.returncode != 0:
        log("syncnet: unavailable on this clip — sync_conf unrecorded "
            f"({(cp.stderr or cp.stdout or '')[-160:]})")
        return None
    try:
        r = json.loads(Path(str(task) + ".result").read_text(encoding="utf-8"))
        return {"mean": r.get("sync_conf_mean"), "p10": r.get("sync_conf_p10"),
                "min": r.get("sync_conf_min"), "windows": r.get("windows_scored")}
    except (OSError, ValueError):
        log("syncnet: result unreadable — sync_conf unrecorded")
        return None


# ------------------------------------------------ 3b · graphics compositor (P7 Head G — TOGGLEABLE)
def compose_graphics(job: "Job", video: Path, spans: list, workdir: Path) -> tuple[Path, list]:
    """OPTIONAL graphics layer. **No-op unless `job.graphics.enabled` is true** — existing
    renders are byte-for-byte unchanged. When on, composites pre-rendered graphic *cues*
    onto the lipsynced video BEFORE captions/label burn (this stage MUST live here: it binds
    to the mastering stage's live segment `spans`, which a finished/purged video can't supply).

    Each cue (in job.graphics.cues):
      mode        "pip" (alpha .mov overlaid, host stays) | "fullscreen" (opaque .mp4 cutaway)
      file        pre-rendered graphic (tier-1 Director = author renders the HyperFrames HTML
                  to a file at the job's fps/res, then references it here); relative to FERRYMAN_HOME
      span        one of:  {"seg": i}  |  {"seg_start": i, "seg_end": j}  |  {"start": s, "end": e}
      pos         overlay x:y (default "0:0"; our comps are full-canvas so the card is placed in-HTML)
    Returns (video_out, provenance[]) — provenance is [] when off, and is hashed into the ledger.
    New oracles here: composite must not change duration or resolution; every cue file must exist."""
    g = getattr(job, "graphics", None) or {}
    if not g.get("enabled") or not g.get("cues"):
        return video, []

    def span_of(cue: dict) -> tuple[float, float]:
        if "start" in cue and "end" in cue:
            return float(cue["start"]), float(cue["end"])
        if "seg" in cue:
            a, b = spans[int(cue["seg"])]
            return float(a), float(b)
        return float(spans[int(cue["seg_start"])][0]), float(spans[int(cue["seg_end"])][1])

    inputs, parts, last, idx, prov = ["-i", str(video)], [], "[0:v]", 1, []
    for cue in g["cues"]:
        a, b = span_of(cue)
        gf = cue.get("file")
        if not gf:
            raise RuntimeError(f"graphics oracle: cue missing 'file' (pre-render the project): {cue}")
        gpath = Path(gf) if Path(gf).is_absolute() else (ROOT / gf)
        if not gpath.exists():
            raise RuntimeError(f"graphics oracle: cue file not found: {gpath}")
        pos = cue.get("pos", "0:0")
        inputs += ["-i", str(gpath)]
        # shift each graphic so its frame 0 lands at the span start, then gate it to [a,b]
        parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{a:.3f}/TB[g{idx}]")
        parts.append(f"{last}[g{idx}]overlay={pos}:enable='between(t,{a:.3f},{b:.3f})':eof_action=pass[v{idx}]")
        last = f"[v{idx}]"
        prov.append({"mode": cue.get("mode"), "file": gf,
                     "span": [round(a, 3), round(b, 3)], "sha256": sha256(gpath)})
        idx += 1

    out = workdir / "composited.mp4"
    dur0, w0, h0 = probe_duration(video), probe_stream(video, "video", "width"), probe_stream(video, "video", "height")
    # QC A-04: composite VIDEO-ONLY (-an). finalize muxes master.wav as the authoritative
    # audio; mapping 0:a here crashed when the lipsync output carried no audio stream and
    # copied stale (pre-master) audio when it did — which finalize discarded anyway.
    vcodec, vq = video_codec(job)
    must(run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(parts),
              "-map", last, "-an",
              "-c:v", vcodec, *vq, "-pix_fmt", "yuv420p",
              str(out)]), "graphics composite")
    if abs(probe_duration(out) - dur0) > 0.15:
        raise RuntimeError(f"graphics oracle: composite duration {probe_duration(out):.2f}s != base {dur0:.2f}s")
    if (probe_stream(out, "video", "width"), probe_stream(out, "video", "height")) != (w0, h0):
        raise RuntimeError("graphics oracle: composite changed resolution")
    log(f"graphics: composited {len(prov)} cue(s) {[p['mode'] for p in prov]} (before captions/label)")
    return out, prov


# ------------------------------------------------ 5 · captions from script (B1) — sentence ASS
def build_ass(segs: list[str], spans: list[tuple[float, float]], out: Path,
              play_w: int, play_h: int, captions: bool = True, label: bool = False,
              total_dur: float | None = None) -> Path:
    """One ASS carries both the captions (B1) and the AI生成 mark (C6) — a single
    subtitle filter, no drawtext, no font-path escaping. libass finds installed
    system fonts by name on Windows (DirectWrite)."""
    def ts(sec: float) -> str:
        # QC A-16: integer centiseconds — float formatting could roll 59.999 into "60.00"
        cs = max(0, round(sec * 100))
        h, rem = divmod(cs, 360000)
        m, rem = divmod(rem, 6000)
        s, c = divmod(rem, 100)
        return f"{h}:{m:02d}:{s:02d}.{c:02d}"
    fs = max(28, int(play_h * 0.055))
    fs_mark = max(20, int(play_h * 0.030))
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        # QC A-17: WrapStyle 0 = libass smart-wraps long CJK lines within the margins;
        # WrapStyle 2 never wraps, so a 100-char segment ran straight off-frame.
        f"PlayResX: {play_w}\nPlayResY: {play_h}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: zh,{CJK_FONT_NAME},{fs},&H00FFFFFF,&H00101010,&H88000000,0,2,1,2,30,30,{int(play_h*0.04)},1\n"
        f"Style: mark,{CJK_FONT_NAME},{fs_mark},&H50FFFFFF,&H00101010,&H70000000,0,0,0,9,20,20,{int(play_h*0.02)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    lines = [head]
    if label:
        # QC A-15: `is not None` (0.0 is falsy) + a floor so the mandated AIGC mark can
        # never collapse to a zero-length cue on a degenerate render.
        end = total_dur if total_dur is not None else (spans[-1][1] if spans else 0.0)
        end = max(end, 0.5)
        lines.append(f"Dialogue: 1,{ts(0)},{ts(end)},mark,AI生成\n")
    if captions:
        for text, (a, b) in zip(segs, spans):
            clean = text.replace("\n", " ").strip()
            lines.append(f"Dialogue: 0,{ts(a)},{ts(b)},zh,{clean}\n")
    out.write_text("".join(lines), encoding="utf-8-sig")
    return out


# ------------------------------------- 4+5+6 · final pass: trim, captions, label (C6), encode
def finalize_video(job: Job, lipsynced: Path, master: Path, ass: Path | None,
                   workdir: Path, outdir: Path) -> Path:
    audio_d = probe_duration(master)
    final = outdir / "final.mp4"
    cmd = ["ffmpeg", "-v", "error", "-i", lipsynced, "-i", master]
    if ass:
        # relative filename + cwd — drive-colons never enter the filtergraph parser
        cmd += ["-vf", f"ass={ass.name}"]
    meta = f"FERRYMAN AIGC content_id={job.job_id}"
    vcodec, vq = video_codec(job)
    cmd += ["-map", "0:v", "-map", "1:a", "-t", f"{audio_d:.3f}",
            "-c:v", vcodec, *vq, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-b:a", "160k",
            "-metadata", f"comment={meta}", "-y", final]
    must(run(cmd, cwd=ass.parent if ass else None), "finalize encode")
    return final


# ---------------------------------------------------------------------- oracles (§10)
def _stream_duration(path: Path, kind: str) -> float | None:
    """Stream-level duration. nvenc mp4 streams sometimes omit the duration tag (QC A-05) —
    for video, derive it from frame count / rate before giving up. None = caller falls back
    to the container duration (and logs that the delta check is partially blind)."""
    s = probe_stream(path, kind, "duration")
    try:
        return float(s)
    except ValueError:
        pass
    if kind == "video":
        nb = probe_stream(path, "video", "nb_frames")
        fr = probe_stream(path, "video", "r_frame_rate")
        try:
            num, den = (fr.split("/") + ["1"])[:2]
            fps = float(num) / float(den or 1)
            if nb and fps > 0:
                return int(nb) / fps
        except (ValueError, ZeroDivisionError):
            pass
    return None


def run_oracles(job: Job, final: Path, master: Path, segs: list[str],
                seg_wavs: list[Path], omet: dict | None = None,
                sync: dict | None = None) -> dict:
    res: dict = {}
    res["sync_conf"] = sync   # V2-P1 report-only (None until the oracle ran); never gates yet
    vd = _stream_duration(final, "video")
    ad = _stream_duration(final, "audio")
    if vd is None or ad is None:
        fmt_d = probe_duration(final)
        log(f"oracles: stream duration tag missing (video={vd}, audio={ad}) — using container "
            f"duration {fmt_d:.2f}s for the missing side (QC A-05)")
        vd = fmt_d if vd is None else vd
        ad = fmt_d if ad is None else ad
    res["av_len_delta_ms"] = round(abs(vd - ad) * 1000, 1)
    res["codec_ok"] = (probe_stream(final, "video", "codec_name") == "h264"
                       and probe_stream(final, "audio", "codec_name") == "aac")
    res["duration_s"] = round(probe_duration(final), 2)
    res["segments"] = len(segs)
    omet = omet or {}
    res["audio_cer"] = omet.get("cer_max")        # worst segment (gate enforced per-seg upstream)
    res["audio_cer_mean"] = omet.get("cer_mean")
    res["speaker_sim"] = omet.get("sim_min")      # worst segment
    res["speaker_sim_mean"] = omet.get("sim_mean")
    # QC A-01: pass covers codec + A/V sync; CER/sim are enforced per-segment upstream
    # (oracle_segments raises). oracle_verified records whether that enforcement RAN —
    # the status line must never say GREEN for an unverified render.
    res["oracle_verified"] = bool(omet.get("verified", omet.get("cer_max") is not None))
    res["pass"] = bool(res["codec_ok"]
                       and res["av_len_delta_ms"] < float(oracle_cfg("av_len_delta_ms_max")))
    return res


# ---------------------------------------------------------------------- ledger (§5/§6)
def ledger_append(record: dict) -> str:
    """Append one hash-chained record. QC A-13: a cross-process lock spans read-prev +
    append so concurrent writers (manual render beside a scheduled batch) can't fork the
    chain, and a torn final line (crash mid-append) is skipped, not fatal."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lockf = open(LEDGER.parent / "ledger.lock", "a+b")
    locked = False
    try:
        try:
            import msvcrt
            lockf.seek(0)
            msvcrt.locking(lockf.fileno(), msvcrt.LK_LOCK, 1)   # blocking (≈10s of retries)
            locked = True
        except ImportError:
            pass
        prev_h = "0" * 32
        needs_nl = False
        if LEDGER.exists():
            raw = LEDGER.read_text(encoding="utf-8")
            needs_nl = bool(raw) and not raw.endswith("\n")   # torn final line has no newline
            for ln in reversed(raw.strip().splitlines()):
                try:
                    prev_h = json.loads(ln).get("h", prev_h)
                    break
                except json.JSONDecodeError:
                    log("ledger: skipping unparseable (torn?) line while resolving prev_h (QC A-13)")
        record["prev_h"] = prev_h
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        record["h"] = hashlib.blake2b((prev_h + payload).encode("utf-8"), digest_size=16).hexdigest()
        with open(LEDGER, "a", encoding="utf-8") as f:
            if needs_nl:
                f.write("\n")   # heal the torn line so the new record is never glued onto it
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record["h"]
    finally:
        if locked:
            try:
                import msvcrt
                lockf.seek(0)
                msvcrt.locking(lockf.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:  # noqa: BLE001
                pass
        lockf.close()


def verify_ledger() -> int:
    """REM-4.4 (R-15): walk runs.jsonl re-deriving every blake2b link. The chain was always
    tamper-EVIDENT; this makes it tamper-CHECKED. Exit 0=intact, 2=broken."""
    if not LEDGER.exists():
        print("verify-ledger: no ledger yet — vacuously intact")
        return 0
    prev = "0" * 32
    n = bad = skipped = 0
    for i, ln in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            print(f"  line {i}: UNPARSEABLE (torn write?) — skipped; chain continues from last good h")
            skipped += 1
            continue
        h = rec.pop("h", None)
        if rec.get("prev_h") != prev:
            print(f"  line {i}: CHAIN BREAK — prev_h {str(rec.get('prev_h'))[:12]}… != expected {prev[:12]}…")
            bad += 1
        payload = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        calc = hashlib.blake2b((str(rec.get("prev_h")) + payload).encode("utf-8"),
                               digest_size=16).hexdigest()
        if calc != h:
            print(f"  line {i}: HASH MISMATCH — record content does not match its h (tampered?)")
            bad += 1
        prev = h or prev
        n += 1
    verdict = "INTACT" if bad == 0 else f"BROKEN ({bad} violation(s))"
    print(f"verify-ledger: {n} record(s), {skipped} torn-skipped — {verdict}")
    return 0 if bad == 0 else 2


# ---------------------------------------------------------------------- render (§7 driver)
def render(job_path: Path) -> Path:
    t0 = time.time()
    job = Job.load(job_path)
    workdir = WORK / job.job_id
    outdir = OUT / job.job_id
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    # QC A-11: heartbeat marks this workdir in-flight; the disk steward skips workdirs
    # whose heartbeat is <48h old (protects a concurrent manual render from a scheduled
    # batch's purge pass).
    (workdir / ".active").write_text(str(os.getpid()), encoding="utf-8")
    log(f"render {job.job_id}: speaker={job.speaker} tier={job.tier} engine={job.voice_engine}")
    route_compute(job)   # v0.2 contract §9: T3/cloud gates evaluate NOW (fail-closed); T2 refuses until D2

    segs = segment_script(job.script)
    log(f"segment: {len(segs)} segments, {sum(len(s) for s in segs)} chars")
    timings = {}
    t = time.time(); seg_wavs = tts_segments(job, segs, workdir); timings["tts"] = round(time.time() - t, 1)
    t = time.time(); omet = oracle_segments(job, segs, seg_wavs, workdir); timings["oracle"] = round(time.time() - t, 1)
    t = time.time(); master, spans = concat_audio(seg_wavs, workdir); timings["audio"] = round(time.time() - t, 1)
    t = time.time(); lipsynced = lipsync_inpaint(job, master, workdir); timings["face"] = round(time.time() - t, 1)
    # V2-P1: score sync on the PURE lipsynced host video (pre-graphics — cutaway frames have
    # no host mouth and would poison windows; pre-captions — clean crops). Report-only.
    t = time.time(); sync = syncnet_score(lipsynced, master, workdir); timings["syncnet"] = round(time.time() - t, 1)
    if sync:
        log(f"syncnet: mean {sync['mean']} p10 {sync['p10']} min {sync['min']} "
            "(report-only; floor lands after V2-P1 multi-clip calibration)")
    t = time.time(); lipsynced, gprov = compose_graphics(job, lipsynced, spans, workdir); timings["graphics"] = round(time.time() - t, 1)

    ass = None
    if job.captions or job.label:
        w = int(probe_stream(lipsynced, "video", "width"))
        h = int(probe_stream(lipsynced, "video", "height"))
        ass = build_ass(segs, spans, workdir / "overlay.ass", w, h,
                        captions=job.captions, label=job.label,
                        total_dur=probe_duration(master))
        if job.captions:
            shutil.copy(ass, outdir / "captions.ass")
    t = time.time(); final = finalize_video(job, lipsynced, master, ass, workdir, outdir)
    timings["assemble"] = round(time.time() - t, 1)
    shutil.copy(master, outdir / "master.wav")

    oracles = run_oracles(job, final, master, segs, seg_wavs, omet, sync)
    spk = job.spk_dir()
    record = {
        "job_id": job.job_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "speaker": job.speaker, "tier": job.tier,
        "inputs": {"script_sha256": hashlib.sha256(job.script.encode("utf-8")).hexdigest(),
                   "voice_ref_sha256": sha256(spk / "ref.wav"),
                   "idle_base": job.idle_base().name},
        "models": {"tts": "IndexTeam/IndexTTS-2@local", "face": "TMElyralab/MuseTalk@v15-local"},
        "params": {"seed": job.seed, "label": job.label, "captions": job.captions,
                   "graphics": gprov or None},
        "outputs": {"final_sha256": sha256(final), "duration_s": oracles["duration_s"]},
        "oracles": oracles, "timings_s": {**timings, "total": round(time.time() - t0, 1)},
    }
    manifest = outdir / "manifest.jsonl"
    h = ledger_append(record)
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    # The content-pack sidecar — the stable filename contract the console (G5
    # canvas) and the site pipeline consume without parsing the ledger.
    (outdir / "meta.json").write_text(json.dumps({
        "job_id": job.job_id, "target": "video", "utc": record["utc"], "ledger_h": h,
        "files": {"final": "final.mp4", "master": "master.wav",
                  **({"captions": "captions.ass"} if job.captions else {})},
        "oracles": oracles}, ensure_ascii=False, indent=2), encoding="utf-8")
    status = ("ALL ORACLES GREEN" if oracles["pass"] and oracles.get("oracle_verified")
              else ("PASS (codec/AV) — SPEECH ORACLES UNVERIFIED (FERRYMAN_ALLOW_UNVERIFIED)"
                    if oracles["pass"] else f"ORACLE FAILURE: {oracles}"))
    log(f"FINALIZE {job.job_id}: {final} | {oracles['duration_s']}s | av_delta {oracles['av_len_delta_ms']}ms "
        f"| ledger {h[:12]} | total {record['timings_s']['total']}s | {status}")
    if not oracles["pass"]:
        raise RuntimeError(status)
    return final


# ---------------------------------------------------------------------- Head D · dubbing (§ target:"dub")
def _asr(audio: Path, lang: str | None = None) -> str:
    """ASR front-end via the earshot organ (whisper.cpp, multilingual). Transcript on stdout.
    QC A-03: run under THIS interpreter (earshot is stdlib-only), never a bare `python`
    from PATH that may be a different version or absent on a foreign box."""
    cmd = [sys.executable, EARSHOT, "--model", "turbo"] + (["--lang", lang] if lang else []) + [audio]
    cp = must(run(cmd), "asr (earshot)")
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip() and not ln.startswith("[earshot]")]
    return " ".join(lines).strip()


def _translate(segs: list[str], src_lang: str, tgt_lang: str, workdir: Path) -> list[str]:
    task = workdir / "mt_task.json"
    task.write_text(json.dumps({"model": str(NLLB_DIR), "src_lang": src_lang, "tgt_lang": tgt_lang,
                                "items": [{"idx": i, "text": t} for i, t in enumerate(segs)]},
                               ensure_ascii=False), encoding="utf-8")
    must(run([VENV_TTS, ROOT / "src" / "stage_translate_nllb.py", task]), "translate (nllb)")  # venv-tts: torch>=2.6
    res = json.loads(Path(str(task) + ".result").read_text(encoding="utf-8"))
    items = sorted(res["items"], key=lambda x: x["idx"])
    # QC B-07: a short result silently produced a misaligned dub — refuse instead.
    if [x["idx"] for x in items] != list(range(len(segs))):
        raise RuntimeError(f"translate: {len(items)} results for {len(segs)} segments — "
                           "refusing a misaligned dub")
    return [x["text"] for x in items]


def _seedvc(source: Path, ref: Path, steps: int, workdir: Path) -> Path:
    outdir = workdir / "seedvc"
    outdir.mkdir(parents=True, exist_ok=True)
    vram_guard()
    must(run([VENV_SEEDVC, "inference.py", "--source", source, "--target", ref,
              "--output", outdir, "--diffusion-steps", str(steps), "--fp16", "True"],
             cwd=SEEDVC_REPO, extra_env={"HF_HUB_DISABLE_XET": "1"}), "seed-vc convert")
    outs = sorted(outdir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not outs:
        raise RuntimeError("dub oracle: seed-vc produced no wav")
    return outs[-1]


def _speaker_sim(wav: Path, ref: Path, text: str, workdir: Path) -> float | None:
    """WeSpeaker cosine (language-agnostic) via stage_oracle — the dub-defining falsifier. The
    FireRedASR2 CER it also computes is only meaningful for zh, so dub uses earshot for
    intelligibility. Tolerant per-audio (returns None if the oracle can't run on THIS audio),
    but fail-closed on a missing oracle env (QC A-09): tolerance is for odd audio, not for
    the gate never being able to run at all."""
    if not VENV_ORACLE.exists():
        if ALLOW_UNVERIFIED:
            log("dub speaker-sim: venv-oracle NOT PRESENT — UNVERIFIED (flag set); sim=None")
            return None
        raise RuntimeError(
            f"dub: venv-oracle not present ({VENV_ORACLE}) — the speaker-sim falsifier cannot "
            "run (fail-closed; QC A-09). Install venv-oracle or set FERRYMAN_ALLOW_UNVERIFIED=1.")
    task = workdir / "dub_sim.json"
    task.write_text(json.dumps({"ref_wav": str(ref), "lexicon": [],
                                "items": [{"wav": str(wav), "text": text or "。"}]},
                               ensure_ascii=False), encoding="utf-8")
    cp = run([VENV_ORACLE, ROOT / "src" / "stage_oracle.py", task])
    if cp.returncode != 0:
        log(f"dub speaker-sim: oracle unavailable on this audio — sim=None ({(cp.stderr or '')[-160:]})")
        return None
    items = json.loads(Path(str(task) + ".result").read_text(encoding="utf-8"))["items"]
    return items[0].get("sim")


def _char_cer(ref: str, hyp: str) -> float:
    """Language-agnostic char-level error rate (normalized): back-transcription vs target text."""
    import re
    norm = lambda s: re.sub(r"[\s\W_]+", "", s.lower())
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return round(prev[-1] / len(r), 4)


def dub(job_path: Path) -> Path:
    """target:"dub" — Head D. same-lang = prosody-preserving VC (seed-vc: keep source words+delivery,
    swap timbre to the enrolled speaker). cross-lingual = ASR (audio source) -> NLLB translate ->
    IndexTTS2 in the target language/voice ("same voice, new language"). Output = out/<id>/dub.wav."""
    t0 = time.time()
    job = Job.load(job_path)
    d = job.dub or {}
    mode = d.get("mode", "cross-lingual")
    workdir, outdir = WORK / job.job_id, OUT / job.job_id
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (workdir / ".active").write_text(str(os.getpid()), encoding="utf-8")   # QC A-11 heartbeat
    spk = job.spk_dir()
    ref = spk / "ref.wav"
    if "source" not in d:
        raise RuntimeError("dub job needs dub.source (audio for same-lang; audio or text for cross-lingual)")
    src = Path(d["source"]) if Path(d["source"]).is_absolute() else (ROOT / d["source"])
    if not src.exists():
        raise FileNotFoundError(f"dub source not found: {src}")
    log(f"dub {job.job_id}: mode={mode} speaker={job.speaker} source={src.name}")

    timings = {}
    if mode == "same-lang":
        steps = int(d.get("diffusion_steps", 30))
        t = time.time(); conv = _seedvc(src, ref, steps, workdir); timings["seedvc"] = round(time.time() - t, 1)  # ref.wav ~45s; seed-vc wants 1-30s
        # QC A-08: NO silence-trim on the VC output — prosody preservation is the whole
        # point of same-lang dub, and a quiet source could trim to zero-length.
        t = time.time(); master, _ = concat_audio([conv], workdir, trim_silence=False); timings["master"] = round(time.time() - t, 1)
        src_text = _asr(src, d.get("lang"))
        hyp = _asr(master, d.get("lang"))
        target_text, intel_cer = src_text, _char_cer(src_text, hyp)
        models_used = {"vc": "Plachtaa/seed-vc@seed-uvit-whisper-small-wavenet"}
    else:  # cross-lingual
        t = time.time()
        src_text = _asr(src, d.get("source_asr_lang")) if src.suffix.lower() in AUDIO_EXTS \
            else src.read_text(encoding="utf-8")
        timings["asr"] = round(time.time() - t, 1)
        segs = segment_script(src_text)
        t = time.time(); tgt_segs = _translate(segs, d["source_lang"], d["target_lang"], workdir); timings["translate"] = round(time.time() - t, 1)
        t = time.time(); wavs = tts_segments(job, tgt_segs, workdir); timings["tts"] = round(time.time() - t, 1)
        t = time.time(); master, _ = concat_audio(wavs, workdir); timings["master"] = round(time.time() - t, 1)
        target_text = " ".join(tgt_segs)
        hyp = _asr(master, None)
        intel_cer = _char_cer(target_text, hyp)
        models_used = {"mt": "facebook/nllb-200-distilled-600M", "tts": "IndexTeam/IndexTTS-2@local"}

    t = time.time()
    sim = _speaker_sim(master, ref, target_text if mode != "same-lang" else "", workdir)
    timings["oracle"] = round(time.time() - t, 1)
    sim_min = float(oracle_cfg("speaker_sim_min"))
    sim_floor = float(d.get("sim_min", round(max(0.60, sim_min - 0.15), 3)))  # dub relaxes: x-lingual/VC lowers sim
    cer_max = float(d.get("cer_max", 0.20))

    dubout = outdir / "dub.wav"
    if job.label:  # C6/GB45438 (audio) — implicit metadata tag; audible notice = TODO
        must(run(["ffmpeg", "-v", "error", "-i", master, "-metadata",
                  f"comment=FERRYMAN AIGC dub mode={mode} content_id={job.job_id}", "-y", dubout]), "dub label")
    else:
        shutil.copy(master, dubout)

    sim_verified = sim is not None   # QC A-09: None here = per-audio oracle failure (venv missing raised)
    ok = (sim is None or sim >= sim_floor) and intel_cer < cer_max
    record = {
        "job_id": job.job_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": "dub", "mode": mode, "speaker": job.speaker,
        "inputs": {"source": str(src), "source_sha256": sha256(src), "voice_ref_sha256": sha256(ref)},
        "dub": {k: d.get(k) for k in ("mode", "source_lang", "target_lang", "diffusion_steps")},
        "models": models_used, "params": {"label": job.label},
        "outputs": {"dub_sha256": sha256(dubout), "duration_s": round(probe_duration(dubout), 2)},
        "oracles": {"speaker_sim": sim, "sim_floor": sim_floor, "sim_verified": sim_verified,
                    "intel_cer": intel_cer, "cer_max": cer_max, "pass": bool(ok)},
        "timings_s": {**timings, "total": round(time.time() - t0, 1)},
    }
    h = ledger_append(record)
    (outdir / "manifest.jsonl").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "meta.json").write_text(json.dumps({   # content-pack sidecar
        "job_id": job.job_id, "target": "dub", "utc": record["utc"], "ledger_h": h,
        "files": {"dub": "dub.wav"}, "oracles": record["oracles"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    status = (("DUB OK" if sim_verified else "DUB OK (speaker-sim UNVERIFIED on this audio)")
              if ok else
              f"DUB ORACLE FAIL (sim={sim} floor={sim_floor} | intel_cer={intel_cer} max={cer_max})")
    log(f"FINALIZE dub {job.job_id}: {dubout} | {record['outputs']['duration_s']}s | sim {sim} | "
        f"intel_cer {intel_cer} | ledger {h[:12]} | {status}")
    if not ok:
        raise RuntimeError(status)
    return dubout


def run_job(job_path: Path) -> Path:
    """Dispatch by target: dub -> Head D, video -> render(). Unknown targets refuse loudly
    — QC E-08: 'infographic'/'audio'/'motion' used to fall silently into the video pipeline."""
    t = Job.load(job_path).target
    if t == "dub":
        return dub(job_path)
    if t == "video":
        return render(job_path)
    raise NotImplementedError(f"job target {t!r} not wired (implemented today: video, dub)")


# ---------------------------------------------------------------------- enroll verbs (§7.0)
def enroll_voice(speaker: str, audio: Path) -> None:
    d = SPEAKERS / speaker
    d.mkdir(parents=True, exist_ok=True)
    ref = d / "ref.wav"
    must(run(["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "24000",
              "-sample_fmt", "s16", "-t", "45", "-y", ref]), "enroll resample")
    full = d / "ref_full.wav"
    must(run(["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "24000",
              "-sample_fmt", "s16", "-y", full]), "enroll full")
    prof = {"speaker": speaker, "lang": "zh", "sample_rate": 24000,
            "enrolled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ref_sha256": sha256(ref), "source": str(audio)}
    (d / "profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"enrolled voice {speaker}: {ref} ({probe_duration(ref):.1f}s ref, {probe_duration(full):.1f}s full)")


def make_idle(speaker: str, src: Path, driving: Path | None = None) -> None:
    d = SPEAKERS / speaker
    d.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
        dst = d / "idle_src.mp4"
        shutil.copy(src, dst)
        log(f"idle (real footage) {speaker}: {dst} ({probe_duration(dst):.1f}s) — B3 primary path")
        return
    # still image → LivePortrait, driven by provided motion or the speaker's own footage
    drv = driving or (d / "idle_src.mp4")
    if not Path(drv).exists():
        raise FileNotFoundError("make-idle from a still needs --driving <mp4> (or enroll footage first)")
    hi = d / "portrait_1080.png"
    must(run(["ffmpeg", "-v", "error", "-i", src, "-vf", "scale=-2:1080", "-y", hi]), "portrait scale")
    outdir = WORK / "_makeidle" / speaker
    outdir.mkdir(parents=True, exist_ok=True)
    vram_guard()
    must(run([VENV_INPAINT, "inference.py", "-s", hi, "-d", drv,
              "--output-dir", outdir, "--flag-crop-driving-video"],
             cwd=LIVEPORTRAIT_REPO), "liveportrait")
    outs = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    picks = [p for p in outs if "concat" not in p.name]
    if not picks:
        raise RuntimeError("make-idle oracle: liveportrait produced no mp4")
    shutil.copy(picks[-1], d / "idle_hi.mp4")
    log(f"idle (LivePortrait) {speaker}: {d / 'idle_hi.mp4'} ({probe_duration(d / 'idle_hi.mp4'):.1f}s)")


def upgrade_idle(speaker: str, footage: Path, force: bool = False) -> None:
    """`ferryman upgrade-idle <speaker> <footage>` — install new (hi-res) idle footage,
    drop-and-go. Validates (>=720p floor, >=20s), ARCHIVES the old
    base (never deletes), installs as idle_hi.mp4, reports before/after. R-01's identity
    sentinel makes the swap self-healing: the idle sha changes, so every stale lipsync
    resume auto-invalidates and re-renders on the new base."""
    d = SPEAKERS / speaker
    if not d.exists():
        raise FileNotFoundError(f"speaker not enrolled: {speaker} ({d})")
    if not footage.exists():
        raise FileNotFoundError(f"footage not found: {footage}")
    w = int(probe_stream(footage, "video", "width"))
    h = int(probe_stream(footage, "video", "height"))
    dur = probe_duration(footage)
    old = d / "idle_hi.mp4"
    old_desc = "none"
    if old.exists():
        old_desc = f"{probe_stream(old, 'video', 'width')}x{probe_stream(old, 'video', 'height')}, {probe_duration(old):.1f}s"
    log(f"upgrade-idle {speaker}: new footage {w}x{h}, {dur:.1f}s (current idle_hi: {old_desc})")
    if dur < 20:
        raise RuntimeError(f"footage is {dur:.1f}s — need >=20s of loopable idle (2-5 min ideal)")
    if h < 720 and not force:
        raise RuntimeError(f"footage is {h}p — below the 720p floor (v0.2 contract §4.3: the base caps "
                           "ALL face quality). Re-capture >=1080p, or pass --force to install anyway.")
    if old.exists():
        arch = d / "_archive"
        arch.mkdir(exist_ok=True)
        dst = arch / f"idle_hi_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        shutil.move(str(old), dst)           # archive, never delete (biometric material)
        log(f"upgrade-idle: previous base archived -> {dst}")
    shutil.copy(footage, old)
    log(f"upgrade-idle: INSTALLED {old} ({w}x{h}, {dur:.1f}s). Stale lipsync resumes will "
        "auto-invalidate (identity sentinel). Re-check: `ferryman doctor` (idle-base-res WARN clears).")


def _acquire_batch_lock() -> bool:
    """Single-holder lock (§9 spirit): scheduled + manual batches must never share the GPU.
    QC A-20: atomic O_EXCL create (no check-then-write race); stale detection uses an EXACT
    tasklist PID match (the old substring test matched PID 123 inside 1234)."""
    lock = ROOT / "ledger" / "batch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    def try_create() -> bool:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}")
        return True

    if try_create():
        return True
    try:
        pid = int(lock.read_text(encoding="utf-8").split()[0])
        cp = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True)
        if any(f'"{pid}"' in ln for ln in (cp.stdout or "").splitlines()):
            return False   # holder is genuinely alive
    except Exception:  # noqa: BLE001 — unreadable lock = stale
        pass
    lock.unlink(missing_ok=True)   # stale — clear, then re-race atomically
    return try_create()


def _release_batch_lock() -> None:
    (ROOT / "ledger" / "batch.lock").unlink(missing_ok=True)


def _keep_awake(on: bool) -> None:
    """Prevent system sleep while a batch renders (P5 polish: a 90-min render once took
    2.5h wall because the box napped mid-render). Process-scoped; no OS settings change."""
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if on else 0))
    except Exception:  # noqa: BLE001 — best-effort nicety, never fatal
        pass


def batch() -> None:
    if not _acquire_batch_lock():
        log("batch: another batch holds the lock — exiting (scheduled overlap is normal)")
        return
    _keep_awake(True)
    try:
        _batch_inner()
    finally:
        _keep_awake(False)
        _release_batch_lock()


def _disk_steward(min_free_gb: float = 40.0) -> None:
    """Operator-granted stewardship (charter v1.1, 2026-07-11): under disk pressure,
    purge in order (1) completed-job workdirs >24h (never _cache), (2) prior renders'
    video files in out\\ oldest-first — manifests, captions and the ledger are NEVER
    touched (every render is reproducible from its manifest)."""
    import shutil as _sh
    def free_gb(path: str) -> float:
        return _sh.disk_usage(path).free / 1e9
    for drive, label in ((str(WORK), "work-drive"), (str(OUT), "out-drive")):
        if free_gb(drive) >= min_free_gb:
            continue
        log(f"disk_steward: {label} below {min_free_gb} GB free — purging")
        now = time.time()
        for d in sorted(WORK.iterdir(), key=lambda p: p.stat().st_mtime):
            if not d.is_dir() or d.name == "_cache":
                continue
            hb = d / ".active"
            if hb.exists() and (now - hb.stat().st_mtime) < 172800:
                continue   # QC A-11: in-flight (or recently active) job — off-limits
            if now - d.stat().st_mtime > 86400:
                _sh.rmtree(d, ignore_errors=True)
                log(f"disk_steward: purged workdir {d.name}")
                if free_gb(drive) >= min_free_gb:
                    break
        if free_gb(drive) < min_free_gb:
            for d in sorted(OUT.iterdir(), key=lambda p: p.stat().st_mtime):
                if not d.is_dir() or d.parent != OUT:
                    continue
                # QC A-12: explicit regenerable basenames only — never an open-ended glob
                for v in (d / "final.mp4", d / "dub.wav", d / "master.wav"):
                    v.unlink(missing_ok=True)
                log(f"disk_steward: purged render media in out/{d.name} (manifest kept)")
                if free_gb(drive) >= min_free_gb:
                    break
        log(f"disk_steward: now {free_gb(drive):.1f} GB free on {label}")


def _batch_inner() -> None:
    _disk_steward()
    inbox = sorted((JOBS / "inbox").glob("*.job.json"), key=lambda p: p.stat().st_mtime)
    # QC F-07 (factory side): a job file written <2s ago may still be mid-write (the console
    # writes non-atomically today) — leave it for the next pass instead of failing it.
    cutoff = time.time() - 2.0
    fresh = [j for j in inbox if j.stat().st_mtime > cutoff]
    if fresh:
        log(f"batch: leaving {len(fresh)} just-written job(s) for the next pass (torn-write guard)")
        inbox = [j for j in inbox if j not in fresh]
    log(f"batch: {len(inbox)} job(s) in inbox")
    for j in inbox:
        try:
            run_job(j)   # dispatches by target: dub -> Head D, else video render()
            (JOBS / "done").mkdir(parents=True, exist_ok=True)
            shutil.move(str(j), JOBS / "done" / j.name)
        except Exception as e:  # noqa: BLE001 — batch must survive a bad job
            (JOBS / "failed").mkdir(parents=True, exist_ok=True)
            (JOBS / "failed" / (j.name + ".err")).write_text(str(e), encoding="utf-8")
            shutil.move(str(j), JOBS / "failed" / j.name)
            log(f"batch: {j.name} FAILED → jobs/failed ({e})")


def _run_ok(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run a probe command; return (success, combined-output). Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, **UTF8_ENV},
        )
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def doctor() -> int:
    """Self-diagnosis: encode every institutional trap + the render prerequisites
    as an actionable check. Verdict READY / DEGRADED / BLOCKED. Also the GUI's
    health panel. Exit code 0=READY, 1=DEGRADED, 2=BLOCKED."""
    checks: list[tuple[str, str, str]] = []  # (name, OK|WARN|FAIL, detail)
    def add(name: str, status: str, detail: str = "") -> None:
        checks.append((name, status, detail))

    # --- config / layout ---
    add("root", "OK" if ROOT.exists() else "FAIL", str(ROOT))
    add("venvs", "OK" if VENVS.exists() else "FAIL", str(VENVS))
    box = CFG.get("box", {}) if isinstance(CFG, dict) else {}
    add("pipeline.toml", "OK" if CFG else "FAIL",
        f"box={box.get('name', '?')}, {len(CFG)} sections" if CFG else "missing/unreadable")

    # --- render runtime: ffmpeg / ffprobe / nvenc ---
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    add("ffmpeg", "OK" if ff else "FAIL", ff or "not on PATH")
    add("ffprobe", "OK" if fp else "FAIL", fp or "not on PATH")
    if ff:
        ok, out = _run_ok([ff, "-hide_banner", "-encoders"], timeout=20)
        has = ok and "h264_nvenc" in out
        add("h264_nvenc", "OK" if has else "WARN",
            "GPU encode available" if has else "no nvenc — CPU x264 fallback (slow)")

    # --- GPU (one heavy model at a time; VRAM matters) ---
    smi = shutil.which("nvidia-smi")
    if smi:
        ok, out = _run_ok([smi, "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader"], timeout=15)
        line = out.strip().splitlines()[0] if (ok and out.strip()) else ""
        vram_ok = True
        try:
            mib = float(line.split(",")[1].strip().split()[0])
            vram_ok = mib >= 12000
        except Exception:  # noqa: BLE001
            pass
        add("gpu", "OK" if (line and vram_ok) else "WARN",
            line or "nvidia-smi query failed")
    else:
        add("gpu", "WARN", "nvidia-smi not found (CPU-only)")

    # --- per-model venvs (existence) + a torch.cuda canary on venv-tts ---
    for label, py in [("venv-tts", VENV_TTS), ("venv-oracle", VENV_ORACLE),
                      ("venv-musetalk", VENV_MUSETALK)]:
        add(label, "OK" if py.exists() else "FAIL",
            "python present" if py.exists() else f"missing: {py}")
    if VENV_TTS.exists():
        ok, out = _run_ok([str(VENV_TTS), "-c",
                           "import torch;print(torch.version.cuda,torch.cuda.is_available())"],
                          timeout=90)
        add("torch.cuda (tts canary)", "OK" if (ok and "True" in out) else "WARN",
            out.strip() or "torch import failed — re-assert the torch pin (trap #3)")

    # --- render core: the MuseTalk vendor layout (QC D-01/E-11 — what lipsync ACTUALLY loads) ---
    mt_unet = MUSETALK_REPO / "models" / "musetalkV15" / "unet.pth"
    add("vendor:MuseTalk", "OK" if MUSETALK_REPO.exists() else "FAIL",
        str(MUSETALK_REPO) if MUSETALK_REPO.exists() else
        f"missing — clone per manifests/vendor.json: {MUSETALK_REPO}")
    add("musetalk layout (unet.pth)", "OK" if mt_unet.exists() else "FAIL",
        str(mt_unet) if mt_unet.exists() else
        "vendor/MuseTalk/models/* junctions missing — run bin/hydrate_weights.ps1 (layout phase)")

    # --- head prerequisites: graphics / dub / make-idle (QC E-11) ---
    node = shutil.which("node")
    node_ok, node_detail = False, "not found — Head G (HyperFrames) unavailable"
    if node:   # REM-4.7 (R-18): the label said >=22; now the CHECK does too
        ok, out = _run_ok([node, "--version"], timeout=10)
        ver = (out or "").strip().lstrip("v")
        try:
            node_ok = ok and int(ver.split(".")[0]) >= 22
        except ValueError:
            node_ok = False
        node_detail = f"{node} v{ver or '?'}" + (
            " · first render fetches npx hyperframes + Chromium (network, once)" if node_ok
            else " — HyperFrames needs Node >=22")
    add("node>=22 (graphics)", "OK" if node_ok else "WARN", node_detail)
    add("earshot (dub ASR)", "OK" if EARSHOT.exists() else "WARN", str(EARSHOT))
    wcli_dir = Path(os.environ.get("EARSHOT_WHISPER_DIR") or r"C:\whisper.cpp")
    wcli = next((wcli_dir / n for n in ("whisper-cli.exe", "main.exe") if (wcli_dir / n).exists()), None)
    add("whisper.cpp engine (dub)", "OK" if wcli else "WARN",
        str(wcli) if wcli else f"no whisper-cli in {wcli_dir} — see manifests/substrate.json")
    ggml_dir = Path(os.environ.get("EARSHOT_MODELS_DIR") or r"C:\models")
    ggml = ggml_dir / "ggml-large-v3-turbo.bin"
    add("whisper turbo model (dub)", "OK" if ggml.exists() else "WARN",
        str(ggml) if ggml.exists() else f"missing {ggml} — bin/fetch_substrate.ps1")
    add("venv-seedvc (dub)", "OK" if VENV_SEEDVC.exists() else "WARN",
        "python present" if VENV_SEEDVC.exists() else f"missing: {VENV_SEEDVC} (same-lang dub down)")
    add("vendor:seed-vc (dub)", "OK" if SEEDVC_REPO.exists() else "WARN",
        str(SEEDVC_REPO) if SEEDVC_REPO.exists() else "clone per manifests/vendor.json")
    add("venv-inpaint (make-idle)", "OK" if VENV_INPAINT.exists() else "WARN",
        "python present" if VENV_INPAINT.exists() else f"missing: {VENV_INPAINT} (stills->idle down)")
    _sn_w = MODELS / "latentsync16" / "stable_syncnet.pt"
    _sn_ok = SYNCNET_STAGE.exists() and _sn_w.exists() and VENV_MUSETALK.exists()
    add("syncnet oracle (V2-P1)", "OK" if _sn_ok else "WARN",
        "report-only sync_conf on every render" if _sn_ok else
        "stage/weights/venv-musetalk missing — sync unmeasured (REM-2.1)")

    # --- substrate: grounding/briefing + image-head vision (QC E-11 — doctor used to say READY
    # while these heads were 100% absent) ---
    llama_dir = Path(os.environ.get("FERRYMAN_LLAMA_DIR") or r"C:\llama.cpp")
    gguf_dir = Path(os.environ.get("FERRYMAN_GGUF_DIR") or r"C:\models")
    add("llama-server (grounding)", "OK" if (llama_dir / "llama-server.exe").exists() else "WARN",
        str(llama_dir / "llama-server.exe") + ("" if (llama_dir / "llama-server.exe").exists()
                                               else " missing — substrate.json"))
    for g, why in [("Qwen3-8B-Q4_K_M.gguf", "grounding chat"),
                   ("qwen3-embedding-0.6b-q8_0.gguf", "grounding embed"),
                   ("Qwen3.5-9B-Q5_K_M.gguf", "vision oracle"),
                   ("mmproj-F16.gguf", "vision mmproj")]:
        add(f"gguf:{g.split('.gguf')[0][:24]}", "OK" if (gguf_dir / g).exists() else "WARN",
            f"{why} — " + (str(gguf_dir / g) if (gguf_dir / g).exists() else "missing (fetch_substrate)"))
    add("llama-mtmd-cli (image)", "OK" if (llama_dir / "llama-mtmd-cli.exe").exists() else "WARN",
        "vision-oracle runner" if (llama_dir / "llama-mtmd-cli.exe").exists() else "missing — substrate.json")
    comfy_base = Path(os.environ.get("COMFYUI_BASE") or
                      (Path(os.environ.get("USERPROFILE", "")) / "Documents" / "ComfyUI"))
    add("ComfyUI (image)", "OK" if comfy_base.exists() else "WARN",
        str(comfy_base) if comfy_base.exists() else "not installed — image head unavailable (optional)")

    # --- assets: CJK font (trap #2/#6), speakers, models ---
    add("CJK font", "OK" if Path(CJK_FONT_FILE).exists() else "WARN",
        f"{CJK_FONT_FILE} (note: libass resolves '{CJK_FONT_NAME}' by NAME — QC A-14)")
    spk = ([d.name for d in SPEAKERS.iterdir() if d.is_dir() and (d / "profile.json").exists()]
           if SPEAKERS.exists() else [])
    add("speakers enrolled", "OK" if spk else "WARN",
        ", ".join(spk) or "none — run `ferryman enroll-voice`")
    # v0.2 contract §4.3: the idle base caps EVERY tier's face quality — surface a low-res base
    # here so the #1 quality lever (a >=1080p re-capture / upscale) is never invisible.
    for s in spk:
        for nm in ("idle_hi.mp4", "idle_src.mp4"):
            p = SPEAKERS / s / nm
            if p.exists():
                try:
                    hgt = int(probe_stream(p, "video", "height"))
                    add(f"idle base res ({s})", "OK" if hgt >= 720 else "WARN",
                        f"{nm} {hgt}p" + ("" if hgt >= 720 else
                        " — base caps ALL face quality (v0.2 contract §4.3): re-capture >=1080p or upscale"))
                except Exception:  # noqa: BLE001
                    add(f"idle base res ({s})", "WARN", f"{nm}: unprobeable")
                break
    for m in ["indextts2", "musetalk", "fireredasr2-aed", "wespeaker-cnceleb-resnet34",
              "dwpose", "face-parse-bisent", "whisper-tiny", "sd-vae-ft-mse", "nllb200"]:
        add(f"model:{m}", "OK" if (MODELS / m).exists() else "WARN", str(MODELS / m))

    # --- traps encoded as standing env (informational) ---
    add("stage env", "OK", "PYTHONUTF8=1 + HF_HUB_DISABLE_XET=1 (baked into every subprocess)")

    icon = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    print(f"FERRYMAN doctor  ·  {ROOT}")
    for name, status, detail in checks:
        print(f"  {icon[status]}  {name:26s} {detail}")
    fails = sum(1 for _, s, _ in checks if s == "FAIL")
    warns = sum(1 for _, s, _ in checks if s == "WARN")
    oks = len(checks) - fails - warns
    verdict = "BLOCKED" if fails else ("DEGRADED" if warns else "READY")
    print(f"\n  VERDICT: {verdict}   ({oks} ok, {warns} warn, {fails} fail)")
    return 2 if fails else (1 if warns else 0)


def write_box() -> None:
    """QC D-09: stamp [box] in pipeline.toml from the REAL GPU on this machine — the
    shipped values describe whatever box built the seed and are wrong-on-arrival."""
    toml_path = ROOT / "config" / "pipeline.toml"
    if not toml_path.exists():
        raise SystemExit(f"write-box: {toml_path} not found")
    cp = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if cp.returncode != 0 or not cp.stdout.strip():
        raise SystemExit("write-box: nvidia-smi unavailable — cannot derive [box]")
    name, mem = [x.strip() for x in cp.stdout.strip().splitlines()[0].split(",")[:2]]
    vram_gb = round(float(mem.split()[0]) / 1024)
    slug = name.lower().replace("nvidia ", "").replace("geforce ", "").replace(" ", "-")
    import re as _re
    text = toml_path.read_text(encoding="utf-8")
    new_box = (f'[box]\nname = "{os.environ.get("COMPUTERNAME", "this-box").lower()}"\n'
               f'gpu = "{slug}"                # written by `ferryman doctor --write-box`\n'
               f'vram_gb = {vram_gb}\n'
               f'torch_cuda = "cu128"              # orchestrator-preferred; per-venv pins may differ\n'
               f'autocast = "fp16"\n')
    # REM-4.5 (R-16): also match a trailing [box] section; refuse loudly on zero matches
    # instead of logging a false "stamped".
    text2, nsub = _re.subn(r"\[box\][\s\S]*?(?=\n\[|\Z)", new_box, text, count=1)
    if nsub == 0:
        raise SystemExit("write-box: no [box] section found in pipeline.toml — nothing stamped")
    toml_path.write_text(text2, encoding="utf-8")
    log(f"write-box: [box] stamped — gpu={slug}, vram_gb={vram_gb}")


def main() -> None:
    # Trap #2, applied to OURSELVES: a bare `python src\ferryman.py …` on a cp1252 console
    # crashes printing CJK/typographic glyphs (the launchers set PYTHONUTF8, direct calls
    # don't). Same fix the organs use: force UTF-8 on the streams, replace-on-error.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="ferryman")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("job")
    sub.add_parser("batch")
    dr = sub.add_parser("doctor")
    dr.add_argument("--write-box", action="store_true",
                    help="stamp [box] in pipeline.toml from nvidia-smi (first run on a new machine)")
    sub.add_parser("verify-ledger", help="re-derive the blake2b hash chain over runs.jsonl")
    pc = sub.add_parser("preflight-cloud", help="show the live §9 cloud-gate checklist for a speaker")
    pc.add_argument("speaker")
    ui = sub.add_parser("upgrade-idle", help="install new (hi-res) idle footage for a speaker")
    ui.add_argument("speaker"); ui.add_argument("footage")
    ui.add_argument("--force", action="store_true", help="install below the 720p floor anyway")
    du = sub.add_parser("dub"); du.add_argument("job")
    ev = sub.add_parser("enroll-voice"); ev.add_argument("speaker"); ev.add_argument("audio")
    mi = sub.add_parser("make-idle"); mi.add_argument("speaker"); mi.add_argument("src")
    mi.add_argument("--driving", default=None)
    a = ap.parse_args()
    if a.cmd == "render":
        run_job(Path(a.job))     # dispatches by target (a dub-target job still dubs)
    elif a.cmd == "dub":
        dub(Path(a.job))
    elif a.cmd == "batch":
        batch()
    elif a.cmd == "doctor":
        if getattr(a, "write_box", False):
            write_box()
        raise SystemExit(doctor())
    elif a.cmd == "verify-ledger":
        raise SystemExit(verify_ledger())
    elif a.cmd == "preflight-cloud":
        ok, unmet = cloud_preflight(Job(job_id="_preflight", speaker=a.speaker, compute="cloud"))
        print("CLOUD PREFLIGHT: " + ("ALL GATES GREEN — only the D4 executor remains" if ok else "gates unmet:"))
        for u in unmet:
            print(f"  - {u}")
        raise SystemExit(0 if ok else 1)
    elif a.cmd == "upgrade-idle":
        upgrade_idle(a.speaker, Path(a.footage), force=a.force)
    elif a.cmd == "enroll-voice":
        enroll_voice(a.speaker, Path(a.audio))
    elif a.cmd == "make-idle":
        make_idle(a.speaker, Path(a.src), Path(a.driving) if a.driving else None)


if __name__ == "__main__":
    main()
