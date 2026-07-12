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

ROOT = Path(os.environ.get("FERRYMAN_HOME") or Path(__file__).resolve().parent.parent)
SPEAKERS = ROOT / "speakers"
JOBS = ROOT / "jobs"
WORK = ROOT / "work"
OUT = ROOT / "out"
LEDGER = ROOT / "ledger" / "runs.jsonl"
VENVS = Path(os.environ.get("FERRYMAN_VENVS") or (ROOT / "venvs"))
VENV_TTS = VENVS / "venv-tts" / "Scripts" / "python.exe"
VENV_ORACLE = VENVS / "venv-oracle" / "Scripts" / "python.exe"
VENV_MUSETALK = VENVS / "venv-musetalk" / "Scripts" / "python.exe"
VENV_INPAINT = VENVS / "venv-inpaint" / "Scripts" / "python.exe"
MUSETALK_REPO = ROOT / "vendor" / "MuseTalk"
LIVEPORTRAIT_REPO = ROOT / "vendor" / "LivePortrait"
MODELS = ROOT / "models"
TTS_CACHE = WORK / "_cache" / "tts"
CJK_FONT_NAME = "Microsoft YaHei"
CJK_FONT_FILE = r"C:/Windows/Fonts/msyh.ttc"

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


def log(msg: str) -> None:
    print(f"[ferryman] {msg}", flush=True)


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


def vram_guard(min_free_gb: float = 1.5) -> None:
    cp = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    if cp.returncode == 0:
        free_gb = int(cp.stdout.strip().split("\n")[0]) / 1024
        if free_gb < min_free_gb:
            raise RuntimeError(f"vram_guard: only {free_gb:.1f} GB free (< {min_free_gb})")
        log(f"vram_guard: {free_gb:.1f} GB free")


# ---------------------------------------------------------------------- job spec (§6)
@dataclass
class Job:
    job_id: str
    speaker: str
    script: str
    lang: str = "zh"
    tier: str = "T1"
    voice_engine: str = "indextts2"
    face_model: str = "musetalk"
    idle: str = "auto"              # auto | hi | src | <path>
    captions: bool = True
    label: bool = True              # C6 — AIGC mark; False only for private outputs
    pinyin_overrides: dict = field(default_factory=dict)   # C3 (consumed when cosyvoice3 lands)
    seed: int = 1234
    output: dict = field(default_factory=lambda: {
        "codec": "h264_nvenc", "fps": 25, "res": "source", "audio": "aac_48k"})

    @staticmethod
    def load(path: Path) -> "Job":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "script_path" in d:
            d["script"] = Path(d.pop("script_path")).read_text(encoding="utf-8")
        known = {f.name for f in Job.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            log(f"job: ignoring unknown fields {sorted(extra)}")
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
    text = re.sub(r"\[\d+\]\s*", "", text)   # strip citation markers like [5]
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
    return [s for s in (x.strip() for x in segs) if s]


# ---------------------------------------------------------------------- 2 · tts (§7.2, B7 cache)
def tts_segments(job: Job, segs: list[str], workdir: Path,
                 take_overrides: dict[int, int] | None = None) -> list[Path]:
    # 'take' feeds only the cache key: IndexTTS2 sampling is stochastic, so a new
    # take = a fresh roll. The oracle retry path bumps takes for failing segments.
    take_overrides = take_overrides or {}
    if job.voice_engine != "indextts2":
        raise NotImplementedError(f"voice_engine {job.voice_engine} not wired (Q4: cosyvoice3 deferred)")
    ref = job.spk_dir() / "ref.wav"
    ref_sha = sha256(ref)[:16]
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    outs, todo = [], []
    for i, text in enumerate(segs):
        take = take_overrides.get(i, job.seed)
        key = hashlib.sha256(
            f"indextts2|{job.speaker}|{ref_sha}|{take}|{text}".encode("utf-8")).hexdigest()[:24]
        cached = TTS_CACHE / f"{key}.wav"
        outs.append(cached)
        if not cached.exists():
            todo.append({"text": text, "out": str(cached)})
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
        log("oracle: venv-oracle not present — SKIPPING (metrics recorded as null)")
        return {"cer_max": None, "cer_mean": None, "sim_min": None, "sim_mean": None, "items": None}
    ref = job.spk_dir() / "ref.wav"

    lex_path = job.spk_dir() / "lexicon.json"
    lexicon = json.loads(lex_path.read_text(encoding="utf-8")) if lex_path.exists() else []

    def grade(pairs: list[tuple[int, Path]], tag: str) -> list[dict]:
        task = workdir / f"oracle_{tag}.json"
        task.write_text(json.dumps(
            {"ref_wav": str(ref), "lexicon": lexicon,
             "items": [{"wav": str(w), "text": segs[i]} for i, w in pairs]},
            ensure_ascii=False), encoding="utf-8")
        vram_guard()
        must(run([VENV_ORACLE, ROOT / "src" / "stage_oracle.py", task]), f"oracle stage ({tag})")
        return json.loads(Path(str(task) + ".result").read_text(encoding="utf-8"))["items"]

    cer_max = float(ORACLE_CFG.get("audio_cer_max", 0.05))
    sim_min = float(ORACLE_CFG.get("speaker_sim_min", 0.0))
    sim_enforce = bool(ORACLE_CFG.get("speaker_sim_enforce", False))

    res = grade(list(enumerate(seg_wavs)), "pass1")

    floor_chars = float(ORACLE_CFG.get("cer_error_floor_chars", 1))

    def bad_idx() -> list[int]:
        # short segments get an absolute floor: one char of ASR ambiguity must not
        # fail a 16-char sign-off (a real garble produces multiple char errors)
        out = []
        for i, r in enumerate(res):
            ref_len = max(len(r.get("ref_norm") or ""), 1)
            thresh = max(cer_max, floor_chars / ref_len)
            if r["cer"] > thresh or (sim_enforce and r["sim"] is not None and r["sim"] < sim_min):
                out.append(i)
        return out

    bad = bad_idx()
    if bad:
        log(f"oracle: segments {bad} failed (cer>{cer_max}"
            + (f" or sim<{sim_min}" if sim_enforce else "") + ") — re-rendering fresh takes")
        wavs2 = tts_segments(job, segs, workdir, take_overrides={i: job.seed + 1000 for i in bad})
        res2 = grade([(i, wavs2[i]) for i in bad], "pass2")
        for (i, r2) in zip(bad, res2):
            if r2["cer"] <= res[i]["cer"]:
                res[i] = r2
                seg_wavs[i] = wavs2[i]
        still = bad_idx()
        if still:
            worst = {i: res[i]["cer"] for i in still}
            raise RuntimeError(f"oracle: segments still failing after retry {worst} — flag to human (§10)")
    cers = [r["cer"] for r in res]
    sims = [r["sim"] for r in res if r["sim"] is not None]
    agg = {"cer_max": max(cers), "cer_mean": round(sum(cers) / len(cers), 4),
           "sim_min": min(sims) if sims else None,
           "sim_mean": round(sum(sims) / len(sims), 4) if sims else None,
           "items": [{"cer": r["cer"], "sim": r["sim"]} for r in res]}
    log(f"oracle: cer max/mean {agg['cer_max']}/{agg['cer_mean']} | sim min/mean {agg['sim_min']}/{agg['sim_mean']}")
    return agg


# ------------------------------------------------------- 2b · concat + mastering (§7.2b, B4)
def concat_audio(seg_wavs: list[Path], workdir: Path, pause_ms: int = 300,
                 lufs: float = -16.0) -> tuple[Path, list[tuple[float, float]]]:
    """Silence-trim each segment, insert deliberate pauses, one soxr resample to 48 kHz,
    one loudnorm pass. Returns (master.wav, [(start_s, end_s) per segment]) — the exact
    offsets that make script-sourced captions (B1) free."""
    audio_dir = workdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trimmed, spans, t = [], [], 0.0
    for i, w in enumerate(seg_wavs):
        tw = audio_dir / f"trim_{i:03d}.wav"
        must(run(["ffmpeg", "-v", "error", "-i", w, "-af",
                  "silenceremove=start_periods=1:start_threshold=-45dB,"
                  "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse",
                  "-ar", "48000", "-ac", "1", "-y", tw]), f"trim seg {i}")
        d = probe_duration(tw)
        spans.append((t, t + d))
        t += d + pause_ms / 1000.0
        trimmed.append(tw)
    silence = audio_dir / "pause.wav"
    must(run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
              f"anullsrc=r=48000:cl=mono", "-t", f"{pause_ms/1000:.3f}", "-y", silence]), "pause gen")
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
    existing = sorted((res_dir / "v15").glob("*.mp4"), key=lambda p: p.stat().st_mtime) \
        if (res_dir / "v15").exists() else []
    if existing:
        # reuse only if the existing render still covers the (possibly re-mastered) audio
        if probe_duration(existing[-1]) >= audio_d - 0.05:
            log(f"lipsync: reusing {existing[-1].name} (idempotent resume; covers {audio_d:.1f}s)")
            return existing[-1]
        log("lipsync: existing render shorter than new master — re-rendering")
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
    return vids[-1]


# ------------------------------------------------ 5 · captions from script (B1) — sentence ASS
def build_ass(segs: list[str], spans: list[tuple[float, float]], out: Path,
              play_w: int, play_h: int, captions: bool = True, label: bool = False,
              total_dur: float | None = None) -> Path:
    """One ASS carries both the captions (B1) and the AI生成 mark (C6) — a single
    subtitle filter, no drawtext, no font-path escaping. libass finds installed
    system fonts by name on Windows (DirectWrite)."""
    def ts(sec: float) -> str:
        h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"
    fs = max(28, int(play_h * 0.055))
    fs_mark = max(20, int(play_h * 0.030))
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: zh,{CJK_FONT_NAME},{fs},&H00FFFFFF,&H00101010,&H88000000,0,2,1,2,30,30,{int(play_h*0.04)},1\n"
        f"Style: mark,{CJK_FONT_NAME},{fs_mark},&H50FFFFFF,&H00101010,&H70000000,0,0,0,9,20,20,{int(play_h*0.02)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    lines = [head]
    if label:
        end = total_dur if total_dur else (spans[-1][1] if spans else 0.0)
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
    cmd += ["-map", "0:v", "-map", "1:a", "-t", f"{audio_d:.3f}",
            "-c:v", job.output.get("codec", "h264_nvenc"), "-cq", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-b:a", "160k",
            "-metadata", f"comment={meta}", "-y", final]
    must(run(cmd, cwd=ass.parent if ass else None), "finalize encode")
    return final


# ---------------------------------------------------------------------- oracles (§10)
def run_oracles(job: Job, final: Path, master: Path, segs: list[str],
                seg_wavs: list[Path], omet: dict | None = None) -> dict:
    res: dict = {}
    vd = float(probe_stream(final, "video", "duration") or 0)
    ad = float(probe_stream(final, "audio", "duration") or 0)
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
    res["pass"] = bool(res["codec_ok"] and res["av_len_delta_ms"] < 100)
    return res


# ---------------------------------------------------------------------- ledger (§5/§6)
def ledger_append(record: dict) -> str:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    prev_h = "0" * 32
    if LEDGER.exists():
        lines = LEDGER.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            prev_h = json.loads(lines[-1]).get("h", prev_h)
    record["prev_h"] = prev_h
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
    record["h"] = hashlib.blake2b((prev_h + payload).encode("utf-8"), digest_size=16).hexdigest()
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record["h"]


# ---------------------------------------------------------------------- render (§7 driver)
def render(job_path: Path) -> Path:
    t0 = time.time()
    job = Job.load(job_path)
    workdir = WORK / job.job_id
    outdir = OUT / job.job_id
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"render {job.job_id}: speaker={job.speaker} tier={job.tier} engine={job.voice_engine}")

    segs = segment_script(job.script)
    log(f"segment: {len(segs)} segments, {sum(len(s) for s in segs)} chars")
    timings = {}
    t = time.time(); seg_wavs = tts_segments(job, segs, workdir); timings["tts"] = round(time.time() - t, 1)
    t = time.time(); omet = oracle_segments(job, segs, seg_wavs, workdir); timings["oracle"] = round(time.time() - t, 1)
    t = time.time(); master, spans = concat_audio(seg_wavs, workdir); timings["audio"] = round(time.time() - t, 1)
    t = time.time(); lipsynced = lipsync_inpaint(job, master, workdir); timings["face"] = round(time.time() - t, 1)

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

    oracles = run_oracles(job, final, master, segs, seg_wavs, omet)
    spk = job.spk_dir()
    record = {
        "job_id": job.job_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "speaker": job.speaker, "tier": job.tier,
        "inputs": {"script_sha256": hashlib.sha256(job.script.encode("utf-8")).hexdigest(),
                   "voice_ref_sha256": sha256(spk / "ref.wav"),
                   "idle_base": job.idle_base().name},
        "models": {"tts": "IndexTeam/IndexTTS-2@local", "face": "TMElyralab/MuseTalk@v15-local"},
        "params": {"seed": job.seed, "label": job.label, "captions": job.captions},
        "outputs": {"final_sha256": sha256(final), "duration_s": oracles["duration_s"]},
        "oracles": oracles, "timings_s": {**timings, "total": round(time.time() - t0, 1)},
    }
    manifest = outdir / "manifest.jsonl"
    h = ledger_append(record)
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "ALL ORACLES GREEN" if oracles["pass"] else f"ORACLE FAILURE: {oracles}"
    log(f"FINALIZE {job.job_id}: {final} | {oracles['duration_s']}s | av_delta {oracles['av_len_delta_ms']}ms "
        f"| ledger {h[:12]} | total {record['timings_s']['total']}s | {status}")
    if not oracles["pass"]:
        raise RuntimeError(status)
    return final


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


def _acquire_batch_lock() -> bool:
    """Single-holder lock (§9 spirit): scheduled + manual batches must never share the GPU."""
    lock = ROOT / "ledger" / "batch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
            alive = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                   capture_output=True, text=True)
            if str(pid) in (alive.stdout or ""):
                return False
        except Exception:  # noqa: BLE001 — unreadable lock = stale
            pass
        lock.unlink(missing_ok=True)
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}", encoding="utf-8")
    return True


def _release_batch_lock() -> None:
    (ROOT / "ledger" / "batch.lock").unlink(missing_ok=True)


def batch() -> None:
    if not _acquire_batch_lock():
        log("batch: another batch holds the lock — exiting (scheduled overlap is normal)")
        return
    try:
        _batch_inner()
    finally:
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
            if now - d.stat().st_mtime > 86400:
                _sh.rmtree(d, ignore_errors=True)
                log(f"disk_steward: purged workdir {d.name}")
                if free_gb(drive) >= min_free_gb:
                    break
        if free_gb(drive) < min_free_gb:
            for d in sorted(OUT.iterdir(), key=lambda p: p.stat().st_mtime):
                if not d.is_dir():
                    continue
                for v in list(d.glob("*.mp4")) + list(d.glob("*.wav")):
                    v.unlink(missing_ok=True)
                log(f"disk_steward: purged render media in out/{d.name} (manifest kept)")
                if free_gb(drive) >= min_free_gb:
                    break
        log(f"disk_steward: now {free_gb(drive):.1f} GB free on {label}")


def _batch_inner() -> None:
    _disk_steward()
    inbox = sorted((JOBS / "inbox").glob("*.job.json"), key=lambda p: p.stat().st_mtime)
    log(f"batch: {len(inbox)} job(s) in inbox")
    for j in inbox:
        try:
            render(j)
            (JOBS / "done").mkdir(parents=True, exist_ok=True)
            shutil.move(str(j), JOBS / "done" / j.name)
        except Exception as e:  # noqa: BLE001 — batch must survive a bad job
            (JOBS / "failed").mkdir(parents=True, exist_ok=True)
            (JOBS / "failed" / (j.name + ".err")).write_text(str(e), encoding="utf-8")
            shutil.move(str(j), JOBS / "failed" / j.name)
            log(f"batch: {j.name} FAILED → jobs/failed ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="ferryman")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("job")
    sub.add_parser("batch")
    ev = sub.add_parser("enroll-voice"); ev.add_argument("speaker"); ev.add_argument("audio")
    mi = sub.add_parser("make-idle"); mi.add_argument("speaker"); mi.add_argument("src")
    mi.add_argument("--driving", default=None)
    a = ap.parse_args()
    if a.cmd == "render":
        render(Path(a.job))
    elif a.cmd == "batch":
        batch()
    elif a.cmd == "enroll-voice":
        enroll_voice(a.speaker, Path(a.audio))
    elif a.cmd == "make-idle":
        make_idle(a.speaker, Path(a.src), Path(a.driving) if a.driving else None)


if __name__ == "__main__":
    main()
