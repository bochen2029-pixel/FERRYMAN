# P8 · APPLIANCE — FERRYMAN as a self-contained, portable, GUI-operated product

**Drafted:** 2026-07-11 (operator-requested). **Goal:** one zip → any suitable Windows machine →
unzip → runs. Operable by a human through a GUI with **no coding harness required**; replicable
**with or without Claude Code**; fully offline once hydrated (sovereignty preserved).

---

## §1 · The three deliverable modes (the "with/without CC" matrix)

| Mode | Artifact | Size | Internet | Claude Code | Use case |
|---|---|---|---|---|---|
| **A · Full-freeze** | `FERRYMAN-full-<ver>.zip` | ~120–150 GB | none (fully offline) | optional | USB/NAS copy to the production box; air-gapped replication |
| **B · Seed + hydrate** | `FERRYMAN-seed-<ver>.zip` | ~150–300 MB | once (weights+wheels) | optional | fresh machine anywhere; repo-shippable |
| **C · CC-native** | Mode B + `REPLICATE.md` | same as B | once | drives the bring-up | agent rebuilds the box like tonight, compressed to a checklist |

All three share one source of truth: the **manifests** (per-venv lockfiles + per-model SHA-256
weight lists + tool inventory). Mode A ships the bytes; Mode B ships the *description* of the
bytes plus a hydrator that fetches/validates them; Mode C is Mode B where the agent is the hydrator.

## §2 · The hard problems and their answers

1. **Venvs are not relocatable** (absolute `home=` in pyvenv.cfg). BUT ferryman always invokes
   venv pythons by absolute path — the only post-move fix is rewriting `pyvenv.cfg` to point at a
   same-version base interpreter. **Answer:** bundle python.org *embeddable* base interpreters
   (3.11, 3.13 — fully relocatable, no installer, no registry) under `runtimes\`, and ship
   `bin\relocate.ps1` that (a) rewrites every venv's pyvenv.cfg to the bundled base, (b) rebuilds
   the junction layout if the operator wants cross-drive data placement, else uses real dirs.
2. **Hardcoded install path.** **Answer:** de-hardcode — `ROOT = FERRYMAN_HOME env ‖ parent of
   src\` (one small refactor touching the constants block + stage scripts). The zip then runs
   from ANY path; a fixed install location becomes a convention, not a requirement.
3. **GPU generation drift** (dev = sm_89 Ada; prod = maybe Blackwell sm_120). cu128 wheels cover
   both; the ONE exception is venv-musetalk (torch 2.0.1+cu118 — no sm_120 kernels). **Answer:**
   compatibility matrix in `doctor`; hydrator flag `--rebuild musetalk-modern` (newer torch +
   mmcv chain; recipe already proven under R1 conditions and documented). Full-freeze README
   states: Ada/Ampere/Hopper run as-is; Blackwell runs everything except MuseTalk until rebuild.
4. **>4 GB zip entries.** PowerShell Compress-Archive chokes. **Answer:** bundle `7za.exe`;
   `freeze.ps1` emits zip64 (single .zip as requested; store-level compression — weights don't
   compress, speed wins).
5. **Windows Scheduled Task can't live in a zip.** **Answer:** `setup.ps1 --enable-schedule`
   recreates it (the P5 one-liner, parameterized).
6. **Institutional knowledge** (xet wedge, --include trap, cp1252, NCCL/gloo, chumpy,
   pyvenv relocation, kingbri1 wheels…). **Answer:** encode as CODE, not prose — every lesson
   becomes either a `doctor` check or hydrator logic. Prose lives in RUNBOOK/REPLICATE as backup.

## §3 · `ferryman doctor` — the no-harness survival kit

Self-diagnosis command (also the GUI's "health" panel). Checks, each with an actionable message:
- Hardware: GPU present, VRAM ≥ threshold per tier, driver ≥ min for bundled CUDA, disk free.
- Runtime: ffmpeg/ffprobe (bundled) exec, nvenc encode smoke, every venv python launches + key
  imports (torch.cuda.is_available per venv), Node 22 (bundled, for HyperFrames/P7).
- Assets: weight manifest audit (SHA-256 spot + size-exact full), speakers enrolled, fonts.
- Config: pipeline.toml parses, per-box [box] matches detected hardware (warn on mismatch),
  charter present, oracles calibrated (else "run `ferryman calibrate <speaker>`").
- Verdict: READY / DEGRADED (what still works) / BLOCKED (exact fix per failure).

## §4 · The GUI — operating FERRYMAN without any coding harness

**Stack:** `pywebview` (Edge WebView2 native window) + FastAPI backend, running in venv-runtime,
importing the existing runtime as a library. Pure Python, no Electron, no new languages, ships in
the same zip. Launch: `FERRYMAN.exe` shim → `bin\gui.cmd` → localhost app in a native window.

**Panels (v1):**
1. **Queue** — drag a `.txt`/`.md` script → job wizard (speaker, tier, captions, label, idle
   source, graphics mode) → writes `jobs\inbox\*.job.json`; live per-job status from PROGRESS/logs;
   inbox/done/failed columns; retry button (move failed→inbox).
2. **Speakers** — enrollment wizard: pick voice file + footage/photo → runs `enroll-voice` /
   `make-idle`; shows ref playback, sim-calibration status, homophone-lexicon editor (the
   proper-noun homophone table as a form).
3. **Monitor** — live render log tail, GPU util/VRAM, stage timeline, ETA; disk steward status.
4. **Review** — finished episodes with video player, oracle scorecard (CER/sim/AV-delta per
   segment), manifest viewer, "copy to publish folder" action (publishing itself stays human).
5. **Ledger** — hash-chain browser; verify-chain button.
6. **Settings** — pipeline.toml form editor; doctor page; scheduled-task on/off; charter viewer.

**The taste/director without CC (P7 tiers, GUI-exposed):** dropdown per job — `none` (default),
`rules` (regex: numbers→chart, lists→kinetic list; ships in-box), `local-llm` (Qwen via bundled
llama.cpp, optional weight download), `agent` (present only when a harness is attached). Sovereignty
default = everything local; any cloud option is off unless explicitly enabled.

**What the GUI does NOT replace:** incident repair of the *environment* (that's `doctor` +
RUNBOOK + optionally CC), and the initial creative setup of new graphic templates (P7 catalog
grows via harness or community templates dropped into `graphics\templates\`).

## §5 · Zip layout (target)

```
FERRYMAN\
  FERRYMAN.exe / FERRYMAN.cmd      ← GUI launcher (doctor-on-first-run)
  setup.ps1                        ← relocate + optional schedule + doctor
  bin\        ffmpeg, 7za, node\ (portable), launchers, freeze/relocate/manifest scripts
  runtimes\   python-3.11-embed\  python-3.13-embed\
  src\        ferryman.py + stage scripts + gui\ (fastapi app)
  venvs\      venv-tts | -inpaint | -musetalk | -oracle | -stableavatar | -longcat | -runtime
  models\     ALL weights (Mode A) | manifests only (Mode B)
  vendor\     index-tts, MuseTalk, LivePortrait, FireRedASR2S, StableAvatar, LongCat-Video,
              hyperframes (P7), each at pinned commit w/ vendored patches applied
  manifests\  weights.sha256.json · venv-*.lock.txt · tools.json · VERSION · box-profiles\
  speakers\   (operator's enrolled speakers — optionally excluded from shareable builds)
  config\ jobs\ work\ out\ ledger\ logs\ docs\{RUNBOOK.md, REPLICATE.md, BUILD_STATE, PROGRESS}
```

## §6 · Replication gate (the falsifier)

A build is DONE only when: unzip on a *different* path/machine → `setup.ps1` → `doctor` = READY →
drop the ep1 job → `ferryman batch` → **ALL ORACLES GREEN with byte-identical oracle verdicts**
(sim/CER within tolerance) — no hand-edits allowed. Dry-run proxy on this box: unzip to
`D:\FERRYMAN_PORTTEST\`, run with FERRYMAN_HOME set, delete after. Real gate: the production box.

## §7 · Phasing

- **P8a — De-hardcode + doctor v1** *(hours)*: FERRYMAN_HOME refactor; `ferryman doctor`;
  per-venv lockfile export; weights.sha256.json generator (extends make_manifest).
- **P8b — Full-freeze pipeline** *(a day)*: freeze.ps1 (stop tasks → prune work → manifests →
  7z zip64) + relocate.ps1 + setup.ps1; **pass the replication gate via D:\ dry-run**.
- **P8c — Seed + hydrate** *(1–2 days)*: hydrate.ps1 (embeddable pythons → uv/pip from locks →
  weights via manifest downloader with xet-off/curl fallback → vendor clones at pinned commits +
  patch application → doctor); absorbs every documented wedge as code. REPLICATE.md (human+agent).
- **P8d — GUI v1** *(2–4 days)*: pywebview+FastAPI, panels §4.1–.6, RUNBOOK.md.
- **P8e — Hardening** *(ongoing)*: Blackwell matrix entry (prod box), musetalk-modern rebuild
  recipe, shareable-build profile (speakers excluded, consent notice), version/update story
  (seed re-hydrate = upgrade path).

**Sequencing vs P6/P7:** P6 finishes first (in flight). P7a-b next (graphics artifact). P8a can
interleave anytime (small, pure-code). P8b+ after P7b so the freeze includes the graphics organ.

## §8 · Open questions (log-and-proceed defaults)

- Q-A: GUI branding/name? *Default: FERRYMAN Console; trivial to change.*
- Q-B: Ship CosyVoice3 venv in freeze despite Q4 (broken inference)? *Default: yes as-is
  (fallback slot), flagged DEGRADED by doctor.*
- Q-C: speakers\ in shareable builds? *Default: excluded profile exists; personal builds include.*
- Q-D: auto-update mechanism? *Default: none in v1 — re-hydrate is the update path.*
