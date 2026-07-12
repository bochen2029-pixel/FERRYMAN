# FERRYMAN — Architecture

The durable asset is not any single model — it's the pipeline, the filesystem contract, the
oracles, and the manifest. Models are swappable behind stage contracts.

## The core decision: decouple audio from video

Render **all** audio first (cheap, scales linearly), then drive video from it. This is what makes
unlimited-length output possible on modest VRAM: one continuous master track, then a mouth
inpainted onto a looping idle base.

## Two rendering paths

| | **Inpainting (default)** | **Diffusion (opt-in, hero)** |
|---|---|---|
| Does | Modifies only the mouth region on a looping idle base clip | Generates full head motion from audio |
| Length | **Unlimited**, constant VRAM | Clip-limited — drifts past ~30–60s |
| VRAM | ~6.5–8 GB, flat | ~12–16 GB |
| Identity | Locked by the idle base — zero drift | Held only by seed/reference across chunks |
| Use for | The body of every episode | Short high-realism inserts only |

The split's justification is cost-per-minute and drift, not just length. The inpainting path is
the production tier; the diffusion path is a garnish.

## Stages

Each stage is a pure function with a CLI shim, writes typed artifacts, and never mutates a prior
stage's output. Stages are idempotent by `(job_id, idx)` — re-running resumes from the last good
artifact.

| # | Stage | Tool binding |
|---|---|---|
| 1 | segment script | CJK-aware sentence/≤N-char splitter |
| 2 | tts per segment | IndexTTS2 (content-addressed cache) |
| 2a | segment oracles | FireRedASR2-AED CER + WeSpeaker similarity, auto-retake |
| 2b | concat + master | FFmpeg: silence-trim → pause policy → 48 kHz → loudnorm |
| 3 | lip-sync | MuseTalk (inpaint) / LivePortrait (idle-from-photo) |
| 4 | captions | script text + segment offsets → ASS (no ASR of synthetic audio) |
| 5 | label + encode | AIGC mark + `h264_nvenc`, `-t` audio length |
| 6 | finalize | hash outputs; append hash-chained ledger record |

## Oracles — non-model verification

The externality doctrine: grade every job with assertions no generative model authored.

- **Audio fidelity** — back-transcribe each segment, compute CER vs the source text (normalized:
  traditional/simplified fold, number normalization, punctuation strip, per-speaker homophone
  lexicon). Fail → retake at a fresh seed, then flag.
- **Speaker similarity** — cosine between each segment's speaker embedding and the reference.
  Calibrate the floor on real recordings of the enrolled speaker.
- **Duration & codec** — A/V length delta under tolerance; every segment matches the standardized
  codec/fps/sample-rate so lossless concat is valid.

Oracles are cheap, deterministic, and the reason a batch can run unattended and still be trusted.

## Captions from the script, not from ASR

The pipeline already possesses the ground truth — the script. Caption **text** = script text;
**timing** comes from the mastered per-segment offsets. ASR's role is the oracle (does the audio
match the text), never the source of captions — transcribing synthetic audio would burn homophone
and name errors permanently into the video.

## Provenance ledger

Every render appends one line to an append-only, hash-chained ledger:
`h = blake2b(prev_h ‖ record)`. Each record carries input SHA-256s, model repos/revisions, seeds,
output SHA-256s, oracle scores, and timings. A published episode is reproducible from its manifest;
the chain is tamper-evident.

## VRAM discipline

One model resident at a time. Each stage loads, runs, and explicitly tears down (the venv
subprocess boundary *is* the teardown) before the next loads. A guard asserts free headroom before
each heavy load and fails loud rather than silently spilling to system RAM.

## Portability

`FERRYMAN_HOME` (repo root) and `FERRYMAN_VENVS` locate everything; nothing is drive-absolute.
Per-model virtual environments are rebuilt per machine from pinned requirements; model weights are
relocatable files. The runtime orchestrator is stdlib + FFmpeg; heavy models run in their own venvs
via subprocess.
