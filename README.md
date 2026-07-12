# FERRYMAN

**A sovereign, fully-local talking-head video factory.**

A text script + a short voice sample + a portrait → a finished, captioned video of that
person speaking the text, in their own cloned voice. Entirely offline — no cloud, no API,
no subscription, no quota. Open-weight models, an FFmpeg runtime, non-model verification
oracles, and a hash-chained provenance ledger. Batchable to hours of content.

> The name is 渡 — *to ferry*: the pipeline carries a still figure across into speech.

## Why

Cloud talking-head and document-to-video tools give you a canned voice, a stranger's face,
and your data on someone else's servers. FERRYMAN inverts all three: **your** enrolled voice,
**your** enrolled face, on **your** hardware, with the source material never leaving the box.
"Sovereign" means the render path has zero online dependency — bring your own speaker.

## How it works

```
INPUT: script.txt  +  voice_ref.wav (~30s)  +  portrait.png / idle.mp4
   │
   ├─ segment script (CJK-aware)
   ├─ TTS per segment (IndexTTS2), content-addressed cache  ── edit a line, re-render only that line
   ├─ ORACLE gate per segment: CER (FireRedASR2-AED, normalized + per-speaker homophone lexicon)
   │                           + speaker similarity (WeSpeaker cosine vs the reference)
   │                           auto-retake on failure, then flag
   ├─ master audio: silence-trim → pause policy → 48kHz → loudnorm -16 LUFS
   ├─ lip-sync: MuseTalk inpaints the mouth onto a looping idle base (unlimited length, flat VRAM)
   ├─ captions from the SCRIPT (never ASR of the synthetic audio) → ASS
   ├─ AIGC label ("AI生成") burned per applicable regulations
   └─ encode → oracle gates (A/V delta <100ms, codec conformance) → out/<job>/final.mp4
                                                                   + manifest + hash-chained ledger
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Requirements

- **Windows 11** (primary target), NVIDIA GPU **≥ 12 GB VRAM** (16 GB recommended), current driver.
- **Python 3.11** (per-model virtual environments; see the model repos for their torch pins).
- **FFmpeg** on PATH, built with `h264_nvenc` and `libass`.
- **Model weights** are NOT included (they're large and public). Download them from their
  original sources into `models/` — see [config/licenses.md](config/licenses.md) for the list
  and each model's license. Point the runtime at any layout via `FERRYMAN_HOME` / `FERRYMAN_VENVS`.

## Quickstart

```bat
:: 1. enroll a speaker (one-time; consent required)
bin\ferryman.cmd enroll-voice your_speaker path\to\voice.mp3
bin\ferryman.cmd make-idle    your_speaker path\to\footage.mp4   :: or a photo + --driving <motion.mp4>

:: 2. drop a job (see jobs/inbox/example.job.json) pointing at your script, then:
bin\ferryman.cmd batch

:: 3. collect out\<job_id>\final.mp4  (+ captions.ass, manifest.jsonl)
```

A job is seven lines of JSON — see [jobs/inbox/example.job.json](jobs/inbox/example.job.json).

## What makes it trustworthy at scale

Every episode is graded by assertions **no generative model authored**, before it counts as done:
per-segment **character-error-rate** (does the voice say every word?), **speaker similarity**
(does it sound like the enrolled person?), **A/V length delta** and **codec conformance**. A job
that fails its oracles goes to `jobs/failed/` with a reason — it never pretends to have worked.
Every render appends a **blake2b hash-chained record** to the ledger with input hashes, model
revisions, seeds, and output hashes, so a published episode is reproducible from its manifest.

## Models & licensing

FERRYMAN composes several open-weight models, each under its own license — some are
**non-commercial**. The pipeline is model-swappable behind stage contracts; **you** choose the
stack and are responsible for complying with each model's license for your use case. The
per-model record lives in [config/licenses.md](config/licenses.md). FERRYMAN's own code is
Apache-2.0 (see [LICENSE](LICENSE)).

## Status & caveats

- Windows-first. Stages run in isolated per-model venvs; a Linux/WSL2 path is feasible per stage.
- The default **inpainting** tier (mouth-on-idle-base) renders unlimited length at flat VRAM with
  zero identity drift — it is the production workhorse. A diffusion "hero" tier for short
  full-motion inserts is optional and drift-limited; see ARCHITECTURE.
- Consent is a hard precondition of enrollment: only clone voices and faces of people who have
  agreed. Keep the AIGC label on for anything published.

## License

Apache-2.0. Model weights carry their own separate licenses.
