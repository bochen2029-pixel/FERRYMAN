# Model license record (re-verify at each model bump)

Record each model's license here. Some bundled models are non-commercial — verify before
any commercial use. Only clone voices/faces of people who have given explicit consent.

> **CAUTION:** if you intend any commercial use, verify every model below first. The default
> voice (IndexTTS2) is non-commercial; commercial use requires a separate license from its
> vendor (Bilibili). License-clean alternatives for the voice stage include CosyVoice3
> (Apache-2.0) and GPT-SoVITS (MIT) — swap before shipping anything commercial.

| Model | License | Status here |
|---|---|---|
| IndexTTS-2 | **Non-commercial** (Bilibili Index model license) | DEFAULT voice — OK under nonprofit stance |
| Fun-CosyVoice3-0.5B | Apache-2.0 | fallback/pinyin reserve (deferred, Q4) |
| MuseTalk 1.5 | openrail-m (use-restricted, commercial-OK) | DEFAULT face |
| LatentSync-1.6 | openrail++ (use-restricted, commercial-OK) | **T2 quality face tier** (512² lip region; wiring in progress) |
| seed-vc | **GPLv3** | same-lang VC dub engine — arm's-length subprocess in its own venv; vendor clone never redistributed, so no copyleft carry; quarantine/swap if monetized |
| InfiniteTalk (MeiGen) | Apache-2.0 | planned T3 cloud V2V dub — the commercial-clean face escalation |
| LivePortrait | per-repo (Kwai; research/non-com leanings — re-verify before any revenue) | idle-from-still only |
| whisper-tiny / faster-whisper | MIT / MIT | features + (future) alignment |
| FireRedASR2-AED | Apache-2.0 (verify at wiring) | CER oracle (P5) |
| WeSpeaker cnceleb | Apache-2.0 | speaker-sim oracle (P5) |
| sd-turbo, sd-vae-ft-mse | stability community licenses | P0 smoke / VAE dep only |

China AIGC labeling (GB 45438-2025, in force 2025-09-01): explicit "AI生成" mark +
implicit metadata are ON by default in the pipeline (`label: true`); per-job override
exists for private/family-only outputs.
