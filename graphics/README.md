# FERRYMAN — Graphics head (P7, Head G)

An **optional** motion-graphics layer: charts, kinetic lists, diagram cutaways composited
onto a talking-head render. **OFF by default** — a job renders exactly as before unless
`graphics.enabled` is set. It is a per-job toggle decided at authoring time, never a global
mode switch.

## Toggle (in the job.json)

Omit `graphics` (or set `enabled:false`) → no graphics, zero change to the render. Turn on:

```jsonc
{
  "job_id": "...", "speaker": "...", "script_path": "...",
  "graphics": {
    "enabled": true,
    "cues": [
      {"mode": "pip",        "file": "graphics/<proj>/pip.mov", "seg": 2},
      {"mode": "fullscreen", "file": "graphics/<proj>/fs.mp4",  "seg_start": 8, "seg_end": 10}
    ]
  }
}
```

**Cue fields**
- `mode`: `"pip"` (alpha overlay — host stays visible) | `"fullscreen"` (opaque cutaway — host replaced, voice continues)
- `file`: a **pre-rendered** graphic (see Authoring), relative to `FERRYMAN_HOME`. PiP must be `.mov` (alpha); fullscreen `.mp4`.
- span — pick one: `{"seg": i}` (bind to segment i's mastering span) · `{"seg_start": i, "seg_end": j}` · `{"start": s, "end": e}` (seconds)
- `pos`: overlay `"x:y"` (default `"0:0"` — our comps are full-canvas, the card is positioned in-HTML)

## Where it runs

Inside `ferryman render()`, **after lipsync, before the captions/label burn**. It binds to the
mastering stage's **live** segment spans (`concat_audio` offsets) — which is why it lives in the
pipeline, not as a post-step on a finished/purged video. Graphics render on CPU/Chromium, in
parallel with the GPU stages.

## Oracles (enforced in `compose_graphics`)

- every cue `file` exists
- composite duration == base (±0.15 s) and resolution unchanged
- provenance `{mode, file, span, sha256}` per cue → ledger record `params.graphics`

## Authoring a graphic (tier-1 Director = author at job time)

1. Make a HyperFrames project dir: `index.html` + `hyperframes.json` + `assets/fonts/NotoSansSC-*.otf`.
   **Template note (QC E-15):** for a video overlay copy `graphics/ep1_pip` or `graphics/ep1_fullscreen`
   (authored at the job's 1206×1080@29); `graphics/test_chart` is a STANDALONE 1920×1080@25 demo —
   fix its `data-width/height/fps` if you start from it.
2. **CJK MUST use the bundled Noto Sans SC via `@font-face`** — HyperFrames maps only Latin web-fonts; Chinese tofus off-box otherwise.
3. **A `fullscreen` cue MUST carry its own `AI生成` mark in the HTML (QC E-14).** The cutaway
   replaces the host frame, so it occludes the burned-in ASS label — an unmarked fullscreen
   graphic is an unlabeled AIGC segment (GB 45438-2025). See `ep1_fullscreen/index.html` /
   `ep1_image_bg/index.html` for the standard corner mark. PiP cues don't need it (the host's
   label stays visible).
4. Author at the job's resolution (1206×1080) with `data-duration` ≈ the target span; keep GSAP deterministic (paused timeline, finite repeats, no `Math.random`/`Date.now`).
5. Render at the job's fps:
   - fullscreen (opaque): `npx hyperframes@0.6.73 render --fps <fps> --output fs.mp4`
   - PiP (alpha): `npx hyperframes@0.6.73 render --fps <fps> --format mov --output pip.mov`  ← **MOV = ProRes 4444 alpha; `webm` silently flattens to yuv420p**
6. Reference the file(s) in `graphics.cues`, bound to segment spans.

**Reference comps:** `graphics/test_chart` (data chart), `graphics/ep1_fullscreen` (flow diagram), `graphics/ep1_pip` (alpha corner card).

## Not yet (P7d / later)

- tier-2 local-LLM Director (auto-EDL from the script)
- scene transitions (the fullscreen cut is a hard cut today)
- a shared graphics-head font asset (fonts are per-project copies today)
- surgical Qwen-Image assets composited into the HTML
- fully-offline vendoring of GSAP + mono numerals (CDN/Google-fetch, cached, today)
