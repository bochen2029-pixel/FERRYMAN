#!/usr/bin/env python3
r"""
hypergen.py — abstract cover ART from pure code (no GPU, no model, no download).

The portability floor for covers: on ANY machine, instantly and for free, draw a
tasteful abstract cover BACKGROUND (no baked title/author text -- composite_cover.py
lays type on top). Think early-PowerPoint templates, but infinite: every image is
(style x palette x seed). Where a GPU box makes bespoke SDXL art, a laptop makes a
clean hypergen cover in a tenth of a second.

Design rule shared with the SDXL path: keep the UPPER THIRD calm/open so the title
has room. Output has no text.

Styles: gradient, duotone, horizon, geometric, contours, halftone, arcs, deco.
Moods:  dark_literary, warm_memoir, cool_scifi, bold_thriller, soft_romance,
        earthy_nonfiction, noir, botanical.

Usage:
  python _tools/hypergen.py --style horizon --mood dark_literary --out cover_art/x.png
  python _tools/hypergen.py --palette "#0a0a14,#1b2a4a,#c9a227" --style deco --seed 7 --out x.png
  python _tools/hypergen.py --contact-sheet --mood cool_scifi --out /tmp/sheet.png
  (Pillow only.)
"""
from __future__ import annotations
import argparse, math, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

STYLES = ["gradient", "duotone", "horizon", "geometric", "contours", "halftone", "arcs", "deco"]

MOODS = {
    "dark_literary":    ["#0b0d14", "#1a2233", "#3a4a63", "#c7b299"],
    "warm_memoir":      ["#2b1d14", "#6b4a2f", "#c98a45", "#f0dcae"],
    "cool_scifi":       ["#05070f", "#0f2a44", "#1f6f8b", "#7fd8d8"],
    "bold_thriller":    ["#0a0a0a", "#3a0d0d", "#a11212", "#e8b04b"],
    "soft_romance":     ["#3a2230", "#7d4a63", "#d08aa6", "#f3d9e4"],
    "earthy_nonfiction":["#1c2016", "#3f4a2e", "#8a9a5b", "#e6e2cf"],
    "noir":             ["#050506", "#17181c", "#4a4d55", "#b9bcc4"],
    "botanical":        ["#0e1a12", "#1f4030", "#4e8d5b", "#e7d9a6"],
}


def _hex(c):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _multi_stop(colors, t):
    if t <= 0:
        return colors[0]
    if t >= 1:
        return colors[-1]
    n = len(colors) - 1
    seg = t * n
    i = int(seg)
    return _lerp(colors[i], colors[min(i + 1, n)], seg - i)


def vgrad(size, colors):
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    # draw as rows (fast enough), then paste to full width
    strip = Image.new("RGB", (1, h))
    sp = strip.load()
    for y in range(h):
        sp[0, y] = _multi_stop(colors, y / max(1, h - 1))
    return strip.resize(size)


def grain(img, amount=14, blur=0.0, r=None):
    # QC C-19: PIL's effect_noise draws from an UNSEEDED RNG — the one layer that broke
    # the (style × palette × seed)-deterministic promise. With a seeded Random we build
    # the noise plane from randbytes (C-speed), scaled to roughly effect_noise's spread.
    if r is not None:
        w, h = img.size
        n = Image.frombytes("L", (w, h), r.randbytes(w * h))
        n = n.point(lambda v: 128 + ((v - 128) * amount) // 96)
    else:
        n = Image.effect_noise(img.size, amount).convert("L")
    if blur:
        n = n.filter(ImageFilter.GaussianBlur(blur))
    noise_rgb = Image.merge("RGB", (n, n, n))
    return Image.blend(img, ImageChops_screen(img, noise_rgb), 0.10)


def ImageChops_screen(a, b):
    from PIL import ImageChops
    return ImageChops.screen(a, b)


def _rng(seed):
    return random.Random(seed)


# ---- styles: each returns an RGB Image; top third kept calm ----------------
def s_gradient(size, pal, seed):
    r = _rng(seed)
    cols = pal[:]
    if r.random() < 0.5:
        cols = list(reversed(cols))
    img = vgrad(size, cols)
    return grain(img, 10, r=r)


def s_duotone(size, pal, seed):
    r = _rng(seed)
    dark, light = pal[0], pal[-1]
    img = vgrad(size, [dark, _lerp(dark, light, 0.5), light])
    d = ImageDraw.Draw(img, "RGBA")
    w, h = size
    for _ in range(3):                       # soft light blooms low in the frame
        cx = r.randint(0, w); cy = r.randint(int(h * 0.55), h)
        rad = r.randint(int(w * 0.3), int(w * 0.7))
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                  fill=light + (26,))
    img = img.filter(ImageFilter.GaussianBlur(w * 0.02))
    return grain(img, 12, r=r)


def s_horizon(size, pal, seed):
    r = _rng(seed)
    w, h = size
    sky = vgrad((w, h), [pal[0], pal[1], _lerp(pal[1], pal[2], 0.4)])
    img = sky.copy()
    d = ImageDraw.Draw(img, "RGBA")
    hz = int(h * r.uniform(0.62, 0.72))       # horizon low -> title room up top
    d.rectangle([0, hz, w, h], fill=pal[0])
    # a disc (sun/moon) sitting near the horizon
    disc = pal[3]
    dr = int(w * r.uniform(0.16, 0.24))
    dx = int(w * r.uniform(0.3, 0.7)); dy = hz - int(dr * r.uniform(0.1, 0.5))
    d.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=disc + (235,))
    # faint reflection
    d.ellipse([dx - dr, hz + (hz - dy) - dr, dx + dr, hz + (hz - dy) + dr], fill=disc + (40,))
    return grain(img, 9, r=r)


def s_geometric(size, pal, seed):
    r = _rng(seed)
    w, h = size
    img = Image.new("RGB", size, pal[0])
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(7):                        # translucent bauhaus shapes, weighted low
        col = _hexpick(r, pal) + (r.randint(60, 130),)
        cy = r.randint(int(h * 0.3), h)
        s = r.randint(int(w * 0.2), int(w * 0.6))
        cx = r.randint(0, w)
        kind = r.random()
        if kind < 0.34:
            d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=col)
        elif kind < 0.67:
            d.rectangle([cx - s, cy - s, cx + s // 2, cy + s // 2], fill=col)
        else:
            d.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=col)
    img = Image.alpha_composite(vgrad(size, [pal[0], pal[1]]).convert("RGBA"),
                                img.convert("RGBA")).convert("RGB")
    return grain(img, 8, r=r)


def s_contours(size, pal, seed):
    r = _rng(seed)
    w, h = size
    img = vgrad(size, [pal[0], pal[1]])
    d = ImageDraw.Draw(img, "RGBA")
    line = pal[3]
    n = 18
    for i in range(n):
        base = h * (0.35 + 0.62 * i / n)      # lines gather lower
        amp = r.uniform(10, 40) * (i / n + 0.3)
        ph = r.uniform(0, math.tau)
        pts = []
        for x in range(0, w + 20, 20):
            y = base + amp * math.sin(x / w * math.tau * r.uniform(1.2, 2.5) + ph)
            pts.append((x, y))
        d.line(pts, fill=line + (70,), width=2)
    return grain(img, 8, r=r)


def s_halftone(size, pal, seed):
    r = _rng(seed)
    w, h = size
    img = vgrad(size, [pal[1], pal[0]])
    d = ImageDraw.Draw(img, "RGBA")
    dot = pal[3]
    step = max(18, w // 46)
    for gy in range(0, h, step):
        t = gy / h                            # dots grow toward the bottom
        for gx in range(0, w, step):
            rad = step * 0.5 * (t ** 1.6) * r.uniform(0.7, 1.1)
            if rad > 0.6:
                d.ellipse([gx - rad, gy - rad, gx + rad, gy + rad], fill=dot + (150,))
    return grain(img, 7, r=r)


def s_arcs(size, pal, seed):
    r = _rng(seed)
    w, h = size
    img = vgrad(size, [pal[0], pal[1]])
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = int(w * r.uniform(0.2, 0.8)), int(h * r.uniform(0.85, 1.1))
    for i in range(14):
        rad = int(w * (0.15 + 0.12 * i))
        col = _multi_stop(pal[1:], i / 14) + (90,)
        d.arc([cx - rad, cy - rad, cx + rad, cy + rad], 180, 360, fill=col, width=max(3, w // 220))
    return grain(img, 8, r=r)


def s_deco(size, pal, seed):
    r = _rng(seed)
    w, h = size
    img = vgrad(size, [pal[0], pal[1]])
    d = ImageDraw.Draw(img, "RGBA")
    gold = pal[3]
    bw = max(4, w // 300)
    for i in range(-h, w, w // 14):           # diagonal deco bands, lower weight
        d.line([(i, h), (i + h, 0)], fill=gold + (46,), width=bw)
    d.rectangle([w * 0.08, h * 0.08, w * 0.92, h * 0.92], outline=gold + (140,), width=bw)
    return grain(img, 7, r=r)


_STYLE_FN = {
    "gradient": s_gradient, "duotone": s_duotone, "horizon": s_horizon,
    "geometric": s_geometric, "contours": s_contours, "halftone": s_halftone,
    "arcs": s_arcs, "deco": s_deco,
}


def _hexpick(r, pal):
    return r.choice(pal)


def render(style, palette_hex, size=(1600, 2400), seed=0):
    pal = [_hex(c) for c in palette_hex]
    while len(pal) < 4:
        pal.append(_lerp(pal[-1], (255, 255, 255), 0.4))
    fn = _STYLE_FN.get(style)
    if not fn:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    return fn(size, pal, seed).convert("RGB")


def contact_sheet(palette_hex, size=(360, 540), seed=0):
    cols = 4
    rows = (len(STYLES) + cols - 1) // cols
    pad = 12
    W = cols * size[0] + (cols + 1) * pad
    H = rows * size[1] + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for i, st in enumerate(STYLES):
        im = render(st, palette_hex, size=size, seed=seed + i)
        x = pad + (i % cols) * (size[0] + pad)
        y = pad + (i // cols) * (size[1] + pad)
        sheet.paste(im, (x, y))
        d.text((x + 6, y + 6), st, fill=(255, 255, 255))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description="Abstract cover art from pure code (no GPU).")
    ap.add_argument("--style", choices=STYLES, default="gradient")
    ap.add_argument("--mood", choices=list(MOODS), help="use a built-in palette")
    ap.add_argument("--palette", help="comma-separated hex colors (overrides --mood)")
    ap.add_argument("--size", default="1600x2400", help="WxH (default 1600x2400, 6:9)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact-sheet", action="store_true", help="render all styles into one grid")
    args = ap.parse_args(argv)

    if args.palette:
        pal = [p.strip() for p in args.palette.split(",") if p.strip()]
    elif args.mood:
        pal = MOODS[args.mood]
    else:
        pal = MOODS["dark_literary"]
    w, h = (int(x) for x in args.size.lower().split("x"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.contact_sheet:
        contact_sheet(pal, seed=args.seed).save(out)
    else:
        render(args.style, pal, size=(w, h), seed=args.seed).save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
