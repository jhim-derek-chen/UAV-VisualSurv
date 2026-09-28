"""Render sample frames with the ground truth and every method's alarm area
drawn together on ONE image, so the four can be compared where they actually
overlap rather than across four separate panels.

Thresholds are computed live, per frame, per method: the top NOTICE_BAR of
THAT frame's own clean pixels -- the same self-calibrated rule benchmark.py
uses for the headline numbers, so the picture matches the tables.

  white outline     = labelled anomaly
  coloured outlines = each method's alarm region, one colour per method

Where two methods flag the same pixels their colours alternate along the shared
contour, so no method can be hidden by another regardless of object size.

    python scripts/survey/benchmark_figures.py
    python scripts/survey/benchmark_figures.py --only B4 --rows 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name

from benchmark import BENCHMARKS, BUILDERS, MIN_COMPONENT, NOTICE_BAR, Item, calibrate  # noqa: E402
from make_report import BENCH_LABEL  # noqa: E402  -- one source for the labels

OUT = REPO_ROOT / "results" / "figures"
SKIP = {"B5"}          # excluded from the report; see make_report.SKIP_BENCH

# Categorical slots 1-4 of the validated data-viz palette, same assignment as
# the recall/false-alarm chart, so a colour means the same method in both.
COLOUR = {
    "prowl-v2": (42, 120, 214),
    "prowl-v3": (235, 104, 52),
    "gdino": (27, 175, 122),
    "rba": (237, 161, 0),
}
LABEL = {
    "prowl-v2": "Self-sup. v2 · DINOv2",
    "prowl-v3": "Self-sup. v3 · DINOv3",
    "gdino": "Language prompt · G-DINO",
    "rba": "Rejected-by-all · SegFormer",
}
# Caption abbreviations. The full labels are in the legend above; repeating them
# per frame overran the 700px cell and clipped the last method's number.
SHORT = {"prowl-v2": "v2", "prowl-v3": "v3", "gdino": "lang", "rba": "rej"}
# Draw order fixes which colour takes which slot in the interleave, so a colour
# means the same method from frame to frame. Primary method first.
ORDER = ["prowl-v2", "prowl-v3", "rba", "gdino"]

GT_COLOUR = (255, 255, 255)
HALO = (14, 14, 16)
PAD, HEAD, LEG = 8, 30, 26


def font(size: int, bold: bool = False):
    names = (("seguisb.ttf", "arialbd.ttf") if bold
             else ("segoeui.ttf", "arial.ttf"))
    for n in names + ("DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick(items: list[Item], n: int) -> list[Item]:
    """Frames whose largest object is closest to the benchmark's median size.

    The biggest object would make every method look good and the smallest would
    make every method look broken; the median is the honest middle. Falls back
    to an even spread if no frame has a labelled object at all."""
    from scipy import ndimage

    areas = []
    for it in items:
        gt, valid = it.masks()
        if gt is None:
            areas.append(None)
            continue
        g = gt & valid
        lab, _ = ndimage.label(g)
        blobs = [int((lab[sl] > 0).sum()) for sl in ndimage.find_objects(lab)]
        blobs = [b for b in blobs if b >= MIN_COMPONENT]
        areas.append(max(blobs) / max(1, int(valid.sum())) if blobs else None)
    good = [(a, i) for i, a in enumerate(areas) if a is not None]
    if not good:
        step = max(1, len(items) // n)
        return items[::step][:n]
    med = float(np.median([a for a, _ in good]))
    good.sort(key=lambda t: abs(t[0] - med))
    # Keep the typically-sized half, then spread across the original ordering:
    # the n closest to the median alone returned neighbouring frames of one scene.
    band = sorted(i for _, i in good[:max(n, len(good) // 2)])
    step = max(1, len(band) // n)
    return [items[i] for i in band[::step][:n]]


def ring(mask: np.ndarray, offset: int, width: int) -> np.ndarray:
    """A band `width` px thick sitting `offset` px outside the region boundary."""
    from scipy import ndimage
    if not mask.any():
        return mask
    outer = ndimage.binary_dilation(mask, iterations=offset + width) if offset + width else mask
    inner = ndimage.binary_dilation(mask, iterations=offset) if offset else mask
    return outer & ~inner


def paint(img: Image.Image, gt: np.ndarray | None, alarms: dict[str, np.ndarray],
          scale: float) -> Image.Image:
    """Ground truth plus every method's alarm region, on one frame.

    All four outlines sit on their own region boundary, and where boundaries
    coincide the colours interleave along the contour instead of stacking.

    Concentric offsets were tried first and are wrong for this data: on a small
    obstacle every method produces nearly the same 15px blob, so a later
    method's dark casing painted straight over an earlier method's colour and
    the primary method vanished from the frame entirely. Interleaving is
    scale-free -- it reads the same on a 15px box and a 300px cow."""
    from scipy import ndimage

    base = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    h, w_img = base.shape[:2]
    w = max(2, int(3 * scale))

    bands = {m: ring(alarms[m], 0, w) for m in ORDER
             if alarms.get(m) is not None and alarms[m].any()}
    if bands:
        union = np.zeros((h, w_img), bool)
        overlap = np.zeros((h, w_img), np.uint8)
        for b in bands.values():
            union |= b
            overlap += b.astype(np.uint8)
        # one casing under the whole set, so no method can erase another
        base[ndimage.binary_dilation(union, iterations=2) & ~union] = HALO
        # dash period ~4px on screen; big enough to see, small enough that a
        # short contour still shows every colour that fired on it
        yy, xx = np.mgrid[0:h, 0:w_img]
        phase = ((xx + yy) // max(3, int(4 * scale))) % len(bands)
        for i, (m, b) in enumerate(bands.items()):
            base[b & ((overlap == 1) | (phase == i))] = COLOUR[m]

    if gt is not None and gt.any():
        gw = max(3, int(4 * scale))
        base[ring(gt, 0, gw + 2)] = HALO
        base[ring(gt, 1, gw)] = GT_COLOUR
    return Image.fromarray(base)


def sheet(bench: str, frames: list[Image.Image], caps: list[str],
          methods: list[str]) -> Image.Image:
    """Two frames side by side under one header and one legend."""
    cfg = BENCHMARKS[bench]
    cell_w = 700
    scaled = []
    for im in frames:
        s = cell_w / im.width
        scaled.append(im.resize((cell_w, max(1, round(im.height * s))), Image.LANCZOS))
    cell_h = max(im.height for im in scaled)
    n = len(scaled)
    W = n * cell_w + (n + 1) * PAD
    H = HEAD + LEG + cell_h + 20 + PAD

    out = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(out)
    f_head, f_leg, f_cap = font(17, True), font(13), font(13)
    name, note = BENCH_LABEL.get(bench, (cfg["title"], ""))
    d.text((PAD, 6), f"{bench}  {name}   |   {note}   |   {cfg['source']}",
           fill=(238, 238, 242), font=f_head)

    x = PAD
    y = HEAD
    d.text((x, y + 4), "White outline = labelled anomaly", fill=(235, 235, 240), font=f_leg)
    x += 230
    for m in methods:
        d.rectangle([x, y + 9, x + 22, y + 12], fill=COLOUR[m])
        d.text((x + 28, y + 4), LABEL[m], fill=(205, 205, 212), font=f_leg)
        x += 240

    for i, (im, cap) in enumerate(zip(scaled, caps)):
        px = PAD + i * (cell_w + PAD)
        out.paste(im, (px, HEAD + LEG))
        d.text((px + 2, HEAD + LEG + cell_h + 3), cap, fill=(198, 198, 206), font=f_cap)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    default=[b for b in BENCHMARKS if b not in SKIP])
    ap.add_argument("--method", nargs="+", default=list(BUILDERS))
    ap.add_argument("--rows", type=int, default=2, help="samples per benchmark")
    args = ap.parse_args()

    rep = json.loads((REPO_ROOT / "results/benchmark.json").read_text(encoding="utf-8"))
    methods = [m for m in args.method if m in rep["methods"]]
    if not methods:
        print("no results -- run scripts/survey/benchmark.py first")
        return 1

    chosen = {b: pick(BENCHMARKS[b]["loader"](BENCHMARKS[b]["cap"]), args.rows)
              for b in args.only}
    chosen = {b: v for b, v in chosen.items() if v}

    # alarms[bench][row][method]. Models are loaded one at a time to stay inside
    # 4 GB of VRAM, so the loop is method-outer and frames are re-read. The
    # threshold is computed live per frame -- top NOTICE_BAR of THAT frame's own
    # clean pixels -- matching benchmark.py's self-calibrated rule exactly.
    alarms: dict = {b: [dict() for _ in v] for b, v in chosen.items()}
    for m in methods:
        print(f"--- {m}", flush=True)
        model = BUILDERS[m]()
        for b, items in chosen.items():
            for r, it in enumerate(items):
                img = Image.open(it.img).convert("RGB")
                gt, valid = it.masks()
                s = model(img)
                if s.shape != valid.shape:
                    s = np.array(Image.fromarray(s).resize(valid.shape[::-1],
                                                           Image.BILINEAR))
                neg = s[valid & ~gt] if gt is not None else s[valid]
                tau = calibrate(neg, NOTICE_BAR)
                alarms[b][r][m] = (s >= tau) & valid
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    for b, items in chosen.items():
        frames, caps = [], []
        for r, it in enumerate(items):
            img = Image.open(it.img).convert("RGB")
            gt, valid = it.masks()
            scale = max(0.6, min(2.2, img.width / 1000))
            frames.append(paint(img, (gt & valid) if gt is not None else None,
                                alarms[b][r], scale))
            bits = [f"{SHORT[m]} {100 * a.mean():.1f}%"
                    for m, a in alarms[b][r].items()]
            name = it.img.stem
            name = name if len(name) <= 24 else name[:23] + "…"
            caps.append(f"{name} — alarm area  " + " · ".join(bits))
        dest = OUT / f"{b}.jpg"
        sheet(b, frames, caps, methods).save(dest, quality=88, optimize=True)
        print(f"saved {dest.relative_to(REPO_ROOT)}  ({len(frames)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
