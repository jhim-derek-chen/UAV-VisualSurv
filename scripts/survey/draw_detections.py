"""Draw Grounding DINO detections onto images so they can be judged by eye.

The aerial data carries no anomaly labels, so no metric can say whether a
detection is right. Looking at the boxes is currently the only way to find out.

Green boxes = ground-truth anomalies, drawn only where labels exist.
Red boxes    = model detections, with the matched prompt phrase and score.

    python scripts/survey/draw_detections.py --set aerial
    python scripts/survey/draw_detections.py --set ground --n 12
    python scripts/survey/draw_detections.py --set aerial --threshold 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name

from eval_transfer import (Detector, aerial_samples, components,  # noqa: E402
                           ground_samples)

OUT = REPO_ROOT / "results" / "detections"


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(img: Image.Image, dets, scores, labels, gt=None) -> Image.Image:
    canvas = img.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    # Scale line width and text with image size, or boxes vanish on 4K frames.
    w = max(2, int(max(canvas.size) / 500))
    f = font(max(14, int(max(canvas.size) / 70)))

    for box in gt or []:
        d.rectangle(list(box), outline=(0, 220, 0), width=w + 1)

    for box, sc, lab in zip(dets, scores, labels):
        x0, y0, x1, y1 = [float(v) for v in box]
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=w)
        tag = f"{lab} {sc:.2f}" if lab else f"{sc:.2f}"
        tb = d.textbbox((0, 0), tag, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y0 - th - 4)
        d.rectangle([x0, ty, x0 + tw + 6, ty + th + 4], fill=(255, 40, 40))
        d.text((x0 + 3, ty + 2), tag, fill=(255, 255, 255), font=f)

    legend = "green = ground truth   red = detection" if gt else "red = detection (no labels exist)"
    lb = d.textbbox((0, 0), legend, font=f)
    d.rectangle([0, 0, lb[2] + 10, lb[3] + 8], fill=(0, 0, 0))
    d.text((5, 4), legend, fill=(255, 255, 255), font=f)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["ground", "aerial"], default="aerial")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--max-width", type=int, default=1600,
                    help="downscale saved images so they open quickly")
    args = ap.parse_args()

    det = Detector()
    OUT.mkdir(parents=True, exist_ok=True)
    n_written = 0

    if args.set == "ground":
        items = ground_samples(args.n)
        print(f"ground, {len(items)} images -> {OUT.relative_to(REPO_ROOT)}")
        for img_p, lab_p, anom in items:
            img = Image.open(img_p).convert("RGB")
            gt = components(np.array(Image.open(lab_p)) == anom)
            boxes, scores, labels = det(img, args.threshold)
            out = draw(img, boxes, scores, labels, gt)
            if out.width > args.max_width:
                out = out.resize((args.max_width, int(out.height * args.max_width / out.width)))
            out.save(OUT / f"ground__{img_p.stem[:45]}.jpg", quality=88)
            n_written += 1
            print(f"  {img_p.stem[:45]:<48} gt={len(gt):<3} det={len(boxes)}")
    else:
        items = aerial_samples(args.n)[: args.n]
        print(f"aerial, {len(items)} images -> {OUT.relative_to(REPO_ROOT)}")
        print("no anomaly labels exist for these; judge the red boxes by eye\n")
        for img_p, stratum in items:
            img = Image.open(img_p).convert("RGB")
            boxes, scores, labels = det(img, args.threshold)
            out = draw(img, boxes, scores, labels)
            if out.width > args.max_width:
                out = out.resize((args.max_width, int(out.height * args.max_width / out.width)))
            tag = stratum.replace("/", "-")
            out.save(OUT / f"aerial__{tag}__{img_p.parent.name}_{img_p.stem}.jpg", quality=88)
            n_written += 1
            print(f"  {stratum:<16} {img_p.parent.name}/{img_p.stem:<12} det={len(boxes)}"
                  f"  scores={' '.join(f'{s:.2f}' for s in sorted(scores)[::-1][:5])}")

    print(f"\n{n_written} images written to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
