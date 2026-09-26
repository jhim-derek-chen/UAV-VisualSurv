"""Result figures: what the pipeline finds versus the labels.

Reads saved predictions (results/road-object-eval/preds_<method>.json) and
the operating threshold from that method's result JSON, so it draws exactly
what was scored. Each figure is the whole frame on the left and a zoom on
the frame's debris on the right.

  blue        label (ground truth)
  green       detection that matches a label
  red         detection that matches nothing (a false alarm)
  light blue  Stage 1 road region (two-stage only)
  yellow ring the debris

Two modes:

  python scripts/draw_detection_examples.py --method owlv2-objectness --which two_stage
      -> results/road-object-eval/examples/: a few frames with large debris
         (tyres, fridges, ladders) that the pipeline found

  python scripts/draw_detection_examples.py --method owlv2-objectness --which two_stage --all
      -> results/synthetic-debris-inference/: one figure per image, found or
         missed, plus summary.json with per-image counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from eval_road_objects import (DATASET, HIT_IOU, OUT, centre_in_any, iou,  # noqa: E402
                               match, stage1_masks)

ALL_OUT = REPO_ROOT / "results" / "synthetic-debris-inference"
BLUE, GREEN, RED, YELLOW = (40, 140, 255), (40, 220, 60), (255, 50, 50), (255, 255, 0)
LARGE = {"tire", "truck-tire", "refrigerator", "ladder", "mattress", "wooden-plank"}


def draw_boxes(d: ImageDraw.ImageDraw, boxes, colour, scale: float, off=(0, 0), width=3, pad=0):
    for b in boxes:
        x, y, w, h = b[:4]
        x0 = (x - off[0]) * scale - pad
        y0 = (y - off[1]) * scale - pad
        d.rectangle([x0, y0, x0 + w * scale + 2 * pad, y0 + h * scale + 2 * pad],
                    outline=colour, width=width)


def assign(gts: list, preds: list) -> tuple[list[bool], list[bool]]:
    """One-to-one matching, highest score first, same rule as the scoring.
    Returns (label found?, detection used?)."""
    order = sorted(range(len(preds)), key=lambda k: -preds[k][4])
    found, used = [False] * len(gts), [False] * len(preds)
    for k in order:
        best, bi = HIT_IOU, -1
        for i, g in enumerate(gts):
            if not found[i]:
                v = iou(g["bbox"], preds[k][:4])
                if v > best:
                    best, bi = v, i
        if bi >= 0:
            found[bi] = used[k] = True
    return found, used


def figure(img: Image.Image, gts: list, preds: list, thr: float, title: str,
           road: np.ndarray | None) -> tuple[Image.Image, dict]:
    font = ImageFont.load_default(22)
    small = ImageFont.load_default(18)
    found, used = assign(gts, preds)
    good = [p for p, u in zip(preds, used) if u]
    bad = [p for p, u in zip(preds, used) if not u]
    di = next(i for i, g in enumerate(gts) if g["category"] != "vehicle")
    debris = gts[di]

    # Left: whole frame, road region tinted.
    H = 900
    s = H / img.height
    left = img.resize((round(img.width * s), H), Image.BILINEAR)
    if road is not None:
        m = Image.fromarray((road * 255).astype(np.uint8)).resize(left.size, Image.NEAREST)
        tint = Image.new("RGB", left.size, (120, 200, 255))
        left = Image.composite(Image.blend(left, tint, 0.28), left, m)
    d = ImageDraw.Draw(left)
    draw_boxes(d, [g["bbox"] for g in gts], BLUE, s, width=2, pad=3)
    draw_boxes(d, good, GREEN, s, width=2)
    draw_boxes(d, bad, RED, s, width=2)
    x, y, w, h = debris["bbox"]
    d.ellipse([(x + w / 2) * s - 28, (y + h / 2) * s - 28, (x + w / 2) * s + 28,
               (y + h / 2) * s + 28], outline=YELLOW, width=3)

    # Right: zoom on the debris at >= native resolution.
    side = int(max(4 * max(w, h), 160))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(min(max(0, cx - side / 2), img.width - side))
    y0 = int(min(max(0, cy - side / 2), img.height - side))
    crop = img.crop((x0, y0, x0 + side, y0 + side))
    zs = H / side
    right = crop.resize((H, H), Image.LANCZOS)
    d2 = ImageDraw.Draw(right)
    near = lambda b: x0 - b[2] < b[0] < x0 + side and y0 - b[3] < b[1] < y0 + side  # noqa: E731
    draw_boxes(d2, [g["bbox"] for g in gts if near(g["bbox"])], BLUE, zs, (x0, y0), 4, pad=6)
    draw_boxes(d2, [p for p in good if near(p)], GREEN, zs, (x0, y0), 4)
    draw_boxes(d2, [p for p in bad if near(p)], RED, zs, (x0, y0), 4)
    hits = [p for p in good if iou(debris["bbox"], p[:4]) > HIT_IOU]
    if found[di] and hits:
        best = max(hits, key=lambda p: iou(debris["bbox"], p[:4]))
        # Fused runs score a gated box 1 + probe; show the probe part.
        conf = best[4] - 1 if best[4] > 1 else best[4]
        verdict = (f"FOUND  {debris['category']}  {w}x{h} px   overlap with label (IoU) "
                   f"{iou(debris['bbox'], best[:4]):.2f}   confidence {conf:.2f}")
        vcol = (255, 255, 255)
    else:
        verdict = f"MISSED  {debris['category']}  {w}x{h} px"
        vcol = (255, 120, 120)
    d2.text((12, H - 64), verdict, fill=vcol, font=font, stroke_width=3, stroke_fill=(0, 0, 0))

    n_veh = sum(1 for g in gts if g["category"] == "vehicle")
    n_veh_found = sum(1 for g, f in zip(gts, found) if f and g["category"] == "vehicle")
    stats = {"vehicles": n_veh, "vehicles_found": n_veh_found,
             "debris_category": debris["category"], "debris_found": bool(found[di]),
             "debris_iou": round(iou(debris["bbox"], best[:4]), 3) if found[di] and hits else None,
             "false_alarms": len(bad)}

    canvas = Image.new("RGB", (left.width + H + 20, H + 90), (24, 24, 24))
    canvas.paste(left, (0, 90))
    canvas.paste(right, (left.width + 20, 90))
    dc = ImageDraw.Draw(canvas)
    dc.text((12, 8), title, fill=(255, 255, 255), font=font)
    dc.text((12, 36), f"vehicles found {n_veh_found}/{n_veh}   debris "
            f"{'found' if found[di] else 'MISSED'}   false alarms {len(bad)}   "
            f"(threshold {thr:.3f})", fill=(255, 255, 255), font=small)
    dc.text((12, 62), "blue = label   green = detection matching a label   red = false alarm   "
            "yellow ring = the debris" + ("   light blue = road region" if road is not None
                                          else ""), fill=(200, 200, 200), font=small)
    return canvas, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="owlv2-objectness",
                    help="run name, as in results/road-object-eval/<run>.json")
    ap.add_argument("--which", default="two_stage", choices=["unrestricted", "two_stage"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--all", action="store_true",
                    help="one figure per image into results/synthetic-debris-inference/")
    args = ap.parse_args()

    res = json.loads((OUT / f"{args.method}.json").read_text(encoding="utf-8"))
    thr = res[args.which]["best_f1"]["threshold"]
    preds_all = json.loads((OUT / f"preds_{args.method}.json").read_text(encoding="utf-8"))
    preds_all = {int(k): v for k, v in preds_all["predictions"].items()}
    which = 0 if args.which == "unrestricted" else 1

    anns = json.loads((DATASET / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in anns["categories"]}
    images = {im["id"]: im for im in anns["images"]}
    gt_by_img: dict[int, list] = {}
    for a in anns["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(
            {"bbox": a["bbox"], "category": cats[a["category_id"]]})
    road = (stage1_masks(sorted(images), images)
            if args.which == "two_stage" and res.get("stage1") == "aeroscapes-mask" else {})

    def kept(img_id):
        ign = images[img_id].get("ignore_regions", [])
        return [p for p in preds_all[img_id][which] if p[4] >= thr and not centre_in_any(p, ign)]

    def loc(img_id):
        return Path(manifest["samples"][img_id]["background_source"]).parts[-2]

    if args.all:
        chosen = sorted(images)
        out_dir = ALL_OUT
    else:
        # Large debris the pipeline found, one per shoot location first.
        found = []
        for img_id, gts in gt_by_img.items():
            hits, _ = match(gts, kept(img_id), thr)
            for g, ok in zip(gts, hits):
                if ok and g["category"] in LARGE:
                    found.append((max(g["bbox"][2:]), img_id, loc(img_id)))
        found.sort(key=lambda t: -t[0])
        chosen, seen = [], set()
        for size, img_id, lc in found:
            if lc not in seen:
                chosen.append(img_id)
                seen.add(lc)
        for size, img_id, lc in found:
            if len(chosen) >= args.n:
                break
            if img_id not in chosen:
                chosen.append(img_id)
        chosen = chosen[:args.n]
        out_dir = OUT / "examples"

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("*.jpg")) + list(out_dir.glob("summary.json")):
        old.unlink()
    summary = {"method": args.method, "which": args.which, "threshold": thr, "images": {}}
    for img_id in chosen:
        img = Image.open(DATASET / "images" / images[img_id]["file_name"]).convert("RGB")
        title = (f"{args.method} ({args.which})   image {img_id:02d}   location {loc(img_id)}")
        fig, stats = figure(img, gt_by_img[img_id], kept(img_id), thr, title, road.get(img_id))
        path = out_dir / f"{img_id:02d}_{stats['debris_category']}.jpg"
        fig.save(path, quality=88)
        summary["images"][f"{img_id:02d}"] = {"location": loc(img_id), **stats}
        print(f"saved: {path.relative_to(REPO_ROOT)}")
    if args.all:
        v = summary["images"].values()
        summary["totals"] = {
            "vehicles": sum(s["vehicles"] for s in v),
            "vehicles_found": sum(s["vehicles_found"] for s in v),
            "debris": len(v), "debris_found": sum(s["debris_found"] for s in v),
            "false_alarms": sum(s["false_alarms"] for s in v)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["totals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
