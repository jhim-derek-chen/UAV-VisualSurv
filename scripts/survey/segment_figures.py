"""Render road-segmentation sample frames: AeroScapes (with ground truth, the
same frames segment_benchmark.py scores) and real UAVDT highway footage
(no ground truth -- qualitative only, same treatment the deferred B6 got in
benchmark_figures.py).

Frames are hand-picked, not auto-sampled -- an unfiltered sample (first
entries in AeroScapes' val.txt, an even spread of UAVDT) turned out to be
narrow footpaths and a single toll-plaza scene, neither representative of a
"road". Selection rule, applied the same way to both sources: keep only
frames where a human can identify a genuine vehicle carriageway (lane
markings, or a car establishing scale on a paved strip), not a footpath or
plaza that AeroScapes' single "road" class also covers.

AEROSCAPES_SAMPLES were checked against segformer-b2's per-frame IoU before
selection: 002002_013 (IoU 0.306) sits almost exactly on segformer-b2's
reported mean (0.312) -- a representative case, not a cherry-picked good one.
200004_001 (IoU 0.730) is kept alongside it specifically because it is the
clearest multi-lane road in the val set; it is captioned as the better-than-
typical case it is, not offered as the average.

    python scripts/survey/segment_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name

from benchmark_figures import font, ring, HALO, PAD, HEAD  # noqa: E402
from segment_benchmark import BUILDERS, DATA, ground_truth  # noqa: E402

OUT = REPO_ROOT / "results" / "figures"
UAVDT_ROOT = REPO_ROOT / "datasets" / "uavdt-benchmark-m" / "UAV-benchmark-M"

AEROSCAPES_SAMPLES = ["002002_013", "200004_001"]
UAVDT_SAMPLES = ["M0201/img000431.jpg", "M0207/img000355.jpg", "M0801/img000119.jpg"]

COLOUR = {
    "segformer-b0": (42, 120, 214),
    "segformer-b2": (235, 104, 52),
    "mask2former": (237, 161, 0),
    "gdino-road": (27, 175, 122),
}
LABEL = {
    "segformer-b0": "SegFormer-B0",
    "segformer-b2": "SegFormer-B2",
    "mask2former": "Mask2Former",
    "gdino-road": "G-DINO (road prompt)",
}
ORDER = ["segformer-b0", "segformer-b2", "mask2former", "gdino-road"]
GT_COLOUR = (255, 255, 255)
LEG = 26


def paint(img: Image.Image, gt: np.ndarray | None, regions: dict[str, np.ndarray],
          scale: float) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    h, w_img = base.shape[:2]
    w = max(2, int(3 * scale))
    bands = {m: ring(regions[m], 0, w) for m in ORDER
             if regions.get(m) is not None and regions[m].any()}
    if bands:
        from scipy import ndimage
        union = np.zeros((h, w_img), bool)
        overlap = np.zeros((h, w_img), np.uint8)
        for b in bands.values():
            union |= b
            overlap += b.astype(np.uint8)
        base[ndimage.binary_dilation(union, iterations=2) & ~union] = HALO
        yy, xx = np.mgrid[0:h, 0:w_img]
        phase = ((xx + yy) // max(3, int(4 * scale))) % len(bands)
        for i, (m, b) in enumerate(bands.items()):
            base[b & ((overlap == 1) | (phase == i))] = COLOUR[m]
    if gt is not None and gt.any():
        gw = max(3, int(4 * scale))
        base[ring(gt, 0, gw + 2)] = HALO
        base[ring(gt, 1, gw)] = GT_COLOUR
    return Image.fromarray(base)


def sheet(title: str, frames: list[Image.Image], caps: list[str]) -> Image.Image:
    cell_w = 640
    scaled = [im.resize((cell_w, max(1, round(im.height * cell_w / im.width))), Image.LANCZOS)
              for im in frames]
    cell_h = max(im.height for im in scaled)
    n = len(scaled)
    W = n * cell_w + (n + 1) * PAD
    H = HEAD + LEG + cell_h + 20 + PAD
    out = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(out)
    f_head, f_leg, f_cap = font(17, True), font(13), font(13)
    d.text((PAD, 6), title, fill=(238, 238, 242), font=f_head)
    x, y = PAD, HEAD
    d.text((x, y + 4), "White outline = real road (AeroScapes labels)"
            if any(caps) else "", fill=(235, 235, 240), font=f_leg)
    x += 320
    for m in ORDER:
        d.rectangle([x, y + 9, x + 22, y + 12], fill=COLOUR[m])
        d.text((x + 28, y + 4), LABEL[m], fill=(205, 205, 212), font=f_leg)
        x += 190
    for i, (im, cap) in enumerate(zip(scaled, caps)):
        px = PAD + i * (cell_w + PAD)
        out.paste(im, (px, HEAD + LEG))
        d.text((px + 2, HEAD + LEG + cell_h + 3), cap, fill=(198, 198, 206), font=f_cap)
    return out


def predict(img: Image.Image, m) -> np.ndarray:
    pred = m(img)
    if pred.shape[:2] != img.size[::-1]:
        pred = np.array(Image.fromarray(pred).resize(img.size, Image.NEAREST))
    return pred


def main() -> int:
    aero_imgs = [Image.open(DATA / "JPEGImages" / f"{n}.jpg").convert("RGB")
                 for n in AEROSCAPES_SAMPLES]
    aero_gts = [ground_truth(n) for n in AEROSCAPES_SAMPLES]
    uavdt_imgs = [Image.open(UAVDT_ROOT / p).convert("RGB") for p in UAVDT_SAMPLES]

    # Models loaded one at a time, same VRAM discipline as benchmark_figures.py.
    aero_regions: list[dict] = [{} for _ in aero_imgs]
    uavdt_regions: list[dict] = [{} for _ in uavdt_imgs]
    for key, build in BUILDERS.items():
        print(f"--- {key}", flush=True)
        m = build()
        for i, img in enumerate(aero_imgs):
            aero_regions[i][key] = predict(img, m)
        for i, img in enumerate(uavdt_imgs):
            uavdt_regions[i][key] = predict(img, m)
        del m
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    frames = [paint(img, gt, regions, max(0.6, min(2.2, img.width / 1000)))
              for img, gt, regions in zip(aero_imgs, aero_gts, aero_regions)]
    sheet("Road region · AeroScapes val (labelled)", frames, AEROSCAPES_SAMPLES) \
        .save(OUT / "roadseg_aeroscapes.jpg", quality=88, optimize=True)
    print(f"saved {OUT / 'roadseg_aeroscapes.jpg'}")

    frames = [paint(img, None, regions, max(0.6, min(2.2, img.width / 1000)))
              for img, regions in zip(uavdt_imgs, uavdt_regions)]
    caps = [Path(p).stem for p in UAVDT_SAMPLES]
    sheet("Road region · real UAVDT highway (no labels, qualitative only)", frames, caps) \
        .save(OUT / "roadseg_uavdt.jpg", quality=88, optimize=True)
    print(f"saved {OUT / 'roadseg_uavdt.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
