"""Training renders for the Stage 2 re-scorer (scripts/see/train_rescorer.py).

The 30-image test set (datasets/synthetic-highway-debris) stays fixed and is
never trained on. This builds a separate set from the background frames the
test set did not use: the same pool (highway_backgrounds.sample(), open
highway only), minus every frame a test sample was composited onto.

Each training image gets several debris objects instead of one, for more
positives per render. Same rendering, scale and placement code as the test
set (build_synthetic_dataset.py), so the debris looks the same.

Vehicle labels here are NOT reviewed by hand: Grounding DINO boxes from
background_vehicles.json at >= 0.35 are labels, 0.22-0.35 are ignore regions
(could be either), plus highway_backgrounds.IGNORE_BANDS. Noisy, but only used
to train the re-scorer, never to score it.

Leakage note: the training frames come from the same four clips as the test
frames. train_rescorer.py therefore evaluates leave-one-location-out: the
re-scorer applied to one location's test images is trained only on the other
three locations' renders.

    python scripts/data/build_training_renders.py --per-image 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
import build_synthetic_dataset as B  # noqa: E402
import highway_backgrounds  # noqa: E402

TEST = REPO_ROOT / "datasets" / "synthetic-highway-debris"
OUT = REPO_ROOT / "datasets" / "synthetic-highway-debris-train"
POSITIVE_MIN, IGNORE_MIN = 0.35, 0.22


class DroneRoad:
    """Placement road mask from the drone-view segmenter
    (train_road_segmenter.py) instead of the Cityscapes one the test set
    used: the Cityscapes model called a container yard road, and these
    frames have no hand review to catch that."""

    def __init__(self):
        import transformers
        from train_road_segmenter import OUT as MODEL_DIR
        self.m = transformers.SegformerForSemanticSegmentation.from_pretrained(
            str(MODEL_DIR)).cuda().eval()

    def __call__(self, img: Image.Image) -> np.ndarray:
        from train_road_segmenter import predict
        return predict(self.m, img) > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--render-res", type=int, default=640)
    args = ap.parse_args()
    random.seed(args.seed)

    test_manifest = json.loads((TEST / "manifest.json").read_text(encoding="utf-8"))
    used = {Path(s["background_source"]).as_posix() for s in test_manifest["samples"]}
    frames = [f for f in highway_backgrounds.sample()
              if f.relative_to(REPO_ROOT).as_posix() not in used]
    print(f"{len(frames)} background frames not used by the test set")

    real_size = B.load_real_size()
    calibration = B.load_calibration()
    debris = B.load_debris_manifest()
    cache = B.load_vehicle_cache()
    raw_cache = json.loads(B.VEHICLE_CACHE.read_text(encoding="utf-8"))
    cat_names = list(debris)
    weights = [B.CATEGORY_WEIGHT[c] for c in cat_names]

    segmenter = DroneRoad()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    images, annotations, samples = [], [], []
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(B.WIDTH_FRAC)] + \
           [{"id": len(B.WIDTH_FRAC) + 1, "name": "vehicle"}]
    cat_id = {c["name"]: c["id"] for c in cats}
    ann_id = 1
    for img_id, bg_path in enumerate(frames):
        key = bg_path.relative_to(REPO_ROOT).as_posix()
        bg = Image.open(bg_path).convert("RGB")
        canvas = bg
        mask_total = np.zeros((bg.height, bg.width), dtype=np.uint8)
        # Debris placed so far blocks later placements, like a vehicle.
        vehicles = [list(v) for v in cache.get(key, [])]
        objs = []
        for k in range(args.per_image):
            category = random.choices(cat_names, weights=weights, k=1)[0]
            pick = random.choice(debris[category])
            model_path = next((B.MODELS_ROOT / category / pick["uid"]).glob("*.gltf"), None)
            if model_path is None:
                continue
            target_w, _ = B.target_width_px(category, bg, bg_path, real_size, calibration)
            cx, cy, placement = B.choose_placement(bg, bg_path, segmenter, vehicles, target_w)
            seed = 100000 + img_id * 10 + k
            try:
                render, ppm = B.render_object(model_path, category, seed=seed, res=args.render_res)
                scale = target_w / (real_size[category] * ppm)
                canvas, m, bbox = B.composite(canvas, render, cx, cy, scale)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {category}: {exc}")
                continue
            mask_total = np.maximum(mask_total, m)
            vehicles.append([*bbox, 0.0])  # score 0: blocks placement, never an anchor
            objs.append({"category": category, "bbox": list(bbox), "placement": placement,
                         "model_uid": pick["uid"], "render_seed": seed})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cat_id[category],
                                "bbox": list(bbox), "area": int((m > 0).sum()), "iscrowd": 0})
            ann_id += 1

        w, h = bg.size
        ignore = [[0, int(h * t), w, int(h * (b - t))]
                  for t, b in highway_backgrounds.IGNORE_BANDS.get(
                      highway_backgrounds.location_key(bg_path), [])]
        for x, y, bw, bh, s in raw_cache.get(key, []):
            if s >= POSITIVE_MIN:
                annotations.append({"id": ann_id, "image_id": img_id,
                                    "category_id": cat_id["vehicle"], "bbox": [x, y, bw, bh],
                                    "area": bw * bh, "iscrowd": 0,
                                    "label_source": "gdino_tiled_unreviewed"})
                ann_id += 1
            elif s >= IGNORE_MIN:
                ignore.append([x, y, bw, bh])
        name = f"{img_id:05d}.jpg"
        canvas.save(OUT / "images" / name, quality=92)
        Image.fromarray(mask_total).save(OUT / "masks" / f"{img_id:05d}.png")
        images.append({"id": img_id, "file_name": name, "width": w, "height": h,
                       "ignore_regions": ignore})
        samples.append({"image_id": img_id, "background_source": key, "objects": objs})
        print(f"  {img_id + 1}/{len(frames)}  {len(objs)} debris", flush=True)

    (OUT / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": cats}, indent=1),
        encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(
        {"per_image": args.per_image, "seed": args.seed, "samples": samples}, indent=1,
        ensure_ascii=False), encoding="utf-8")
    print(f"saved: {OUT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
