"""Vehicle labels for the synthetic debris dataset, so the pipeline is scored
on "find everything on the highway" (vehicles + debris).

Two steps, run in this order:

  python scripts/add_vehicle_labels.py --backgrounds
      Detect vehicles on every clean background frame in the pool
      (highway_backgrounds.sample()) and cache them in
      background_vehicles.json. build_synthetic_dataset.py reads this cache
      to anchor debris placement next to real traffic.

  python scripts/add_vehicle_labels.py
      Write the "vehicle" annotations of every sample from that cache plus
      the manual review in vehicle_label_review.json. Idempotent: existing
      vehicle annotations are dropped and rebuilt, debris annotations are
      never touched.

Why detect on the background, not the composite: run on the composite, the
detector labelled a rendered ladder as a vehicle.

Why a manual review: the background pool is all Pexels footage, which has no
annotation anywhere. Grounding DINO runs on overlapping 800 px tiles at
native resolution (one whole-frame pass shrunk to 1024 px missed most distant
cars). No threshold is clean on its own: at 0.30 it misses motion-blurred
trucks, at 0.22 it also boxes trees, pallets and parked trailers. So every
candidate at 0.22 was checked by eye on the frames the dataset uses. The
review file lists, per background, which candidates to drop, which missed
vehicles to add by hand, and which real vehicles to ignore because they are
off the highway (an overpass, a side road). Hand-added boxes are approximate;
the benchmark's hit criterion (IoU > 0.1) tolerates that.

Each image also gets `ignore_regions` (x, y, w, h) from
highway_backgrounds.IGNORE_BANDS: areas off the highway, such as parking
lots. A vehicle centred inside one is not labelled, and a prediction centred
inside one is neither a hit nor a false alarm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import highway_backgrounds  # noqa: E402

DATASET = REPO_ROOT / "datasets" / "synthetic-highway-debris"
CACHE = DATASET / "background_vehicles.json"
REVIEW = DATASET / "vehicle_label_review.json"

VEHICLE_PROMPT = "car. truck. bus. van. trailer. lorry."
CANDIDATE_THRESHOLD = 0.22   # everything the review looks at
ANCHOR_THRESHOLD = 0.30      # confident enough to anchor debris placement
TILE, OVERLAP = 800, 200


def _origins(size: int) -> list[int]:
    if size <= TILE:
        return [0]
    step = TILE - OVERLAP
    o = list(range(0, size - TILE + 1, step))
    if o[-1] + TILE < size:
        o.append(size - TILE)
    return o


def merge_boxes(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Greedy NMS on xyxy boxes that also drops a box mostly contained in a
    higher-scoring one: a car cut by a tile edge comes back as a fragment
    of the full box from the neighbouring tile, and plain IoU misses that."""
    keep: list[int] = []
    for i in np.argsort(-scores):
        b = boxes[i]
        ab = (b[2] - b[0]) * (b[3] - b[1])
        dup = False
        for j in keep:
            k = boxes[j]
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            ak = (k[2] - k[0]) * (k[3] - k[1])
            if inter / (ab + ak - inter) > 0.5 or inter / max(1.0, min(ab, ak)) > 0.7:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return keep


class TiledGdino:
    def __init__(self):
        import torch
        import transformers
        p = REPO_ROOT / "models" / "grounding-dino-tiny"
        self.torch = torch
        self.proc = transformers.AutoProcessor.from_pretrained(str(p))
        self.m = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(p), dtype=torch.float32).to("cuda").eval()

    def __call__(self, img: Image.Image) -> list[list[float]]:
        """[[x, y, w, h, score], ...] in full-image pixels."""
        allb, alls = [], []
        for y0 in _origins(img.height):
            for x0 in _origins(img.width):
                crop = img.crop((x0, y0, min(img.width, x0 + TILE), min(img.height, y0 + TILE)))
                inp = self.proc(images=crop, text=VEHICLE_PROMPT, return_tensors="pt").to("cuda")
                with self.torch.inference_mode():
                    out = self.m(**inp)
                r = self.proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=CANDIDATE_THRESHOLD, text_threshold=0.2,
                    target_sizes=[crop.size[::-1]])[0]
                for b, s in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()):
                    # A box spanning most of a tile is a road or lane, not a vehicle.
                    if b[2] - b[0] > 0.5 * crop.width or b[3] - b[1] > 0.5 * crop.height:
                        continue
                    allb.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                    alls.append(float(s))
        if not allb:
            return []
        boxes, scores = np.array(allb), np.array(alls)
        out = []
        for i in merge_boxes(boxes, scores):
            x0, y0, x1, y1 = (int(round(v)) for v in boxes[i])
            out.append([x0, y0, x1 - x0, y1 - y0, round(float(scores[i]), 3)])
        return out


def bg_key(path) -> str:
    """Background path relative to the repo, forward slashes: the key used in
    the cache, the review file and manifest.json alike."""
    p = Path(path)
    if p.is_absolute():
        p = p.relative_to(REPO_ROOT)
    return p.as_posix()


def ignore_regions(bg: str, w: int, h: int) -> list[list[int]]:
    loc = highway_backgrounds.location_key(Path(bg))
    return [[0, int(h * t), w, int(h * (b - t))]
            for t, b in highway_backgrounds.IGNORE_BANDS.get(loc, [])]


def centre_in(box, regions) -> bool:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return any(x <= cx < x + w and y <= cy < y + h for x, y, w, h in regions)


def detect_backgrounds() -> int:
    frames = highway_backgrounds.sample()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    todo = [f for f in frames if bg_key(f) not in cache]
    print(f"{len(frames)} background frames, {len(todo)} not yet cached")
    if todo:
        det = TiledGdino()
        for i, f in enumerate(todo):
            cache[bg_key(f)] = det(Image.open(f).convert("RGB"))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"saved: {CACHE.relative_to(REPO_ROOT)}")
    return 0


def write_labels() -> int:
    anns = json.loads((DATASET / "annotations.json").read_text(encoding="utf-8"))
    m = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.is_file() else {}
    bg_by_img = {s["image_id"]: bg_key(s["background_source"]) for s in m["samples"]}

    old = {c["id"] for c in anns["categories"] if c["name"] == "vehicle"}
    anns["categories"] = [c for c in anns["categories"] if c["name"] != "vehicle"]
    anns["annotations"] = [a for a in anns["annotations"] if a["category_id"] not in old]
    vehicle_cat_id = max(c["id"] for c in anns["categories"]) + 1
    anns["categories"].append({"id": vehicle_cat_id, "name": "vehicle"})
    ann_id = max(a["id"] for a in anns["annotations"]) + 1

    unreviewed, counts = [], {"detector": 0, "manual": 0}
    for meta in sorted(anns["images"], key=lambda im: im["id"]):
        bg = bg_by_img[meta["id"]]
        w, h = meta["width"], meta["height"]
        ign = ignore_regions(bg, w, h)
        rv = review.get(bg)
        if rv is None:
            unreviewed.append(meta["id"])
            rv = {}
        drop = set(rv.get("drop", [])) | set(rv.get("ignore_ids", []))
        # Real vehicles that are not on the highway (overpass, side road):
        # neither targets nor false alarms.
        ign = ign + [cache[bg][i][:4] for i in rv.get("ignore_ids", [])] + rv.get("ignore", [])
        boxes = [(c[:4], "gdino_tiled_reviewed") for i, c in enumerate(cache[bg])
                 if i not in drop]
        boxes += [(b, "manual") for b in rv.get("add", [])]

        meta["ignore_regions"] = ign
        meta["vehicle_label_source"] = "gdino_tiled + manual review"
        for (x, y, bw, bh), src in boxes:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0 or centre_in((x0, y0, x1 - x0, y1 - y0), ign):
                continue
            anns["annotations"].append({
                "id": ann_id, "image_id": meta["id"], "category_id": vehicle_cat_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0, "label_source": src,
            })
            ann_id += 1
            counts["detector" if src != "manual" else "manual"] += 1

    (DATASET / "annotations.json").write_text(json.dumps(anns, indent=2), encoding="utf-8")
    print(f"vehicles: {counts['detector']} detector-proposed and kept, "
          f"{counts['manual']} added by hand")
    if unreviewed:
        print(f"[warn] {len(unreviewed)} image(s) not in {REVIEW.name}: {unreviewed}")
    print(f"saved: {(DATASET / 'annotations.json').relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", action="store_true",
                    help="detect and cache vehicles on every background frame")
    args = ap.parse_args()
    return detect_backgrounds() if args.backgrounds else write_labels()


if __name__ == "__main__":
    raise SystemExit(main())
