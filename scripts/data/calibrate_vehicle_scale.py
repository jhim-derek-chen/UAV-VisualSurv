"""Calibrate pixels-per-metre for every background location, using real cars
already visible in the frame as the ruler.

Root cause this fixes: build_synthetic_dataset.py originally sized every
debris object as a fixed fraction of the *background image's pixel width*.
That is only correct if every background was shot at the same altitude and
lens -- ours were not (UAVDT is a low-altitude research rig; some Pexels
clips are wide-angle establishing shots from much higher up). The same
"4% of image width" is a small suitcase in one frame and a truck-sized slab
in another. A spot check (image_id 285/130 in the original 300-sample run)
confirmed this: a suitcase rendered wider than a real truck three lanes away
in the same frame.

Fix: detect real cars/trucks in a handful of frames per location with
Grounding DINO (already used for Stage 1, see segment_benchmark.py), take
the median of each box's shorter side as a stand-in for a car's ~1.8 m
width, and derive pixels-per-metre from that -- independent of image
resolution or camera altitude. Debris is then sized in real metres
(render_debris.REAL_SIZE, the same figures Blender already normalizes each
3D model to) times this location's pixels-per-metre, instead of a fraction
of image width.

Locations with too few detected vehicles (empty rural frames) fall back to
the old width-fraction heuristic, flagged in the output so that is visible
rather than silently assumed correct.

    python scripts/data/calibrate_vehicle_scale.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
import highway_backgrounds  # noqa: E402

MODELS = REPO_ROOT / "models"
OUT_PATH = REPO_ROOT / "datasets" / "synthetic-highway-debris" / "scale_calibration.json"
CAR_WIDTH_M = 1.8       # typical passenger car width, the calibration anchor
FRAMES_PER_LOCATION = 6
MIN_DETECTIONS = 3      # below this, a location's calibration isn't trusted
PROMPT = "car. truck."
THRESHOLD = 0.3


def group_by_location(frames: list[Path]) -> dict[str, list[Path]]:
    by_loc = defaultdict(list)
    for f in frames:
        by_loc[highway_backgrounds.location_key(f)].append(f)
    return by_loc


def main() -> int:
    frames = highway_backgrounds.sample()
    by_loc = group_by_location(frames)
    print(f"{len(frames)} frames across {len(by_loc)} locations")

    p = MODELS / "grounding-dino-tiny"
    proc = transformers.AutoProcessor.from_pretrained(str(p))
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
        str(p), dtype=torch.float32).to("cuda").eval()

    calibration = {}
    for loc, loc_frames in sorted(by_loc.items()):
        import random
        random.seed(0)
        sample_frames = random.sample(loc_frames, min(FRAMES_PER_LOCATION, len(loc_frames)))
        widths = []
        for fp in sample_frames:
            img = Image.open(fp).convert("RGB")
            sc = min(1.0, 1024 / max(img.size))
            small = img.resize((int(img.width * sc), int(img.height * sc))) if sc < 1 else img
            inp = proc(images=small, text=PROMPT, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**inp)
            r = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=THRESHOLD, text_threshold=0.2,
                target_sizes=[small.size[::-1]])[0]
            for b in r["boxes"].cpu().numpy():
                x0, y0, x1, y1 = b
                w, h = (x1 - x0) / sc, (y1 - y0) / sc
                widths.append(float(min(w, h)))

        if len(widths) >= MIN_DETECTIONS:
            car_px = float(np.median(widths))
            px_per_metre = car_px / CAR_WIDTH_M
            calibration[loc] = {
                "px_per_metre": px_per_metre, "n_detections": len(widths),
                "car_px_median": car_px, "fallback": False,
            }
            print(f"{loc:30s} n={len(widths):3d}  car_px={car_px:6.1f}  "
                  f"px/m={px_per_metre:6.1f}")
        else:
            calibration[loc] = {"px_per_metre": None, "n_detections": len(widths),
                                 "fallback": True}
            print(f"{loc:30s} n={len(widths):3d}  -- too few detections, "
                  f"falls back to width-fraction sizing")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "car_width_m": CAR_WIDTH_M, "locations": calibration,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT_PATH.relative_to(REPO_ROOT)}")
    n_fallback = sum(1 for v in calibration.values() if v["fallback"])
    print(f"{len(calibration) - n_fallback}/{len(calibration)} locations calibrated, "
          f"{n_fallback} fall back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
