"""Build the synthetic highway-debris dataset: real UAV backgrounds (see
scripts/data/highway_backgrounds.py) + rendered 3D debris (see
scripts/data/render_debris.py), composited with a placement location chosen by
Stage 1's own road segmenter -- so the object lands on the road, not on the
grass or a building roof next to it.

Object scale is real metres (render_debris.REAL_SIZE) times the background's
own pixels-per-metre, calibrated per shoot location from real cars/trucks
already visible in the frame -- see scripts/data/calibrate_vehicle_scale.py. Run
that script first; its output feeds this one. An earlier version sized
objects as a fixed fraction of the background image's pixel width, which
does not hold across the three background sources (UAVDT, VisDrone, Pexels
were shot at different altitudes and lenses with no shared calibration): a
suitcase sized that way came out wider than a real truck three lanes away in
one Pexels frame (image_id 285 in the original 300-sample run). WIDTH_FRAC
below survives only as the fallback for a location calibration couldn't
measure (too few detected vehicles).

Two more fixes from a later QA pass, both in main(): (1) category is chosen
with CATEGORY_WEIGHT, not uniformly -- a uniform choice across 12 categories
made genuinely-large, legible objects (truck-tire, refrigerator, ...) rare
enough in a 300-sample run to look absent. (2) backgrounds are drawn from a
shuffled queue with no replacement instead of `random.choice` each time, so
a run no longer reuses the same background frame for multiple samples (the
background pool, ~420 frames, comfortably covers the default 100).

Output: datasets/synthetic-highway-debris/
    images/<id>.jpg         composite, same size as the source background
    masks/<id>.png          single-channel, 255 = debris pixel
    annotations.json        COCO-style: images, categories, one segmentation
                             + bbox per object (pixel-exact, taken directly
                             from the compositing alpha -- no manual labelling
                             anywhere in this pipeline)
    manifest.json            generation parameters, per-sample provenance
                             (background source, 3D model + its Sketchfab
                             credit, placement, seed) for reproducibility

    python scripts/data/build_synthetic_dataset.py --n 30
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name

import highway_backgrounds  # noqa: E402
# Not `from render_debris import REAL_SIZE`: that module does `import bpy` at
# top level, which only exists inside Blender's embedded interpreter and
# would crash this plain-Python process. load_real_size() below parses that
# dict out of render_debris.py's source text instead (and cross-checks its
# keys against WIDTH_FRAC), so the two can't drift silently.

BLENDER = REPO_ROOT / "tools" / "blender-5.2.2-windows-x64" / "blender.exe"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "data" / "render_debris.py"
MODELS_ROOT = REPO_ROOT / "datasets" / "debris-3d-models"
OUT = REPO_ROOT / "datasets" / "synthetic-highway-debris"
TMP_RENDER = REPO_ROOT / ".cache" / "tmp" / "debris_render.png"
CALIBRATION_PATH = OUT / "scale_calibration.json"

# Fallback only, used for a location scripts/data/calibrate_vehicle_scale.py could
# not calibrate (too few detected vehicles). Primary sizing is real metres
# (render_debris.REAL_SIZE) times that location's measured pixels-per-metre
# -- see calibrate_vehicle_scale.py's docstring for why: a fixed fraction of
# image width was found to size the same category wildly differently
# depending on a background's altitude/lens (a suitcase rendered wider than
# a real truck three lanes away in one Pexels frame, id 285 in the original
# 300-sample run -- not a rare fluke, every location's calibration differs by
# up to 7.5x). Ratios here still match REAL_SIZE (e.g. mattress/tire ~=
# 1.9/0.65, 0.065/0.022 here) so the fallback stays internally consistent.
WIDTH_FRAC = {
    "tire": 0.022,
    "cardboard-box": 0.020,
    "suitcase": 0.024,
    "traffic-cone": 0.024,
    "barrel": 0.030,
    "wooden-pallet": 0.040,
    "mattress": 0.065,
    "trash-can": 0.028,
    "wooden-plank": 0.068,
    "ladder": 0.068,
    "truck-tire": 0.037,
    "refrigerator": 0.059,
}

# Category sampling weight -- uniform choice made "genuinely small in real
# life" and "genuinely large" equally likely, so a user QA pass on the
# calibrated-size dataset found large, legible objects (truck-tire etc.)
# "vanishingly rare" even though every category had a roughly equal share.
# Three tiers by real-world size (render_debris.REAL_SIZE), not by category
# identity, so this stays principled rather than hand-tuned per category:
# small (<=0.9m) keeps some representation (this is still meant to cover
# small real debris), large (>=1.75m, plus truck-tire) is weighted up most.
# All categories still appear -- weighting, not exclusion.
_SMALL, _MEDIUM, _LARGE = 1, 3, 5
CATEGORY_WEIGHT = {
    "tire": _SMALL, "cardboard-box": _SMALL, "suitcase": _SMALL,
    "traffic-cone": _SMALL, "barrel": _SMALL, "trash-can": _SMALL,
    "wooden-pallet": _MEDIUM,
    "mattress": _LARGE, "wooden-plank": _LARGE, "ladder": _LARGE,
    "truck-tire": _LARGE, "refrigerator": _LARGE,
}
# User still found tires "barely visible" scrolling through the size-tier-
# weighted batch above, even with truck-tire in the large tier: it's the
# *smallest* of the large tier by real size (1.10m vs 1.75-2.0m for
# mattress/plank/ladder/fridge), so it was still the least prominent among
# them. This is a direct "I want to see more tires" content request, not a
# size-physics one, so it overrides the tier weights above rather than
# extending them -- both tire categories become the two most frequent in
# the set.
CATEGORY_WEIGHT["truck-tire"] = 9
CATEGORY_WEIGHT["tire"] = 4

# Same request, the size half: "轮胎又多又大" (tires, more AND bigger) --
# not just more frequent (CATEGORY_WEIGHT above) but individually larger
# than their calibrated real-world size would put them. This is a
# deliberate, explicit departure from the vehicle-calibration in
# target_width_px() below for exactly these two categories, not a revival
# of the old blanket _SIZE_BOOST that was removed for being uncalibrated
# guesswork -- that one applied to every category with no stated reason;
# this one is scoped to tire/truck-tire only, at the user's direct request.
VISIBILITY_BOOST = {"tire": 2.0, "truck-tire": 1.8}

# A flat multiplier still comes out small on a location with a low measured
# pixels-per-metre (a high-altitude/wide shot) -- one run drew mostly such
# locations for truck-tire by chance and its median landed at 12px (the
# generic floor) despite the 1.8x boost, while tire's 12 samples happened to
# draw mostly close-up locations and came out at 33px median. Same physics,
# opposite luck. A per-category minimum removes the luck-dependence: tires
# are now reliably at least this size regardless of which background they
# land on, and still scale up further from calibration+boost on a close-up
# location (up to ~57px seen so far).
MIN_WIDTH_PX = {"tire": 24, "truck-tire": 28}


def load_debris_manifest() -> dict:
    return json.loads((MODELS_ROOT / "manifest.json").read_text(encoding="utf-8"))


def load_real_size() -> dict:
    """render_debris.py can't be imported here (it does `import bpy`), so its
    REAL_SIZE dict -- metres, the same figures Blender normalizes every 3D
    model to -- is parsed out of the source text instead of duplicated by
    hand where it could silently drift."""
    import ast
    src = (REPO_ROOT / "scripts" / "data" / "render_debris.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "REAL_SIZE" for t in node.targets):
            real_size = ast.literal_eval(node.value)
            if set(real_size) != set(WIDTH_FRAC):
                raise SystemExit(
                    "WIDTH_FRAC and render_debris.REAL_SIZE category lists have "
                    f"diverged: {set(real_size).symmetric_difference(WIDTH_FRAC)}")
            return real_size
    raise SystemExit("could not find REAL_SIZE in render_debris.py to cross-check")


def load_calibration() -> dict:
    if not CALIBRATION_PATH.is_file():
        print(f"[warn] no {CALIBRATION_PATH.name} -- run "
              f"scripts/data/calibrate_vehicle_scale.py first; falling back to "
              f"width-fraction sizing for every background")
        return {}
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["locations"]


def target_width_px(category: str, bg: Image.Image, bg_path: Path,
                     real_size: dict, calibration: dict) -> tuple[int, str]:
    """Real metres x this location's measured pixels-per-metre, falling back
    to the old image-width-fraction heuristic only where no vehicle-based
    calibration exists for the location. Returns (pixels, source-label) --
    the label is recorded per sample so a fallback is visible, not silent."""
    loc = highway_backgrounds.location_key(bg_path)
    boost = VISIBILITY_BOOST.get(category, 1.0)
    floor = MIN_WIDTH_PX.get(category, 12)
    cal = calibration.get(loc)
    if cal and not cal.get("fallback") and cal.get("px_per_metre"):
        w = real_size[category] * cal["px_per_metre"] * boost
        return max(floor, round(w)), "vehicle_calibration"
    w = bg.width * WIDTH_FRAC[category] * boost
    return max(floor, round(w)), "width_fraction_fallback"


def render_object(model_path: Path, category: str, seed: int,
                  res: int) -> tuple[Image.Image, float]:
    """Returns the RGBA render (object plus shadow) and its pixels per metre.
    The object-only alpha from render_debris.py's second pass is attached as
    `im.info["object_alpha"]` (an L image); the label is cut from that."""
    TMP_RENDER.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BLENDER), "--background", "--python", str(RENDER_SCRIPT), "--",
           "--model", str(model_path), "--category", category,
           "--out", str(TMP_RENDER), "--seed", str(seed), "--res", str(res)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not TMP_RENDER.is_file():
        raise RuntimeError(f"render failed for {model_path}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    im = Image.open(TMP_RENDER).convert("RGBA")
    obj_path = TMP_RENDER.with_name(TMP_RENDER.stem + "_obj.png")
    im.info["object_alpha"] = Image.open(obj_path).convert("RGBA").split()[3]
    meta = TMP_RENDER.with_suffix(".json")
    px_per_metre = json.loads(meta.read_text(encoding="utf-8"))["px_per_metre"]
    for f in (TMP_RENDER, obj_path, meta):
        f.unlink()
    return im, px_per_metre


def road_mask_for(bg: Image.Image, segmenter) -> np.ndarray:
    """Reuses Stage 1's own SegFormer-B2 (segment_benchmark.SegformerRoad) --
    the model this project already measured and picked as the more reliable
    of the two Stage 1 candidates for consistency across frames -- to decide
    where "the road" is in an arbitrary background, rather than guessing a
    fixed image region."""
    pred = segmenter(bg.convert("RGB"))
    if pred.shape != (bg.height, bg.width):
        pred = np.array(Image.fromarray(pred).resize((bg.width, bg.height), Image.NEAREST))
    return pred


def pick_placement(mask: np.ndarray, margin_frac: float = 0.08,
                   band: tuple[float, float] | None = None) -> tuple[int, int]:
    """A random point on the predicted road, away from the frame edge (an
    object half-cropped at the image border is not a useful training
    sample). Falls back to the image's lower-centre band -- most of these
    backgrounds were framed with the road filling that region -- if the
    segmenter found no road at all in this frame."""
    h, w = mask.shape
    mx, my = int(w * margin_frac), int(h * margin_frac)
    interior = np.zeros_like(mask)
    interior[my:h - my, mx:w - mx] = True
    if band is not None:  # highway_backgrounds.PLACEMENT_BAND
        interior[:int(h * band[0])] = False
        interior[int(h * band[1]):] = False
    candidates = np.argwhere(mask & interior)
    if len(candidates) == 0:
        return (w // 2 + random.randint(-w // 6, w // 6),
                int(h * random.uniform(0.55, 0.85)))
    y, x = candidates[random.randrange(len(candidates))]
    return int(x), int(y)


VEHICLE_CACHE = OUT / "background_vehicles.json"
ANCHOR_THRESHOLD = 0.30


VEHICLE_REVIEW = OUT / "vehicle_label_review.json"


def load_vehicle_cache() -> dict:
    """Placement anchors per background, keyed by repo-relative path:
    [x, y, w, h, score] vehicles detected by
    `add_vehicle_labels.py --backgrounds`.

    Where a background has been reviewed by hand (vehicle_label_review.json),
    the reviewed set is used instead: dropped detections go, hand-added
    vehicles come in with score 1. An unreviewed detector box at 0.41 turned
    out to be a stack of pallets in a yard, and debris anchored to it landed
    in the yard."""
    if not VEHICLE_CACHE.is_file():
        raise SystemExit(f"{VEHICLE_CACHE.name} missing -- run "
                         f"scripts/data/add_vehicle_labels.py --backgrounds first")
    cache = json.loads(VEHICLE_CACHE.read_text(encoding="utf-8"))
    review = (json.loads(VEHICLE_REVIEW.read_text(encoding="utf-8"))
              if VEHICLE_REVIEW.is_file() else {})
    out = {}
    for bg, dets in cache.items():
        rv = review.get(bg)
        if rv is None:
            out[bg] = dets
            continue
        gone = set(rv.get("drop", [])) | set(rv.get("ignore_ids", []))
        out[bg] = ([d for i, d in enumerate(dets) if i not in gone]
                   + [list(b) + [1.0] for b in rv.get("add", [])])
    return out


def pick_placement_near_traffic(mask: np.ndarray, vehicles: list, target_w: int,
                                band: tuple[float, float] | None = None,
                                margin_frac: float = 0.05,
                                tries: int = 400) -> tuple[int, int] | None:
    """A point in the same lane as a real vehicle: 1.2-4 vehicle lengths ahead
    of or behind it along its long axis, with a little sideways jitter.

    Why: the road segmenter alone put debris on a grass verge, a bridge
    parapet and a shipping-container roof in a QA pass of the first
    open-highway batch. A lane that real traffic is driving in is road by
    definition. The segmenter mask is still required at the point, and the
    debris must not overlap any detected vehicle. Returns None when no
    confident vehicle offers a valid spot; the caller then falls back to
    pick_placement()."""
    h, w = mask.shape
    mx, my = int(w * margin_frac), int(h * margin_frac)
    y_lo, y_hi = (int(h * band[0]), int(h * band[1])) if band else (my, h - my)
    anchors = []
    for x, y, bw, bh, score in vehicles:
        if score < ANCHOR_THRESHOLD or max(bw, bh) < 1.25 * min(bw, bh):
            continue
        cx, cy = x + bw / 2, y + bh / 2
        if not y_lo <= cy < y_hi:
            continue
        # The segmenter labels the vehicle itself "car", not "road", so test
        # the pavement around it: a box grown by half its size on each side.
        rx0, ry0 = max(0, int(x - bw / 2)), max(0, int(y - bh / 2))
        rx1, ry1 = min(w, int(x + 1.5 * bw)), min(h, int(y + 1.5 * bh))
        if mask[ry0:ry1, rx0:rx1].mean() >= 0.25:
            anchors.append((cx, cy, bw, bh))
    if not anchors:
        return None
    pad = target_w
    blocked = [(x - pad, y - pad, x + bw + pad, y + bh + pad) for x, y, bw, bh, _ in vehicles]
    for _ in range(tries):
        cx, cy, bw, bh = random.choice(anchors)
        length, width = max(bw, bh), min(bw, bh)
        along = random.choice((-1, 1)) * random.uniform(1.2, 4.0) * length
        side = random.uniform(-0.6, 0.6) * width
        px, py = (cx + along, cy + side) if bw >= bh else (cx + side, cy + along)
        px, py = int(px), int(py)
        if not (mx <= px < w - mx and max(my, y_lo) <= py < min(h - my, y_hi)):
            continue
        if not mask[py, px]:
            continue
        if any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in blocked):
            continue
        # The straight line from the vehicle to the point must stay on road.
        # On a diagonal carriageway a step along the image axis can cross the
        # verge into a yard the segmenter also calls "road"; the line test
        # catches the verge in between. Starts 0.6 lengths out, past the
        # vehicle body, which the segmenter labels "car", not "road".
        t = np.linspace(0.6 * length / max(1.0, abs(along)), 1.0, 24)
        lx = (cx + (px - cx) * t).astype(int).clip(0, w - 1)
        ly = (cy + (py - cy) * t).astype(int).clip(0, h - 1)
        if not mask[ly, lx].all():
            continue
        return px, py
    return None


def choose_placement(bg: Image.Image, bg_path: Path, segmenter, vehicles: list,
                     target_w: int) -> tuple[int, int, str]:
    loc = highway_backgrounds.location_key(bg_path)
    band = highway_backgrounds.PLACEMENT_BAND.get(loc)
    mask = road_mask_for(bg, segmenter)
    p = pick_placement_near_traffic(mask, vehicles, target_w, band=band)
    if p is not None:
        return p[0], p[1], "near_traffic"
    cx, cy = pick_placement(mask, band=band)
    return cx, cy, "road_mask_only"


OBJECT_ALPHA = 128  # object-only render alpha above this is the object


def composite(bg: Image.Image, obj: Image.Image, cx: int, cy: int,
              scale: float) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Alpha-composite `obj` (square RGBA render, object plus shadow) onto
    `bg` centred on the object at (cx, cy), resized by `scale` (background
    px/m divided by render px/m).

    The first version resized the render so its whole alpha extent was
    `target_w` wide. That extent is mostly faint shadow and haze: a
    mattress's alpha > 10 spanned 385 x 640 px of a 640 px render while the
    mattress itself was 120 px, so every object came out at roughly a third
    of its real size. Scaling by the two known pixels-per-metre values is
    exact.

    The mask and box come from the object-only render
    (obj.info["object_alpha"]), not from the composite's alpha. A second
    version thresholded the composite alpha at 200 to drop the shadow; a
    dark shadow is as opaque as the object, so a truck tyre's label still
    took in its shadow and was twice the tyre's width."""
    obj_alpha = obj.info["object_alpha"]
    oa = np.array(obj_alpha)
    ys, xs = np.where(oa > OBJECT_ALPHA)
    if len(xs) == 0:
        raise RuntimeError("empty render (no opaque object pixels)")
    ocx, ocy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    new_side = max(1, round(obj.width * scale))
    obj_resized = obj.resize((new_side, new_side), Image.LANCZOS)
    alpha_resized = obj_alpha.resize((new_side, new_side), Image.LANCZOS)
    paste_x = int(round(cx - ocx * scale))
    paste_y = int(round(cy - ocy * scale))

    canvas = bg.convert("RGBA").copy()
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.paste(obj_resized, (paste_x, paste_y))
    canvas.alpha_composite(layer)
    composite_rgb = canvas.convert("RGB")

    mask_full = Image.new("L", bg.size, 0)
    mask_full.paste(alpha_resized.point(lambda a: 255 if a > OBJECT_ALPHA else 0),
                    (paste_x, paste_y))
    mask_arr = np.array(mask_full)

    ys2, xs2 = np.where(mask_arr > 0)
    if len(xs2) == 0:
        raise RuntimeError("object fell outside the frame")
    bbox = (int(xs2.min()), int(ys2.min()),
            int(xs2.max() - xs2.min() + 1), int(ys2.max() - ys2.min() + 1))
    return composite_rgb, mask_arr, bbox


def rerender_all(args) -> int:
    """Rebuild every image, mask and debris box in place from the manifest:
    same background, 3D model, placement and render seed, so the scene is
    unchanged and only the compositing code's output can differ."""
    real_size = load_real_size()
    anns = json.loads((OUT / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["samples"]:
        img_id = s["image_id"]
        # Samples made before render_seed was recorded: main() seeded with
        # the sample id, a redo with id + 1000 * redo count.
        seed = s.get("render_seed", img_id + 1000 * s.get("redo_count", 0))
        bg = Image.open(REPO_ROOT / s["background_source"]).convert("RGB")
        model_path = next((MODELS_ROOT / s["category"] / s["model_uid"]).glob("*.gltf"))
        render, render_ppm = render_object(model_path, s["category"], seed=seed,
                                           res=args.render_res)
        scale = s["target_width_px"] / (real_size[s["category"]] * render_ppm)
        cx, cy = s["placement_xy"]
        comp, mask_arr, bbox = composite(bg, render, cx, cy, scale)
        comp.save(OUT / "images" / f"{img_id:05d}.jpg", quality=92)
        Image.fromarray(mask_arr).save(OUT / "masks" / f"{img_id:05d}.png")
        for a in anns["annotations"]:
            if a["image_id"] == img_id and a.get("segmentation_mask"):
                a["bbox"] = list(bbox)
                a["area"] = int((mask_arr > 0).sum())
        s["render_seed"] = seed
        print(f"  {img_id:2d} {s['category']:13s} bbox {bbox[2]}x{bbox[3]}", flush=True)
    (OUT / "annotations.json").write_text(json.dumps(anns, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return 0


def redo_samples(args) -> int:
    """Re-place and re-render the listed samples in place. Background,
    category and 3D model stay the same; only the placement (and the render
    seed) change. Used after a visual QA pass finds a badly placed object.
    Vehicle labels must be rewritten afterwards (add_vehicle_labels.py)."""
    real_size = load_real_size()
    calibration = load_calibration()
    vehicle_cache = load_vehicle_cache()
    anns = json.loads((OUT / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    samples = {s["image_id"]: s for s in manifest["samples"]}
    from segment_benchmark import SegformerRoad
    segmenter = SegformerRoad("segformer-b2-cityscapes")
    for img_id in args.redo:
        s = samples[img_id]
        bg_path = REPO_ROOT / s["background_source"]
        bg = Image.open(bg_path).convert("RGB")
        category = s["category"]
        model_path = next((MODELS_ROOT / category / s["model_uid"]).glob("*.gltf"))
        s["redo_count"] = s.get("redo_count", 0) + 1
        random.seed(args.seed * 100003 + img_id * 101 + s["redo_count"])
        target_w, scale_source = target_width_px(category, bg, bg_path, real_size, calibration)
        if args.at:
            if len(args.redo) != 1:
                raise SystemExit("--at needs exactly one --redo id")
            (cx, cy), placement = args.at, "manual"
        else:
            cx, cy, placement = choose_placement(
                bg, bg_path, segmenter,
                vehicle_cache.get(bg_path.relative_to(REPO_ROOT).as_posix(), []), target_w)
        seed = img_id + 1000 * s["redo_count"]
        render, render_ppm = render_object(model_path, category, seed=seed,
                                           res=args.render_res)
        scale = target_w / (real_size[category] * render_ppm)
        comp, mask_arr, bbox = composite(bg, render, cx, cy, scale)
        comp.save(OUT / "images" / f"{img_id:05d}.jpg", quality=92)
        Image.fromarray(mask_arr).save(OUT / "masks" / f"{img_id:05d}.png")
        for a in anns["annotations"]:
            if a["image_id"] == img_id and a.get("segmentation_mask"):
                a["bbox"] = list(bbox)
                a["area"] = int((mask_arr > 0).sum())
        s.update({"placement_xy": [cx, cy], "target_width_px": target_w,
                  "scale_source": scale_source, "placement": placement,
                  "render_seed": seed})
        print(f"  redid {img_id}: {category} at ({cx}, {cy}) via {placement}")
    del segmenter
    (OUT / "annotations.json").write_text(json.dumps(anns, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-res", type=int, default=640)
    ap.add_argument("--redo", type=int, nargs="+", metavar="ID",
                    help="re-place and re-render only these sample ids (same "
                         "background, category and 3D model), keep the rest")
    ap.add_argument("--rerender", action="store_true",
                    help="re-render and re-composite every sample with its "
                         "recorded placement and render seed (same background, "
                         "model, pose, lighting); used after a labelling fix")
    ap.add_argument("--at", type=int, nargs=2, metavar=("X", "Y"),
                    help="with a single --redo id: place the object centre at "
                         "this pixel, picked by eye on a lane, instead of "
                         "sampling. For backgrounds where the road segmenter "
                         "is wrong (recorded as placement 'manual')")
    args = ap.parse_args()
    if args.rerender:
        return rerender_all(args)
    if args.redo:
        return redo_samples(args)

    real_size = load_real_size()
    calibration = load_calibration()
    random.seed(args.seed)
    debris = load_debris_manifest()
    backgrounds = highway_backgrounds.sample()
    if not backgrounds:
        print("no background frames found -- run the fetch/download scripts first")
        return 1
    if len(backgrounds) < args.n:
        print(f"[warn] only {len(backgrounds)} background frames for {args.n} "
              f"requested samples -- some will have to repeat")
    bg_queue = list(backgrounds)
    random.shuffle(bg_queue)
    bg_idx = 0
    print(f"{len(backgrounds)} background frames, "
          f"{sum(len(v) for v in debris.values())} debris models")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    vehicle_cache = load_vehicle_cache()
    from segment_benchmark import SegformerRoad  # heavy import, only needed here
    segmenter = SegformerRoad("segformer-b2-cityscapes")

    images, annotations, categories = [], [], []
    cat_ids = {name: i + 1 for i, name in enumerate(WIDTH_FRAC)}
    for name, cid in cat_ids.items():
        categories.append({"id": cid, "name": name})

    manifest = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_requested": args.n, "samples": []}

    cat_names = list(debris)
    cat_weights = [CATEGORY_WEIGHT[c] for c in cat_names]

    # A category drawn fresh via random.choices() every attempt can vanish
    # entirely by chance -- CATEGORY_WEIGHT gives each "small" category only
    # ~3% of the draws, and a 100-sample run hit exactly this: plain "tire"
    # (the category the whole project started from) came up zero times.
    # Guarantee every category a floor, then fill the rest by weight, so
    # coverage never depends on luck. Even with the weight bump above,
    # weighted-random *count* still swings a lot run to run (one 100-sample
    # run drew 29 tire+truck-tire between them, the next only 16, same
    # weights) -- a per-category minimum COUNT removes that swing the same
    # way MIN_WIDTH_PX removed the swing in per-sample SIZE, for the two
    # categories the user explicitly asked to see reliably.
    min_per_cat = max(1, args.n // (2 * len(cat_names)))
    min_count_frac = {"tire": 0.10, "truck-tire": 0.15}
    cat_plan = []
    for c in cat_names:
        n_c = max(min_per_cat, round(args.n * min_count_frac.get(c, 0)))
        cat_plan += [c] * n_c
    if len(cat_plan) > args.n:
        cat_plan = cat_plan[:args.n]  # only triggers for a very small --n
    remaining = max(0, args.n - len(cat_plan))
    cat_plan += random.choices(cat_names, weights=cat_weights, k=remaining)
    random.shuffle(cat_plan)
    cat_plan_idx = 0

    ann_id = 1
    made = 0
    attempts = 0
    while made < args.n and attempts < args.n * 3:
        attempts += 1
        if bg_idx >= len(bg_queue):
            print(f"[warn] exhausted all {len(bg_queue)} unique backgrounds after "
                  f"{made} samples -- reshuffling and reusing")
            random.shuffle(bg_queue)
            bg_idx = 0
        bg_path = bg_queue[bg_idx]
        bg_idx += 1
        if cat_plan_idx < len(cat_plan):
            category = cat_plan[cat_plan_idx]
            cat_plan_idx += 1
        else:  # only reached if earlier attempts failed and had to be retried
            category = random.choices(cat_names, weights=cat_weights, k=1)[0]
        pick = random.choice(debris[category])
        model_dir = MODELS_ROOT / category / pick["uid"]
        model_path = next(model_dir.glob("*.gltf"), None)
        if model_path is None:
            continue

        try:
            bg = Image.open(bg_path).convert("RGB")
        except Exception:
            continue

        target_w, scale_source = target_width_px(category, bg, bg_path, real_size, calibration)
        cx, cy, placement = choose_placement(
            bg, bg_path, segmenter,
            vehicle_cache.get(bg_path.relative_to(REPO_ROOT).as_posix(), []), target_w)

        try:
            render, render_ppm = render_object(model_path, category, seed=made,
                                               res=args.render_res)
            scale = target_w / (real_size[category] * render_ppm)
            comp, mask_arr, bbox = composite(bg, render, cx, cy, scale)
        except Exception as exc:
            print(f"[skip] {category}/{pick['uid']}: {exc}")
            continue

        img_id = made
        img_name = f"{img_id:05d}.jpg"
        comp.save(OUT / "images" / img_name, quality=92)
        Image.fromarray(mask_arr).save(OUT / "masks" / f"{img_id:05d}.png")

        images.append({"id": img_id, "file_name": img_name,
                       "width": comp.width, "height": comp.height})
        rle_area = int((mask_arr > 0).sum())
        annotations.append({
            "id": ann_id, "image_id": img_id, "category_id": cat_ids[category],
            "bbox": list(bbox), "area": rle_area, "iscrowd": 0,
            "segmentation_mask": f"masks/{img_id:05d}.png",
        })
        manifest["samples"].append({
            "image_id": img_id, "category": category, "model_uid": pick["uid"],
            "model_credit": pick["credit"], "background_source": str(
                bg_path.relative_to(REPO_ROOT)), "placement_xy": [cx, cy],
            "target_width_px": target_w, "scale_source": scale_source,
            "placement": placement, "render_seed": made,
        })
        ann_id += 1
        made += 1
        if made % 25 == 0:
            print(f"  {made}/{args.n}", flush=True)

    del segmenter

    (OUT / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": categories},
        indent=2), encoding="utf-8")
    manifest["n_made"] = made
    manifest["n_attempts"] = attempts
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    # Remove any leftover image/mask files this run didn't (re)write --
    # e.g. ids 100-299 stranded on disk after a 300 -> 100 run, orphaned
    # from annotations.json/manifest.json but still taking up space. Done
    # here, after a successful write, rather than wiping the folders at
    # the start of the run: wiping first left the dataset empty if the run
    # was interrupted (killed mid-render) before writing anything back.
    kept_ids = {im["id"] for im in images}
    removed = 0
    for sub, ext in (("images", "*.jpg"), ("masks", "*.png")):
        for f in (OUT / sub).glob(ext):
            if int(f.stem) not in kept_ids:
                f.unlink()
                removed += 1
    if removed:
        print(f"removed {removed} stale file(s) from a previous, larger run")

    print(f"\nmade {made}/{args.n} samples ({attempts} attempts)")
    print(f"saved: {OUT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
