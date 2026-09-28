"""Scene-relation set: frames where the right risk depends on where objects
are relative to the lanes, the traffic and each other, for testing
architecture C (scene-level analysis, scripts/assess/scene_assess.py).

Every background gets up to five variants:

  clear      the background alone
  lane       one large rigid object in a running lane, near traffic
  shoulder   the same object (same 3D model, same pose) moved sideways onto
             the hard shoulder, level with where it lay in the lane
  spill      3-4 objects of one kind strewn behind a lorry, in its lane
             (the last one may lie in the next lane)
  blockage   two large objects side by side in two adjacent lanes

The expected answers are fixed here, before any model is run, from the risk
rubric the per-box risk step already uses (assess_risk.RISK_PROMPT): a rigid
or heavy object in a traffic lane is high; a large object on the hard
shoulder is medium. Only large rigid objects are used (tyres, fridges,
pallets, planks, ladders), so each placement has one right answer:

  clear      scene none
  lane       object high; scene high; 1 lane blocked
  shoulder   object medium; scene medium; 0 lanes blocked
  spill      every item high and all in one group, dropped by that lorry;
             scene high; lanes blocked = lanes holding an item
  blockage   both objects high; scene high; 2 lanes blocked

Backgrounds are real frames from the open-highway pool
(highway_backgrounds.py). Test backgrounds are frames the 30-image test set
used, so their vehicle labels are the hand-checked ones; dev backgrounds are
training-render frames (vehicle labels not hand-checked). The 3D debris,
rendering and scale are the test set's (build_synthetic_dataset.py).

Lane geometry. Pexels 19851623 is a near-static camera over a straight
six-lane motorway with a hard shoulder on each side; right-hand traffic
(the lorries' cabs lead: top carriageway moves left, bottom moves right).
Its four solid lines are fitted per frame from the image (the camera turns
slightly between frames), and the lanes are the three equal strips between
them. In 12306893 frame_014 the two solid lines are traced row by row where
nothing covers them (y 700-1340; the camera is slightly tilted, so the road
narrows towards the top of the frame). Its outermost strip is bounded by a
solid line but carries traffic in other frames of the clip (a dynamic hard
shoulder), so it gets no shoulder variant. A shoulder object must fit inside
the shoulder: at most 70% of its width, clear of the edge line.

    python scripts/data/build_scene_relations.py
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

TEST_SET = REPO_ROOT / "datasets" / "synthetic-highway-debris"
TRAIN_SET = REPO_ROOT / "datasets" / "synthetic-highway-debris-train"
OUT = {"test": REPO_ROOT / "datasets" / "scene-relations",
       "dev": REPO_ROOT / "datasets" / "scene-relations-dev"}
PX = "datasets/pexels-highway"
BACKGROUNDS = {
    "test": [f"{PX}/19851623/frame_003.jpg", f"{PX}/19851623/frame_006.jpg",
             f"{PX}/19851623/frame_008.jpg", f"{PX}/19851623/frame_012.jpg",
             f"{PX}/12306893/frame_014.jpg"],
    "dev": [f"{PX}/19851623/frame_004.jpg", f"{PX}/19851623/frame_010.jpg"],
}
# Large rigid objects only (see the docstring). Lane/shoulder objects must fit
# a 2 m shoulder lying along the road; tried in this order from a per-frame start.
PAIR_CATS = ["wooden-pallet", "wooden-plank", "ladder", "refrigerator", "truck-tire"]
SPILL_CATS = ["tire", "wooden-pallet", "wooden-plank"]
# No ladders here: seen edge-on at 16 px/m a ladder renders as a 5 px stick.
BLOCK_CATS = ["refrigerator", "truck-tire", "wooden-pallet"]
RENDER_CACHE = REPO_ROOT / ".cache" / "scene_renders"
SHOULDER_FILL = 0.7
ELONGATED = {"refrigerator", "wooden-plank", "ladder", "mattress", "wooden-pallet"}
LORRY_M = 9.0
RENDER_RES = 640


# ------------------------------------------------------------------ lane geometry
class Road:
    """Lanes of one carriageway as straight strips. Coordinates: `a` along the
    road, `c` across it. For a horizontal road a = x, c = y; for a vertical
    one a = y, c = x. bounds[k](a) is the across-road coordinate of line k
    (0 = outer edge line, 3 = median edge line); lane k lies between lines
    k - 1 and k. `ahead` is +1 or -1: the direction of travel along `a`.
    shoulder(a) is the across-road (low, high) band of the hard shoulder, or
    None; `line_side` says which side of the band is the edge line."""

    def __init__(self, name, horizontal, bounds, ahead, shoulder=None, a_range=None):
        self.name, self.horizontal, self.bounds = name, horizontal, bounds
        self.ahead, self.shoulder, self.a_range = ahead, shoulder, a_range

    def shoulder_fits(self, a, box):
        """The box lies inside the shoulder, clear of both edges, and uses at
        most SHOULDER_FILL of its width."""
        lo, hi = self.shoulder(a)
        c0, c1 = (box[1], box[1] + box[3]) if self.horizontal else (box[0], box[0] + box[2])
        return c0 >= lo + 2 and c1 <= hi - 3 and (c1 - c0) <= SHOULDER_FILL * (hi - lo)

    def xy(self, a, c):
        return (a, c) if self.horizontal else (c, a)

    def ac(self, x, y):
        return (x, y) if self.horizontal else (y, x)

    def lane_centre(self, k, a):
        return (self.bounds[k - 1](a) + self.bounds[k](a)) / 2

    def lane_width(self, a):
        return abs(self.bounds[3](a) - self.bounds[0](a)) / 3

    def lane_of(self, x, y):
        a, c = self.ac(x, y)
        lo, hi = sorted((self.bounds[0](a), self.bounds[3](a)))
        if not lo <= c <= hi:
            return None
        for k in (1, 2, 3):
            b0, b1 = sorted((self.bounds[k - 1](a), self.bounds[k](a)))
            if b0 <= c <= b1:
                return k
        return None


def fit_line(gray, y_at_1200, slope=-0.0105, win=12):
    xs, ys = [], []
    for x in range(60, gray.shape[1] - 40, 40):
        ye = int(y_at_1200 + slope * (x - 1200))
        col = gray[ye - win:ye + win + 1, x - 2:x + 3].mean(1)
        k = int(np.argmax(col))
        if col[k] > 150:
            xs.append(x)
            ys.append(ye - win + k)
    xs, ys = np.array(xs), np.array(ys, float)
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        a, b = np.polyfit(xs[keep], ys[keep], 1)
        keep = np.abs(ys - (a * xs + b)) < 2.5
    return lambda x, a=a, b=b: a * x + b


def trace_line(gray, x_start, y_start, y_lo, y_hi, win=10):
    """A near-vertical solid line followed row by row from (x_start, y_start),
    fitted as a straight line x(y) over y_lo..y_hi."""
    ys, xs = [], []
    for step in (10, -10):
        prev = x_start
        for y in range(y_start, y_hi if step > 0 else y_lo, step):
            row = gray[y - 2:y + 3, int(prev) - win:int(prev) + win + 1].mean(0)
            k = int(np.argmax(row))
            if row[k] > 170:
                prev = int(prev) - win + k
                ys.append(y)
                xs.append(prev)
    a, b = np.polyfit(ys, xs, 1)
    return lambda y, a=a, b=b: a * y + b


def roads_for(bg_path: str, bg: Image.Image) -> list[Road]:
    if "19851623" in bg_path:
        g = np.asarray(bg.convert("L"), dtype=float)
        top_out, top_med = fit_line(g, 915), fit_line(g, 1051)
        bot_med, bot_out = fit_line(g, 1134), fit_line(g, 1268)

        def split(outer, median):
            return [lambda x, t=t: outer(x) + t / 3 * (median(x) - outer(x)) for t in range(4)]
        # Shoulder widths measured by hand at x = 1000-1400: 43 px on top, 33 px below.
        return [Road("top", True, split(top_out, top_med), -1,
                     shoulder=lambda x: (top_out(x) - 43, top_out(x) - 2), a_range=(300, 3540)),
                Road("bottom", True, split(bot_out, bot_med), +1,
                     shoulder=lambda x: (bot_out(x) + 2, bot_out(x) + 33), a_range=(300, 3540))]
    if bg_path.endswith("12306893/frame_014.jpg"):
        # Left carriageway, traffic moving up (the lorry's cab leads, the gantry
        # signs face it). Solid lines at x = 383 (outer) and 556 (median) at y = 1300.
        g = np.asarray(bg.convert("L"), dtype=float)
        outer, median = trace_line(g, 383, 1300, 700, 1340), trace_line(g, 556, 1300, 700, 1340)
        bounds = [lambda y, t=t: outer(y) + t / 3 * (median(y) - outer(y)) for t in range(4)]
        return [Road("left", False, bounds, -1, shoulder=None, a_range=(720, 1320))]
    raise SystemExit(f"no lane geometry for {bg_path}")


# ------------------------------------------------------------------ labels of the source frame
def source_labels(bg_path: str):
    """Vehicle boxes and ignore regions of the source frame, from the set that
    used it: the test set (hand-checked) or the training renders."""
    for root in (TEST_SET, TRAIN_SET):
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        anns = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in anns["categories"]}
        for s in man["samples"]:
            if Path(s["background_source"]).as_posix() == bg_path:
                i = s["image_id"]
                im = next(x for x in anns["images"] if x["id"] == i)
                veh = [a["bbox"] for a in anns["annotations"]
                       if a["image_id"] == i and cats[a["category_id"]] == "vehicle"]
                return veh, im.get("ignore_regions", []), root.name
    raise SystemExit(f"{bg_path} is in neither the test set nor the training renders")


# ------------------------------------------------------------------ rendering
class Renderer:
    def __init__(self):
        self.real = B.load_real_size()
        self.cal = B.load_calibration()
        self.models = B.load_debris_manifest()

    @staticmethod
    def _render(model, category, seed):
        """B.render_object, cached on disk: a Blender render takes seconds."""
        f = RENDER_CACHE / f"{category}_{model.parent.name}_{seed}.png"
        if f.is_file():
            im = Image.open(f).convert("RGBA")
            im.info["object_alpha"] = Image.open(f.with_suffix(".alpha.png")).convert("L")
            return im, json.loads(f.with_suffix(".json").read_text())["ppm"]
        im, ppm = B.render_object(model, category, seed=seed, res=RENDER_RES)
        RENDER_CACHE.mkdir(parents=True, exist_ok=True)
        im.save(f)
        im.info["object_alpha"].save(f.with_suffix(".alpha.png"))
        f.with_suffix(".json").write_text(json.dumps({"ppm": ppm}))
        return im, ppm

    def render(self, category, seed, bg, bg_path, along_horizontal):
        """Render one object; for elongated ones, retry seeds until its long
        side lies along the road. Returns (rgba, scale, uid, seed)."""
        pick = self.models[category][seed % len(self.models[category])]
        model = next((B.MODELS_ROOT / category / pick["uid"]).glob("*.gltf"))
        target_w, _ = B.target_width_px(category, bg, Path(bg_path), self.real, self.cal)
        for s in range(seed, seed + 12):
            im, ppm = self._render(model, category, s)
            oa = np.array(im.info["object_alpha"]) > B.OBJECT_ALPHA
            ys, xs = np.nonzero(oa)
            w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            along, across = (w, h) if along_horizontal else (h, w)
            if category not in ELONGATED or along >= 1.4 * across:
                return im, target_w / (self.real[category] * ppm), pick["uid"], s
        return im, target_w / (self.real[category] * ppm), pick["uid"], s


# ------------------------------------------------------------------ placement
def clear_of(box, others, gap):
    x, y, w, h = box
    return all(x + w + gap < ox or ox + ow + gap < x or y + h + gap < oy or oy + oh + gap < y
               for ox, oy, ow, oh in others)


def est_box(x, y, render, scale):
    """Box an object rendered as `render` would get if centred at (x, y)."""
    oa = np.array(render.info["object_alpha"]) > B.OBJECT_ALPHA
    ys, xs = np.nonzero(oa)
    w, h = (xs.max() - xs.min() + 1) * scale, (ys.max() - ys.min() + 1) * scale
    return [x - w / 2, y - h / 2, w, h]


def vehicles_in_lane(road, k, vehicles):
    return [v for v in vehicles if road.lane_of(v[0] + v[2] / 2, v[1] + v[3] / 2) == k]


def length_m(v, road, ppm):
    return (v[2] if road.horizontal else v[3]) / ppm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "dev", "both"], default="both")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    ren = Renderer()
    cal = {k: v["px_per_metre"] for k, v in ren.cal.items()}
    cat_names = list(B.WIDTH_FRAC) + ["vehicle"]
    cat_id = {n: i + 1 for i, n in enumerate(cat_names)}
    for split in (["test", "dev"] if args.split == "both" else [args.split]):
        rng = random.Random(args.seed + (split == "dev"))
        out = OUT[split]
        (out / "images").mkdir(parents=True, exist_ok=True)
        images, annotations, samples = [], [], []
        ann_id = 1
        for b_idx, bg_path in enumerate(BACKGROUNDS[split]):
            bg = Image.open(REPO_ROOT / bg_path).convert("RGB")
            ppm = cal[highway_backgrounds.location_key(Path(bg_path))]
            vehicles, ignore, source = source_labels(bg_path)
            roads = roads_for(bg_path, bg)
            gap = 0.8 * ppm

            def free(box, placed):
                return clear_of(box, vehicles + placed, gap) and not any(
                        rx <= box[0] + box[2] / 2 <= rx + rw and ry <= box[1] + box[3] / 2 <= ry + rh
                        for rx, ry, rw, rh in ignore)

            variants = {"clear": []}
            # ---- lane + shoulder pair (roads that have a shoulder), else lane only
            spot = None
            start = b_idx + (split == "dev") * 2
            for cat in [PAIR_CATS[(start + t) % len(PAIR_CATS)] for t in range(len(PAIR_CATS))]:
                pair_roads = [r for r in roads if r.shoulder] or roads
                rng.shuffle(pair_roads)
                for road in pair_roads:
                    render, scale, uid, seed = ren.render(cat, 1000 * b_idx + 11, bg, bg_path,
                                                          road.horizontal)
                    lane_k = 1 if road.shoulder else rng.choice((2, 3))
                    anchors = vehicles_in_lane(road, lane_k, vehicles)
                    for _ in range(3000):
                        if anchors and rng.random() < 0.8:
                            v = rng.choice(anchors)
                            va, _c = road.ac(v[0] + v[2] / 2, v[1] + v[3] / 2)
                            a = va + road.ahead * rng.uniform(15, 60) * ppm
                        else:
                            a = rng.uniform(*road.a_range)
                        if not road.a_range[0] <= a <= road.a_range[1]:
                            continue
                        if not free(est_box(*road.xy(a, road.lane_centre(lane_k, a)), render, scale), []):
                            continue
                        if road.shoulder:
                            sb = est_box(*road.xy(a, sum(road.shoulder(a)) / 2), render, scale)
                            if not (free(sb, []) and road.shoulder_fits(a, sb)):
                                continue
                        spot = a
                        break
                    if spot is not None:
                        break
                if spot is not None:
                    break
            if spot is None:
                raise SystemExit(f"{bg_path}: no free spot for the lane object")
            obj = {"category": cat, "uid": uid, "seed": seed, "render": render, "scale": scale}
            variants["lane"] = [dict(obj, xy=road.xy(spot, road.lane_centre(lane_k, spot)),
                                     zone="lane", lane=f"{road.name}-{lane_k}", group=None,
                                     risk="high")]
            if road.shoulder:
                variants["shoulder"] = [dict(obj, xy=road.xy(spot, sum(road.shoulder(spot)) / 2),
                                             zone="shoulder", lane=None, group=None,
                                             risk="medium")]
            # ---- spill behind a lorry
            lorries = [(r, k, v) for r in roads for k in (1, 2, 3)
                       for v in vehicles_in_lane(r, k, vehicles) if length_m(v, r, ppm) >= LORRY_M]
            rng.shuffle(lorries)
            spill_cat = SPILL_CATS[(b_idx + (split == "dev")) % len(SPILL_CATS)]
            for r, k, v in lorries:
                n = rng.choice((3, 4))
                va0, va1 = sorted(r.ac(v[0], v[1])[0:1] + r.ac(v[0] + v[2], v[1] + v[3])[0:1])
                rear = va0 if r.ahead > 0 else va1
                items, placed, a = [], [], rear - r.ahead * rng.uniform(2, 5) * ppm
                ok = True
                for j in range(n):
                    rend, sc, u, sd = ren.render(spill_cat, 1000 * b_idx + 101 + j, bg, bg_path,
                                                 r.horizontal)
                    for _ in range(200):
                        kk = k if j < n - 1 or rng.random() < 0.5 else \
                            (k + 1 if k < 3 else k - 1)
                        side = rng.uniform(-0.3, 0.3) * r.lane_width(a)
                        p = r.xy(a, r.lane_centre(kk, a) + side)
                        box = est_box(*p, rend, sc)
                        if r.a_range[0] <= a <= r.a_range[1] and free(box, placed):
                            break
                        a -= r.ahead * 0.5 * ppm
                    else:
                        ok = False
                        break
                    placed.append(box)
                    items.append({"category": spill_cat, "uid": u, "seed": sd, "render": rend,
                                  "scale": sc, "xy": p, "zone": "lane", "lane": f"{r.name}-{kk}",
                                  "group": 1, "risk": "high", "source_vehicle": v})
                    a -= r.ahead * rng.uniform(4, 10) * ppm
                if ok:
                    variants["spill"] = items
                    break
            # ---- blockage: two objects side by side in adjacent lanes
            cats = rng.sample(BLOCK_CATS, 2)
            rends = [ren.render(c, 1000 * b_idx + 201 + j, bg, bg_path, roads[0].horizontal)
                     for j, c in enumerate(cats)]
            for _ in range(2000):
                r = rng.choice(roads)
                k = rng.choice((1, 2))
                a = rng.uniform(*r.a_range)
                boxes = []
                for kk, (rend, sc, _u, _s) in zip((k, k + 1), rends):
                    box = est_box(*r.xy(a, r.lane_centre(kk, a)), rend, sc)
                    if not free(box, boxes):
                        break
                    boxes.append(box)
                if len(boxes) == 2:
                    variants["blockage"] = [
                        {"category": c, "uid": u, "seed": sd, "render": rend, "scale": sc,
                         "xy": r.xy(a, r.lane_centre(kk, a)), "zone": "lane",
                         "lane": f"{r.name}-{kk}", "group": None, "risk": "high"}
                        for kk, c, (rend, sc, u, sd) in zip((k, k + 1), cats, rends)]
                    break

            # ---- composite and label every variant
            for name, objs in variants.items():
                img_id = len(images)
                canvas = bg
                obj_anns = []
                for o in objs:
                    canvas, m, bbox = B.composite(canvas, o["render"], int(o["xy"][0]),
                                                  int(o["xy"][1]), o["scale"])
                    obj_anns.append({"id": ann_id, "image_id": img_id,
                                     "category_id": cat_id[o["category"]], "bbox": list(bbox),
                                     "area": int((m > 0).sum()), "iscrowd": 0})
                    o["ann_id"] = ann_id
                    ann_id += 1
                annotations.extend(obj_anns)
                veh_ids = []
                for v in vehicles:
                    annotations.append({"id": ann_id, "image_id": img_id,
                                        "category_id": cat_id["vehicle"],
                                        "bbox": [int(t) for t in v], "area": int(v[2] * v[3]),
                                        "iscrowd": 0})
                    veh_ids.append((v, ann_id))
                    ann_id += 1
                fname = f"{img_id:05d}.jpg"
                canvas.save(out / "images" / fname, quality=92)
                images.append({"id": img_id, "file_name": fname, "width": bg.width,
                               "height": bg.height, "ignore_regions": ignore})
                lanes = sorted({o["lane"] for o in objs if o["lane"]})
                groups = {}
                for o in objs:
                    if o["group"] is not None:
                        groups.setdefault(o["group"], []).append(o["ann_id"])
                src = [vid for v, vid in veh_ids for o in objs[:1]
                       if o.get("source_vehicle") is v]
                expected = {
                    "scene_risk": max((o["risk"] for o in objs),
                                      key=["none", "low", "medium", "high"].index, default="none"),
                    "lanes_blocked": len(lanes), "lanes": lanes,
                    "objects": [{"ann_id": o["ann_id"], "category": o["category"],
                                 "zone": o["zone"], "lane": o["lane"], "risk": o["risk"]}
                                for o in objs],
                    "groups": list(groups.values()),
                    "spill_source_vehicle": src[0] if src else None,
                }
                samples.append({"image_id": img_id, "background_source": bg_path,
                                "labels_from": source, "scenario": name,
                                "objects": [{"ann_id": o["ann_id"], "category": o["category"],
                                             "model_uid": o["uid"], "render_seed": o["seed"],
                                             "xy": [int(o["xy"][0]), int(o["xy"][1])]}
                                            for o in objs],
                                "expected": expected})
                print(f"  {split} {img_id:02d} {Path(bg_path).parent.name}/{Path(bg_path).stem} "
                      f"{name}: {[o['category'] for o in objs]} lanes {lanes}", flush=True)
        (out / "annotations.json").write_text(json.dumps(
            {"images": images, "annotations": annotations,
             "categories": [{"id": i, "name": n} for n, i in cat_id.items()]}, indent=1),
            encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(
            {"seed": args.seed, "backgrounds": BACKGROUNDS[split], "samples": samples}, indent=1),
            encoding="utf-8")
        print(f"saved: {out.relative_to(REPO_ROOT)}/ ({len(images)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
