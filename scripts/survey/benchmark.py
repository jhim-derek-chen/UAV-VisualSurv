"""Factorial benchmark: which factor actually breaks aerial anomaly detection?

The target task is UAV viewpoint + highway scene + small real physical object.
No public dataset has all three; every dataset on disk has exactly two. A single
"best model" number would therefore confound viewpoint, scene and object type.

Object size below is MEASURED -- median object area as a fraction of the scored
region -- because it reordered the sets against intuition: a FOD-A bolt fills
4.4% of its 300x300 frame, 27x more than a SMIYC obstacle fills its road.

  B1  ground      / road          / object 0.16%, 10.6% of frame anomalous
  B2  ground      / road          / object 0.15%,  0.7% of road anomalous
  B3  near-ground / runway        / object 4.40%                    FOD-A
  B4  UAV ~10 m   / rail corridor / object 3.67%                    UAV-RSOD
  B5  UAV high    / streets       / damage 20.4%                    RescueNet

B3 -> B4 is the clean viewpoint contrast: object size and background material
are matched, only the camera moves.

B5 is still scored here but is excluded from the report: post-hurricane street
damage is too far from a highway to inform the choice. It is kept because it is
the one set where anomalies are the MAJORITY of the road, which is exactly where
a prototype-distance method's premise fails -- a boundary worth being able to
re-check.

This stage asks only "can the model tell the object apart from the road around
it", nothing about how often it would cry wolf on a clean feed -- that is a
separate question for once a road-region mask and world-model reasoning exist
to consume a false-alarm number responsibly. Concretely:

  * Methods still emit RAW scores (per-image min-max would put a maximum-score
    pixel in every frame and make ranking meaningless).
  * A labelled object counts as NOTICED if its peak score is in the top
    `NOTICE_BAR` of that SAME IMAGE's own clean pixels -- self-calibrated per
    image, nothing pooled or shared across images, benchmarks or methods. No
    false-alarm budget, no operating point to transfer, no dataset asked to
    stand in for "clean" on another dataset's behalf.
  * AUROC/AUPR (rank-based, already threshold-free) are reported alongside as
    the complementary "can it rank the anomaly above this photo's own
    background at all" number.

UAVDT (ex-B6: UAV + highway, no anomalies, no road-region labels) is deferred,
not deleted -- see the comment above `b6_uav_clean`. Scoring it needs a
false-alarm concept, which this stage deliberately does not use: without a
road mask, every off-road detection (billboards, buildings, trees) reads as
a false alarm just because the model was never told to restrict itself to the
carriageway, which is not a fact about the model. Revisit once road-region
masking (or the world-model reasoning stage) can consume that number fairly.

    python scripts/survey/benchmark.py                    # everything
    python scripts/survey/benchmark.py --only B4 --method prowl-v3
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "datasets"
OUT = REPO_ROOT / "results"

PROMPT = "debris. tire. cardboard box. luggage. rock. fallen object. obstacle."
# "road" is deliberately absent: Grounding DINO grounds sub-phrases, so a prompt
# containing that word makes it box the road surface itself. Measured on an
# earlier prompt: 19 of 48 aerial detections fired on the bare token "road",
# 7 of them covering >30% of the frame.

NOTICE_BAR = 0.10      # "the model noticed it" = object's peak score is in the top
                       # 10% of THAT SAME IMAGE's own clean-pixel scores. Self-
                       # calibrated per image -- no shared threshold, no false-alarm
                       # budget, nothing borrowed from another benchmark.
NEG_SAMPLE = 20000     # negative pixels kept per image, for AUROC/AUPR and NOTICE_BAR
MIN_COMPONENT = 25     # px; below this a labelled blob is annotation noise


# ---------------------------------------------------------------- data
@dataclass
class Item:
    img: Path
    masks: Callable[[], tuple[np.ndarray | None, np.ndarray]]  # -> (gt, valid)


def _boxes_to_mask(boxes, h, w) -> np.ndarray:
    """Boxes -> binary mask, so box-annotated sets score with the same pixel
    metrics as mask-annotated ones. This inflates every object to its bounding
    rectangle, which is generous to all methods equally, but it caps achievable
    AUPR: absolute AUPR on a box set is not comparable to a mask set, only the
    ranking between methods is."""
    m = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = True
    return m


def _voc_boxes(xml: Path) -> tuple[list, int, int]:
    t = ET.parse(xml).getroot()
    sz = t.find("size")
    h, w = int(sz.find("height").text), int(sz.find("width").text)
    boxes = [tuple(float(o.find("bndbox").find(k).text)
                   for k in ("xmin", "ymin", "xmax", "ymax"))
             for o in t.findall("object")]
    return boxes, h, w


def _mask_loader(lab_path: Path, anom_val: int, void_val):
    def fn():
        lab = np.array(Image.open(lab_path))
        if lab.ndim == 3:
            lab = lab[..., 0]
        gt = lab == anom_val
        valid = np.ones_like(gt) if void_val is None else (lab != void_val)
        return gt, valid
    return fn


def _box_loader(boxes, h, w):
    def fn():
        return _boxes_to_mask(boxes, h, w), np.ones((h, w), bool)
    return fn


def b1_ground_large(limit: int) -> list[Item]:
    """Ground road scenes, mid-to-large anomalies: animals, cones, tents, crates.

    EPFL labels carry no void class, so the whole frame is scored. SMIYC marks
    everything off-road as 255 and that is honoured -- an obstacle detector is
    not asked about the sky."""
    items = []
    root = DATA / "road-anomaly-epfl/RoadAnomaly_jpg/frames"
    for lab in sorted(root.rglob("labels_semantic.png")):
        img = lab.parent.parent / (lab.parent.name.replace(".labels", "") + ".jpg")
        if img.is_file():
            items.append(Item(img, _mask_loader(lab, 2, None)))
    root = DATA / "smiyc-road-anomaly21/dataset_AnomalyTrack"
    for lab in sorted((root / "labels_masks").glob("*_labels_semantic.png")):
        img = root / "images" / (lab.name.replace("_labels_semantic.png", "") + ".jpg")
        if img.is_file():
            items.append(Item(img, _mask_loader(lab, 1, 255)))
    return items[:limit] if limit else items


def b2_ground_small(limit: int) -> list[Item]:
    """Ground road scenes, small objects lying ON the road surface -- the closest
    ground analogue of the target task. Off-road pixels are void (255)."""
    root = DATA / "smiyc-road-obstacle21/dataset_ObstacleTrack"
    items = []
    for lab in sorted((root / "labels_masks").glob("*_labels_semantic.png")):
        img = root / "images" / (lab.name.replace("_labels_semantic.png", "") + ".webp")
        if img.is_file():
            items.append(Item(img, _mask_loader(lab, 1, 255)))
    return items[:limit] if limit else items


def b3_runway_tiny(limit: int) -> list[Item]:
    """FOD-A: airport-runway foreign objects, 300x300 frames, camera near ground.

    Classes are small hardware (bolt, washer, nut, wrench). Included not because
    a bolt threatens a car, but because it is the only set on disk whose objects
    are a few dozen pixels -- it isolates object size from viewpoint."""
    root = DATA / "fod-a-voc/FODPascalVOCFormat-V.2.1/VOC2007"
    split = root / "ImageSets/Main/test.txt"
    if not split.is_file():
        return []
    ids = [s.strip() for s in split.read_text().split() if s.strip()]
    step = max(1, len(ids) // max(1, limit)) if limit else 1
    items = []
    for i in ids[::step]:
        xml, img = root / "Annotations" / f"{i}.xml", root / "JPEGImages" / f"{i}.jpg"
        if not (xml.is_file() and img.is_file()):
            continue
        boxes, h, w = _voc_boxes(xml)
        if boxes:
            items.append(Item(img, _box_loader(boxes, h, w)))
    return items[:limit] if limit else items


def b4_uav_rail(limit: int) -> list[Item]:
    """UAV-RSOD test split: obstacles on a rail corridor, flown at ~10.5 m.

    Not a highway, but the only labelled set on disk that is genuinely
    UAV + discrete small object, which is what the viewpoint contrast needs."""
    items = []
    for xml in sorted((DATA / "uav-rsod-detection").rglob("test/*.xml")):
        img = xml.with_suffix(".jpg")
        if not img.is_file():
            continue
        boxes, h, w = _voc_boxes(xml)
        if boxes:
            items.append(Item(img, _box_loader(boxes, h, w)))
    return items[:limit] if limit else items


# RescueNet 11-class ids, verified against the labels rather than assumed:
#   0 Background  1 Water  2 Building-No-Damage  3 Building-Minor
#   4 Building-Major  5 Building-Destroyed  6 Vehicle  7 Road-Clear
#   8 Road-Blocked  9 Tree  10 Pool
RESCUENET_ROAD_BLOCKED, RESCUENET_ROAD_CLEAR = 8, 7


def b5_uav_damage(limit: int) -> list[Item]:
    """RescueNet val: post-hurricane debris blocking residential streets.

    Scored on road pixels only. Frames where the whole road is blocked are
    dropped: with no clear road there are no negatives, so a method that fires
    everywhere would look perfect. 40 of 63 candidate frames are like that."""
    root = DATA / "rescuenet-val"
    if not root.is_dir():
        return []
    labs = {p.stem.replace("_lab", ""): p for p in root.rglob("*.png")}
    items = []
    for img in sorted(root.rglob("*.jpg")):
        lp = labs.get(img.stem)
        if not lp:
            continue
        lab = np.array(Image.open(lp))
        if lab.ndim == 3:
            lab = lab[..., 0]
        gt = lab == RESCUENET_ROAD_BLOCKED
        road = gt | (lab == RESCUENET_ROAD_CLEAR)
        if gt.any() and gt.sum() < 0.999 * road.sum():
            items.append(Item(img, (lambda g=gt, r=road: (g, r))))
    return items[:limit] if limit else items


UAVDT_ATTRS = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt",
               "front-view", "side-view", "bird-view", "long-term"]


def _uavdt_masks(img_path: Path, boxes) -> tuple[None, np.ndarray]:
    w, h = Image.open(img_path).size
    valid = np.ones((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        valid[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = False
    return None, valid


# Deferred, not deleted: UAVDT is UAV + highway, but has no road-region labels
# and no anomalies -- it can only ever measure false alarms, which this stage
# does not score (see module docstring). Scoring it fairly needs a road mask,
# so that off-road detections (billboards, buildings, trees) stop being counted
# against a model that was never told to restrict itself to the carriageway.
# Revisit once a road-region mask or the world-model reasoning stage exists to
# consume that number responsibly. Kept out of BENCHMARKS below; run() also
# assumes gt is never None, which this loader's masks() violates, so plug it
# back in only alongside restoring that handling.
def b6_uav_clean(limit: int) -> list[Item]:
    """UAVDT daylight frames, stratified by altitude x view. The only set that
    is actually UAV + road traffic; has no anomaly labels at all.

    Annotated vehicles and the dataset's own ignore regions are cut out of the
    scored area. A car IS an object on the road; counting a detection on one as
    a false positive would punish a method for being right."""
    root = DATA / "uavdt-benchmark-m/UAV-benchmark-M"
    attr_dir = DATA / "uavdt-attributes/M_attr"
    gt_dir = DATA / "uavdt-motd/UAV-benchmark-MOTD_v1.0/GT"
    if not (root.is_dir() and attr_dir.is_dir()):
        return []
    attrs = {f.stem.replace("_attr", "").strip():
             dict(zip(UAVDT_ATTRS, [int(x) for x in f.read_text().strip().split(",")]))
             for f in attr_dir.rglob("*_attr.txt")}

    def excluded(seq: str) -> dict[int, list]:
        """frame index -> boxes to cut out (annotated vehicles + ignore regions)."""
        per: dict[int, list] = {}
        for suffix in ("_gt.txt", "_gt_ignore.txt"):
            f = gt_dir / f"{seq}{suffix}"
            if not f.is_file():
                continue
            for line in f.read_text().splitlines():
                p = line.split(",")
                if len(p) < 6:
                    continue
                fr = int(p[0])
                x, y, w, h = (int(float(v)) for v in p[2:6])
                per.setdefault(fr, []).append((x, y, x + w, y + h))
        return per

    items, seen = [], set()
    for alt in ["low-alt", "medium-alt", "high-alt"]:
        for view in ["front-view", "side-view", "bird-view"]:
            seqs = [s for s, a in sorted(attrs.items())
                    if a[alt] and a[view] and a["daylight"] and (root / s).is_dir()]
            for seq in seqs[:4]:
                frames = sorted((root / seq).glob("*.jpg"))
                if not frames:
                    continue
                per = excluded(seq)
                # Spread within the sequence; consecutive frames are near-identical.
                for f in frames[len(frames) // 5::max(1, len(frames) // 5)][:4]:
                    if f in seen:
                        continue
                    seen.add(f)
                    idx = int(f.stem.replace("img", ""))
                    items.append(Item(f, (lambda b=per.get(idx, []), p=f:
                                          _uavdt_masks(p, b))))
    return items[:limit] if limit else items


# Object size below is the MEASURED median object area as a fraction of the
# scored region, not an eyeball judgement. It reorders the sets: FOD-A objects
# look tiny in a 300x300 crop but occupy 4.4% of it, which is 27x more of the
# frame than a SMIYC obstacle. Grouping by measured size is what makes B3->B4 a
# clean viewpoint contrast; grouping by intuition would not have.
BENCHMARKS = {
    "B1": dict(title="Ground road, salient anomalies", view="ground", scene="road",
               anomaly="discrete object", obj_size="0.16% of frame",
               source="RoadAnomaly-EPFL + SMIYC Anomaly21",
               loader=b1_ground_large, cap=0),
    "B2": dict(title="Ground road, rare small obstacles", view="ground", scene="road",
               anomaly="discrete object", obj_size="0.15% of road",
               source="SMIYC RoadObstacle21",
               loader=b2_ground_small, cap=0),
    "B3": dict(title="Runway foreign objects", view="near-ground", scene="paved runway",
               anomaly="discrete object", obj_size="4.4% of frame", source="FOD-A",
               loader=b3_runway_tiny, cap=120),
    "B4": dict(title="UAV rail corridor obstacles", view="UAV ~10 m",
               scene="paved rail corridor", anomaly="discrete object",
               obj_size="3.7% of frame", source="UAV-RSOD",
               loader=b4_uav_rail, cap=120),
    "B5": dict(title="UAV streets, scene-scale damage", view="UAV high", scene="streets",
               anomaly="scene-scale damage", obj_size="20.4% of road",
               source="RescueNet", loader=b5_uav_damage, cap=0),
}

# Each contrast holds everything else roughly fixed and moves one factor, so a
# drop between its two sides is attributable to that factor.
CONTRASTS = [
    dict(factor="viewpoint", a="B3", b="B4",
         note="object occupies 4.4% vs 3.7% of the frame and both backgrounds are "
              "paved corridors; the camera moves from near-ground to airborne"),
    dict(factor="object size", a="B2", b="B3",
         note="0.15% -> 4.4% of the scored region, both near-ground"),
    dict(factor="anomaly prevalence", a="B1", b="B2",
         note="10.6% -> 0.7% of the scored region is anomalous per frame, "
              "same viewpoint, scene and object size"),
    dict(factor="anomaly type", a="B4", b="B5",
         note="discrete object -> scene-scale damage, both UAV"),
]


# ---------------------------------------------------------------- methods
class GDino:
    name = "gdino"
    kind = "open-vocabulary detector, ground-trained"

    def __init__(self, max_side=1024):
        import transformers
        p = MODELS / "grounding-dino-tiny"
        self.proc = transformers.AutoProcessor.from_pretrained(str(p))
        self.m = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(p), dtype=torch.float32).to("cuda").eval()
        self.max_side = max_side

    def __call__(self, img: Image.Image) -> np.ndarray:
        sc = min(1.0, self.max_side / max(img.size))
        small = img.resize((int(img.width * sc), int(img.height * sc))) if sc < 1 else img
        inp = self.proc(images=small, text=PROMPT, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.m(**inp)
        # Low box threshold on purpose: the operating point is set later by the
        # shared calibration, so filtering hard here would pre-empt it.
        r = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.05, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        heat = np.zeros(img.size[::-1], dtype=np.float32)
        for b, s in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()):
            x0, y0, x1, y1 = (int(max(0, v)) for v in b)
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], float(s))
        return heat


class Prowl:
    """PROWL-style: normal road is whatever dominates the frame, anomalies are
    what sits far from it in feature space. No training, no anomaly examples.
    Raw output is cosine distance to the frame's own prototype, in [0, 2), which
    is meaningful across images -- unlike a per-image min-max rescale."""
    kind = "self-supervised features + prototype distance"

    def __init__(self, key: str, side=896):
        import transformers
        p = MODELS / key
        self.name = "prowl-v3" if "dinov3" in key else "prowl-v2"
        cfg = transformers.AutoConfig.from_pretrained(str(p))
        # DINOv3 returns all-NaN in fp16; bf16 has fp32's exponent range.
        self.dtype = torch.bfloat16
        self.m = transformers.AutoModel.from_pretrained(str(p), dtype=self.dtype).to("cuda").eval()
        self.proc = transformers.AutoImageProcessor.from_pretrained(str(p))
        self.skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.side = side

    def __call__(self, img: Image.Image) -> np.ndarray:
        # do_resize/size/do_center_crop must be forced. Both DINO processors
        # default to shortest-edge 256 + centre crop 224, which silently undoes
        # any resize done beforehand and leaves a 16x16 patch grid -- at that
        # resolution a small road obstacle is a fraction of ONE patch and the
        # method cannot see it at all. Passing the size through here is what
        # makes `side` mean anything.
        inp = self.proc(images=img, return_tensors="pt", do_resize=True,
                        size={"height": self.side, "width": self.side},
                        do_center_crop=False).to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            h = self.m(**inp).last_hidden_state[0, self.skip:].float()
        h = F.normalize(h, dim=-1)
        n = int(np.sqrt(h.shape[0]))
        # Prototype = mean of the half of the patches closest to the global mean.
        # Taking the closest half rather than all patches keeps the anomaly from
        # being absorbed into the definition of "normal".
        g = F.normalize(h.mean(0, keepdim=True), dim=-1)
        sim = (h @ g.T).squeeze(1)
        proto = F.normalize(h[sim >= sim.median()].mean(0, keepdim=True), dim=-1)
        dist = (1 - (h @ proto.T).squeeze(1)).reshape(1, 1, n, n)
        up = F.interpolate(dist, size=img.size[::-1], mode="bilinear", align_corners=False)
        return up[0, 0].cpu().numpy()


class Rba:
    """RbA: a pixel rejected by every known class is an unknown object. Raw score
    is -max(class logit), so low confidence everywhere reads as 'none of my
    classes explain this'. Comparable across images without rescaling."""
    name = "rba"
    kind = "closed-set segmenter, rejected-by-all"

    def __init__(self, key="segformer-b2-cityscapes"):
        import transformers
        p = MODELS / key
        self.m = transformers.SegformerForSemanticSegmentation.from_pretrained(
            str(p), dtype=torch.float16).to("cuda").eval()
        self.proc = transformers.SegformerImageProcessor.from_pretrained(str(p))

    def __call__(self, img: Image.Image) -> np.ndarray:
        inp = self.proc(images=img, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            lg = self.m(**inp).logits.float()
        lg = F.interpolate(lg, size=img.size[::-1], mode="bilinear", align_corners=False)
        return (-lg.max(1).values)[0].cpu().numpy()


BUILDERS = {"gdino": GDino,
            "prowl-v2": lambda: Prowl("dinov2-large"),
            "prowl-v3": lambda: Prowl("dinov3-vitl16"),
            "rba": Rba}


# ---------------------------------------------------------------- metrics
def rank_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """AUPR / AUROC by sorting rather than sklearn -- these are megapixel arrays.
    Threshold-free, so they answer 'can this method rank an anomaly above clean
    road at all', independently of where the operating point sits."""
    if pos.size == 0 or neg.size == 0:
        return {"aupr": None, "auroc": None}
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(pos.size, bool), np.zeros(neg.size, bool)])
    order = np.argsort(-v, kind="mergesort")
    v, y = v[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    # Collapse each block of tied scores to its last index. No threshold can
    # separate pixels that share a score, so evaluating inside a tie is not a
    # real operating point. This matters: Grounding DINO's map is one constant
    # per box and 0 elsewhere, and ranking positives first within the tie handed
    # it a perfect AUROC of 1.000 on FOD-A that no threshold could deliver.
    last = np.r_[np.diff(v) != 0, True]
    tp, fp = tp[last], fp[last]
    recall = np.r_[0.0, tp / pos.size]
    fpr = np.r_[0.0, fp / neg.size]
    precision = tp / np.maximum(tp + fp, 1)
    aupr = float(np.sum(np.diff(recall) * precision))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {"aupr": aupr, "auroc": float(trapz(recall, fpr))}


def components(gt: np.ndarray, s: np.ndarray) -> list[tuple[int, float]]:
    """(area, peak score) per labelled blob. Pixel AUPR is dominated by big
    objects; a benchmark about small debris needs the per-object view too."""
    from scipy import ndimage
    lab, _ = ndimage.label(gt)
    out = []
    for sl in ndimage.find_objects(lab):
        m = lab[sl] > 0
        a = int(m.sum())
        if a >= MIN_COMPONENT:
            out.append((a, float(s[sl][m].max())))
    return out


def calibrate(neg: np.ndarray, frac: float) -> float:
    """Smallest threshold such that at most `frac` of `neg` scores at or above it.

    A plain quantile is wrong here: Grounding DINO's score map is exactly 0 over
    most of the frame, so quantile(0.99) can land on 0.0 and then fire on the
    entire image. Stepping above the tie block gives the honest operating point.
    Used per-image with NOTICE_BAR -- `neg` is one image's own clean pixels, not
    a pool shared across images."""
    if neg.size == 0:
        return float("inf")
    d = np.sort(neg)[::-1]
    tau = float(d[max(0, int(frac * d.size) - 1)])
    if float((neg >= tau).mean()) <= frac:
        return tau
    higher = neg[neg > tau]
    return float(higher.min()) if higher.size else float(tau) + 1e-6


def size_bucket(a: int) -> str:
    return ("very small" if a < 32**2 else "small" if a < 96**2
            else "medium" if a < 256**2 else "large")


# ---------------------------------------------------------------- run
def run(method, items: list[Item], tag: str) -> dict:
    """One method over one benchmark, scored per image against that image's own
    clean pixels only -- nothing pooled or compared across images.

    per_img: threshold-free AUROC/AUPR (does the anomaly outrank this photo's
    own background at all). comps: (area, peak, hit) per labelled object, where
    hit means the object's peak score is in NOTICE_BAR of ITS OWN image."""
    rng = np.random.default_rng(0)
    per_img, comps = [], []
    t0 = time.time()
    for i, it in enumerate(items):
        img = Image.open(it.img).convert("RGB")
        gt, valid = it.masks()
        s = method(img)
        if s.shape != valid.shape:
            s = np.array(Image.fromarray(s).resize(valid.shape[::-1], Image.BILINEAR))
        pos_mask = gt & valid
        neg_all = s[valid & ~gt]
        if neg_all.size == 0 or not pos_mask.any():
            continue
        sub = (neg_all if neg_all.size <= NEG_SAMPLE
               else neg_all[rng.integers(0, neg_all.size, NEG_SAMPLE)]).astype(np.float32)
        per_img.append(rank_metrics(s[pos_mask], sub))
        tau_img = calibrate(sub, NOTICE_BAR)
        comps += [(a, p, bool(p >= tau_img)) for a, p in components(pos_mask, s)]
        if (i + 1) % 25 == 0:
            print(f"      {tag} {i + 1}/{len(items)}", flush=True)
    return {"per_img": per_img, "comps": comps,
            "sec_per_image": (time.time() - t0) / max(1, len(items)),
            "images": len(items)}


def summarise(raw: dict) -> dict:
    def avg(k):
        v = [a[k] for a in raw["per_img"] if a[k] is not None]
        return float(np.mean(v)) if v else None

    by_size: dict[str, list] = {}
    for a, _, hit in raw["comps"]:
        by_size.setdefault(size_bucket(a), []).append(hit)
    hits = [hit for _, _, hit in raw["comps"]]

    return {
        "images": raw["images"],
        "auroc": avg("auroc"), "aupr": avg("aupr"),
        "object_recall": float(np.mean(hits)) if hits else None,
        "objects": len(hits),
        "sec_per_image": raw["sec_per_image"],
        "recall_by_size": {k: {"n": len(v), "recall": float(np.mean(v))}
                           for k, v in sorted(by_size.items())},
        "objects_detail": [[int(a), round(float(p), 5), bool(h)]
                           for a, p, h in raw["comps"]],
    }


def profile(items: list[Item]) -> dict:
    """Measured shape of a benchmark: how big its objects are and how much of a
    frame is anomalous. Reported alongside the scores because a method's number
    means nothing without knowing which regime it was measured in -- PROWL
    assumes anomalies are a minority of the frame, and B5 violates that."""
    rel, frac = [], []
    for it in items:
        gt, valid = it.masks()
        if gt is None:
            continue
        v = int(valid.sum())
        g = gt & valid
        frac.append(float(g.sum()) / max(1, v))
        from scipy import ndimage
        lab, _ = ndimage.label(g)
        for sl in ndimage.find_objects(lab):
            a = int((lab[sl] > 0).sum())
            if a >= MIN_COMPONENT:
                rel.append(a / max(1, v))
    if not rel:
        return {"objects": 0}
    return {"objects": len(rel),
            "object_area_frac_p50": float(np.median(rel)),
            "object_area_frac_p10": float(np.percentile(rel, 10)),
            "object_area_frac_p90": float(np.percentile(rel, 90)),
            "anomalous_frac_per_frame": float(np.mean(frac))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--method", nargs="+", default=list(BUILDERS))
    args = ap.parse_args()

    sets = {k: BENCHMARKS[k]["loader"](BENCHMARKS[k]["cap"]) for k in args.only}
    for k, v in sets.items():
        print(f"{k}  {BENCHMARKS[k]['title']:<38} {len(v):>4} images   "
              f"[{BENCHMARKS[k]['source']}]")
    if not any(sets.values()):
        print("no images found -- are the datasets extracted?")
        return 1
    print()

    report = {"notice_bar": NOTICE_BAR, "prompt": PROMPT,
              "contrasts": CONTRASTS,
              "benchmarks": {k: {kk: vv for kk, vv in BENCHMARKS[k].items()
                                 if kk != "loader"}
                             | {"images": len(sets[k])} | profile(sets[k])
                             for k in args.only},
              "methods": {}}

    for name in args.method:
        print(f"--- {name} ---", flush=True)
        try:
            method = BUILDERS[name]()
        except Exception as exc:
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")
            continue
        kind = getattr(method, "kind", "")
        raws = {k: run(method, v, k) for k, v in sets.items() if v}
        del method
        torch.cuda.empty_cache()

        report["methods"][name] = {
            "kind": kind,
            "results": {k: summarise(v) for k, v in raws.items()},
        }
        pct = lambda v, w=6, d=1: f"{v:>{w}.{d}%}" if v is not None else f"{'-':>{w}}"
        for k, r in report["methods"][name]["results"].items():
            auroc = f"{r['auroc']:.3f}" if r["auroc"] is not None else "  -  "
            print(f"    {k}  noticed {pct(r['object_recall'])}  "
                  f"auroc {auroc}  {r['sec_per_image']:.2f}s/img", flush=True)
        print()

    OUT.mkdir(exist_ok=True)
    dest = OUT / "benchmark.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
