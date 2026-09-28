"""Task A: which model finds the highway/road region, before Task B (the
anomaly benchmark in benchmark.py) looks for debris inside it.

No UAV+highway dataset on disk has pixel-level road labels: UAVDT (the real
target domain) has none at all, and RescueNet's Road-Clear/-Blocked is
post-hurricane residential streets, not highway. AeroScapes is the closest
real fit found by search: actual drone footage at 5-50 m altitude -- the
operational range this project cares about -- with a pixel-level "road"
class among 11. It is not highway-specific (parks, campuses, streets), so
its number answers "does this model find paved road from a drone at all",
not a highway number specifically -- the same kind of proxy FOD-A (a runway,
not a highway) already plays for the anomaly benchmark's object-size axis.

UAVDT frames (real highway, no labels) get a qualitative-only pass: rendered
with each model's predicted road region overlaid, eyeballed rather than
scored -- the same treatment the deferred B6 got in benchmark.py, reusing
its loader for exactly the real-highway imagery it already curated.

AeroScapes' road class index (10) was verified by eye, not assumed: its
label PNGs use a bespoke palette, not the standard VOC one, so pixel value 10
was matched against Visualizations/<name>.png (green in the visible image =
road, index 10, the whole paved area) for a sample frame before trusting it
across the set.

    python scripts/survey/segment_benchmark.py
    python scripts/survey/segment_benchmark.py --only segformer-b0 --cap 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "datasets" / "aeroscapes" / "aeroscapes"
OUT = REPO_ROOT / "results"

ROAD_INDEX = 10          # AeroScapes SegmentationClass value for "road" -- verified by eye
CITYSCAPES_ROAD_ID = 0   # trainId 0 in every Cityscapes-pretrained checkpoint here
ROAD_PROMPT = "road. highway. pavement. street. driveway. carriageway."
GDINO_THRESHOLD = 0.3    # a real labelled task now, so a plain confidence bar applies
                          # (unlike benchmark.py's anomaly score, which is calibrated
                          # post-hoc because there is no ground truth to threshold against)


def load_split(cap: int) -> list[str]:
    names = (DATA / "ImageSets" / "val.txt").read_text().split()
    if not cap or cap >= len(names):
        return names
    step = len(names) / cap
    return [names[int(i * step)] for i in range(cap)]


def ground_truth(name: str) -> np.ndarray:
    seg = np.array(Image.open(DATA / "SegmentationClass" / f"{name}.png"))
    return seg == ROAD_INDEX


# ---------------------------------------------------------------- methods
class SegformerRoad:
    kind = "closed-set segmenter (Cityscapes), argmax == road"

    def __init__(self, key: str):
        import transformers
        p = MODELS / key
        self.name = key
        cfg = json.loads((p / "config.json").read_text())
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.m = transformers.SegformerForSemanticSegmentation.from_pretrained(
            str(p), dtype=torch.float16).to("cuda").eval()
        self.proc = transformers.SegformerImageProcessor.from_pretrained(str(p))

    def classes(self, img: Image.Image) -> np.ndarray:
        """Full per-pixel class-id map, not just the road bit -- what run() scores,
        and what confusion_over_road() below uses to see what a miss actually gets
        predicted as."""
        inp = self.proc(images=img, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            logits = self.m(**inp).logits.float()
        logits = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].cpu().numpy()

    def __call__(self, img: Image.Image) -> np.ndarray:
        return self.classes(img) == CITYSCAPES_ROAD_ID


class Mask2FormerRoad:
    name = "mask2former"
    kind = "closed-set segmenter (Cityscapes), argmax == road"

    def __init__(self, key: str = "mask2former-swin-tiny-cityscapes"):
        import transformers
        p = MODELS / key
        self.m = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(
            str(p), dtype=torch.float32).to("cuda").eval()
        self.proc = transformers.Mask2FormerImageProcessor.from_pretrained(str(p))

    def __call__(self, img: Image.Image) -> np.ndarray:
        inp = self.proc(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.m(**inp)
        pred = self.proc.post_process_semantic_segmentation(
            out, target_sizes=[img.size[::-1]])[0]
        return pred.cpu().numpy() == CITYSCAPES_ROAD_ID


class GDinoRoad:
    name = "gdino-road"
    kind = "open-vocabulary detector, prompted for road words"

    def __init__(self, max_side=1024):
        import transformers
        p = MODELS / "grounding-dino-tiny"
        self.proc = transformers.AutoProcessor.from_pretrained(str(p))
        self.m = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(p), dtype=torch.float32).to("cuda").eval()
        self.max_side = max_side

    def detect(self, img: Image.Image) -> list[tuple[tuple[int, int, int, int], float]]:
        """Raw detected boxes ((x0,y0,x1,y1), score), before they're collapsed
        into a mask -- kept as its own method so a caller that wants to see
        exactly what Grounding DINO found (e.g. to draw each box) doesn't have
        to reimplement the inference call."""
        sc = min(1.0, self.max_side / max(img.size))
        small = img.resize((int(img.width * sc), int(img.height * sc))) if sc < 1 else img
        inp = self.proc(images=small, text=ROAD_PROMPT, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.m(**inp)
        r = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=GDINO_THRESHOLD, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        boxes = []
        for b, s in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()):
            x0, y0, x1, y1 = (int(max(0, v)) for v in b)
            boxes.append(((x0, y0, x1, y1), float(s)))
        return boxes

    def __call__(self, img: Image.Image) -> np.ndarray:
        mask = np.zeros(img.size[::-1], dtype=bool)
        for (x0, y0, x1, y1), _ in self.detect(img):
            mask[y0:y1, x0:x1] = True
        return mask


BUILDERS = {
    "segformer-b0": lambda: SegformerRoad("segformer-b0-cityscapes"),
    "segformer-b2": lambda: SegformerRoad("segformer-b2-cityscapes"),
    "mask2former": Mask2FormerRoad,
    "gdino-road": GDinoRoad,
}


# ---------------------------------------------------------------- scoring
def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return {
        "iou": inter / union if union else None,
        "precision": inter / pred.sum() if pred.sum() else None,
        "recall": inter / gt.sum() if gt.sum() else None,
    }


def run(method, names: list[str]) -> dict:
    per_img = []
    t0 = time.time()
    for i, name in enumerate(names):
        img = Image.open(DATA / "JPEGImages" / f"{name}.jpg").convert("RGB")
        gt = ground_truth(name)
        pred = method(img)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST))
        per_img.append(score(pred, gt))
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(names)}", flush=True)
    dt = (time.time() - t0) / max(1, len(names))

    def avg(k):
        v = [r[k] for r in per_img if r[k] is not None]
        return float(np.mean(v)) if v else None

    return {"images": len(names), "iou": avg("iou"), "precision": avg("precision"),
            "recall": avg("recall"), "sec_per_image": dt}


def confusion_over_road(key: str, names: list[str]) -> dict:
    """Diagnostic, not a scoring function: across every true-road pixel in the
    split, what class did the closed-set segmenter actually predict? Answers
    "what does a miss get called instead of road", which the IoU/precision/
    recall numbers above cannot -- built to check whether the low scores are a
    preprocessing artefact or a genuine semantic failure (see module docstring
    update / README for the finding this produced)."""
    m = BUILDERS[key]()
    counts: dict[int, int] = {}
    total = 0
    for name in names:
        gt = ground_truth(name)
        if gt.sum() == 0:
            continue
        img = Image.open(DATA / "JPEGImages" / f"{name}.jpg").convert("RGB")
        cls = m.classes(img)
        if cls.shape != gt.shape:
            cls = np.array(Image.fromarray(cls.astype(np.int32)).resize(
                gt.shape[::-1], Image.NEAREST))
        vals, c = np.unique(cls[gt], return_counts=True)
        for v, n in zip(vals.tolist(), c.tolist()):
            counts[v] = counts.get(v, 0) + n
        total += int(gt.sum())
    id2label = m.id2label
    del m
    torch.cuda.empty_cache()
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return {"total_road_px": total,
            "top": [{"class": id2label[k], "frac": v / total} for k, v in ranked[:5]]}


def gdino_uavdt_hit_rate(n: int = 40) -> dict:
    """How often does Grounding DINO's road prompt return zero detections on
    real UAVDT highway frames? No ground truth exists there (see module
    docstring), so this measures reliability, not accuracy: a frame with zero
    boxes hands Stage 2 nothing to look at, which is a worse failure than an
    imprecise box. Reuses benchmark.b6_uav_clean's curated highway frames --
    stratified across altitude x view, daylight only -- rather than a
    hand-picked illustrative sample."""
    sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
    from benchmark import b6_uav_clean

    items = b6_uav_clean(n)
    m = GDinoRoad()
    zero = 0
    for it in items:
        img = Image.open(it.img).convert("RGB")
        pred = m(img)
        if not pred.any():
            zero += 1
    del m
    torch.cuda.empty_cache()
    return {"frames": len(items), "zero_detection_frames": zero,
            "zero_detection_rate": zero / len(items) if items else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(BUILDERS))
    ap.add_argument("--cap", type=int, default=150)
    args = ap.parse_args()

    if not (DATA / "ImageSets" / "val.txt").is_file():
        print("AeroScapes not found -- run scripts/setup/download_datasets.py --key aeroscapes")
        return 1
    names = load_split(args.cap)
    print(f"AeroScapes val: {len(names)} images (road = index {ROAD_INDEX})\n")

    report = {"road_index": ROAD_INDEX, "source": "aeroscapes val split", "methods": {}}
    for key in args.only:
        print(f"--- {key} ---", flush=True)
        try:
            method = BUILDERS[key]()
        except Exception as exc:
            print(f"[skip] {key}: {type(exc).__name__}: {exc}")
            continue
        r = run(method, names)
        r["kind"] = getattr(method, "kind", "")
        del method
        torch.cuda.empty_cache()
        report["methods"][key] = r
        print(f"    IoU {r['iou']:.3f}  precision {r['precision']:.3f}  "
              f"recall {r['recall']:.3f}  {r['sec_per_image']:.2f}s/img\n")

    report["diagnostics"] = {}
    if "segformer-b2" in args.only:
        print("--- confusion_over_road(segformer-b2) ---", flush=True)
        diag = confusion_over_road("segformer-b2", names)
        report["diagnostics"]["segformer-b2_road_confusion"] = diag
        for row in diag["top"]:
            print(f"    {row['frac']:.1%}  predicted as '{row['class']}'")

    if "gdino-road" in args.only:
        print("--- gdino_uavdt_hit_rate ---", flush=True)
        hit = gdino_uavdt_hit_rate()
        report["diagnostics"]["gdino-road_uavdt_hit_rate"] = hit
        print(f"    {hit['zero_detection_frames']}/{hit['frames']} real UAVDT frames "
              f"got zero detections ({hit['zero_detection_rate']:.0%})")

    OUT.mkdir(exist_ok=True)
    dest = OUT / "segmentation_benchmark.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
