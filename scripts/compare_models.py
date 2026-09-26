"""Which model detects anomalies best from a UAV viewpoint?

Runs every model that can produce an anomaly signal over labelled UAV data and
scores them identically, so the numbers are comparable.

Four methods, from the six checkpoints on disk:

  gdino     Grounding DINO, open-vocabulary text prompts (skill.md section 4).
  prowl-v2  DINOv2 dense features, PROWL-style: build a "normal road" prototype
            from the most common patches, score every patch by distance from it.
  prowl-v3  Same, on DINOv3.
  rba       SegFormer / Mask2Former turned into an anomaly detector the RbA way:
            a pixel that every known Cityscapes class rejects is an unknown
            object. Turns a closed-set segmenter into an open-set detector.

Every method emits a per-pixel anomaly score in [0,1], so all four are scored
with the same metrics -- AUPR, AUROC, FPR@95TPR, and component-level recall.

    python scripts/compare_models.py --dataset uav-rsod --limit 100
    python scripts/compare_models.py --dataset rescuenet --method prowl-v3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "datasets"
OUT = REPO_ROOT / "results"

PROMPT = "debris. tire. cardboard box. luggage. rock. fallen object. obstacle."
# "road" is deliberately absent: Grounding DINO grounds sub-phrases, so a prompt
# containing that word makes it box the road surface itself. Measured on the
# earlier prompt: 19 of 48 aerial detections fired on the bare token "road",
# 7 of them covering >30% of the frame.


# ---------------------------------------------------------------- metrics
def score_pixels(scores: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """AUPR / AUROC / FPR@95TPR over valid pixels, computed by sorting rather
    than sklearn: these arrays are millions of pixels and this stays cheap."""
    s, y = scores[valid].astype(np.float64), gt[valid].astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return {"aupr": None, "auroc": None, "fpr95": None, "n_pos": n_pos, "n_pix": int(y.size)}

    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    n_neg = y.size - n_pos

    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    aupr = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    tpr, fpr = recall, fp / n_neg
    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    i95 = int(np.searchsorted(tpr, 0.95))
    fpr95 = float(fpr[min(i95, len(fpr) - 1)])
    return {"aupr": aupr, "auroc": auroc, "fpr95": fpr95, "n_pos": n_pos, "n_pix": int(y.size)}


def component_recall(scores: np.ndarray, gt: np.ndarray, thr: float) -> tuple[int, int, list]:
    """How many annotated objects were hit at all, and how big were they?

    Pixel AUPR is dominated by large objects; a benchmark about small debris
    needs the per-object view as well.
    """
    from scipy import ndimage

    lab, n = ndimage.label(gt)
    hit, sizes = 0, []
    for i in range(1, n + 1):
        m = lab == i
        a = int(m.sum())
        if a < 25:
            continue
        sizes.append((a, bool((scores[m] >= thr).any())))
        hit += sizes[-1][1]
    return hit, len(sizes), sizes


# ---------------------------------------------------------------- datasets
def _boxes_to_mask(boxes, h, w) -> np.ndarray:
    """VOC boxes -> binary mask, so a box-annotated set can be scored with the
    same pixel metrics as a mask-annotated one.

    This inflates every object to its bounding rectangle, which is generous to
    all four methods equally. It caps achievable AUPR (a perfect segmentation of
    a thin branch still loses against its box), so absolute AUPR here is not
    comparable to a mask-labelled dataset -- only the ranking between methods is.
    """
    m = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def load_uav_rsod(limit: int) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """UAV-RSOD obstacle detection split: image + VOC XML boxes.

    Only the test split is used. Train is left alone even though nothing is
    trained here -- keeping the split honest costs nothing and keeps the door
    open for a later fine-tuning experiment.
    """
    import xml.etree.ElementTree as ET

    root = DATA / "uav-rsod-detection"
    if not root.is_dir():
        return []
    pairs = []
    for xml in sorted(root.rglob("test/*.xml")):
        img = xml.with_suffix(".jpg")
        if not img.is_file():
            continue
        t = ET.parse(xml).getroot()
        sz = t.find("size")
        h, w = int(sz.find("height").text), int(sz.find("width").text)
        boxes = []
        for o in t.findall("object"):
            b = o.find("bndbox")
            boxes.append(tuple(int(float(b.find(k).text))
                               for k in ("xmin", "ymin", "xmax", "ymax")))
        if boxes:
            m = _boxes_to_mask(boxes, h, w)
            pairs.append((img, m, np.ones_like(m)))  # whole frame is valid
    return pairs[:limit] if limit else pairs


# RescueNet 11-class ids, verified against the val labels rather than assumed:
#   0 Background      1 Water            2 Building-No-Damage  3 Building-Minor
#   4 Building-Major  5 Building-Destroyed  6 Vehicle          7 Road-Clear
#   8 Road-Blocked    9 Tree            10 Pool
# Positives are Road-Blocked (8). Note 4 is Building-Major-Damage -- scoring
# against it would silently benchmark building damage instead of road blockage.
RESCUENET_ROAD_BLOCKED = 8
RESCUENET_ROAD_CLEAR = 7


def load_rescuenet(limit: int) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """RescueNet val: image + pixel labels, positives = Road-Blocked."""
    root = DATA / "rescuenet-val"
    if not root.is_dir():
        return []
    labs = {}
    for p in root.rglob("*.png"):
        labs[p.stem.replace("_lab", "")] = p
    pairs = []
    for img in sorted(root.rglob("*.jpg")):
        lp = labs.get(img.stem)
        if not lp:
            continue
        lab = np.array(Image.open(lp))
        if lab.ndim == 3:
            lab = lab[..., 0]
        gt = lab == RESCUENET_ROAD_BLOCKED
        if gt.any():
            # Score only on road. Off-road pixels have no ground truth for
            # "obstacle on the road", so counting them would measure scene
            # segmentation, not the task.
            road = gt | (lab == RESCUENET_ROAD_CLEAR)
            # Skip frames where the entire road is labelled blocked. With no
            # clear road in the frame there are no negatives, so AUPR/AUROC are
            # undefined-in-spirit and would flatter any method that simply
            # fires everywhere. 40 of 63 val frames are like this; 23 survive.
            if gt.sum() < 0.999 * road.sum():
                pairs.append((img, gt, road))
    return pairs[:limit] if limit else pairs


# ---------------------------------------------------------------- methods
class GDino:
    name = "gdino"

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
        r = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.15, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        # Rasterise boxes into a score map so a detector can be scored with the
        # same pixel metrics as the segmentation methods.
        heat = np.zeros(img.size[::-1], dtype=np.float32)
        for b, s in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()):
            x0, y0, x1, y1 = [int(max(0, v)) for v in b]
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], float(s))
        return heat


class Prowl:
    """PROWL-style: normal road is whatever dominates the frame; anomalies are
    what sits far from it in feature space. No training, no anomaly examples."""

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
        inp = self.proc(images=img.resize((self.side, self.side)), return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            h = self.m(**inp).last_hidden_state[0, self.skip:].float()
        h = F.normalize(h, dim=-1)
        n = int(np.sqrt(h.shape[0]))
        # Prototype = mean of the half of the patches closest to the global mean.
        # Using the median-closest half rather than all patches keeps the
        # anomaly itself from being absorbed into the "normal" prototype.
        g = F.normalize(h.mean(0, keepdim=True), dim=-1)
        sim = (h @ g.T).squeeze(1)
        proto = F.normalize(h[sim >= sim.median()].mean(0, keepdim=True), dim=-1)
        dist = (1 - (h @ proto.T).squeeze(1)).reshape(1, 1, n, n)
        up = F.interpolate(dist, size=img.size[::-1], mode="bilinear", align_corners=False)
        d = up[0, 0].cpu().numpy()
        return (d - d.min()) / (d.ptp() + 1e-8) if hasattr(d, "ptp") else \
               (d - d.min()) / (np.ptp(d) + 1e-8)


class Rba:
    """RbA: a pixel rejected by every known class is an unknown object.
    Score = -max over class logits, so low confidence everywhere reads as
    'none of my classes explain this'."""

    def __init__(self, key="segformer-b2-cityscapes"):
        import transformers
        self.name = "rba"
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
        d = (-lg.max(1).values)[0].cpu().numpy()
        return (d - d.min()) / (np.ptp(d) + 1e-8)


def build(names: list[str]) -> list:
    made = []
    for n in names:
        try:
            if n == "gdino":
                made.append(GDino())
            elif n == "prowl-v2":
                made.append(Prowl("dinov2-large"))
            elif n == "prowl-v3":
                made.append(Prowl("dinov3-vitl16"))
            elif n == "rba":
                made.append(Rba())
        except Exception as exc:
            print(f"[skip] {n}: {type(exc).__name__}: {exc}")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["uav-rsod", "rescuenet"], default="uav-rsod")
    ap.add_argument("--method", nargs="+",
                    default=["gdino", "prowl-v2", "prowl-v3", "rba"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    pairs = load_uav_rsod(args.limit) if args.dataset == "uav-rsod" else load_rescuenet(args.limit)
    if not pairs:
        print(f"no labelled pairs found for {args.dataset} -- is it downloaded and extracted?")
        return 1
    print(f"dataset : {args.dataset}  ({len(pairs)} labelled images)\n")

    report = {"dataset": args.dataset, "images": len(pairs), "prompt": PROMPT, "methods": {}}

    for method in build(args.method):
        t0 = time.time()
        agg, hits, objs, sizes = [], 0, 0, []
        for img_p, gt, valid in pairs:
            img = Image.open(img_p).convert("RGB")
            valid = valid.astype(bool)
            s = method(img)
            if s.shape != gt.shape:
                s = np.array(Image.fromarray(s).resize(gt.shape[::-1]))
            agg.append(score_pixels(s, gt, valid))
            h, n, sz = component_recall(s, gt, args.thr)
            hits += h; objs += n; sizes += sz
        dt = (time.time() - t0) / len(pairs)

        def avg(k):
            v = [a[k] for a in agg if a[k] is not None]
            return float(np.mean(v)) if v else None

        by_size = {}
        for a, ok in sizes:
            b = ("very small" if a < 32**2 else "small" if a < 96**2
                 else "medium" if a < 256**2 else "large")
            by_size.setdefault(b, []).append(ok)

        report["methods"][method.name] = {
            "aupr": avg("aupr"), "auroc": avg("auroc"), "fpr95": avg("fpr95"),
            "component_recall": hits / objs if objs else None,
            "objects": objs, "sec_per_image": dt,
            "recall_by_size": {k: {"n": len(v), "recall": float(np.mean(v))}
                               for k, v in sorted(by_size.items())},
        }
        r = report["methods"][method.name]
        print(f"{method.name:<10} AUPR {r['aupr'] or 0:.3f}  AUROC {r['auroc'] or 0:.3f}  "
              f"FPR95 {r['fpr95'] or 0:.3f}  obj-recall {r['component_recall'] or 0:.1%}  "
              f"{dt:.2f}s/img")

    OUT.mkdir(exist_ok=True)
    dest = OUT / f"comparison_{args.dataset}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved: {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
