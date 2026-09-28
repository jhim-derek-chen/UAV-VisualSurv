"""Does a ground-trained open-vocabulary detector survive an aerial viewpoint?

Grounding DINO is prompted with the skill.md section 4 phrases and run on two
sets. The two sides measure different things, because only one of them has
labels -- stating that plainly is the point, not a shortcoming to paper over.

  A. GROUND, labelled (RoadAnomaly-EPFL + SMIYC). Anomaly masks exist, so this
     yields real recall: of the annotated anomalies, how many does a detection
     box actually cover? This is the control -- it says whether the method works
     at all in the domain it was built for.

  B. AERIAL, unlabelled (UAVDT, stratified by altitude x view). No anomaly
     annotations exist, and these are ordinary traffic scenes with no planted
     obstacles. So recall is not computable; what *is* computable is how much
     the detector fires, and how confidently.

     Caveat that must travel with every number from side B: UAVDT roads carry
     real vehicles, and a prompt like "object on road" legitimately matches a
     car. Firing here is therefore not automatically a false positive. The
     comparable quantity across A and B is the *rate and confidence* of firing,
     not its correctness.

    python scripts/survey/eval_transfer.py
    python scripts/survey/eval_transfer.py --limit 20 --threshold 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "datasets"
OUT = REPO_ROOT / "results"

# skill.md section 4 prompts. Grounding DINO wants lowercase, period-separated.
PROMPT = "object on road. foreign object. debris on road. unexpected object. obstacle."

ATTR_NAMES = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt",
              "front-view", "side-view", "bird-view", "long-term"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def ground_samples(limit: int) -> list[tuple[Path, Path, int]]:
    """(image, semantic label, anomaly value). Only labelled frames."""
    out = []
    for lab in sorted((DATA / "road-anomaly-epfl/RoadAnomaly_jpg/frames").rglob("labels_semantic.png")):
        img = lab.parent.parent / (lab.parent.name.replace(".labels", "") + ".jpg")
        if img.is_file():
            out.append((img, lab, 2))  # EPFL encodes anomaly as 2
    for lab in sorted((DATA / "smiyc-road-obstacle21/dataset_ObstacleTrack/labels_masks").glob("*_labels_semantic.png")):
        img = lab.parent.parent / "images" / (lab.name.replace("_labels_semantic.png", "") + ".webp")
        if img.is_file():
            out.append((img, lab, 1))  # SMIYC encodes anomaly as 1, 255 = void
    for lab in sorted((DATA / "smiyc-road-anomaly21/dataset_AnomalyTrack/labels_masks").glob("*_labels_semantic.png")):
        img = lab.parent.parent / "images" / (lab.name.replace("_labels_semantic.png", "") + ".jpg")
        if img.is_file():
            out.append((img, lab, 1))
    return out[:limit] if limit else out


def aerial_samples(per_stratum: int) -> list[tuple[Path, str]]:
    root = DATA / "uavdt-benchmark-m/UAV-benchmark-M"
    attr_dir = DATA / "uavdt-attributes/M_attr"
    if not root.is_dir() or not attr_dir.is_dir():
        return []
    attrs = {}
    for f in attr_dir.rglob("*_attr.txt"):
        attrs[f.stem.replace("_attr", "").strip()] = dict(
            zip(ATTR_NAMES, [int(x) for x in f.read_text().strip().split(",")]))

    chosen: dict[Path, str] = {}
    for alt in ["low-alt", "medium-alt", "high-alt"]:
        for view in ["front-view", "side-view", "bird-view"]:
            seqs = sorted(s for s, a in attrs.items()
                          if a[alt] and a[view] and a["daylight"] and (root / s).is_dir())
            for seq in seqs[:per_stratum]:
                frames = sorted((root / seq).glob("*.jpg"))
                if not frames:
                    continue
                # Spread within the sequence; consecutive frames are near-identical.
                for f in frames[len(frames) // 4::max(1, len(frames) // 3)][:3]:
                    chosen.setdefault(f, f"{alt[:-4]}/{view[:-5]}")
    return sorted(chosen.items(), key=lambda kv: kv[1])


def components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Connected anomaly regions as boxes. scipy avoids an opencv dependency."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(lab):
        y, x = sl
        if (y.stop - y.start) * (x.stop - x.start) >= 25:  # drop annotation specks
            boxes.append((x.start, y.start, x.stop, y.stop))
    return boxes


def hit(gt: tuple, dets: np.ndarray, thr: float = 0.25) -> tuple[bool, bool]:
    """Two verdicts for one ground-truth box, because they answer different questions.

    noticed : intersection over the GT area. "Did anything get flagged here at
              all?" -- skill.md section 9's obstacle recall. Forgiving about a
              detection box that is much larger than the object.
    located : standard IoU. "Was it localised?" -- much stricter.

    Reporting only the first inflates recall roughly 2x on this data, and it
    inverts the recall-vs-size trend: tiny GT boxes are cheaply covered by any
    overlapping detection, so they score *higher* than large ones, which is
    backwards. Both numbers must travel together.
    """
    if len(dets) == 0:
        return False, False
    x0, y0, x1, y1 = gt
    gt_area = max(1, (x1 - x0) * (y1 - y0))
    ix0 = np.maximum(dets[:, 0], x0); iy0 = np.maximum(dets[:, 1], y0)
    ix1 = np.minimum(dets[:, 2], x1); iy1 = np.minimum(dets[:, 3], y1)
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    d_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    iou = inter / np.maximum(d_area + gt_area - inter, 1)
    return bool((inter / gt_area).max() >= thr), bool(iou.max() >= thr)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Detector:
    """Grounding DINO. fp32 only -- its text tower is fp32 and autocast measured
    worse than plain fp32 (3.81 vs 3.49 GB), so there is nothing to gain."""

    def __init__(self, key="grounding-dino-tiny", max_side=1024):
        import transformers

        path = MODELS / key
        self.proc = transformers.AutoProcessor.from_pretrained(str(path))
        self.model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(path), dtype=torch.float32).to("cuda").eval()
        self.max_side = max_side

    def __call__(self, img: Image.Image, threshold: float):
        # Downscale to keep 4 GB VRAM; Grounding DINO trained near 800x1333, so
        # this is also closer to its native scale than the full frame would be.
        scale = min(1.0, self.max_side / max(img.size))
        small = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale)))) \
            if scale < 1.0 else img
        inputs = self.proc(images=small, text=PROMPT, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=threshold, text_threshold=0.25,
            target_sizes=[small.size[::-1]])[0]
        boxes = res["boxes"].cpu().numpy() / scale
        # text_labels says which prompt phrase fired -- essential when judging
        # aerial boxes by eye, since "obstacle" and "object on road" mean very
        # different things about what the model thinks it found.
        labels = res.get("text_labels") or res.get("labels") or [""] * len(boxes)
        return boxes, res["scores"].cpu().numpy(), list(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="cap ground images (0 = all)")
    ap.add_argument("--per-stratum", type=int, default=2)
    args = ap.parse_args()

    det = Detector()
    report: dict = {"prompt": PROMPT, "threshold": args.threshold}

    # ---- A. ground, labelled -> real recall -------------------------------
    ground = ground_samples(args.limit)
    print(f"A. GROUND (labelled): {len(ground)} images")
    hits = hits_iou = total = 0
    n_det = []
    scores_g = []
    per_size: dict[str, list[int]] = {}
    for img_p, lab_p, anom_val in ground:
        img = Image.open(img_p).convert("RGB")
        gt = components(np.array(Image.open(lab_p)) == anom_val)
        boxes, sc, _ = det(img, args.threshold)
        n_det.append(len(boxes))
        scores_g.extend(sc.tolist())
        for b in gt:
            total += 1
            noticed, located = hit(b, boxes)
            hits += noticed
            hits_iou += located
            a = (b[2] - b[0]) * (b[3] - b[1])
            bucket = ("very small" if a < 32**2 else "small" if a < 96**2
                      else "medium" if a < 256**2 else "large")
            per_size.setdefault(bucket, []).append((int(noticed), int(located)))
    report["ground"] = {
        "images": len(ground), "gt_objects": total,
        "recall_noticed": hits / total if total else None,
        "recall_located_iou": hits_iou / total if total else None,
        "det_per_image": float(np.mean(n_det)) if n_det else 0.0,
        "mean_score": float(np.mean(scores_g)) if scores_g else None,
        "recall_by_size": {k: {"n": len(v),
                               "noticed": float(np.mean([a for a, _ in v])),
                               "located": float(np.mean([b for _, b in v]))}
                           for k, v in sorted(per_size.items())},
    }
    g = report["ground"]
    print(f"   {total} annotated objects")
    print(f"   recall  noticed {g['recall_noticed']:.1%}   located(IoU) {g['recall_located_iou']:.1%}")
    print(f"   detections/image {g['det_per_image']:.1f}, mean score {g['mean_score']:.3f}")
    for k, v in g["recall_by_size"].items():
        print(f"     {k:<11} n={v['n']:<4} noticed {v['noticed']:>6.1%}   located {v['located']:>6.1%}")

    # ---- B. aerial, unlabelled -> firing rate only ------------------------
    aerial = aerial_samples(args.per_stratum)
    print(f"\nB. AERIAL (unlabelled): {len(aerial)} images")
    by_stratum: dict[str, list] = {}
    for img_p, stratum in aerial:
        img = Image.open(img_p).convert("RGB")
        boxes, sc, _ = det(img, args.threshold)
        by_stratum.setdefault(stratum, []).append((len(boxes), sc))
    report["aerial"] = {}
    print(f"   {'STRATUM':<18}{'IMGS':>5}{'DET/IMG':>9}{'MEAN SCORE':>12}")
    for k, v in sorted(by_stratum.items()):
        counts = [c for c, _ in v]
        allsc = np.concatenate([s for _, s in v]) if any(len(s) for _, s in v) else np.array([])
        report["aerial"][k] = {
            "images": len(v), "det_per_image": float(np.mean(counts)),
            "mean_score": float(allsc.mean()) if allsc.size else None,
        }
        ms = f"{allsc.mean():.3f}" if allsc.size else "-"
        print(f"   {k:<18}{len(v):>5}{np.mean(counts):>9.1f}{ms:>12}")

    OUT.mkdir(exist_ok=True)
    (OUT / "transfer_eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved: results/transfer_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
