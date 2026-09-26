"""Stage 2 re-scorer: a linear probe on OWLv2's own box features.

Why: after the Stage 1 fixes and size-routed scales, every debris object has
an OWLv2 box on it, but objectness alone ranks the harder ones (a flat
mattress, a small suitcase) below street lamps, gantry parts and people on a
footpath. Text queries for known background did not separate them either:
debris matched "a painted arrow" or "a traffic light" as strongly as the
real lamps did. What does separate them is learned from examples.

What: logistic regression on [objectness logit, 512-d OWLv2 class embedding]
per box, "target on the road" (vehicle or debris) vs "anything else".
Training boxes come only from the training renders
(build_training_renders.py), never from the 30 test images.

Evaluation is leave-one-location-out. The test set has 4 shoot locations;
the boxes of each location's test images are scored by a probe trained on
the other three locations' renders, so every test image is scored by a model
that never saw its location. The regularisation strength is chosen the same
way inside the training set only.

Output: results/road-object-eval/preds_owlv2-rescored.json in the format
eval_road_objects.py reads; then it runs that script with --rescore so the
numbers are computed exactly like every other method's.

    python scripts/train_rescorer.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import eval_road_objects as E  # noqa: E402
import highway_backgrounds  # noqa: E402

TEST = REPO_ROOT / "datasets" / "synthetic-highway-debris"
TRAIN = REPO_ROOT / "datasets" / "synthetic-highway-debris-train"
FEAT = E.OUT / "features"
RUN = "owlv2-rescored"
MIN_OBJ = 0.02          # boxes below this objectness are never candidates
POS_IOU, NEG_IOU = 0.3, 0.1
# Fusion (run "owlv2-fused"): a box is kept if the probe is confident on its
# own, OR if objectness passes the first version's operating threshold and
# the probe does not reject it. Why: tested leave-one-location-out, the probe
# under-scores the heavily motion-blurred traffic of the static-camera clip
# (the other three clips have none to learn from), while objectness had been
# reliable on vehicles all along. OBJ_GATE is the first version's held-out
# operating point, PROBE_GATE the probability midpoint; neither is tuned
# here. The one free threshold (on the probe) is swept by
# eval_road_objects.py like every other method's.
OBJ_GATE, PROBE_GATE = 0.15, 0.5
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)


def load_set(root: Path) -> tuple[dict, dict, dict]:
    anns = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in anns["categories"]}
    images = {im["id"]: im for im in anns["images"]}
    gt: dict[int, list] = {i: [] for i in images}
    for a in anns["annotations"]:
        gt[a["image_id"]].append({"bbox": a["bbox"], "category": cats[a["category_id"]]})
    loc = {s["image_id"]: highway_backgrounds.location_key(Path(s["background_source"]))
           for s in man["samples"]}
    return images, gt, loc


BLUR_DIR = "images_motionblur"
BLUR_FRAC = 0.3  # blur length as a share of the vehicle's length


def motion_blur_copies(images: dict, gt: dict) -> None:
    """Training augmentation: a copy of every training render with each
    labelled vehicle smeared along its long axis (a line kernel BLUR_FRAC of
    its length); debris is left sharp, since it does not move.

    Why: tested leave-one-location-out, the probe under-scored the static-
    camera clip's traffic, which is heavily motion-blurred, and none of the
    other three clips has blurred traffic to learn from. Drone video of a
    motorway has this blur in general, so it belongs in the training data.
    The test images are not touched."""
    import cv2
    out_dir = TRAIN / BLUR_DIR
    out_dir.mkdir(exist_ok=True)
    for i, im in images.items():
        dst = out_dir / im["file_name"]
        if dst.is_file():
            continue
        a = cv2.imread(str(TRAIN / "images" / im["file_name"]))
        H, W = a.shape[:2]
        for g in gt[i]:
            if g["category"] != "vehicle":
                continue
            x, y, w, h = (int(v) for v in g["bbox"])
            k = max(3, int(BLUR_FRAC * max(w, h)) | 1)
            kern = np.zeros((k, k), np.float32)
            if w >= h:
                kern[k // 2, :] = 1.0 / k
            else:
                kern[:, k // 2] = 1.0 / k
            X0, Y0 = max(0, x - k), max(0, y - k)
            X1, Y1 = min(W, x + w + k), min(H, y + h + k)
            blurred = cv2.filter2D(a[Y0:Y1, X0:X1], -1, kern)
            # Only the vehicle and a margin of half the kernel pick up the
            # smear, so the road around it stays sharp.
            m = k // 2
            bx0, by0 = max(0, x - m) - X0, max(0, y - m) - Y0
            bx1, by1 = min(W, x + w + m) - X0, min(H, y + h + m) - Y0
            a[Y0 + by0:Y0 + by1, X0 + bx0:X0 + bx1] = blurred[by0:by1, bx0:bx1]
        cv2.imwrite(str(dst), a, [cv2.IMWRITE_JPEG_QUALITY, 92])


def extract(root: Path, images: dict, tag: str,
            subdir: str = "images") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per image: boxes (N x 9, Owlv2Scored rows) and embeddings (N x 512).
    Cached per image file size and mtime."""
    d = FEAT / tag
    d.mkdir(parents=True, exist_ok=True)
    out, det = {}, None
    t0, n_run = time.time(), 0
    for i, im in sorted(images.items()):
        f = root / subdir / im["file_name"]
        st = f.stat()
        cf = d / f"{f.stem}_{st.st_size}_{int(st.st_mtime)}.npz"
        if not cf.is_file():
            if det is None:
                det = E.Owlv2Scored()
                print(f"extracting OWLv2 features: {tag}")
            rows, emb = det.detect(Image.open(f).convert("RGB"), with_embeddings=True)
            np.savez_compressed(cf, rows=np.array(rows, dtype=np.float32).reshape(-1, 9), emb=emb)
            n_run += 1
        z = np.load(cf)
        out[i] = (z["rows"], z["emb"].astype(np.float32))
    if det is not None:
        import torch
        del det
        torch.cuda.empty_cache()
        print(f"  {n_run} images, {(time.time() - t0) / max(1, n_run):.1f} s/image")
    return out


def features(rows: np.ndarray, emb: np.ndarray) -> np.ndarray:
    o = np.clip(rows[:, 5], 1e-4, 1 - 1e-4)
    return np.hstack([np.log(o / (1 - o))[:, None], emb])


def labels(rows: np.ndarray, gts: list, ignore: list) -> tuple[np.ndarray, np.ndarray]:
    """1 = on a vehicle or debris (IoU >= POS_IOU), 0 = on nothing (IoU <
    NEG_IOU with every label, centre outside ignore regions), -1 = skip.
    Also returns whether each positive is debris, for weighting."""
    y = np.full(len(rows), -1, dtype=np.int8)
    is_debris = np.zeros(len(rows), dtype=bool)
    for k, r in enumerate(rows):
        best, cat = 0.0, None
        for g in gts:
            v = E.iou(g["bbox"], r[:4])
            if v > best:
                best, cat = v, g["category"]
        if best >= POS_IOU:
            y[k] = 1
            is_debris[k] = cat != "vehicle"
        elif best < NEG_IOU and not E.centre_in_any(r, ignore):
            y[k] = 0
    return y, is_debris


def training_matrix(feats: dict, images: dict, gt: dict, masks: dict, loc: dict):
    X, Y, W, G = [], [], [], []
    for i, (rows, emb) in feats.items():
        keep = rows[:, 5] >= MIN_OBJ
        keep &= np.array([E.on_road(r, masks[i]) for r in rows], dtype=bool)
        rows, emb = rows[keep], emb[keep]
        y, deb = labels(rows, gt[i], images[i].get("ignore_regions", []))
        m = y >= 0
        X.append(features(rows[m], emb[m]))
        Y.append(y[m])
        W.append(np.where(deb[m], 1.0, 0.0))
        G += [loc[i]] * int(m.sum())
    X, Y, D, G = np.vstack(X), np.concatenate(Y), np.concatenate(W).astype(bool), np.array(G)
    # Weights: debris positives, vehicle positives and negatives each carry
    # a third of the total, so 400 negatives per image and a few hundred
    # vehicles cannot drown out 90 debris.
    w = np.ones(len(Y))
    for m in (Y == 0, (Y == 1) & ~D, (Y == 1) & D):
        if m.any():
            w[m] = len(Y) / (3 * m.sum())
    return X, Y, w, D, G


def fit(X, Y, w, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(X, Y, logisticregression__sample_weight=w)
    return clf


def choose_c(X, Y, w, D, G) -> float:
    """Leave-one-location-out inside the training renders only: the C whose
    held-out locations rank debris and vehicles above background best
    (mean of the two average precisions)."""
    from sklearn.metrics import average_precision_score
    best, best_c = -1.0, C_GRID[0]
    for C in C_GRID:
        scores = []
        for g in np.unique(G):
            tr, te = G != g, G == g
            if not (Y[te] == 1).any():
                continue
            p = fit(X[tr], Y[tr], w[tr], C).predict_proba(X[te])[:, 1]
            neg = Y[te] == 0
            for pos in ((Y[te] == 1) & D[te], (Y[te] == 1) & ~D[te]):
                if pos.any():
                    m = pos | neg
                    scores.append(average_precision_score(pos[m], p[m]))
        s = float(np.mean(scores))
        print(f"  C={C:<6} held-out-location AP {s:.3f}")
        if s > best:
            best, best_c = s, C
    return best_c


PROBE_DIR = REPO_ROOT / "models" / "owlv2-rescorer"


def save_probe(clf, C: float) -> None:
    """The deployable probe: trained on all four locations' renders (the
    leave-one-location-out probes exist only to score the test set honestly).
    Stored as plain arrays, not a pickle."""
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROBE_DIR / "probe.npz", mean=sc.mean_, scale=sc.scale_,
             coef=lr.coef_[0], intercept=lr.intercept_)
    (PROBE_DIR / "config.json").write_text(json.dumps({
        "features": "[logit(objectness), 512-d OWLv2 class embedding] per box, "
                    "from eval_road_objects.Owlv2Scored.detect(with_embeddings=True)",
        "score": "sigmoid(coef . (x - mean) / scale + intercept)",
        "fusion": "keep if score >= threshold, or objectness >= obj_gate and score >= probe_gate",
        "obj_gate": OBJ_GATE, "probe_gate": PROBE_GATE, "C": C,
        "min_objectness": MIN_OBJ,
        "trained_on": str(TRAIN.relative_to(REPO_ROOT)) + " (plus motion-blur copies)",
    }, indent=2), encoding="utf-8")
    print(f"saved: {PROBE_DIR.relative_to(REPO_ROOT)}/")


def main() -> int:
    te_images, te_gt, te_loc = load_set(TEST)
    tr_images, tr_gt, tr_loc = load_set(TRAIN)
    tr_feats = extract(TRAIN, tr_images, "train")
    motion_blur_copies(tr_images, tr_gt)
    blur_feats = extract(TRAIN, tr_images, "train_motionblur", subdir=BLUR_DIR)
    te_feats = extract(TEST, te_images, "test")
    tr_masks = E.stage1_masks(sorted(tr_images), tr_images, dataset=TRAIN)

    parts = [training_matrix(f, tr_images, tr_gt, tr_masks, tr_loc)
             for f in (tr_feats, blur_feats)]
    X, Y, D, G = (np.concatenate([p[k] for p in parts]) for k in (0, 1, 3, 4))
    w = np.ones(len(Y))
    for m in (Y == 0, (Y == 1) & ~D, (Y == 1) & D):
        if m.any():
            w[m] = len(Y) / (3 * m.sum())
    print(f"training boxes: {len(Y)}  (debris {int((D & (Y == 1)).sum())}, "
          f"vehicle {int((~D & (Y == 1)).sum())}, background {int((Y == 0).sum())})")
    C = choose_c(X, Y, w, D, G)
    print(f"chosen C = {C}")

    preds, log = {}, {}
    for L in sorted(set(te_loc.values())):
        m = G != L
        clf = fit(X[m], Y[m], w[m], C)
        log[L] = {"trained_on": sorted(set(G[m])), "boxes": int(m.sum())}
        for i in [i for i in te_images if te_loc[i] == L]:
            rows, emb = te_feats[i]
            keep = rows[:, 5] >= MIN_OBJ
            rows, emb = rows[keep], emb[keep]
            p = clf.predict_proba(features(rows, emb))[:, 1] if len(rows) else np.zeros(0)
            boxes = [[int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(s), float(r[5])]
                     for r, s in zip(rows, p)]
            preds[i] = [boxes, boxes]
        print(f"  {L}: probe trained on {log[L]['trained_on']}")

    (E.OUT / f"preds_{RUN}.json").write_text(json.dumps(
        {"sec_per_image": None, "predictions": preds, "C": C, "folds": log}), encoding="utf-8")
    save_probe(fit(X, Y, w, C), C)
    fused = {}
    for i, (boxes, _) in preds.items():
        # Gated boxes score 1 + probe: kept at every probe threshold,
        # still ranked among themselves by the probe.
        b = [[*x[:4], (1.0 + x[4]) if (x[5] >= OBJ_GATE and x[4] >= PROBE_GATE) else x[4], x[5]]
             for x in boxes]
        fused[i] = [b, b]
    (E.OUT / "preds_owlv2-fused.json").write_text(json.dumps(
        {"sec_per_image": None, "predictions": fused, "C": C, "folds": log,
         "obj_gate": OBJ_GATE, "probe_gate": PROBE_GATE,
         # The free parameter is the probe threshold, in [0, 1]; above 1 the
         # sweep would start dropping gated boxes, which the rule keeps.
         "threshold_max": 1.0}), encoding="utf-8")
    rc = 0
    for run in (RUN, "owlv2-fused"):
        rc |= subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "eval_road_objects.py"),
                               "--method", run, "--rescore"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
