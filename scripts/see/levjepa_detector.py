"""A single-pass detector on LeVJEPA features (architecture D, "fast").

OWLv2 (architectures A-C) reads a 4K frame as about 18 overlapping tiles at
960 px each, one after another: 4.3 s per frame. Here the whole frame is
resized once (long side LONG_SIDE px), encoded once by LeVJEPA's ViT-L/16
(0.45 s at 1344 x 768 on the RTX 3050 Ti), and every 16 px patch is
classified background / vehicle / debris by a linear probe on its 1024-d
feature. Connected patches of one class become one box, scored by the
highest class probability inside it.

LeVJEPA (Kuhn, Maes, ..., LeCun, Balestriero, Buettner, 2026,
arXiv:2608.27395) is a video encoder trained with the LeJEPA objective; its
savings are in pretraining compute, not inference. The only released
checkpoint is ViT-L/16, galilai-group/LeVJEPA-VideoMix-Large, weights under
CC BY-NC 4.0. A single frame is fed as a 1-frame clip (its RoPE positions
accept any frame count and grid size).

The probe is trained on the training renders (dev set) and applied
leave-one-location-out: a frame from location L is scored by a probe fitted
without L's frames, as for the OWLv2 re-scorer.

    python scripts/see/levjepa_detector.py            # features, probe, evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
import eval_road_objects as E  # noqa: E402
import train_rescorer as T  # noqa: E402

MODEL = REPO_ROOT / "models" / "levjepa-videomix-large"
FEAT = REPO_ROOT / ".cache" / "levjepa_feats"
OUT = E.OUT
RUN = "levjepa-patch"
LONG_SIDE = 1344
PATCH = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ("background", "vehicle", "debris")
COVER = 0.25       # a patch is labelled with an object that covers this much of it
REGION_T = 0.3     # patches above this class probability can belong to an object
PEAK_T = 0.5       # an object is seeded at a local maximum above this


class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.m = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True,
                                           dtype=torch.float16).cuda().eval()

    def __call__(self, img: Image.Image):
        """Patch features (Hp, Wp, 1024) float16 and the resize factor."""
        torch = self.torch
        s = LONG_SIDE / max(img.size)
        w, h = round(img.width * s), round(img.height * s)
        Wp, Hp = -(-w // PATCH), -(-h // PATCH)
        arr = np.asarray(img.resize((w, h), Image.BILINEAR), np.float32) / 255.0
        pad = np.tile(MEAN, (Hp * PATCH, Wp * PATCH, 1))
        pad[:h, :w] = arr
        x = torch.from_numpy((pad - MEAN) / STD).permute(2, 0, 1)[None, :, None].cuda().half()
        with torch.inference_mode():
            tok = self.m(pixel_values=x).last_hidden_state[0, 1:]
        return tok.reshape(Hp, Wp, -1).float().cpu().numpy().astype(np.float16), s


def features(root: Path, images: dict, tag: str, enc: Encoder | None = None):
    """Cached per image file size and mtime, like train_rescorer.extract."""
    d = FEAT / tag
    d.mkdir(parents=True, exist_ok=True)
    out, t_run, n_run = {}, 0.0, 0
    for i, im in sorted(images.items()):
        f = root / "images" / im["file_name"]
        st = f.stat()
        cf = d / f"{f.stem}_{st.st_size}_{int(st.st_mtime)}_{LONG_SIDE}.npz"
        if not cf.is_file():
            enc = enc or Encoder()
            t0 = time.time()
            feat, s = enc(Image.open(f).convert("RGB"))
            t_run += time.time() - t0
            n_run += 1
            np.savez_compressed(cf, feat=feat, scale=s)
        z = np.load(cf)
        out[i] = (z["feat"], float(z["scale"]))
    if n_run:
        print(f"  LeVJEPA features {tag}: {n_run} images, {t_run / n_run:.2f} s/image "
              f"(incl. resize)")
    return out


def patch_labels(feat_shape, scale, gts, ignore):
    """0 background, 1 vehicle, 2 debris, -1 skip, per patch."""
    Hp, Wp = feat_shape[:2]
    lab = np.zeros((Hp, Wp), np.int8)
    side = PATCH / scale  # patch side in original pixels
    for g in gts:
        x, y, w, h = g["bbox"]
        c = 1 if g["category"] == "vehicle" else 2
        c0, c1 = int(x // side), int((x + w) // side)
        r0, r1 = int(y // side), int((y + h) // side)
        for r in range(max(0, r0), min(Hp, r1 + 1)):
            for q in range(max(0, c0), min(Wp, c1 + 1)):
                ix = max(0, min(x + w, (q + 1) * side) - max(x, q * side))
                iy = max(0, min(y + h, (r + 1) * side) - max(y, r * side))
                centre = (q * side <= x + w / 2 < (q + 1) * side and
                          r * side <= y + h / 2 < (r + 1) * side)
                if ix * iy >= COVER * side * side or centre:
                    lab[r, q] = max(lab[r, q], c) if lab[r, q] >= 0 else c
                elif lab[r, q] == 0:
                    lab[r, q] = -1
    for rx, ry, rw, rh in ignore:
        lab[int(ry // side):int((ry + rh) // side) + 1,
            int(rx // side):int((rx + rw) // side) + 1] = -1
    return lab


class Probe:
    """Softmax regression on standardised features, class-balanced, L2 weight
    `1 / (C * n)` as in scikit-learn's LogisticRegression; trained full-batch
    with L-BFGS on the GPU (scikit-learn took minutes per fit at 2048-d)."""

    def __init__(self, X, y, C=0.05, steps=150):
        import torch
        self.torch = torch
        dev = "cuda"
        Xt = torch.from_numpy(X).to(dev)
        self.mean, self.std = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = (Xt - self.mean) / self.std
        yt = torch.from_numpy(y.astype(np.int64)).to(dev)
        counts = torch.bincount(yt, minlength=3).float()
        wcls = len(yt) / (3 * counts.clamp(min=1))
        self.lin = torch.nn.Linear(X.shape[1], 3).to(dev)
        opt = torch.optim.LBFGS(self.lin.parameters(), lr=1, max_iter=steps, line_search_fn="strong_wolfe")
        lam = 1.0 / (C * len(yt))

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.lin(Xt), yt, weight=wcls) + \
                lam * 0.5 * (self.lin.weight ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)

    def predict_proba(self, X):
        torch = self.torch
        with torch.no_grad():
            Xt = (torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda() - self.mean) / self.std
            return torch.softmax(self.lin(Xt), -1).cpu().numpy()


def fit_probe(X, y, C=0.05):
    return Probe(X, y, C)


def context(f):
    """Each patch feature with the mean feature of its 3 x 3 neighbourhood."""
    from scipy import ndimage
    f = f.astype(np.float32)
    return np.concatenate([f, ndimage.uniform_filter(f, size=(3, 3, 1), mode="nearest")], -1)


def boxes_from_probs(P, scale):
    """One object per local maximum of a class probability map: watershed
    from the maxima above PEAK_T over the patches above REGION_T, so
    vehicles side by side in adjacent lanes are split. Box in original
    pixels; score = the peak probability."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    side = PATCH / scale
    out = []
    for c in (1, 2):
        p = P[..., c]
        mask = p >= REGION_T
        if not mask.any():
            continue
        peaks = peak_local_max(p, min_distance=1, threshold_abs=PEAK_T, labels=mask.astype(int),
                               exclude_border=False)
        markers = np.zeros(p.shape, np.int32)
        for k, (r, q) in enumerate(peaks, 1):
            markers[r, q] = k
        lab = watershed(-p, markers, mask=mask)
        for k in range(1, len(peaks) + 1):
            rr, qq = np.nonzero(lab == k)
            if not len(rr):
                continue
            x0, y0 = qq.min() * side, rr.min() * side
            x1, y1 = (qq.max() + 1) * side, (rr.max() + 1) * side
            out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0),
                        float(p[peaks[k - 1][0], peaks[k - 1][1]]), c])
    return out


def train_matrix(feats, gt, images, loc, ctx=True, bg_per_image=1500, seed=0):
    rng = np.random.default_rng(seed)
    X, y, G = [], [], []
    for i, (f, s) in feats.items():
        lab = patch_labels(f.shape, s, gt[i], images[i].get("ignore_regions", []))
        f = context(f) if ctx else f
        flat, fl = f.reshape(-1, f.shape[-1]), lab.reshape(-1)
        idx = np.nonzero(fl > 0)[0]
        bg = np.nonzero(fl == 0)[0]
        idx = np.concatenate([idx, rng.choice(bg, min(bg_per_image, len(bg)), replace=False)])
        X.append(flat[idx].astype(np.float32))
        y.append(fl[idx])
        G += [loc[i]] * len(idx)
    return np.concatenate(X), np.concatenate(y), np.array(G)


def evaluate(feats, gt, images, loc, probes, masks, ctx=True):
    """Held-out-halves protocol of eval_road_objects: the score threshold is
    chosen on half the frames and applied to the other half, both ways."""
    preds, t_head = {}, []
    for i, (f, s) in feats.items():
        t0 = time.time()
        g = context(f) if ctx else f.astype(np.float32)
        P = probes[loc[i]].predict_proba(g.reshape(-1, g.shape[-1])).reshape(f.shape[0], f.shape[1], 3)
        u = boxes_from_probs(P, s)
        t_head.append(time.time() - t0)
        preds[i] = [u, [b for b in u if E.on_road(b, masks[i])]]
    ids = sorted(feats)
    gt_by = {i: gt[i] for i in ids}
    ign_by = {i: images[i].get("ignore_regions", []) for i in ids}
    res = {"head_s_per_image": round(float(np.mean(t_head)), 3)}
    for which, label in ((0, "unrestricted"), (1, "two_stage")):
        rows = E.sweep(ids, gt_by, preds, ign_by, which)
        halves = (ids[0::2], ids[1::2])
        held = []
        for tune, test in (halves, halves[::-1]):
            t = E.best_f1(E.sweep(tune, gt_by, preds, ign_by, which))["threshold"]
            held.append(dict(threshold=t, **E.score_at(test, gt_by, preds, ign_by, t, which)))
        pooled = {}
        for k in ("vehicle", "debris"):
            n = sum(h[f"n_{k}"] for h in held)
            pooled[f"recall_{k}"] = round(sum(h[f"recall_{k}"] * h[f"n_{k}"] for h in held) / n, 3)
        tp_fp = [(h["precision"] or 0, h["predictions_per_image"]) for h in held]
        pooled["precision"] = round(sum(p * n for p, n in tp_fp) / max(1e-9, sum(n for _, n in tp_fp)), 3)
        res[label] = {"held_out": held, "held_out_pooled": pooled, "best_f1": E.best_f1(rows),
                      "max_recall": {k: round(rows[0][f"recall_{k}"], 3) for k in ("vehicle", "debris")}}
    return res, preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true",
                    help="compare probe settings on the dev set (leave-one-location-out)")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()
    dev_images, dev_gt, dev_loc = T.load_set(T.TRAIN)
    dev_f = features(T.TRAIN, dev_images, "train")
    dev_masks = E.stage1_masks(sorted(dev_images), dev_images, dataset=T.TRAIN)
    if args.select:
        for ctx in (False, True):
            X, y, G = train_matrix(dev_f, dev_gt, dev_images, dev_loc, ctx)
            for C in (0.01, 0.05, 0.2):
                probes = {L: fit_probe(X[G != L], y[G != L], C) for L in sorted(set(G))}
                res, _ = evaluate(dev_f, dev_gt, dev_images, dev_loc, probes, dev_masks, ctx)
                print(json.dumps({"context": ctx, "C": C, **res["two_stage"]["held_out_pooled"],
                                  "max_recall": res["two_stage"]["max_recall"]}), flush=True)
        return 0
    ctx = not args.no_context
    test_images, test_gt, test_loc = T.load_set(T.TEST)
    test_f = features(T.TEST, test_images, "test")
    X, y, G = train_matrix(dev_f, dev_gt, dev_images, dev_loc, ctx)
    probes = {L: fit_probe(X[G != L], y[G != L], args.C) for L in sorted(set(G))}
    masks = E.stage1_masks(sorted(test_images), test_images)
    res, preds = evaluate(test_f, test_gt, test_images, test_loc, probes, masks, ctx)
    result = {"method": RUN, "long_side": LONG_SIDE, "C": args.C, "context": ctx, **res}
    for label in ("unrestricted", "two_stage"):
        print(label, res[label]["held_out_pooled"], "max recall", res[label]["max_recall"])
    (OUT / f"preds_{RUN}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")
    (OUT / f"{RUN}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
