"""Qualitative smoke test on real imagery.

Reported percentages are predicted-road pixels over the WHOLE IMAGE.
They are NOT recall or IoU: none of these sets carry road labels, so
accuracy is not computable here. A 0% cell means the model predicted no
road at all -- which may be a failure, or may be correct if the frame has
no road in it. Only the overlays can tell the two apart.

Every model check so far ran on random noise, which proves the models fit in
VRAM and execute -- and nothing about whether their output means anything. This
runs them on real road images and saves overlays to look at.

Deliberately NOT a benchmark: no metrics, no scores. The question is only
"is this obviously broken?" before we invest in the real evaluation.

    python scripts/survey/smoke_test.py
    python scripts/survey/smoke_test.py --n 8 --model segformer-b2-cityscapes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
OUT = REPO_ROOT / "results" / "smoke_test"
LABELS: dict = {}

# Cityscapes class ids. 0=road, 1=sidewalk. The models are Cityscapes-trained,
# so "road" is the class we care about for Task A.
ROAD_ID, SIDEWALK_ID = 0, 1


def sample_images(n: int) -> list[Path]:
    """A spread across the ground-view sets we already have on disk."""
    pools = [
        sorted((REPO_ROOT / "datasets/road-anomaly-epfl/RoadAnomaly_jpg/frames").glob("*.jpg")),
        sorted((REPO_ROOT / "datasets/smiyc-road-obstacle21/dataset_ObstacleTrack/images").glob("*.webp")),
        sorted((REPO_ROOT / "datasets/smiyc-road-anomaly21/dataset_AnomalyTrack/images").glob("*.jpg")),
    ]
    picked: list[Path] = []
    per = max(1, n // len([p for p in pools if p]))
    for pool in pools:
        if not pool:
            continue
        # Spread across the pool rather than taking the first few, which are
        # alphabetically clustered and would all show the same scene.
        step = max(1, len(pool) // per)
        picked.extend(pool[::step][:per])
    return picked[:n]


ATTR_NAMES = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt",
              "front-view", "side-view", "bird-view", "long-term"]


def load_uavdt_attributes() -> dict[str, dict]:
    """Sequence-level attribute flags. Not mutually exclusive within a group --
    a sequence can carry more than one view flag."""
    out = {}
    for f in (REPO_ROOT / "datasets/uavdt-attributes/M_attr").rglob("*_attr.txt"):
        vals = [int(x) for x in f.read_text().strip().split(",")]
        out[f.stem.replace("_attr", "").strip()] = dict(zip(ATTR_NAMES, vals))
    return out


def sample_uav(per_stratum: int = 2) -> list[tuple[Path, str]]:
    """Frames stratified by altitude x view, daylight only.

    Stratified rather than random: the question is whether viewpoint or altitude
    is what breaks a ground-trained model, and a random draw would drown that in
    the crowded medium-altitude front-view cell.

    A sequence can carry several view flags at once, so the same frame can land
    in more than one stratum. Deduplicate by path and label it with every view
    that applies, otherwise the table double-counts and hides other strata.
    """
    root = REPO_ROOT / "datasets/uavdt-benchmark-m/UAV-benchmark-M"
    if not root.is_dir():
        return []
    attrs = load_uavdt_attributes()
    chosen: dict[Path, str] = {}
    for alt in ["low-alt", "medium-alt", "high-alt"]:
        for view in ["front-view", "side-view", "bird-view"]:
            seqs = sorted(s for s, a in attrs.items()
                          if a[alt] and a[view] and a["daylight"] and (root / s).is_dir())
            for seq in seqs[:per_stratum]:
                frames = sorted((root / seq).glob("*.jpg"))
                if not frames:
                    continue
                f = frames[len(frames) // 2]  # mid-sequence; early frames show take-off
                if f in chosen:
                    continue
                views = "+".join(v[:-5] for v in ["front-view", "side-view", "bird-view"]
                                 if attrs[seq][v])
                chosen[f] = f"{alt[:-4]:<6} {views:<16} {seq}"
    return sorted(chosen.items(), key=lambda kv: kv[1])


def overlay(img: Image.Image, mask: np.ndarray, colour=(255, 0, 0), alpha=0.45) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    tint = np.zeros_like(base)
    tint[..., 0], tint[..., 1], tint[..., 2] = colour
    m = mask[..., None].astype(np.float32) * alpha
    return Image.fromarray((base * (1 - m) + tint * m).astype(np.uint8))


def run_segformer(key: str, images: list[Path]) -> list[dict]:
    import transformers

    path = MODELS / key
    model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        str(path), dtype=torch.float16).to("cuda").eval()
    proc = transformers.SegformerImageProcessor.from_pretrained(str(path))
    id2label = model.config.id2label

    rows = []
    for p in images:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            logits = model(**inputs).logits
        # Logits come back at 1/4 resolution; put them back on the image grid.
        logits = torch.nn.functional.interpolate(
            logits.float(), size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()

        road = (pred == ROAD_ID)
        drivable = road | (pred == SIDEWALK_ID)
        overlay(img, road).save(OUT / f"{key}__{p.stem[:40]}.jpg", quality=88)

        top = np.bincount(pred.ravel(), minlength=len(id2label)).argsort()[::-1][:3]
        rows.append({
            "image": LABELS.get(p, p.stem[:40])[:40],
            # Fraction of the WHOLE IMAGE predicted as road -- not recall.
            # UAVDT has no road labels, so accuracy is not computable here;
            # 0% means "predicted nothing", which could be correct or a failure.
            "road_px_frac": float(road.mean()),
            "drivable_px_frac": float(drivable.mean()),
            "top_classes": [id2label[int(i)] for i in top],
        })
    del model
    torch.cuda.empty_cache()
    return rows


def run_dinov2(key: str, images: list[Path]) -> list[dict]:
    """Do the dense features carry structure, or are they degenerate?

    A collapsed backbone gives near-identical patch vectors everywhere. PCA over
    the patch grid exposes that: real features spread variance across components,
    collapsed ones put nearly all of it in the first.
    """
    import transformers

    path = MODELS / key
    cfg = transformers.AutoConfig.from_pretrained(str(path))
    # DINOv3 overflows fp16 and returns 100% NaN -- silently, with the right
    # shape and no error. bfloat16 has fp32's exponent range, so it is safe at
    # the same memory cost. Never run this family in fp16.
    dtype = torch.bfloat16
    model = transformers.AutoModel.from_pretrained(str(path), dtype=dtype).to("cuda").eval()
    proc = transformers.AutoImageProcessor.from_pretrained(str(path))
    # Drop CLS *and* register tokens; DINOv3 has 4 registers, DINOv2 has none.
    # Counting them as patches would corrupt the patch grid and the prototypes.
    skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)

    rows = []
    for p in images:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            feats = model(**inputs).last_hidden_state[0, skip:].float().cpu().numpy()

        if not np.isfinite(feats).all():
            rows.append({"image": p.stem[:40], "patches": int(feats.shape[0]),
                         "pc1_var": float("nan"), "top10_var": float("nan")})
            print(f"  !! {p.stem[:40]}: non-finite features -- dtype problem")
            continue

        f = feats - feats.mean(0)
        # Singular values squared are proportional to variance per component.
        sv = np.linalg.svd(f, compute_uv=False)
        var = sv**2 / (sv**2).sum()

        n = int(np.sqrt(feats.shape[0]))
        if n * n == feats.shape[0]:
            u, _, vt = np.linalg.svd(f, full_matrices=False)
            pc = (f @ vt[:3].T).reshape(n, n, 3)
            pc = (pc - pc.min((0, 1))) / (np.ptp(pc, axis=(0, 1)) + 1e-8)  # ndarray.ptp gone in numpy 2
            Image.fromarray((pc * 255).astype(np.uint8)).resize(
                img.size, Image.NEAREST).save(OUT / f"{key}__{p.stem[:40]}_pca.jpg", quality=88)

        rows.append({
            "image": p.stem[:40],
            "patches": int(feats.shape[0]),
            "pc1_var": float(var[0]),
            "top10_var": float(var[:10].sum()),
        })
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--uav", action="store_true",
                    help="sample UAVDT stratified by altitude x view instead")
    ap.add_argument("--model", nargs="+",
                    default=["segformer-b2-cityscapes", "dinov2-large"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    labels: dict[Path, str] = {}
    if args.uav:
        pairs = sample_uav()
        images = [p for p, _ in pairs]
        LABELS.update({p: lab for p, lab in pairs})
    else:
        images = sample_images(args.n)
    if not images:
        print("no images found -- are the datasets downloaded?")
        return 1
    print(f"images: {len(images)}\noutput: {OUT.relative_to(REPO_ROOT)}\n")

    for key in args.model:
        if not (MODELS / key).is_dir():
            print(f"[skip] {key}: not downloaded")
            continue
        print(f"--- {key} ---")
        if key.startswith("segformer"):
            rows = run_segformer(key, images)
            print(f"{'IMAGE':<42}{'ROAD-PX%':>10}{'DRIVE-PX%':>11}  TOP CLASSES")
            for r in rows:
                print(f"{r['image']:<42}{r['road_px_frac']:>9.1%}{r['drivable_px_frac']:>11.1%}"
                      f"  {', '.join(r['top_classes'])}")
        elif key.startswith("dinov"):
            rows = run_dinov2(key, images)
            print(f"{'IMAGE':<42}{'PATCHES':>9}{'PC1 var':>9}{'top10 var':>11}")
            for r in rows:
                print(f"{r['image']:<42}{r['patches']:>9}{r['pc1_var']:>9.3f}{r['top10_var']:>11.3f}")
        print()

    print(f"overlays written to {OUT.relative_to(REPO_ROOT)} -- open them and look")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
