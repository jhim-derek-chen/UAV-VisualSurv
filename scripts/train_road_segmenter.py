"""Fine-tune SegFormer-B2 into a drone-view road segmenter (Stage 1).

Why: every zero-shot Stage 1 candidate fails from the air. On AeroScapes
(real drone footage, 5-50 m) the Cityscapes SegFormer-B2 reached road IoU
0.31 and Grounding DINO's road boxes 0.32 (results/segmentation_benchmark.json).
On the synthetic highway set, Grounding DINO's boxes covered anywhere from 2%
to 92% of a frame from the same location and missed 17% of the objects, and
SegFormer labelled a container yard as road while missing most of a
motorway.

What: binary "drivable surface" = AeroScapes road (10) plus car (3), so a
vehicle on the road is inside the mask rather than a hole in it. Starts from
the Cityscapes SegFormer-B2 weights (already on disk), new 2-class head.
AeroScapes train split only; its val split is held out and scored.

Budget: batch 4 at 512 px with fp16 autocast. Peak VRAM is printed; the
project rule is that anything above ~3 GB counts as a failure (Windows spills
into shared memory instead of raising OOM).

    python scripts/train_road_segmenter.py            # ~15 min on an RTX 3050 Ti
    python scripts/train_road_segmenter.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "datasets" / "aeroscapes" / "aeroscapes"
BASE = REPO_ROOT / "models" / "segformer-b2-cityscapes"
OUT = REPO_ROOT / "models" / "segformer-b2-aeroscapes-road"
POSITIVE = (10, 3)  # road, car
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pair(name: str) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(DATA / "JPEGImages" / f"{name}.jpg").convert("RGB")
    seg = np.array(Image.open(DATA / "SegmentationClass" / f"{name}.png"))
    return img, np.isin(seg, POSITIVE).astype(np.uint8)


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a).permute(2, 0, 1)


def augment(img: Image.Image, lab: np.ndarray, crop: int):
    """Random scale (so road width in pixels varies like altitude does),
    random crop, horizontal and vertical flips (nadir views have no up)."""
    s = random.uniform(0.5, 1.2)
    w, h = img.size
    nw, nh = max(crop, int(w * s)), max(crop, int(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    lab = np.array(Image.fromarray(lab).resize((nw, nh), Image.NEAREST))
    x0, y0 = random.randint(0, nw - crop), random.randint(0, nh - crop)
    img = img.crop((x0, y0, x0 + crop, y0 + crop))
    lab = lab[y0:y0 + crop, x0:x0 + crop]
    if random.random() < 0.5:
        img, lab = img.transpose(Image.FLIP_LEFT_RIGHT), lab[:, ::-1]
    if random.random() < 0.5:
        img, lab = img.transpose(Image.FLIP_TOP_BOTTOM), lab[::-1]
    return to_tensor(img), torch.from_numpy(np.ascontiguousarray(lab)).long()


def load_model(path: Path):
    import transformers
    if path == BASE:
        return transformers.SegformerForSemanticSegmentation.from_pretrained(
            str(path), num_labels=2, id2label={0: "other", 1: "drivable"},
            label2id={"other": 0, "drivable": 1}, ignore_mismatched_sizes=True)
    return transformers.SegformerForSemanticSegmentation.from_pretrained(str(path))


@torch.inference_mode()
def predict(model, img: Image.Image, long_side: int = 1280) -> np.ndarray:
    """Drivable-surface probability at the input's own resolution. The image
    is resized so its long side is `long_side`, the AeroScapes frame width
    the model was trained around."""
    w, h = img.size
    sc = long_side / max(w, h)
    small = img.resize((max(32, round(w * sc / 32) * 32), max(32, round(h * sc / 32) * 32)),
                       Image.BILINEAR)
    x = to_tensor(small)[None].cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(pixel_values=x).logits
    prob = logits.float().softmax(1)[:, 1:2]
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return prob[0, 0].cpu().numpy()


def evaluate(model, names: list[str]) -> dict:
    model.eval()
    inter = union = tp = fp = fn = 0
    for n in names:
        img, lab = load_pair(n)
        pred = predict(model, img) > 0.5
        gt = lab.astype(bool)
        inter += int((pred & gt).sum())
        union += int((pred | gt).sum())
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    return {"iou": inter / union, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "images": len(names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    train = (DATA / "ImageSets" / "trn.txt").read_text().split()
    val = (DATA / "ImageSets" / "val.txt").read_text().split()

    if args.eval_only:
        model = load_model(OUT).cuda()
        print(json.dumps(evaluate(model, val), indent=2))
        return 0

    model = load_model(BASE).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        batch = [augment(*load_pair(random.choice(train)), args.crop) for _ in range(args.batch)]
        x = torch.stack([b[0] for b in batch]).cuda()
        y = torch.stack([b[1] for b in batch]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(pixel_values=x).logits
        logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear",
                               align_corners=False)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 100 == 0 or it == 1:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"iter {it:5d}  loss {loss.item():.4f}  peak {peak:.2f} GB  "
                  f"{(time.time() - t0) / it:.2f} s/iter", flush=True)
            if peak > 3.0:
                raise SystemExit("peak VRAM above 3 GB -- lower --batch or --crop")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    metrics = evaluate(model, val)
    metrics.update({"iters": args.iters, "batch": args.batch, "crop": args.crop,
                    "positive_classes": "aeroscapes road + car",
                    "train_minutes": round((time.time() - t0) / 60, 1)})
    (OUT / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
