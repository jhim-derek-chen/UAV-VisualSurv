"""How much of the self-supervised method's score is decided by input size?

PROWL sees an image as a grid of side/patch cells. A small road obstacle can be
a fraction of ONE cell, and at that point no amount of feature quality helps --
the object and its surroundings share a vector. This sweeps the input size on
the hardest benchmark (B2: objects are 0.15% of the road) and records what each
grid buys, so the report can state the effect instead of asserting it.

"recall" here means the same thing benchmark.py's NOTICE_BAR does: an object's
peak score lands in the top slice of THAT SAME IMAGE's own clean pixels --
self-calibrated per image, no false-alarm budget involved.

    python scripts/resolution_sweep.py            -> results/resolution_sweep.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark as B  # noqa: E402

BENCH = "B2"
SIDES = (224, 448, 672, 896)
DEST = REPO_ROOT / "results" / "resolution_sweep.json"


def measure(key: str, patch: int, items, side: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    m = B.Prowl(key, side=side)
    rng = np.random.default_rng(0)
    hits, aur = [], []
    t0 = time.time()
    for it in items:
        img = Image.open(it.img).convert("RGB")
        gt, valid = it.masks()
        s = m(img)
        if s.shape != valid.shape:
            s = np.array(Image.fromarray(s).resize(valid.shape[::-1], Image.BILINEAR))
        pos = gt & valid
        na = s[valid & ~gt]
        if not pos.any() or na.size == 0:
            continue
        sub = (na if na.size <= B.NEG_SAMPLE
               else na[rng.integers(0, na.size, B.NEG_SAMPLE)]).astype(np.float32)
        tau_img = B.calibrate(sub, B.NOTICE_BAR)
        hits += [p >= tau_img for _, p in B.components(pos, s)]
        aur.append(B.rank_metrics(s[pos], sub)["auroc"])
    dt = (time.time() - t0) / len(items)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m
    torch.cuda.empty_cache()

    return {"side": side, "grid": side // patch,
            "patches": (side // patch) ** 2,
            "recall": float(np.mean(hits)) if hits else None,
            "auroc": float(np.mean(aur)) if aur else None,
            "vram_gb": float(peak), "sec_per_image": float(dt)}


def main() -> int:
    items = B.BENCHMARKS[BENCH]["loader"](B.BENCHMARKS[BENCH]["cap"])
    if not items:
        print(f"{BENCH} not available")
        return 1
    print(f"{BENCH}: {len(items)} images, notice bar {B.NOTICE_BAR:.0%}\n")
    out = {"benchmark": BENCH, "images": len(items), "notice_bar": B.NOTICE_BAR, "runs": {}}
    for key, patch in (("dinov2-large", 14), ("dinov3-vitl16", 16)):
        rows = [measure(key, patch, items, s) for s in SIDES]
        out["runs"][key] = rows
        print(f"--- {key} (patch {patch})")
        print(f"{'side':>6}{'grid':>9}{'VRAM':>8}{'s/img':>8}{'recall':>9}{'AUROC':>8}")
        for r in rows:
            print(f"{r['side']:>6}{f'{r['grid']}x{r['grid']}':>9}{r['vram_gb']:>7.2f}G"
                  f"{r['sec_per_image']:>8.2f}{r['recall']:>8.0%}{r['auroc']:>8.3f}")
        print()
    DEST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {DEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
