"""Verify every downloaded checkpoint loads and runs, and measure real VRAM.

Downloading a checkpoint proves nothing about whether it fits. This measures
peak VRAM with torch.cuda.max_memory_allocated() so registry estimates can be
replaced with facts.

It also answers the question the pipeline design depends on: at what resolution
does each model stop fitting? That decides whether tiling is mandatory, and at
what tile size -- rather than assuming it is.

    python scripts/setup/check_models.py
    python scripts/setup/check_models.py --sizes 1024x1024 4096x2160
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# key -> transformers auto-class
SPECS = [
    ("segformer-b0-cityscapes", "SegformerForSemanticSegmentation"),
    ("segformer-b2-cityscapes", "SegformerForSemanticSegmentation"),
    ("mask2former-swin-tiny-cityscapes", "Mask2FormerForUniversalSegmentation"),
    ("dinov2-large", "AutoModel"),
    ("dinov3-vitl16", "AutoModel"),
    ("grounding-dino-tiny", "AutoModelForZeroShotObjectDetection"),
    ("sam2.1-hiera-base-plus", "Sam2Model"),
]


def reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_once(model, cls_name: str, path: Path, h: int, w: int) -> float:
    """One forward pass. Returns peak GB, or raises."""
    import transformers

    reset()
    x = torch.randn(1, 3, h, w, dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        if cls_name == "AutoModelForZeroShotObjectDetection":
            # Grounding DINO takes a text prompt too. The text tower stays fp32
            # in this checkpoint, so the whole model must run fp32 to match.
            tok = transformers.AutoTokenizer.from_pretrained(str(path))
            ids = tok(["object on road."], return_tensors="pt").to("cuda")
            model(pixel_values=x.float(), **ids)
        elif cls_name == "Sam2Model":
            model(pixel_values=x)
        else:
            model(pixel_values=x)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del x
    reset()
    return round(peak, 2)


def probe(key: str, cls_name: str, sizes: list[tuple[int, int]]) -> dict:
    import transformers

    path = MODELS_DIR / key
    out: dict = {"key": key, "class": cls_name, "sizes": {}}

    if not path.is_dir():
        out["load"] = "MISSING"
        return out

    reset()
    # Grounding DINO's text tower is fp32; loading the whole thing fp16 breaks it.
    dtype = torch.float32 if cls_name == "AutoModelForZeroShotObjectDetection" else torch.float16
    try:
        model = getattr(transformers, cls_name).from_pretrained(str(path), dtype=dtype).to("cuda").eval()
        out["load"] = "OK"
        out["dtype"] = str(dtype).replace("torch.", "")
    except Exception as exc:
        out["load"] = f"FAILED: {type(exc).__name__}: {str(exc)[:100]}"
        return out

    for h, w in sizes:
        label = f"{w}x{h}"
        try:
            out["sizes"][label] = run_once(model, cls_name, path, h, w)
        except torch.cuda.OutOfMemoryError:
            out["sizes"][label] = "OOM"
            reset()
        except Exception as exc:
            out["sizes"][label] = f"{type(exc).__name__}: {str(exc)[:70]}"
            reset()

    del model
    reset()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["1024x1024", "2048x1080", "4096x2160"],
                    help="WxH resolutions to probe")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable -- nothing to measure")
        return 1

    sizes = []
    for s in args.sizes:
        w, h = s.lower().split("x")
        sizes.append((int(h), int(w)))

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu   : {name}  ({total:.1f} GB)")
    print(f"torch : {torch.__version__}\n")

    rows = [probe(k, c, sizes) for k, c in SPECS]

    labels = [f"{w}x{h}" for h, w in sizes]
    header = f"{'MODEL':<34} {'DTYPE':<8}" + "".join(f"{l:>13}" for l in labels)
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["load"] != "OK":
            print(f"{r['key']:<34} {r['load']}")
            continue
        cells = "".join(f"{(f'{v:.2f} GB' if isinstance(v, float) else str(v)[:12]):>13}"
                        for v in (r["sizes"][l] for l in labels))
        print(f"{r['key']:<34} {r.get('dtype', ''):<8}{cells}")

    (REPO_ROOT / "results").mkdir(exist_ok=True)
    dest = REPO_ROOT / "results" / "vram_check.json"
    dest.write_text(json.dumps({"gpu": name, "total_gb": round(total, 1), "results": rows},
                               indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved: {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
