"""Download model checkpoints from their official Hugging Face repositories.

Weights go to models/<key>/. The resolved commit SHA is recorded in
models/manifest.json so runs pin exact weights rather than a moving branch.

Variant choices are driven by the 4 GB VRAM budget of the target GPU
(RTX 3050 Ti Laptop). `vram_gb` estimates peak inference VRAM at our intended
tile size, not the weight size.

    python scripts/download_models.py --list
    python scripts/download_models.py                    # everything
    python scripts/download_models.py --key dinov2-large
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.json"

# Skip formats we never load; halves the transfer on repos shipping duplicates.
IGNORE = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.onnx_data"]


@dataclass(frozen=True)
class Model:
    key: str
    repo_id: str
    group: str
    licence: str
    vram_gb: float
    why: str
    gated: bool = False


MODELS = [
    # --- Task A: road / drivable-surface segmentation ---
    Model(
        "segformer-b0-cityscapes",
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "segmentation",
        "NVIDIA Source Code License (non-commercial)",
        0.5,
        "Fast smoke test for the tiling core before spending time on heavier checkpoints.",
    ),
    Model(
        "segformer-b2-cityscapes",
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "segmentation",
        "NVIDIA Source Code License (non-commercial)",
        1.2,
        "Primary road-segmentation baseline. B2 not B5: B5 peaks near 3.5 GB at "
        "1024x1024, leaving no headroom on a 4 GB card.",
    ),
    Model(
        "mask2former-swin-tiny-cityscapes",
        "facebook/mask2former-swin-tiny-cityscapes-semantic",
        "segmentation",
        "CC-BY-NC-4.0",
        2.0,
        "Second segmentation architecture. Swin-T not Swin-L, which does not fit "
        "4 GB at native resolution. Also previews how Mask2Anomaly and RbA will behave.",
    ),
    # --- Vision foundation-model baselines ---
    Model(
        "dinov2-large",
        "facebook/dinov2-large",
        "foundation",
        "Apache-2.0",
        1.0,
        "Dense-feature backbone for the PROWL-style prototype/OOD test. PROWL was "
        "built on DINOv2, so this is the faithful reproduction target.",
    ),
    Model(
        "dinov3-vitl16",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "foundation",
        "DINOv3 License (gated)",
        1.4,
        "Newer backbone generation. Tests whether better self-supervised features "
        "close the ground-to-aerial viewpoint gap. ViT-L/16 fp16 fits 4 GB; ViT-H+ does not.",
        gated=True,
    ),
    Model(
        "grounding-dino-tiny",
        "IDEA-Research/grounding-dino-tiny",
        "foundation",
        "Apache-2.0",
        1.5,
        "Open-vocabulary detection baseline for text prompts such as 'debris on road'.",
    ),
    Model(
        "sam2.1-hiera-base-plus",
        "facebook/sam2.1-hiera-base-plus",
        "foundation",
        "Apache-2.0",
        2.2,
        "Class-agnostic mask refinement after a candidate region is found, and the "
        "second half of the Grounding DINO -> SAM baseline. base-plus not large, to "
        "stay inside 4 GB alongside the detector.",
    ),
    Model(
        "owlv2-base-ensemble",
        "google/owlv2-base-patch16-ensemble",
        "foundation",
        "Apache-2.0",
        1.5,
        "Class-agnostic objectness head: scores 'is this box an object' with no "
        "class list, which is exactly Stage 2's job. Base not large, for 4 GB.",
    ),
    # --- Objective 2: identify each detection and assess its risk ---
    Model(
        "qwen3.5-2b",
        "Qwen/Qwen3.5-2B",
        "vlm",
        "Apache-2.0",
        2.5,
        "Vision-language model for naming each detected object and explaining its "
        "risk. 2B in 8-bit fits the 4 GB card with room for image tokens.",
    ),
    Model(
        "qwen3.5-4b",
        "Qwen/Qwen3.5-4B",
        "vlm",
        "Apache-2.0",
        3.0,
        "Larger sibling of qwen3.5-2b, 4-bit, to test whether size buys "
        "identification accuracy on small aerial crops.",
    ),
]

BY_KEY = {m.key: m for m in MODELS}
GROUPS = sorted({m.group for m in MODELS})


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    cached = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return cached.read_text(encoding="utf-8").strip() if cached.is_file() else None


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", nargs="+", help="download only these keys")
    ap.add_argument("--group", nargs="+", choices=GROUPS, help="download only these groups")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if recorded")
    args = ap.parse_args()

    if args.list:
        w = max(len(m.key) for m in MODELS)
        for m in MODELS:
            print(f"{m.key.ljust(w)}  {m.group:<12} {m.vram_gb:>4.1f}G  {m.repo_id}"
                  f"{' [gated]' if m.gated else ''}")
        return 0

    if args.key:
        unknown = [k for k in args.key if k not in BY_KEY]
        if unknown:
            sys.exit(f"unknown key(s): {', '.join(unknown)}")
        targets = [BY_KEY[k] for k in args.key]
    elif args.group:
        targets = [m for m in MODELS if m.group in args.group]
    else:
        targets = list(MODELS)

    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.exit("huggingface_hub missing -- pip install -r requirements.txt")

    tok = token()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {"models": {}}
    ok, skipped = [], []

    print(f"target : {MODELS_DIR}")
    print(f"token  : {'present' if tok else 'absent (gated repos will be skipped)'}\n")

    for m in targets:
        dest = MODELS_DIR / m.key
        if m.key in manifest["models"] and dest.is_dir() and not args.force:
            print(f"[have] {m.key} @ {manifest['models'][m.key]['revision'][:12]}")
            ok.append(m.key)
            continue

        if m.gated and not tok:
            reason = f"gated; accept the licence at https://huggingface.co/{m.repo_id} and set HF_TOKEN"
            print(f"[skip] {m.key} -- {reason}")
            skipped.append((m.key, reason))
            continue

        print(f"[get ] {m.key} <- {m.repo_id}")
        try:
            snapshot_download(repo_id=m.repo_id, local_dir=str(dest), token=tok, ignore_patterns=IGNORE)
        except GatedRepoError:
            # A valid token is not enough: the licence must also be granted.
            reason = f"licence not granted for {m.repo_id} on this account"
            print(f"[skip] {m.key} -- {reason}")
            skipped.append((m.key, reason))
            continue
        except RepositoryNotFoundError:
            reason = f"not found: {m.repo_id} (renamed or withdrawn upstream?)"
            print(f"[FAIL] {m.key} -- {reason}")
            skipped.append((m.key, reason))
            continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {m.key} -- {reason}")
            skipped.append((m.key, reason))
            continue

        # Pin the exact commit on disk, not the branch tip.
        try:
            revision = HfApi().model_info(m.repo_id, token=tok).sha
        except Exception:
            revision = "unknown"

        size = dir_size(dest)
        manifest["models"][m.key] = {
            "repo_id": m.repo_id,
            "revision": revision,
            "group": m.group,
            "licence": m.licence,
            "size_bytes": size,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"       ok {size / 1024**2:.0f} MB @ {revision[:12]}")
        ok.append(m.key)

    print(f"\ndownloaded/present: {len(ok)}")
    for key, reason in skipped:
        print(f"  skipped {key}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
