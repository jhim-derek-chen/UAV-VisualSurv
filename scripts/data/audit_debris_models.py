"""One-off QA sweep: render every downloaded 3D debris model at several seeds
and flag any whose rendered footprint area varies wildly across seeds.

Why this is the right check: with only a Z-axis (yaw) spin and a near-nadir
camera (0-25 degree tilt, see render_debris.py build_camera), a genuinely
flat-lying object's silhouette area should stay roughly constant regardless
of which way it's facing -- unlike a *standing* object, whose silhouette
collapses to a thin sliver from most yaw angles and balloons from others.
A big area swing across seeds is therefore a strong, cheap signal for the
"merged multi-state mesh renders as a standing sliver" bug already confirmed
for suitcase/9490e779... (see project notes), without eyeballing every model
by hand.

    python scripts/data/audit_debris_models.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER = REPO_ROOT / "tools" / "blender-5.2.2-windows-x64" / "blender.exe"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "data" / "render_debris.py"
MODELS_ROOT = REPO_ROOT / "datasets" / "debris-3d-models"
TMP = REPO_ROOT / ".cache" / "tmp" / "audit"
SEEDS = [0, 1, 2, 3, 4, 5]
RES = 300


def render(model_path: Path, category: str, seed: int) -> np.ndarray | None:
    out = TMP / f"{category}_{seed}.png"
    cmd = [str(BLENDER), "--background", "--python", str(RENDER_SCRIPT), "--",
           "--model", str(model_path), "--category", category,
           "--out", str(out), "--seed", str(seed), "--res", str(RES)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not out.is_file():
        return None
    arr = np.array(Image.open(out).convert("RGBA"))
    return arr[..., 3]


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODELS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    flagged = []
    for category, picks in manifest.items():
        for pick in picks:
            uid = pick["uid"]
            model_dir = MODELS_ROOT / category / uid
            model_path = next(model_dir.glob("*.gltf"), None)
            if model_path is None:
                print(f"[skip] {category}/{uid}: no gltf")
                continue
            areas = {}
            for seed in SEEDS:
                alpha = render(model_path, category, seed)
                if alpha is None:
                    print(f"[FAIL] {category}/{uid} seed={seed}: render produced no file")
                    continue
                areas[seed] = int((alpha > 10).sum())
            if len(areas) < 2:
                print(f"[warn] {category}/{uid}: too few successful renders to judge")
                continue
            lo, hi = min(areas.values()), max(areas.values())
            ratio = hi / max(lo, 1)
            flag = " <-- FLAG" if ratio > 3.0 else ""
            print(f"{category:14s} {uid[:8]}  {pick['name'][:35]:35s}  "
                  f"area {lo:6d}-{hi:6d}  ratio {ratio:5.1f}x{flag}")
            if ratio > 3.0:
                flagged.append((category, uid, pick["name"], ratio, areas))

    print("\n=== flagged models (area ratio > 3x across seeds) ===")
    for category, uid, name, ratio, areas in flagged:
        print(f"{category}/{uid}  {name}  ratio={ratio:.1f}x  areas={areas}")
    if not flagged:
        print("none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
