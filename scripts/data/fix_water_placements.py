"""One-off patch: regenerate the synthetic-dataset samples whose background
was M0208 (sea-crossing bridge) and whose object landed on open water or a
breakwater instead of the road surface -- found by a manual spot check, see
highway_backgrounds.py's docstring for the root cause. M0208 has already been
removed from the background pool, so this only needs to redraw a background
for these image_ids and re-run the existing render/composite/write pipeline
that build_synthetic_dataset.py already contains -- no new logic.

    python scripts/data/fix_water_placements.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

import build_synthetic_dataset as bsd
import highway_backgrounds

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
OUT = bsd.OUT

# 6 clearly floating on open water + 2 on a breakwater (not road either).
# image_ids 77, 115, 248, 271 also used M0208 but landed correctly on the
# bridge deck itself and are left alone.
#
# Second pass (this run): of the first 8 regenerated, id 12 (mattress) landed
# off the shoulder below an overpass and id 100 (tire) landed in trees next
# to a house -- both still wrong, redrawn again here. The other 6 (0, 37,
# 90, 212, 231, 234) were checked against their full frame and are correctly
# on the road surface, so they are left alone this time.
BAD_IDS = [12, 100]


def main() -> int:
    bsd.check_category_lists_agree()
    random.seed(4242)  # distinct from both the original run (42) and pass 1 (999)

    anns = json.loads((OUT / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    debris = bsd.load_debris_manifest()
    backgrounds = highway_backgrounds.sample()
    assert not any("M0208" in str(p) for p in backgrounds), "M0208 still in pool"

    from segment_benchmark import SegformerRoad
    segmenter = SegformerRoad("segformer-b2-cityscapes")

    ann_by_img = {a["image_id"]: a for a in anns["annotations"]}
    sample_by_img = {s["image_id"]: s for s in manifest["samples"]}
    cat_ids = {c["name"]: c["id"] for c in anns["categories"]}

    for img_id in BAD_IDS:
        category = ann_by_img[img_id]["category_id"]
        category = next(n for n, i in cat_ids.items() if i == category)
        print(f"[redo] id={img_id} category={category}")

        for attempt in range(20):
            bg_path = random.choice(backgrounds)
            pick = random.choice(debris[category])
            model_dir = bsd.MODELS_ROOT / category / pick["uid"]
            model_path = next(model_dir.glob("*.gltf"), None)
            if model_path is None:
                continue
            try:
                bg = Image.open(bg_path).convert("RGB")
            except Exception:
                continue

            mask_pred = bsd.road_mask_for(bg, segmenter)
            cx, cy = bsd.pick_placement(mask_pred)
            target_w = max(12, round(bg.width * bsd.WIDTH_FRAC[category]))
            try:
                render = bsd.render_object(model_path, category, seed=10_000 + img_id, res=640)
                comp, mask_arr, bbox = bsd.composite(bg, render, cx, cy, target_w)
            except Exception as exc:
                print(f"  retry after: {exc}")
                continue
            break
        else:
            raise RuntimeError(f"could not regenerate id {img_id} after 20 attempts")

        img_name = f"{img_id:05d}.jpg"
        comp.save(OUT / "images" / img_name, quality=92)
        Image.fromarray(mask_arr).save(OUT / "masks" / f"{img_id:05d}.png")

        for im in anns["images"]:
            if im["id"] == img_id:
                im["width"], im["height"] = comp.width, comp.height
        ann_by_img[img_id]["bbox"] = list(bbox)
        ann_by_img[img_id]["area"] = int((mask_arr > 0).sum())

        s = sample_by_img[img_id]
        s["model_uid"] = pick["uid"]
        s["model_credit"] = pick["credit"]
        s["background_source"] = str(bg_path.relative_to(REPO_ROOT))
        s["placement_xy"] = [cx, cy]

    del segmenter

    (OUT / "annotations.json").write_text(json.dumps(anns, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nregenerated {len(BAD_IDS)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
