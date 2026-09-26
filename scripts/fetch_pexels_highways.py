"""Download curated Pexels drone-highway clips and extract frames as extra
background diversity for the synthetic debris pipeline.

UAVDT's 7 genuine highway sequences (scripts/highway_backgrounds.py) and a
VisDrone slice add up to only 8 physical locations, all in China. These clips
add different countries, road types (mountain, coastal, night, interchange)
and, unlike UAVDT/VisDrone, are shot by hobbyists rather than a research rig,
which is closer to what a deployed inspection drone's footage would actually
look like.

License: Pexels grants "an irrevocable, worldwide, perpetual, non-exclusive
and royalty-free right to download, use, copy, modify or adapt the Content
for commercial or non-commercial purposes" (checked against pexels.com/license
directly, not assumed) -- no attribution required, but credited anyway in
manifest.json for traceability. The only restriction is not redistributing
the clips themselves as stock content, which extracting frames into a
synthetic-debris training set does not do.

Video IDs were found by searching pexels.com for drone/highway footage and
picking ones that are actually highway (not an intersection or a parking
lot) by reading each clip's own description -- not auto-selected.

    python scripts/fetch_pexels_highways.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "pexels-highway"
FRAMES_PER_CLIP = 15  # evenly spaced; clips are ~15-40s, so this avoids near-duplicates

CLIPS = {
    "12571926": "Highway with traffic, intersection + greenery",
    "19851623": "Aerial view of a highway with cars driving on it",
    "5382494": "Cars driving on highway, rural landscape",
    "4062948": "Highway at night, city lights",
    "5598972": "Winding mountain highway, autumn foliage",
    "32831945": "Detroit highway skyline",
    "17671627": "Coastal city highway",
    "14191353": "Dhaka-Mawa-Bhanga Expressway, Bangladesh",
    "7277385": "Downtown expressway",
    "8177425": "Highway through forest",
    # Added 2026-09-24 for the open-highway-only background pool (see
    # highway_backgrounds.OPEN_HIGHWAY): both are steep, close-range views of
    # an open motorway segment, no toll plaza, no horizon.
    "12306893": "Top-down multi-lane motorway with sign gantry (portrait)",
    "8742752": "Low-altitude motorway, close range, no sky in frame",
}


def download_clip(vid: str, dest: Path) -> bool:
    r = requests.get(f"https://www.pexels.com/download/video/{vid}/",
                      headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    if r.status_code != 200 or not r.content:
        return False
    dest.write_bytes(r.content)
    return True


def extract_frames(video_path: Path, out_dir: Path, n: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    idxs = [int(i * total / n) for i in range(n)]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
    cap.release()
    return saved


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for vid, desc in CLIPS.items():
        dest_dir = OUT_DIR / vid
        if dest_dir.is_dir() and any(dest_dir.glob("frame_*.jpg")):
            n = len(list(dest_dir.glob("frame_*.jpg")))
            print(f"[have] {vid}  ({n} frames)  {desc}")
        else:
            tmp = OUT_DIR / f"{vid}.mp4"
            print(f"[get ] {vid}  {desc}", flush=True)
            if not download_clip(vid, tmp):
                print(f"[FAIL] {vid}: download failed")
                continue
            n = extract_frames(tmp, dest_dir, FRAMES_PER_CLIP)
            tmp.unlink(missing_ok=True)
            print(f"       {n} frames extracted")
            time.sleep(0.5)
        manifest[vid] = {
            "description": desc,
            "source_url": f"https://www.pexels.com/video/{vid}/",
            "license": "Pexels License (free for commercial/non-commercial use, "
                       "no attribution required) -- https://www.pexels.com/license/",
        }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {(OUT_DIR / 'manifest.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
