"""Real, clean UAV/drone highway frames for synthetic debris compositing,
pooled from three sources.

1. UAVDT (datasets/uavdt-benchmark-m). Its own attribute files record only
   weather / altitude / camera-view / duration per sequence -- nothing
   distinguishing a genuine highway from the dense urban intersections that
   make up most of the dataset. No metadata shortcut, so all 50
   UAV-benchmark-M sequences were visually triaged (one representative frame
   each, then every remaining sub-sequence in the promising groups): most
   are urban intersections or night footage. Seven are real, clean, daylight
   highway:

     key     frames  what it shows
     M0201     1076  divided multi-lane highway
     M0209     1576  toll plaza approach
     M0210      583  toll station, daylight
     M0301      325  divided highway, coastal tree-lined
     M0601      372  toll road, industrial surroundings
     M0602      480  expressway approach to a toll station

   An eighth, M0208 (sea-crossing bridge / causeway, 265 frames), was tried
   and then dropped: most of its frame is open sea, and Stage 1's road
   segmenter -- never trained on marine scenes -- mislabelled large patches
   of water as "road" at a high rate (6/12 synthetic placements landed on
   open water in a spot check, see datasets/synthetic-highway-debris commit
   notes). Excluded rather than patched, since the bridge deck itself is a
   small fraction of the frame either way.

2. VisDrone2019-DET val (datasets/visdrone-det-val, via the Voxel51 FiftyOne
   mirror on HuggingFace). Its scene_type label only has three coarse values
   ("road" / "intersection" / "sporting event"); of the two "road"-labelled
   video sequences, one turned out to be a normal shopping street once
   viewed, the other a genuine highway interchange -- checked by eye, not
   trusted from the label alone.

3. Pexels drone-highway clips (datasets/pexels-highway, via
   scripts/fetch_pexels_highways.py). Ten hand-picked clips (not a random
   sample -- each one's own description was read first) covering different
   countries and conditions (Bangladesh, US, European mountains, coastal,
   night) that neither research dataset provides, since UAVDT and VisDrone
   are both single-country research captures. Licensed for free commercial
   and non-commercial use; see that script's docstring. Nine of the ten are
   used -- see PEXELS_EXCLUDED_CLIPS below for the one dropped and why.

Video frames are near-duplicates of their neighbours, so `sample()` strides
through UAVDT and VisDrone rather than taking every frame; Pexels frames are
already pre-subsampled to 15 per clip during extraction.

    python scripts/highway_backgrounds.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UAVDT_ROOT = REPO_ROOT / "datasets" / "uavdt-benchmark-m" / "UAV-benchmark-M"
VISDRONE_ROOT = REPO_ROOT / "datasets" / "visdrone-det-val"
PEXELS_ROOT = REPO_ROOT / "datasets" / "pexels-highway"

HIGHWAY_SEQUENCES = {
    "M0201": "divided multi-lane highway",
    "M0209": "toll plaza approach",
    "M0210": "toll station, daylight",
    "M0301": "divided highway, coastal tree-lined",
    "M0601": "toll road, industrial surroundings",
    "M0602": "expressway approach to a toll station",
}
# M0208 (sea-crossing bridge) was dropped -- see module docstring: mostly
# open water, which the road segmenter mislabels at a high rate.

VISDRONE_SEQUENCES = {
    "uav0000268_05773_v": "highway interchange / on-ramp, elevated expressway",
}


def _uavdt_frames(stride: int) -> list[Path]:
    out = []
    for key in HIGHWAY_SEQUENCES:
        d = UAVDT_ROOT / key
        out += sorted(d.glob("img*.jpg"))[::stride]
    return out


def _visdrone_frames(stride: int) -> list[Path]:
    f = VISDRONE_ROOT / "samples.json"
    if not f.is_file():
        return []
    samples = json.loads(f.read_text(encoding="utf-8"))["samples"]
    out = []
    for key in VISDRONE_SEQUENCES:
        paths = sorted(VISDRONE_ROOT / s["filepath"] for s in samples
                       if s["scene_id"] == key)
        out += paths[::stride]
    return out


# 17671627 ("Coastal city highway", Chicago Lake Shore Drive) dropped: in
# nearly every frame the road segmenter prefers the wide lakefront
# pedestrian/bike trail running alongside the real highway over the highway
# itself -- a spot check found 9/11 synthetic placements landed on that
# trail, not a driving lane (see datasets/synthetic-highway-debris commit
# notes). Same failure mode as M0208 (segmenter drawn to a large uniform
# paved area that isn't the road), excluded rather than patched for the
# same reason.
PEXELS_EXCLUDED_CLIPS = {"17671627"}


def _pexels_frames() -> list[Path]:
    if not PEXELS_ROOT.is_dir():
        return []
    return sorted(p for p in PEXELS_ROOT.glob("*/frame_*.jpg")
                  if p.parent.name not in PEXELS_EXCLUDED_CLIPS)


# Scope rule (project spec, 2026-09-24): open highway segments only. Oblique
# views, distant shots and toll stations are out. Every location was triaged
# by eye on three frames each; only these four pass.
OPEN_HIGHWAY = {
    "pexels/12571926": "top-down motorway with an overpass",
    "pexels/19851623": "top-down divided motorway, static camera",
    "pexels/12306893": "top-down multi-lane motorway, sign gantry",
    "pexels/8742752": "low-altitude motorway, close range",
}

# Why each other location fails the rule. Kept in the pool code (not just
# deleted) so the triage can be re-checked.
EXCLUDED = {
    "uavdt/M0201": "oblique, horizon in frame, far road",
    "uavdt/M0209": "toll plaza",
    "uavdt/M0210": "toll station",
    "uavdt/M0301": "oblique, horizon in frame, far road",
    "uavdt/M0601": "toll station",
    "uavdt/M0602": "toll station",
    "visdrone/uav0000268_05773_v": "urban arterial with junction, oblique",
    "pexels/14191353": "oblique, horizon in frame",
    "pexels/32831945": "distant oblique skyline shot",
    "pexels/4062948": "night, distant oblique",
    "pexels/5382494": "urban street, not a highway",
    "pexels/5598972": "distant oblique mountain shot",
    "pexels/7277385": "junction and parking lot, not a highway",
    "pexels/8177425": "oblique, sky in frame",
}

# Where debris may be placed, as a (top, bottom) fraction of image height.
# Only for static-camera clips whose road mask also covers non-highway
# pavement: 19851623 has large parking lots above and below the motorway,
# which the Cityscapes road segmenter labels as road.
PLACEMENT_BAND = {
    "pexels/19851623": (0.415, 0.61),
}


# Areas excluded from scoring, as (top, bottom) fractions of image height,
# for static-camera clips only. Vehicles parked there are not on the
# highway, so finding one is neither a hit nor a false alarm.
IGNORE_BANDS = {
    # the two parking lots and the service road beside each
    "pexels/19851623": [(0.0, 0.35), (0.64, 1.0)],
}


def sample(uavdt_stride: int = 20, visdrone_stride: int = 15) -> list[Path]:
    """The background pool: every frame of an OPEN_HIGHWAY location.
    UAVDT and VisDrone are still enumerated so the triage can be re-run,
    but none of their locations pass the open-highway rule."""
    frames = (_uavdt_frames(uavdt_stride) + _visdrone_frames(visdrone_stride)
              + _pexels_frames())
    return [f for f in frames if location_key(f) in OPEN_HIGHWAY]


def location_key(path: Path) -> str:
    """Groups a background frame by physical shoot location / camera setup
    (one UAVDT sequence, the one VisDrone sequence, or one Pexels clip) --
    frames from the same location share a camera, lens and altitude, so a
    size calibration computed once per location (see
    scripts/calibrate_vehicle_scale.py) is valid for every frame in it."""
    parts = Path(path).parts
    if "uavdt-benchmark-m" in parts:
        return f"uavdt/{Path(path).parent.name}"
    if "visdrone-det-val" in parts:
        return "visdrone/uav0000268_05773_v"
    if "pexels-highway" in parts:
        return f"pexels/{Path(path).parent.name}"
    return f"other/{Path(path).parent.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--uavdt-stride", type=int, default=20)
    ap.add_argument("--visdrone-stride", type=int, default=15)
    args = ap.parse_args()

    if args.list:
        print("--- UAVDT ---")
        for key, note in HIGHWAY_SEQUENCES.items():
            d = UAVDT_ROOT / key
            n = len(list(d.glob("img*.jpg"))) if d.is_dir() else 0
            print(f"{key}  {n:>5} frames  {note}")
        print("--- VisDrone ---")
        for key, note in VISDRONE_SEQUENCES.items():
            print(f"{key}  {note}")
        print("--- Pexels ---")
        clips = sorted({p.parent.name for p in _pexels_frames()})
        print(f"{len(clips)} clips, {len(_pexels_frames())} frames")
        return 0

    frames = sample(args.uavdt_stride, args.visdrone_stride)
    print(f"{len(frames)} background frames in the open-highway pool")
    for loc, note in OPEN_HIGHWAY.items():
        n = sum(1 for f in frames if location_key(f) == loc)
        print(f"  {loc:20s} {n:3d}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
