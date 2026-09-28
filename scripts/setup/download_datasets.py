"""Download benchmark datasets from their official sources.

Only datasets with a scriptable official source live here. UAVid and UAVDT
need a registration form or a Google Drive flow and are handled separately --
see docs/datasets.md.

Downloads to datasets/<key>/, verifies size, extracts, and records provenance
in datasets/manifest.json (url, sha256, size, licence, date).

    python scripts/setup/download_datasets.py --list
    python scripts/setup/download_datasets.py --key smiyc-road-obstacle21
    python scripts/setup/download_datasets.py --group control
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "datasets"
MANIFEST = DATA_DIR / "manifest.json"


@dataclass(frozen=True)
class Dataset:
    key: str
    url: str
    group: str
    mb: float
    licence: str
    paper: str
    why: str


DATASETS = [
    # Ground-view control. Not just a comparison point: these are the sets
    # Mask2Anomaly, RbA and PROWL publish on, so reproducing their numbers here
    # validates our pipeline before any UAV result is trusted.
    Dataset(
        "smiyc-road-anomaly21",
        "https://zenodo.org/record/5270237/files/dataset_AnomalyTrack.zip?download=1",
        "control",
        50.2,
        "see dataset LICENSE; research use",
        "https://arxiv.org/abs/2104.14812",
        "Anomalous objects on or near the road, ground view. 262 GT components.",
    ),
    Dataset(
        "smiyc-road-obstacle21",
        "https://zenodo.org/record/5281633/files/dataset_ObstacleTrack.zip?download=1",
        "control",
        198.0,
        "see dataset LICENSE; research use",
        "https://arxiv.org/abs/2104.14812",
        "Obstacles placed on the road ahead, ground view. 388 GT components. "
        "The closest published analogue to our task, minus the UAV viewpoint.",
    ),
    # Ground-view control with PUBLIC labels. SMIYC withholds most of its
    # ground truth (only 40 of 552 images are labelled), so it cannot carry a
    # quantitative pipeline check on its own -- this can.
    Dataset(
        "road-anomaly-epfl",
        "https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly_jpg.zip",
        "control",
        23.7,
        "research use; cite Lis et al. ICCV 2019",
        "https://arxiv.org/abs/1904.07595",
        "60 real road-anomaly images with public labels. Widely reported by "
        "Mask2Anomaly, RbA and PROWL, so it is the set on which our pipeline "
        "can be checked against published numbers before any UAV result.",
    ),
    # UAV viewpoint WITH annotated obstacles -- the combination the benchmark is
    # actually about. Rail corridor rather than highway, and flown at only
    # 10.5 m, but the task formulation matches: linear paved corridor plus
    # arbitrary foreign objects, annotated.
    Dataset(
        "uav-rsod-detection",
        "https://zenodo.org/api/records/12606374/files/V2%20UAV-RSOD_Dataset%20for%20Obstacle%20Detection.zip/content",
        "uav-anomaly",
        816.5,
        "CC BY 4.0",
        "https://www.nature.com/articles/s41597-024-03952-3",
        "2002 UAV images, 1920x1080. Obstacles: boulder, barrel, branch, jerry "
        "can, iron rod, person. VOC + YOLO boxes. First set that yields a real "
        "aerial-view recall number.",
    ),
    Dataset(
        "uav-rsod-segmentation",
        "https://zenodo.org/api/records/12606374/files/V1%20UAV-RSOD_Dataset%20for%20Segmentation.zip/content",
        "uav-anomaly",
        771.9,
        "CC BY 4.0",
        "https://www.nature.com/articles/s41597-024-03952-3",
        "Same source, pixel masks instead of boxes -- needed for anomaly "
        "segmentation metrics (AUPR, IoU) rather than detection only.",
    ),
    # UAV road segmentation WITH a blocked-road class -- the only public set
    # that annotates "road obstructed by obstacles" from the air.
    # Train (17.8 GB) is deliberately skipped: the first experiment is
    # zero-shot, so nothing is trained and val+test are the evaluation data.
    Dataset(
        "rescuenet-val",
        "https://ndownloader.figshare.com/files/40582916",
        "uav-anomaly",
        2263.9,
        "CC BY-NC-ND 4.0",
        "https://www.nature.com/articles/s41597-023-02799-4",
        "Post-hurricane UAV imagery, pixel labels. Classes include Road-Clear "
        "and Road-Blocked -- normal road and obstructed road, which is exactly "
        "the positive/negative split the benchmark needs.",
    ),
    Dataset(
        "rescuenet-test",
        "https://ndownloader.figshare.com/files/40583084",
        "uav-anomaly",
        2276.6,
        "CC BY-NC-ND 4.0",
        "https://www.nature.com/articles/s41597-023-02799-4",
        "Held-out split of the same set.",
    ),
    # Real highway/road imagery at ~50 m -- the only set in this list flown near
    # the intended deployment altitude. Its annotations are pavement *distress*
    # (cracks, potholes), not foreign objects, so it is not a positives set for
    # our task. Its value is the opposite: cracks, patches and joints are the
    # textures most likely to be mistaken for debris, which makes this the right
    # data for the section 9 false-positive-on-normal-road metric.
    Dataset(
        "highrpd",
        "https://data.mendeley.com/public-files/datasets/sywswj7djj/files/518cfb67-17f4-44a9-92a4-6d9d64cf8b83/file_downloaded",
        "uav-road",
        1625.5,
        "see Mendeley record; CC BY",
        "https://doi.org/10.17632/sywswj7djj.1",
        "11,696 UAV road images at 640x640 (cropped from 8192x5460), ~50 m "
        "altitude, Shanxi China. YOLO boxes: line cracks, block cracks, potholes.",
    ),
]

# Google Drive hosted. gdown handles the confirmation-token dance that plain
# HTTP cannot.
GDRIVE = [
    Dataset(
        "fod-a-voc",
        "1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x",
        "anomaly-objects",
        412.0,
        "MIT",
        "https://arxiv.org/abs/2110.03072",
        "FOD-A in Pascal VOC format, 300x300 crops. Note these are tight crops, "
        "not full scenes -- useful as an object appearance library, not as a "
        "stand-in for wide-area highway imagery.",
    ),
    Dataset(
        "uavdt-benchmark-m",
        "1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc",
        "uav-road",
        12000.0,
        "research use; cite Du et al. ECCV 2018",
        "https://arxiv.org/abs/1804.00518",
        "UAVDT main image set. Vehicle labels are not the draw -- this is our "
        "source of normal UAV road appearance, i.e. the negatives that the "
        "false-positive rate is computed on.",
    ),
    Dataset(
        "uavdt-attributes",
        "1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1",
        "uav-road",
        1.0,
        "research use; cite Du et al. ECCV 2018",
        "https://arxiv.org/abs/1804.00518",
        "Per-sequence attribute labels: altitude (low 10-30m / medium 30-70m / "
        "high >70m), camera view (front / side / bird), weather. Small file, "
        "high value -- it is what makes the section 7 altitude and viewpoint "
        "stratification measurable rather than guessed.",
    ),
    Dataset(
        "uavdt-motd",
        "19498uJd7T9w4quwnQEy62nibt3uyT9pq",
        "uav-road",
        40.0,
        "research use; cite Du et al. ECCV 2018",
        "https://arxiv.org/abs/1804.00518",
        "MOT annotations and the official DET/MOT toolkit.",
    ),
    Dataset(
        "uavdt-benchmark-s",
        "1661_Z_zL1HxInbsA2Mll9al-Ax6Py1rG",
        "uav-road",
        2500.0,
        "research use; cite Du et al. ECCV 2018",
        "https://arxiv.org/abs/1804.00518",
        "Single-object-tracking split. Extra UAV road imagery; not required for "
        "the benchmark but cheap to keep.",
    ),
    # More UAV highway background diversity for the synthetic-debris pipeline
    # (scripts/data/highway_backgrounds.py): UAVDT's genuine highway footage all
    # comes from just 7 physical locations. VisDrone-VID covers 14 Chinese
    # cities and explicitly includes highway sequences -- val split first
    # (smaller) to triage which sequences are actually highway before
    # committing to the 7.5 GB train split. No license stated on the repo;
    # treated like UAVDT/FOD-A here (research use, cite the paper).
    Dataset(
        "visdrone-vid-val",
        "1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B",
        "uav-road",
        1490.0,
        "research use; cite Zhu et al., VisDrone-VID",
        "https://github.com/VisDrone/VisDrone-Dataset",
        "VisDrone Object-Detection-in-Videos validation split, 14 Chinese "
        "cities incl. highway sequences. No per-sequence highway/urban label "
        "-- triaged visually like UAVDT (see scripts/data/highway_backgrounds.py).",
    ),
    # Task A (road-region segmentation) ground truth. Nothing else on disk has
    # pixel-level road labels from an actual drone at low altitude: RescueNet's
    # Road-Clear/-Blocked is UAV-high over residential streets, and UAVDT has no
    # road labels at all. AeroScapes is real drone footage at 5-50 m -- the
    # operational altitude range -- with an explicit "road" class among 11.
    Dataset(
        "aeroscapes",
        "1WmXcm0IamIA0QPpyxRfWKnicxZByA60v",
        "uav-road-seg",
        1600.0,
        "CC BY-SA 4.0",
        "https://www.cs.cmu.edu/~deva/papers/aeroscapes.pdf",
        "3269 720p drone images, 11-class pixel masks (incl. road), 5-50 m "
        "altitude. JPEGImages/ + SegmentationClass/ + ImageSets/ train/val split.",
    ),
]

DATASETS = DATASETS + GDRIVE
GDRIVE_KEYS = {d.key for d in GDRIVE}
BY_KEY = {d.key: d for d in DATASETS}
GROUPS = sorted({d.group for d in DATASETS})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expect_mb: float) -> None:
    """Stream to disk with a progress line. Raises on truncated transfer."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Some hosts (Mendeley) reject urllib's default User-Agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1024**2:7.1f} / {total / 1024**2:.1f} MB"
                          f"  ({done / total:.0%})", end="", flush=True)
    print()
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise IOError(f"truncated: got {done} of {total} bytes")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", nargs="+")
    ap.add_argument("--group", nargs="+", choices=GROUPS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-zip", action="store_true", help="do not delete the archive after extracting")
    args = ap.parse_args()

    if args.list:
        for d in DATASETS:
            print(f"{d.key:<26} {d.group:<10} {d.mb:>7.1f} MB  {d.why}")
        return 0

    if args.key:
        unknown = [k for k in args.key if k not in BY_KEY]
        if unknown:
            sys.exit(f"unknown key(s): {', '.join(unknown)}")
        targets = [BY_KEY[k] for k in args.key]
    elif args.group:
        targets = [d for d in DATASETS if d.group in args.group]
    else:
        targets = list(DATASETS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {"datasets": {}}

    for d in targets:
        dest_dir = DATA_DIR / d.key
        if d.key in manifest["datasets"] and dest_dir.is_dir():
            print(f"[have] {d.key}")
            continue

        print(f"[get ] {d.key}  ({d.mb:.0f} MB)")
        zip_path = DATA_DIR / f"{d.key}.zip"
        try:
            if d.key in GDRIVE_KEYS:
                import gdown
                gdown.download(id=d.url, output=str(zip_path), quiet=False)
            else:
                fetch(d.url, zip_path, d.mb)
        except Exception as exc:
            print(f"[FAIL] {d.key}: {type(exc).__name__}: {exc}")
            continue

        digest = sha256(zip_path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Not every host that serves a ".zip"-named URL actually sends a zip --
        # sniff the real format from its magic bytes instead of trusting the
        # extension. gzip (tar.gz) is the other format seen in the wild here.
        with zip_path.open("rb") as f:
            magic = f.read(4)
        try:
            if magic[:2] == b"\x1f\x8b":
                with tarfile.open(zip_path, mode="r:gz") as t:
                    t.extractall(dest_dir, filter="data")
            else:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(dest_dir)
        except (zipfile.BadZipFile, tarfile.TarError):
            print(f"[FAIL] {d.key}: archive is corrupt")
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(dest_dir, ignore_errors=True)
            continue

        n_files = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())

        manifest["datasets"][d.key] = {
            "url": d.url,
            "sha256": digest,
            "group": d.group,
            "licence": d.licence,
            "paper": d.paper,
            "files": n_files,
            "size_bytes": size,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)
        print(f"       ok  {n_files} files, {size / 1024**2:.0f} MB  sha256 {digest[:12]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
