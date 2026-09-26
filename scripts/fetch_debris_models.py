"""Search and download 3D models of common highway debris from Sketchfab, for
rendering synthetic anomalies (see scripts/highway_backgrounds.py for the
real-photo side of the synthesis pipeline).

Category list follows HazardNet (Choe et al., CVPR 2023 Workshop on
Autonomous Driving) -- "Road Debris Detection by Augmentation of Synthetic
Models" -- the closest published precedent for this exact task, trimmed to
objects plausible on a highway rather than its full 20-class list (e.g.
"detached trailer" dropped).

Sketchfab requires an authenticated account to download even CC0 models --
confirmed against their API docs, not assumed. Token read from
.secrets/sketchfab_token.txt (outside any published output; never printed).

Only CC0 and CC-BY results are kept: CC0 needs no attribution, CC-BY needs a
credit line recorded alongside the model (this script writes one into each
model's manifest entry so it is never separately tracked down later).

    python scripts/fetch_debris_models.py --search          # list candidates only
    python scripts/fetch_debris_models.py --download        # fetch the chosen picks
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = REPO_ROOT / ".secrets" / "sketchfab_token.txt"
OUT_DIR = REPO_ROOT / "datasets" / "debris-3d-models"
CANDIDATES_FILE = REPO_ROOT / "datasets" / "debris-3d-models-candidates.json"
API = "https://api.sketchfab.com/v3"

# The search API's license object has no machine-readable slug -- only a
# free-text `label` (verified against a live response, not assumed). Exact
# match on the two permissive labels; "CC Attribution-NonCommercial" and
# "-ShareAlike"/"-NoDerivs" variants are deliberately excluded.
ALLOWED_LICENSES = {"CC0 (Public Domain)", "CC Attribution"}

# category -> search query. Plain object name, not "highway debris X", because
# most 3D asset libraries are not populated with driving-scene-specific tags.
# "tire"/"car tire" alone return whole cars (SDC/Lamborghini scenes tagged
# with "tire") almost exclusively -- checked by inspecting results, not
# assumed; "old"/"dirty" qualifiers are what actually surfaces a standalone
# tire, and happen to match real roadside debris better than a clean one.
CATEGORIES = {
    "tire": "old dirty tire",
    "cardboard-box": "cardboard box",
    "suitcase": "suitcase",
    "traffic-cone": "traffic cone",
    "barrel": "oil barrel",
    "wooden-pallet": "wooden pallet",
    "mattress": "old mattress",
    "trash-can": "trash can",
    "wooden-plank": "wood plank",
    "ladder": "ladder ",
    # Added after a user QA pass found the calibrated (real-metre) sizes made
    # the dataset's small categories hard to see in wide/high-altitude
    # frames -- rather than inflate those sizes past their real ones, add
    # categories that are genuinely large in real life, which stay legible
    # under the same honest calibration. Both are common real highway
    # spillage (a lost truck/bus wheel; an appliance off a moving van).
    "truck-tire": "truck wheel",
    "refrigerator": "fridge",
}

# Hand-picked from CATEGORIES search results (results/debris-3d-models-candidates
# after --search): sorting by likeCount alone surfaces whatever is popular that
# happens to mention the query word, not necessarily a single free-standing
# object -- e.g. "tire" is dominated by whole supercar scenes, "mattress" by
# full bedroom sets. Each entry here was checked by name/face-count/license
# before being trusted, not auto-selected. All are "CC Attribution" -- credit
# is generated into datasets/debris-3d-models/manifest.json, never separately
# tracked by hand.
CURATED_PICKS = {
    "tire": [
        ("1929673829054c2b8085d743e652acab", "Old Tires - Dirt (Low Poly)"),
        ("ec06c0af626746e7a44de65436037173", "Old Dirty Tire"),
    ],
    "cardboard-box": [
        ("8986ba512f704ac5b253286a0d1ad8bb", "Set of Cardboard Boxes"),
        ("3c8b08023b61484aa583994da991274c", "Box"),
    ],
    # 9490e779... ("Briefcase / suitcase, low poly, game ready") dropped: its
    # glTF scene merges a "closed empty case" mesh with a separate "open
    # lid + stacks of cash" mesh (an alternate display state, not meant to
    # be shown together) -- import_and_normalize()'s blind join glues both
    # into one object, which some yaws render as a tall standing sliver
    # instead of a suitcase. Confirmed by direct re-render at the seed that
    # produced it (dataset image_id 130 in the original 300-sample run).
    "suitcase": [
        ("73505b69be8f4be09194deafdd96c684", "Suitcase"),
    ],
    # e03b1d2d... ("Traffic Cones") and 426e2d6e... ("Barrier & Traffic Cone
    # Pack - Part 2") both dropped: both are multi-object Sketchfab "kit"
    # scenes (a grid of ~15 cones/bollards/panels; a barrier-arm mechanism),
    # not a single cone -- scaling either to one cone's real size renders a
    # small cluster of miniature unrelated objects. Confirmed by direct
    # render. Replaced with two verified single-cone models.
    "traffic-cone": [
        ("66b282e65d3b4da8a883ae8579bfb33c", "Traffic Cone"),
        ("77da362c4cb44295a838fd778f70f31b", "Orange street cone"),
    ],
    # 8df46c47... ("Low Poly | Closed Barrels") dropped: a Sketchfab colour-
    # variant showcase, 16 separate barrels arranged in a grid, not one
    # barrel -- scaled to one barrel's real size it renders as a cluster of
    # 16 miniature barrels. Confirmed by direct render.
    "barrel": [
        ("ec37cc2a844f4f0ca0873d85a8b3b976", "Oil barrel"),
    ],
    "wooden-pallet": [
        ("dba5c00928cd400796d9f6fffdd724b3", "Wooden Pallets"),
        ("e12fde1101b84b6da40056336a7b309d", "Wooden Crate"),
    ],
    "mattress": [
        ("e914df64a021484a91613b38c28ee538", "Old dirty mattress"),
        ("f8bca903fda348568e18adc33884103d", "Old Mattress"),
    ],
    "trash-can": [
        ("3836ff67877248c2a820e6a969984aac", "Trash can"),
    ],
    "wooden-plank": [
        ("1901d8f1e7354ea289860d29c6ba0046", "Set of wood Planks"),
        ("1b31413d89a4485e93f4bdb9eb735d88", "Planches - Planks"),
    ],
    "ladder": [
        ("adf229c426194740ab0ec9cdb00262b4", "Ladder"),
        ("4f0a875a047d451cb65316f590bbcdc2", "Fiberglass Step Ladder 6' (New)"),
    ],
    "truck-tire": [
        ("6ebc61a1e30a4472ae456e8153d0fc82", "Offroad Truck Wheel"),
        ("f2c40e1a5e664d24b80c28d9e4abaedf", "Low-Poly Treaded Tire"),
    ],
    "refrigerator": [
        ("fe9bd6617f924ac096cb37e2157c4d2a", "Not-too-modern fridge"),
        ("2071bda681b642218b6829b82e4fd93b", "Refrigerator - Grey Polished Metal"),
    ],
}


def token() -> str:
    if not TOKEN_FILE.is_file():
        raise SystemExit(f"no token at {TOKEN_FILE} -- see scripts/fetch_debris_models.py")
    return TOKEN_FILE.read_text().strip()


def headers() -> dict:
    return {"Authorization": f"Token {token()}"}


def search(query: str, count: int = 24) -> list[dict]:
    """Downloadable models only; license filtered client-side because the
    search API's `licenses` param takes numeric slugs that vary by license
    and are not worth hard-coding when the response already states them."""
    r = requests.get(f"{API}/search", headers=headers(), params={
        "type": "models", "q": query, "downloadable": "true",
        "sort_by": "-likeCount", "count": count,
    }, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("results", []):
        lic = (m.get("license") or {}).get("label", "")
        if lic not in ALLOWED_LICENSES:
            continue
        out.append({
            "uid": m["uid"], "name": m["name"], "license": lic,
            "faceCount": m.get("faceCount"), "likeCount": m.get("likeCount"),
            "viewCount": m.get("viewCount"),
            "thumbnail": (m.get("thumbnails", {}).get("images") or [{}])[-1].get("url"),
            "url": m.get("viewerUrl"),
            "user": (m.get("user") or {}).get("username"),
        })
    return out


def get_download_url(uid: str) -> str | None:
    r = requests.get(f"{API}/models/{uid}/download", headers=headers(), timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    # Prefer glTF (self-contained, easy to load in Blender); fall back to source.
    for fmt in ("gltf", "source", "usdz"):
        if fmt in data:
            return data[fmt]["url"]
    return None


def cmd_search() -> int:
    all_candidates = {}
    for key, query in CATEGORIES.items():
        print(f"--- {key} ({query!r}) ---", flush=True)
        results = search(query)
        all_candidates[key] = results
        print(f"    {len(results)} CC0/CC-BY downloadable candidates")
        time.sleep(0.3)  # be polite to the API
    CANDIDATES_FILE.write_text(json.dumps(all_candidates, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    print(f"\nsaved: {CANDIDATES_FILE.relative_to(REPO_ROOT)}")
    return 0


def get_model_info(uid: str) -> dict:
    r = requests.get(f"{API}/models/{uid}", headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def cmd_download() -> int:
    manifest = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, picks in CURATED_PICKS.items():
        for uid, expected_name in picks:
            dest = OUT_DIR / key / uid
            info = get_model_info(uid)
            lic = (info.get("license") or {}).get("label", "?")
            author = (info.get("user") or {}).get("username", "?")
            source_url = info.get("viewerUrl", f"https://sketchfab.com/3d-models/{uid}")
            if dest.is_dir() and any(dest.iterdir()):
                print(f"[have] {key}/{uid}  ({expected_name})")
            else:
                url = get_download_url(uid)
                if not url:
                    print(f"[FAIL] {key}/{uid} ({expected_name}): no download url returned")
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                zpath = dest / "model.zip"
                r = requests.get(url, timeout=120)
                zpath.write_bytes(r.content)
                import zipfile
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(dest)
                zpath.unlink()
                print(f"[get ] {key}/{uid}  ({expected_name}, {lic}, by {author})")
            manifest.setdefault(key, []).append({
                "uid": uid, "name": expected_name, "license": lic, "author": author,
                "source_url": source_url,
                "credit": f'"{expected_name}" by {author}, {lic}, via Sketchfab ({source_url})',
            })
            time.sleep(0.3)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {(OUT_DIR / 'manifest.json').relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="discover candidates (writes candidates.json)")
    ap.add_argument("--download", action="store_true", help="fetch CURATED_PICKS")
    args = ap.parse_args()

    if args.search:
        return cmd_search()
    if args.download:
        return cmd_download()
    print("pass --search or --download")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
