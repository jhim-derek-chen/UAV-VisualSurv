# Dataset registry and access status

Records the official source, licence, access route and size for every dataset in
skill.md section 1, plus the outcome of the availability investigation for the
two datasets flagged as uncertain.

Nothing here is downloaded yet — batch 1 is models only. This file is the plan
that `scripts/download_datasets/` will implement.

---

## Priority A — UAV road/highway imagery

### UAVDT — *available, high value*

| | |
|---|---|
| Paper | *The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking*, ECCV 2018 — <https://arxiv.org/abs/1804.00518> |
| Source | Official project page; image data hosted on Google Drive |
| Size | ~12 GB (DET + MOT); ~50 k annotated frames |
| Licence | Research use, cite the paper |

**UAVDT is more valuable to this project than its vehicle labels suggest**, and
this changes how it should be used. It carries per-sequence **attribute
annotations** that map directly onto skill.md section 7:

- **Altitude:** low (10–30 m), medium (30–70 m), high (>70 m)
- **Camera view:** front (23,601 img), side (17,672 img), **bird (10,737 img)**
- Weather: daylight / night / fog

That means the altitude and viewpoint stratification the benchmark needs is
**already labelled**, rather than something we must estimate from image content.
UAVDT should therefore be the backbone of the section 7 viewpoint/scale
analysis, not just a source of road appearance. The 10,737 bird-view frames are
the primary top-down pool.

### UAVid — *available, registration required*

| | |
|---|---|
| Paper | *UAVid: A Semantic Segmentation Dataset for UAV Imagery*, ISPRS 2020 — <https://arxiv.org/abs/1810.10438> |
| Source | <https://uavid.nl/> (reachable; download behind a request form) |
| Size | ~10–20 GB, 42 sequences, 4096×2160 |
| Licence | CC BY-NC-SA 4.0 |

Role is Task A road-region segmentation only. Urban/oblique rather than
highway, so it is a segmentation pretraining and baseline set, **not** the
deployment domain — treat its numbers accordingly.

Native resolution 4096×2160 is 8.8 Mpx per image; on 4 GB VRAM every model must
run tiled. This is the dataset that sets the tiling requirement.

### DRTV30K — *not locatable under this name*

Searched by exact name and by description. **No dataset called "DRTV30K" is
publicly discoverable.** The name may be a mis-transcription from a paper, or
the dataset may be request-only or not yet released.

Nearest candidates for the same role (UAV top-down, road/highway, altitude
metadata):

| Candidate | Fit | Notes |
|---|---|---|
| **UAVDT bird-view + high-altitude split** | **Best** | Already planned, already labelled for altitude and view. Covers most of the intended role. |
| Kust4K | Partial | 4,024 RGB-TIR pairs, urban roads, day+night. CC BY-NC-ND 4.0, on figshare: <https://doi.org/10.6084/m9.figshare.29476610.v3>. No altitude metadata, urban not highway. |
| SynDrone | Partial | Synthetic, multi-modal UAV urban, with *controlled* altitudes (20/50/80 m) — <https://arxiv.org/abs/2308.10491>. Useful for a clean altitude sweep, but synthetic appearance. |
| VisDrone | Partial | Large and varied, but no explicit altitude annotation. |

**Recommendation:** do not block on DRTV30K. UAVDT's attribute splits deliver
the altitude stratification it was wanted for. If a controlled altitude sweep
is later needed with the confound removed, SynDrone is the cleaner instrument
precisely because altitude is a controlled variable there.

*Open question for you:* if DRTV30K came from a specific paper, the citation
would let me find the real name and source.

---

## Priority B — foreign-object / anomaly imagery

### FOD-A — *available*

| | |
|---|---|
| Paper | *FOD-A: A Dataset for Foreign Object Debris in Airports* — <https://arxiv.org/abs/2110.03072> |
| Source | <https://github.com/FOD-UNOmaha/FOD-data> (200 OK); mirror on Kaggle |
| Size | ~31 k annotated images, a few GB |
| Licence | See repo; research use |

Domain is runway/taxiway, not highway, but the task formulation is the one this
project cares about: large uniform paved surface plus arbitrary foreign object.
Its main benchmark value is as a **source of anomaly objects with known pixel
sizes**, feeding the section 9 "recall vs obstacle pixel size" metric.

### FODAnomalyData — *confirmed unavailable*

<https://github.com/FOD-UNOmaha/FODAnomalyData> exists, but the repository
states the data **is no longer available**. The authors suggest reproducing it
using normal images of common concrete.

This is exactly the contingency skill.md section 1 anticipates, so it is not a
blocker: use the paper (*Foreign Object Debris Detection for Airport Pavement
Images based on Self-supervised Localization and Vision Transformer*,
<https://arxiv.org/abs/2210.16901>) as a **design reference** for the
normal-pavement-only training formulation, and build the equivalent split from
FOD-A plus normal UAVDT road frames.

Its formulation is worth copying regardless: train on normal surface only,
detect arbitrary foreign objects, no class supervision. That is the closest
published match to the product goal.

---

## The gap this leaves

None of the available datasets is *UAV highway imagery containing suspicious
objects* — the exact combination the benchmark is about. Available data covers:

- UAV highway, **no** obstacles → UAVDT, UAVid
- Obstacles on pavement, **not** UAV highway → FOD-A
- Ground-view road obstacles → Lost&Found, RoadAnomaly, SMIYC (not yet listed
  in skill.md, but needed as the ground-view control condition for section 7)

So `datasets/custom_uav_obstacles/` is not optional — it is the only source of
true positives in the target domain, and its size will cap the statistical
power of every headline number. Worth deciding its target size early.

**Suggested addition:** pull one ground-view road-anomaly set (RoadAnomaly or
SMIYC) as the *control*. Without it, a poor UAV result cannot be separated from
a model that simply performs poorly everywhere — which would undermine the
central "does viewpoint break these models?" question.
