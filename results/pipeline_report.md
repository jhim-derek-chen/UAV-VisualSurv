# Pipeline comparison report (raw, living document)

Last updated: 2026-09-26. Three ways to run the full chain, frame to risk level:

| Path | Objective 1 (see) | Objective 2 (identify + assess risk) | Status |
|---|---|---|---|
| 1 | road mask + OWLv2 + re-scorer | local Qwen3.5-2B (4-bit), answers taken as they are | done |
| 2 | same | local Qwen3.5-2B + context gate before the risk step | done |
| 3 | same | commercial VLM (Claude API) | not started: needs an API key |

Objective 1 is identical in all three paths and is not re-tuned per path.

## 1. Test set and protocol

- **Test set:** `datasets/synthetic-highway-debris`, 30 real UAV frames of open
  highway from 4 shoot locations, 342 hand-checked vehicles, 30 rendered debris
  objects.
- **Development set ("dev"):** `datasets/synthetic-highway-debris-train`, 30 other
  frames from the same four clips, 90 debris, vehicle labels not hand-checked.
  Every model choice, prompt, rule and threshold of Objective 2 is chosen here.
- **Learned parts** (re-scorer, detector opinion) are scored leave-one-location-out.
  A test frame is always scored by a model that never saw its shoot location.
- **Caveat:** dev and test use the same 22 debris 3D models, and debris is
  always placed 1.2 to 4 vehicle lengths from a real vehicle in its lane.
  Anything that learns object appearance or uses distance to traffic is
  therefore flattered here.

## 2. Accuracy

### Objective 1 (shared by all paths)

Held-out halves, two-stage, IoU > 0.1: vehicles found 95.2%, debris found 90.0%,
precision 84.1%. At IoU > 0.5: 92.8% / 80.0% / 81.3%. See the README.

### Objective 2: identification (paths 1 and 2 share it)

Scored on the boxes Objective 1 keeps on the 30 test frames (414 boxes).

| | Path 1 | Path 2 | Path 3 |
|---|---|---|---|
| debris named with the right category | 56% (14 / 25) | 56% | – |
| tyres named "tire" | 7 / 9 | 7 / 9 | – |
| debris sent to risk assessment at all | 84% | 84% | – |
| vehicles named "vehicle" | 96.0% (316 / 329) | 96.0% | – |

### Objective 2: false alarms in the risk step (goal 2)

"False" means the box is not a debris object: a vehicle, a part of one, or
background. A box on a debris object's shadow counts as that debris.

| | Path 1 | Path 2, round 1 | Path 2, round 2 | Path 3 |
|---|---|---|---|---|
| false boxes sent to risk assessment | 34 | 3 | 1 | – |
| of which rated high risk | 31 | 3 | 1 | – |
| debris boxes assessed (of 27) | 22 | 22 | 22 | – |
| dev: false boxes assessed / debris assessed (of 87) | 33 / 70 | 2 / 68 | 2 / 68 | – |

- **Round 1** froze every threshold on dev before the test run, so it is the
  clean held-out result.
- **Round 2** added two rules after inspecting the three round-1 test failures.
  Both were checked on dev and remove no dev debris, but the test set is no
  longer a clean hold-out for them.
- **Goal 2 (no false alarm rated high) is not fully met.** One false alarm
  remains on the test set: the red cab of an articulated lorry, taken for a
  barrel.

### The path-2 context gate

Applied after identification, before the risk step. It only ever removes a
candidate; it never adds one. In order:

1. **Frame edge.** A box touching the frame edge is an incomplete view. It is
   assessed in the next frame instead.
2. **Far from traffic.** A box more than 12.5 m sideways from every lane that
   has a vehicle in it is roadside. The distance is measured along the
   carriageway direction, which comes from the road mask's main axis.
3. **Detector opinion.** A background / vehicle / debris probe on Objective 1's
   own box feature must not rule out debris. The box is removed if
   P(debris) < 0.3, or (round 2) if debris is not the most likely class. This
   costs nothing: the feature is already computed in Objective 1.
4. **Part of a vehicle.** A box under 0.6 m wholly inside a vehicle's box is a
   plate, light or mirror. A larger box inside a vehicle's box, at most 20% of
   its area, is removed if the VLM, asked directly, says it is part of that
   vehicle. (Round 2) A box end-on against a vehicle in its lane, as wide as
   it, is removed as a cab or trailer when the detector gives P(debris) < 0.5.

On the test set the gate removed 33 candidates. The largest groups were
vehicle parts (wheels, plates), vehicles cut by the frame edge, vehicles the
VLM misnamed, and footpath pedestrians more than 12.5 m from traffic.

## 3. Compute

Measured end-to-end from the raw frame on the RTX 3050 Ti Laptop (4 GB),
`scripts/time_pipeline.py`. It uses the deployable re-scorer trained on all
four locations, so box counts differ slightly from the accuracy runs.

**Staged mode** runs Objective 1 on every frame, unloads its models, then runs
the VLM stages. It suits processing a flight's frames after landing.

| seconds per frame (mean over 30) | Path 1 | Path 2 | Path 3 |
|---|---|---|---|
| road mask | 0.38 | 0.32 | – |
| detection (OWLv2 + features) | 4.62 | 4.33 | – |
| re-scorer + filter | < 0.01 | < 0.01 | – |
| VLM identification | 40.81 | 49.19 | – |
| context gate | – | 0.09 | – |
| VLM risk step | 7.75 | 4.45 | – |
| **total** | **53.55** | **58.38** | – |
| slowest frame | 148.18 | 139.44 | – |
| all 30 frames | 1607 s (26.8 min) | 1751 s (29.2 min) | – |
| boxes per frame / risk calls per frame | 13.3 / 1.57 | 13.3 / 0.80 | – |

| GPU memory (peak allocated) | Path 1 | Path 2 | Path 3 |
|---|---|---|---|
| Objective 1 pass | 2.19 GB | 2.19 GB | 2.19 GB (local) |
| VLM pass | 1.97 GB | 1.97 GB | none (remote) |
| all models loaded together | 4.05 GB (6-frame sample) | – | – |

Model load time: road segmenter 3.8 s, OWLv2 1.2 s, Qwen3.5-2B 10.2 s.

- **Identification dominates.** Every kept box costs a VLM call of about
  3 s, and physics re-asks add more. A frame with 30 boxes takes over two
  minutes.
- **Path 2 does the same identification work as path 1.** Its identification
  time was 49.2 s against 40.8 s, but that gap is run-to-run variation of a
  laptop GPU (thermal throttling), not extra work. The real difference is
  the risk step: 0.8 instead of 1.6 VLM risk calls per frame, 4.5 s instead
  of 7.8 s. The context gate itself costs 0.09 s per frame, including the
  rare part-of-vehicle question.
- **Co-resident is not an option on this card.** With every model loaded at once the peak is 4.05 GB, above the card. Windows spills into shared memory instead of failing, and the 6-frame sample ran at 72.5 s per frame against 53.6 s staged. A live system on this card would need a smaller detector or VLM, or a bigger GPU.

## 4. Deployment considerations

| | Path 1 / 2 (local) | Path 3 (commercial API) |
|---|---|---|
| hardware | one 4 GB laptop GPU is enough, staged | same GPU for Objective 1 |
| runs offline | yes | no, needs a network link |
| images leave the device | no | crops of each detection are sent to the provider |
| cost per frame | electricity only | about US$0.2 per frame on Claude Opus 5, estimated (below) |
| latency | about 54 s per frame, far from real time | set by the network and the provider; calls can run in parallel |
| identification quality | 56% of debris named right | to be measured |

**Estimated API cost for path 3.** This must be confirmed with token counting
before a real run. One identification call sends two 448 x 448 crops at 256
visual tokens each (`ceil(448/28)^2`), plus about 350 tokens of text, so about
860 input tokens. Output is about 300 tokens, allowing for thinking. At Claude
Opus 5's $5 / $25 per million input / output tokens, that is about $0.012 per
call. The 30 test frames need about 414 identification calls and about 60
risk calls, so about US$6, or about $0.20 per frame. Claude Sonnet 5 at $2 /
$10 per million would cost about 40% of that. Model choice is the user's call.

**Licences block commercial use as things stand.** They are no obstacle for
research use.
- The road segmenter starts from NVIDIA's SegFormer weights, released for
  non-commercial use only.
- It is fine-tuned on AeroScapes, a research-only dataset.
- A product needs a replacement road model trained on commercially usable
  data.
- OWLv2 and Qwen3.5 are Apache-2.0.

**Per-location calibration.** Ground resolution (pixels per metre) comes from a
per-location calibration here. On a real flight it would come from altitude and
focal length. Thresholds were chosen on four locations; a new site needs a
check.

## 5. Change log

- 2026-09-26: report created. Path 1 and path 2 (two rounds) measured; path 3
  pending an API key.
