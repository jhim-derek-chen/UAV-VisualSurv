# Pipeline report (raw, living document)

Last updated: 2026-09-26. Three architectures, frame to risk level:

| | Objective 1 (see) | Objective 2 (identify, context gate, assess risk) | Status |
|---|---|---|---|
| A | road mask + OWLv2 + re-scorer | local Qwen3.5-2B (4-bit) | done |
| B | same | OpenAI gpt-5.4, same prompts, checks and gate as A | done |
| C | same | B + one scene-level analysis per frame | being built; tested first on the scene-relation set |

- **The context gate is part of every architecture.** It checks each
  candidate's position against the rest of the frame before the risk step.
  It was first measured as an option (paths 1 to 3, 2026-09-26); it cut false
  alarms sent to risk assessment from 34 to 1 with the local model and from
  23 to 0 with gpt-5.4, so it is no longer compared as an option.
- **The gate-free runs are archived, not deleted.** On GitHub they are under
  the tag [`four-paths`](https://github.com/jhim-derek-chen/UAV-VisualSurv/tree/four-paths/results),
  with the report as it stood then. Locally they are in `archive/`.
- **Objective 1 is identical in all three** and is not re-tuned per
  architecture.

## 1. Test sets and protocol

- **Test set:** `datasets/synthetic-highway-debris`, 30 real UAV frames of open
  highway from 4 shoot locations, 342 hand-checked vehicles, 30 rendered debris
  objects.
- **Development set ("dev"):** `datasets/synthetic-highway-debris-train`, 30 other
  frames from the same four clips, 90 debris, vehicle labels not hand-checked.
  Every model choice, prompt, rule and threshold of Objective 2 is chosen here.
- **Scene-relation set:** being built for architecture C (section 5).
- **Learned parts** (re-scorer, detector opinion) are scored leave-one-location-out.
  A test frame is always scored by a model that never saw its shoot location.
- **Caveat:** dev and test use the same 22 debris 3D models, and debris is
  always placed 1.2 to 4 vehicle lengths from a real vehicle in its lane.
  Anything that learns object appearance or uses distance to traffic is
  therefore flattered here.

## 2. Accuracy on the test set

### Objective 1 (shared)

Held-out halves, two-stage, IoU > 0.1: vehicles found 95.2%, debris found 90.0%,
precision 84.1%. At IoU > 0.5: 92.8% / 80.0% / 81.3%. See the README.

### Identification

Scored on the boxes Objective 1 keeps on the 30 test frames (414 boxes).

| | A (Qwen3.5-2B, local) | B (gpt-5.4) |
|---|---|---|
| debris named with the right category | 56% (14 / 25) | 64% (16 / 25) |
| tyres named "tire" | 7 / 9 | 6 / 9 |
| debris sent on towards risk assessment | 84% | 80% |
| vehicles named "vehicle" | 96.0% (316 / 329) | 96.4% (317 / 329) |
| false-alarm boxes rejected by the VLM itself | 24% (9 / 38) | 37% (14 / 38) |
| answers overruled by the physics check | 48 | 10 |

**B is better, but far from the near-100% target.** Looking at the crops it
got wrong splits them in two:
- **Clear model errors.** A tyre with visible tread was called a shadow, and a
  mattress was called a roadside structure.
- **Crops that carry too little information.** A 22 x 15 px bin is a
  green-black blob. A fridge, a suitcase and a cardboard box seen straight
  from above are all dark boxes.

On this test set, a better model can fix only the first kind. The second kind
needs more pixels on the object: a lower flight or a longer lens. The prompt
was chosen for Qwen on the dev set and reused unchanged for gpt-5.4.

### False alarms in the risk step (goal 2)

"False" means the box is not a debris object: a vehicle, a part of one, or
background. A box on a debris object's shadow counts as that debris.

| | A | B |
|---|---|---|
| false boxes sent to risk assessment | 1 | **0** |
| of which rated high risk | 1 | **0** |
| debris boxes assessed (of 27) | 22 | 21 |
| candidates removed by the gate | 33 | 24 |
| real debris removed by the gate | 0 | 0 |

- **B meets goal 2 on the test set.** A misses it by one: the red cab of an
  articulated lorry, taken for a barrel.
- **Risk levels barely vary.** A rated 21 of its 22 assessed debris high. B
  rated 15 high, 5 medium and 1 low. Each box is judged on its own, so the
  answer is mostly "an object on the road is dangerous". This is the gap
  architecture C is meant to close.
- **The gate was developed in two rounds.** Round 1 froze every threshold on
  dev before the test run (3 false alarms reached the risk step with A).
  Round 2 added two rules after inspecting those three; both were checked on
  dev and remove no dev debris, but the test set is no longer a clean
  hold-out for them.

### The context gate

Applied after identification, before the risk step. It only ever removes a
candidate; it never adds one. In order:

1. **Frame edge.** A box touching the frame edge is an incomplete view. It is
   assessed in the next frame instead.
2. **Far from traffic.** A box more than 12.5 m sideways from every lane that
   has a vehicle in it is roadside. The distance is measured across the
   carriageway, whose direction comes from the road mask's main axis.
3. **Detector opinion.** A background / vehicle / debris probe on Objective 1's
   own box feature must not rule out debris. The box is removed if
   P(debris) < 0.3, or if debris is not the most likely class. This costs
   nothing: the feature is already computed in Objective 1.
4. **Part of a vehicle.** A box under 0.6 m wholly inside a vehicle's box is a
   plate, light or mirror. A larger box inside a vehicle's box, at most 20% of
   its area, is removed if the VLM, asked directly, says it is part of that
   vehicle. A box end-on against a vehicle in its lane, as wide as it, is
   removed as a cab or trailer when the detector gives P(debris) < 0.5.

| removals on the test set | A | B |
|---|---|---|
| frame edge | 4 | 4 |
| far from traffic | 2 | 1 |
| detector opinion | 24 | 17 |
| part of a vehicle (small, inside) | 1 | 0 |
| part of a vehicle (VLM asked) | 2 | 2 |

## 3. Compute

Measured end-to-end from the raw frame on the RTX 3050 Ti Laptop (4 GB),
`scripts/time_pipeline.py`. It uses the deployable re-scorer trained on all
four locations, so box counts differ slightly from the accuracy runs.

**Staged mode** runs Objective 1 on every frame, unloads its models, then runs
the VLM stages. It suits processing a flight's frames after landing.

| seconds per frame (mean over 30) | A | B |
|---|---|---|
| road mask | 0.32 | 0.32 (local, as A) |
| detection (OWLv2 + features) | 4.33 | 4.33 (local, as A) |
| re-scorer + filter | < 0.01 | < 0.01 |
| VLM identification | 49.19 | 22.0 (API) |
| context gate | 0.09 | 0.13 |
| VLM risk step | 4.45 | about 1.3 |
| **total** | **58.38** | **about 28** |
| slowest frame | 139.44 | not measured |
| all 30 frames | 1751 s (29.2 min) | about 14 min |
| boxes per frame / risk calls per frame | 13.3 / 0.80 | 13.8 / 0.70 |

| GPU memory (peak allocated) | A | B |
|---|---|---|
| Objective 1 pass | 2.19 GB | 2.19 GB (local) |
| VLM pass | 1.97 GB | none (remote) |
| all models loaded together | 4.05 GB | – |

Model load time: road segmenter 3.8 s, OWLv2 1.2 s, Qwen3.5-2B 10.2 s.

- **Identification dominates.** Every kept box costs a VLM call, about 3.7 s
  locally and 1.6 s on the API, and physics re-asks add more. A frame with
  30 boxes takes over two minutes with A.
- **Detection plus the gate is about 4.7 s per frame.** The gate itself costs
  0.09 s: its rules are box geometry, and the detector-opinion rule reuses
  Objective 1's features. Only the occasional part-of-vehicle question calls
  the VLM.
- **B's time is network time.** Its VLM calls are made one after another and
  do not use the GPU; parallel calls would cut them further. B's risk-step
  time is estimated from its per-call time (1.8 s) and 0.7 calls per frame.
- **Co-resident is not an option on this card.** With every model loaded at
  once the peak is 4.05 GB, above the card. Windows spills into shared memory
  instead of failing, and a 6-frame sample ran at 72.5 s per frame against
  53.6 s staged (measured before the gate existed; the gate adds no GPU
  memory). A live system on this card would need a smaller detector or VLM,
  or a bigger GPU.

## 4. Deployment considerations

| | A (local) | B (OpenAI API) |
|---|---|---|
| hardware | one 4 GB laptop GPU is enough, staged | same GPU for Objective 1 |
| runs offline | yes | no, needs a network link |
| images leave the device | no | crops of each detection are sent to the provider |
| cost per frame | electricity only | US$0.036 on gpt-5.4, measured (below) |
| latency | about 58 s per frame, far from real time | about 28 s per frame, sequential calls |
| identification quality | 56% of debris named right | 64% of debris named right |

**Measured API cost for B.**
- **Tokens.** On the 30 test frames B makes 447 gpt-5.4 calls: 298,391 input
  tokens and 21,953 output tokens. gpt-5.4 used no reasoning tokens at its
  default setting.
- **Cost.** At $2.5 / $15 per M tokens, that is US$1.08 per 30 frames, or
  US$0.036 per frame.

| model | input / output per M tokens | 30 frames, estimated |
|---|---|---|
| gpt-4.1 | $2 / $8 | about $0.8 |
| gpt-5.4 (chosen) | $2.5 / $15 | $1.08 measured |
| gpt-5.5 | $5 / $30 | about $2.2 |

**Free options were tried first and do not cover a run.**
- **GitHub Models** was retired on 2026-07-30.
- **Gemini free tier** measured 20 requests per model per day, and a run
  needs about 450.
- **Gemini with billing enabled** is supported by the code:
  `UAV_API_PROVIDER=gemini`.

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

## 5. Architecture C: scene-level analysis

**Why.** A and B judge each box on its own. Many highway hazards are about
relations between objects: the same tyre is urgent in a lane with traffic
coming and minor on the hard shoulder; several items strewn behind a lorry
are one spilled load, not unrelated debris.

**Design.** C keeps everything in B, including the context gate: removing
false alarms stays the gate's job, since the VLM on its own let 23 through.
C adds one VLM call per frame after the per-box steps. It gets the whole
frame with every remaining candidate numbered, a close view of each
candidate, and a table of measured facts. It returns an overall risk level,
a risk for each candidate, which candidates belong together, and the lanes
affected.

**Test.** The 30-image test set cannot show whether C is better: its debris
is placed at random, with no designed relation to the traffic, and nothing
says what the right risk is. C is therefore tested first on a new
scene-relation set with the expected answers written down before any run.
A and B are re-run on it later if C proves worthwhile.

## 6. Change log

- 2026-09-26: report created. Local Qwen measured with and without the
  context gate (two gate rounds).
- 2026-09-26: gpt-5.4 measured with and without the gate. Goal 2 met with
  gpt-5.4 + gate.
- 2026-09-26: the gate made standard. Report restructured as architectures
  A, B, C; gate-free runs archived (tag `four-paths`). B's records reproduced
  exactly by the new single-pass code, from cached answers.
