# Pipeline report (raw, living document)

Last updated: 2026-09-28. Four architectures, frame to risk level:

| | Objective 1 (see) | Objective 2 (identify, context gate, assess risk) | Status |
|---|---|---|---|
| A | road mask + OWLv2 + re-scorer | local Qwen3.5-2B (4-bit) | done |
| B | same | OpenAI gpt-5.4, same prompts, checks and gate as A | done |
| C | same | B + one scene-level analysis per frame | tested on the scene-relation set (section 5) |
| D | same, code sped up (identical output) | B, but the detector screens first: only boxes it rates debris go to gpt-5.4 | done (section 6): 6.3 s per frame, same accuracy as B |

- **The context gate is part of every architecture.** It checks each
  candidate's position against the rest of the frame before the risk step.
  It was first measured as an option (paths 1 to 3, 2026-09-26); it cut false
  alarms sent to risk assessment from 34 to 1 with the local model and from
  23 to 0 with gpt-5.4, so it is no longer compared as an option.
- **The gate-free runs are archived, not deleted.** On GitHub they are under
  the tag [`four-paths`](https://github.com/jhim-derek-chen/UAV-VisualSurv/tree/four-paths/results),
  with the report as it stood then. Locally they are in `archive/`.
- **Objective 1 is identical in all four** and is not re-tuned per
  architecture. Its code was sped up on 2026-09-28 (section 6) with
  bit-identical output, which benefits all four.

## 1. Test sets and protocol

- **Test set:** `datasets/synthetic-highway-debris`, 30 real UAV frames of open
  highway from 4 shoot locations, 342 hand-checked vehicles, 30 rendered debris
  objects.
- **Development set ("dev"):** `datasets/synthetic-highway-debris-train`, 30 other
  frames from the same four clips, 90 debris, vehicle labels not hand-checked.
  Every model choice, prompt, rule and threshold of Objective 2 is chosen here.
- **Scene-relation set:** `datasets/scene-relations`, 23 frames where the right
  risk depends on relations between objects; built for architecture C
  (section 5).
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
`scripts/report/time_pipeline.py`. It uses the deployable re-scorer trained on all
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
relations between objects: the same pallet is urgent in a lane with traffic
and minor on the hard shoulder; several items strewn behind a lorry are one
spilled load, not unrelated debris.

**Design** (`scripts/assess/scene_assess.py`).
- **C keeps everything in B, including the context gate.** Removing false
  alarms stays the gate's job, since the VLM on its own let 23 through on the
  test set.
- **One more gpt-5.4 call per frame**, after the per-box steps. It gets:
  - the frame cropped to the road, with every candidate boxed in red and
    numbered, and every vehicle boxed in blue;
  - a 20 m close view of each candidate;
  - a table of measured facts, including each candidate's first-pass risk.
- **It returns**, for each candidate, where it lies (running lane, hard
  shoulder, off the road) and a revised risk. It also returns which
  candidates belong to one event and which vehicle it came from, the number
  of running lanes blocked, a scene risk and one instruction for the control
  room.
- **It uses the same risk rubric as the per-box step.** That rubric already
  says a large object on the hard shoulder is medium, so B and C are held to
  the same standard.
- **A frame with no candidate left is "none" by rule**, without a call.

**The scene-relation set** (`scripts/data/build_scene_relations.py`).
- **Frames:** 23 test frames on 5 real backgrounds, and 8 dev frames on 2
  others.
- **Five variants of each background:**

| variant | what is placed | expected answer |
|---|---|---|
| clear | nothing | scene none |
| lane | one large rigid object in a running lane, near traffic | high; 1 lane blocked |
| shoulder | the same object, same pose, moved sideways onto the hard shoulder | medium; 0 lanes blocked |
| spill | 3-4 objects of one kind strewn behind a lorry, in its lane | all high, one group, that lorry; high |
| blockage | two large objects side by side in adjacent lanes | both high; 2 lanes blocked |

- **The answers were fixed in the build script before any model run.** They
  follow the per-box rubric, and only large rigid objects are used (tyres,
  fridges, pallets, planks, ladders), so each placement has one right answer.
- **Lane geometry is measured from each frame.** The solid lines are fitted
  per frame, and every placement was checked by eye and against the fitted
  lines. A shoulder object fills at most 70% of the shoulder, clear of the
  edge line.
- **Backgrounds.** Four of the five test backgrounds are one near-static
  camera (Pexels 19851623): a straight six-lane motorway with a hard
  shoulder each side. The fifth is 12306893. It gets no shoulder variant,
  because its outer strip carries traffic in other frames (a dynamic hard
  shoulder).

**Results on the 23 test frames.** A and B's scene risk is their highest
per-box risk. B's records (`scene-relations/b_gpt/`) were written from the
cached answers of C's run, since C runs every B step first.

| | A (per box) | B (per box) | C (scene) |
|---|---|---|---|
| scene risk right | 16 / 23 | 16 / 23 | 17 / 23 |
| clear frames with no alarm | 5 / 5 | 5 / 5 | 5 / 5 |
| lane frames | 3 / 5 | 3 / 5 | 3 / 5 |
| shoulder frames | 0 / 4 | 0 / 4 | 1 / 4 |
| spill frames | 4 / 4 | 3 / 4 | 3 / 4 |
| blockage frames | 4 / 5 | 5 / 5 | 5 / 5 |
| false alarms rated high | 2 | 1 | 1 |

These are answers only C gives:

| | C |
|---|---|
| where each object lies (lane or shoulder), for objects that reached it | 24 / 25 |
| running lanes blocked, exact | 16 / 23 |
| spill items grouped as one event | 4 / 4 |
| spill traced to the right lorry | 2 / 4 |

- **C's risk levels are only slightly better than B's here.** C changed four
  first-pass levels:
  - **One fix.** A ladder on the shoulder went from high to medium, with the
    reason "it lies on the paved strip outside the solid edge line, on the
    hard shoulder rather than in a live running lane".
  - **Three items C lowered from medium to low.** The VLM had named them
    cardboard boxes, and the rubric makes a cardboard box low. They were
    pallets, so the answer key counts them wrong.
- **Most failures happen before the scene step.**
  - **Detection** missed 5 of the 33 placed objects. Four were pallets (about
    22 px at this camera's scale) and one was a fridge.
  - **Identification** stopped 3 more. It named a pallet a roadside
    structure, and two tyres vegetation or a roadside structure.
  - **Misnamed objects that passed** set the level for B and C alike: pallets
    named cardboard boxes (low), and fridges named mattresses or bins
    (medium).
  - **Shoulder frames were hit hardest.** Only 2 of the 4 shoulder objects
    reached the scene step. C got one right (the ladder) and put the other (a
    pallet) in a lane.
- **C over-counts lanes for spills.** In 3 of the 4 spill frames it said 2
  lanes where the items lay in 1.
- **The one false alarm rated high survived both B and C.** It was part of a
  vehicle, named "barrel or drum". It passed the gate, and C did not
  overrule it.
- **The strength of C is structure.** It can say "four planks, one spilled
  load, most likely from the open-load lorry V9, two lanes, close them", and
  a control room can act on that. B's separate boxes cannot say it.

**Cost and time.** One scene call per frame that has a candidate (15 of the
23 frames). Each call is about 3,200 input and 160 output tokens: about
US$0.011 and 2.8 s. The whole C run on the 23 frames made 154 new calls and
cost US$0.48. Identification answers repeat across the variants of a
background and were reused from the cache.

**Caveats.**
- **Small and narrow.** 23 frames, 4 of the 5 backgrounds from one camera,
  and 4 shoulder frames. The numbers show direction, not precision.
- **The rendered objects and their shadows are not always convincing**, the
  fridge especially, which hurts identification more than it would on real
  debris.
- **This camera's calibration looks too high.** Its lanes measure 2.4 m at
  the calibrated 18.6 px/m, where real lanes are about 3.5 m. The sizes
  given to the VLM are therefore about 30% small at this location, for
  every architecture.

**A on the scene set.** A matches B on scene risk (16 / 23) but misses
more: it named the shoulder ladder and two spilled planks roadside
structures. It also rated two false alarms high: the vehicle part the others
also missed, and a whole lorry named a wooden pallet. Its VLM steps took 16
minutes for the 23 frames, locally and at no cost.

**C on the fixed 30-image test set.** Identification answers came from B's
cache, so only the scene calls were new: 21 calls (the 21 frames with a
candidate), US$0.21 in total and about 3.5 s each; 2.4 s per frame on average.
- **False alarms rated high: 0**, as for B.
- **C changed one level.** A fridge lying across the edge line onto the hard
  shoulder went from high to medium. Checked in the frame: it does lie beyond
  the edge line, pushed there by the placement's sideways jitter.
- **Timing and cost for C:** B's plus 2.4 s and US$0.007 per frame. That is
  about 30.5 s and US$0.043 per frame.

## 6. Architecture D: fast, two tiers

**Goal (2026-09-28).** Total time per frame under 8 s with no loss of
performance, and the first, every-frame tier as short as possible: the drone
must be able to keep up.

**Two tiers.**
- **Tier 1, look, every frame, on the device:** road segmentation, detection,
  re-scoring, and the detector's background / vehicle / debris probe (the
  context gate's detector-opinion rule), run as a screen.
- **Tier 2, assess, only when tier 1 flags a candidate:** gpt-5.4
  identification of the flagged boxes, the context gate, the risk step.
- **A to C have no trigger.** They send every box, mostly ordinary cars, to
  the VLM, so tier 2 runs on every frame.

**Why the screen loses nothing.** In B, a box can reach the risk step only if
the gate's detector-opinion rule lets it through: debris must be the probe's
most likely class and P(debris) >= 0.3. D applies that same test before the
VLM instead of after it.
- **Coverage.** All 21 boxes that reached B's risk step pass the screen, and
  all 27 debris boxes are among the 43 flagged (1.43 per frame).
- **Calls.** Identification calls on the 30 test frames: 414 in B, 43 in D.
- **Other boxes** take the probe's class, vehicle or background. The gate's
  geometry rules use the probe's vehicles.

**Accuracy on the 30 test frames (`results/risk-assessment/d_fast/`).**

| | B | D |
|---|---|---|
| debris named with the right category | 64% | 64% |
| vehicles recognised | 96.4% (VLM) | 98.5% (detector probe) |
| false alarms rated high | 0 | 0 |
| debris assessed / rated high (of 27) | 21 / 15 | 21 / 15 |
| risk levels given (high / medium / low) | 15 / 5 / 1 | 15 / 5 / 1 |

**Time, measured live with `python scripts/report/time_pipeline.py --arch d`.** The
API cache is off, calls for a frame's candidates run in parallel, and the
laptop has an RTX 3050 Ti (4 GB).

| seconds per frame (30 test frames) | D |
|---|---|
| road segmentation ∥ detection (run side by side) | 2.36 |
| re-scoring + screen | < 0.01 |
| **tier 1 total** | **2.36** (max 2.56) |
| identification (gpt-5.4, flagged boxes) | 2.07 |
| context gate | 0.05 |
| risk step | 1.85 |
| **tier 2 total, per frame** | **3.97** (4.41 on the 27 frames that trigger it) |
| **end to end** | **6.33** (median 6.52, 90th percentile 7.88, max 9.24) |
| API cost | US$0.0056 per frame (69 calls, 42k input and 4k output tokens per 30 frames) |

- **Every test frame contains debris**, so tier 2 runs on 27 of 30. On a
  patrol most frames hold none, and the average falls towards tier 1's 2.4 s.
- **Start-up** costs 2.5 s to load the models and 12.6 s to connect to the
  API, once.

**Engineering behind the times.**
- **Detection, 4.33 → 1.96 s, bit-identical output.** Checked on 30 / 30
  frames, boxes and embeddings.
  - The duplicate-box filter was a Python double loop over about 1,500 boxes
    (2 to 3 s) and is now vectorised.
  - Tiles are resized on CPU threads while the GPU runs the previous tile.
  - Road segmentation runs alongside detection, since the mask is only needed
    after it.
- **API latency.**
  - **Cause.** A few calls took 12 to 15 s. This laptop's DNS resolver takes
    about 11 s on a cache miss (curl: 11.3 s, then 0.2 s), and the HTTP
    client closed idle connections after 5 s.
  - **Fix.** Connections are now kept open for 10 minutes and opened once at
    start-up. The slowest call dropped from 14.5 s to 4.3 s, with a median of
    2.1 s.
  - **Hedging.** A request not answered in 4 s is sent a second time; it
    fired once in 69 calls.

**LeVJEPA, tried and not adopted.**
- **What it is.** LeVJEPA (Kuhn, ..., LeCun, Balestriero, Buettner, 2026,
  arXiv:2608.27395) is a video JEPA encoder. Its gain is 5.6 to 20.8 times
  less *pretraining* compute. At inference it is an ordinary ViT; the only
  released checkpoint is ViT-L/16, with weights under CC BY-NC 4.0.
- **Test.** The whole frame, resized to a long side of 1344 px, was encoded
  in one pass (0.54 s per frame). A linear background / vehicle / debris
  probe was fitted on each 16 px patch (`scripts/see/levjepa_detector.py`).
- **Result.** Detection fell far short of OWLv2.
  - **Setting.** Chosen on dev: no neighbourhood context, C = 0.01. Objects
    are split by watershed from local maxima.
  - **Test set, held-out halves:** vehicles 40%, debris 67%, precision 47%.
  - **At the lowest threshold:** vehicles 87%, debris 87%.
  - **OWLv2 for comparison:** 95.2%, 90.0% and 84.1%.
  - **Results file:** `results/road-object-eval/levjepa-patch.json`.
- **Why.** At this scale a patch is 46 px of the 4K frame, larger than most
  debris and than a lane, so neighbouring cars merge.
- **Decision.** Replacing OWLv2 with it would break the rule that
  performance must not drop, so D keeps OWLv2.

## 7. Change log

- 2026-09-26: report created. Local Qwen measured with and without the
  context gate (two gate rounds).
- 2026-09-26: gpt-5.4 measured with and without the gate. Goal 2 met with
  gpt-5.4 + gate.
- 2026-09-26: the gate made standard. Report restructured as architectures
  A, B, C; gate-free runs archived (tag `four-paths`). B's records reproduced
  exactly by the new single-pass code, from cached answers.
- 2026-09-26: scene-relation set built (23 test, 8 dev frames). Architecture
  C run on it: scene risk right 17 / 23 against B's 16 / 23; spills grouped
  4 / 4; lane or shoulder right 24 / 25. Most errors come from detection and
  identification before the scene step.
- 2026-09-26: A run on the scene set (16 / 23), B's scene records written
  from cache, C run on the 30-image test set (0 false alarms rated high, one
  level changed). Stage report written to the SKILL.md guideline:
  `report/stage_report.md`, compiled with `python scripts/report/build_report.py`.
- 2026-09-27: stage report rebuilt as a 9-slide deck (presentation mode):
  overview on the stage's two aims (chain connected, architectures compared),
  one evaluation table, and scene analysis kept to a single add-on slide at
  the end.
- 2026-09-28: architecture D (tier 1 screen + gpt-5.4 on flagged boxes only):
  same accuracy as B, 6.33 s per frame end to end, tier 1 2.36 s, US$0.0056
  per frame. Objective 1 code sped up for all architectures with
  bit-identical output. LeVJEPA tried as a one-pass detector and not adopted.
