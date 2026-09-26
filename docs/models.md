# Model registry

Source of truth for code is `scripts/download_models/registry.py`; resolved
commit SHAs land in `models/manifest.json` after a download run. This file
records the reasoning and the access status.

All variant choices are driven by one constraint: **RTX 3050 Laptop, 4 GB VRAM.**

## Batch 1 — downloaded

| Key | Repo | Licence | Est. VRAM | Role |
|---|---|---|---|---|
| `segformer-b0-cityscapes` | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | NVIDIA SCL (non-commercial) | 0.5 G | Smoke test for the tiling core |
| `segformer-b2-cityscapes` | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | NVIDIA SCL (non-commercial) | 1.2 G | Task A primary baseline |
| `mask2former-swin-tiny-cityscapes` | `facebook/mask2former-swin-tiny-cityscapes-semantic` | CC-BY-NC-4.0 | 2.0 G | Task A second architecture |
| `dinov2-large` | `facebook/dinov2-large` | Apache-2.0 | 1.0 G | PROWL-style dense features |
| `grounding-dino-tiny` | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | 1.5 G | Open-vocabulary detection |
| `sam2.1-hiera-base-plus` | `facebook/sam2.1-hiera-base-plus` | Apache-2.0 | 2.2 G | Class-agnostic mask refinement |

**Licence note.** SegFormer and Mask2Former are released under non-commercial
terms. Fine for a research benchmark; they would need replacing before any
product deployment. DINOv2, Grounding DINO and SAM 2.1 are Apache-2.0 and carry
no such restriction.

## DINOv3 — gate resolved

`facebook/dinov3-vitl16-pretrain-lvd1689m` is `gated: manual`. Access was
requested and **granted** on 2026-09-05 for account `jhimderekchen`; a direct
fetch of `config.json` now returns 200. It is included in batch 1.

The gate has two independent parts, and having only one produces a confusing
failure. For reference if a second machine or account ever needs setting up:

1. **Accept the licence** on the model page while signed in — the slow part,
   reviewed manually by the repo owner.
2. **Supply a `read` token** via `HF_TOKEN`.

With a valid token but no grant the API reports `gatedGranted: ''` and file
fetches return 401/403 — which looks like a token problem but is not. See
`docs/hf_token_setup.md`.

Community re-uploads of these weights exist but were **deliberately not used**:
skill.md section 11 item 3 forbids silently substituting unofficial mirrors when
an official source exists.

## Verified on hardware (2026-09-05)

`scripts/check_models.py`, RTX 3050 Ti Laptop 4 GB, torch 2.6.0+cu124,
transformers 5.16.1. Raw output in `results/vram_check.json`.

All seven checkpoints load and execute. Estimates in the registry were
uniformly **too high** — e.g. Mask2Former estimated 2.0 GB, measured 0.64 GB at
1024x1024. See README for the full measured table.

### Resolved: Mask2Former "MISSING" keys

Loading logs `pixel_level_module.encoder.swin.layernorm.{weight,bias}` as
MISSING and newly initialised. A forward hook confirms the module *is called*,
which looks alarming — but being called is not the same as feeding the output.

Definitive test: scale those weights by 37x and shift the bias by 11, then
compare mask logits. **Max change: 0.000000.** HF's Swin computes a final
layernorm intended for classification heads; Mask2Former consumes multi-scale
features from intermediate stages and discards it. The layer is dead code here.

**Verdict: harmless, no action needed.** Recorded because the warning will
reappear on every load and would otherwise be re-investigated.

### Deferred: SAM 2.1 config type mismatch

The repo's `config.json` declares `sam2_video` while we load `Sam2Model` (image
variant), and transformers warns the combination "can yield errors". It loads
and runs at 1024x1024 (0.40 GB); other input sizes raise RuntimeError, which is
expected — SAM 2 has a fixed 1024 image size by design, not a bug.

Deferred because SAM 2 is not on the critical path: skill.md section 10 places
it after road segmentation, PROWL, Grounding DINO and the multimodal baselines.
The video-vs-image distinction only matters for temporal/memory features, which
single-image mask refinement does not use. Revisit at the Grounding DINO -> SAM
stage.

### Grounding DINO: fp32 only, and tiling is correct anyway

The text tower is fp32, so the whole model must be. `torch.autocast` fp16 was
tested and is **counterproductive** — 3.81 GB versus 3.49 GB for plain fp32,
because autocast retains fp32 master weights alongside fp16 casts.

It OOMs at 4096x2160, which sounds like a constraint but is not one: Grounding
DINO was trained around 800x1333, so feeding it a full 8.8 Mpx aerial frame
would be methodologically wrong regardless of memory. Tiling at ~1024 (2.02 GB)
is both what fits and what the model expects.

## Measured download sizes

Sizes resolved from the HF tree API, excluding `.bin` duplicates where
safetensors exist. Observed throughput on this connection: **~1.3 MB/s**
(SegFormer-B0, 14.3 MB in 11 s) — roughly 13x faster than this machine's PyPI
throughput, so model downloads are not the bottleneck that PyPI was.

| Model | Download |
|---|---:|
| `segformer-b0-cityscapes` | 14.3 MB |
| `segformer-b2-cityscapes` | 104.5 MB |
| `mask2former-swin-tiny-cityscapes` | 362.5 MB |
| `dinov2-large` | 2,322.3 MB |
| `grounding-dino-tiny` | 1,318.2 MB |
| `sam2.1-hiera-base-plus` | 617.1 MB |
| `dinov3-vitl16` | 1,156.4 MB |
| **Total** | **~5.9 GB** |

## Deferred to WSL2 (detectron2 stack)

These have no supported native Windows build and pin an older torch that
conflicts with the main environment.

| Model | Official repo | Backbone | 4 GB status |
|---|---|---|---|
| Mask2Anomaly | `shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation` | Swin-L | Does **not** fit at native 2048x1024. Tiled half-res. |
| RbA | `NazirNayal8/RbA` | Swin-L / Swin-B | Same. Swin-B variant may fit tiled at full res — test first. |
| JSR-Net | `vojirt/JSRNet` | ResNet-101 | Lighter; likely fits. Older torch pin is the real issue. |

**Recorded caveat:** running Swin-L models at half resolution will depress
small-object recall, which is one of the quantities the benchmark exists to
measure (skill.md section 7). Their small-object numbers are therefore a lower
bound, not a clean measurement, and every table reporting them must say so.
Where a result matters, re-run those images with CPU offload at full resolution
to separate the VRAM artefact from the genuine model limitation.

## PROWL / Finding DINO

Highest-priority dedicated baseline per skill.md section 3. Prototype-based on
frozen self-supervised features, so `dinov2-large` (and later `dinov3-vitl16`)
supplies the backbone; only the prototype/scoring layer is method-specific.
Compute-feasible on 4 GB. Code-release status to be confirmed before batch 2.

## Multimodal API baselines (section 5)

No download required — these are API calls, and exact model versions must be
recorded at evaluation time rather than pinned here. Keys not yet configured.
