"""Evaluation harness: find the road, then detect EVERYTHING on it --
vehicles and the rendered debris alike, with no class needed. Pluggable
Stage 2 methods, all scored on the same dataset
(datasets/synthetic-highway-debris, vehicle labels from
scripts/add_vehicle_labels.py).

Metrics, reported for all objects and separately for vehicles and debris:
  recall     share of ground-truth boxes found (IoU > HIT_IOU, one-to-one)
  precision  share of predicted boxes that are a real object
Predictions centred in an image's ignore_regions count as neither.

Methods that output a score are swept over their threshold. The reported
operating point is the best F1 of precision and the mean of vehicle and
debris recall (so the 342 vehicles cannot drown out the 30 debris), plus a
held-out check: the threshold is picked on half the images and applied to
the other half. Everything is reported both unrestricted (Stage 2 alone)
and two-stage (prediction centre inside Stage 1's road region).

    python scripts/eval_road_objects.py --method owlv2-objectness
    python scripts/eval_road_objects.py --method owlv2-objectness --rescore
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DATASET = REPO_ROOT / "datasets" / "synthetic-highway-debris"
OUT = REPO_ROOT / "results" / "road-object-eval"
HIT_IOU = 0.1


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = b
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_in_region(box, region_boxes) -> bool:
    """Center-in-any-region-box -- matches how Stage 1 coverage was already
    measured earlier (README/chat), kept consistent rather than reinvented."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in region_boxes)


def road_mask_from_boxes(road_boxes, size: tuple[int, int]) -> np.ndarray:
    """size = (width, height), matching PIL's Image.size."""
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in road_boxes:
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return mask


# ---------------------------------------------------------------- Stage 2 methods
class GdinoObjects:
    """Stage 2 as detection, not anomaly-scoring: Grounding DINO itself,
    prompted for the actual object vocabulary (vehicles + every debris
    category), since it already IS a detector -- DINOv2's prototype-distance
    was only ever built to flag "the one thing that differs most", not
    enumerate several different known object types at once."""
    name = "gdino-objects"

    def __init__(self, prompt: str, threshold: float = 0.25, max_side: int = 1024):
        import transformers
        import torch
        self.torch = torch
        p = REPO_ROOT / "models" / "grounding-dino-tiny"
        self.proc = transformers.AutoProcessor.from_pretrained(str(p))
        self.m = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(p), dtype=torch.float32).to("cuda").eval()
        self.prompt = prompt
        self.threshold = threshold
        self.max_side = max_side

    def __call__(self, img: Image.Image) -> list[tuple[int, int, int, int]]:
        sc = min(1.0, self.max_side / max(img.size))
        small = img.resize((int(img.width * sc), int(img.height * sc))) if sc < 1 else img
        inp = self.proc(images=small, text=self.prompt, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.m(**inp)
        r = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.threshold, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        boxes = []
        for b in r["boxes"].cpu().numpy():
            x0, y0, x1, y1 = (int(max(0, v)) for v in b)
            boxes.append((x0, y0, x1 - x0, y1 - y0))
        return boxes

    def predict_both(self, img: Image.Image, road_boxes: list):
        """GDINO detects over the whole image regardless of region; "in
        road" is just which of those boxes' centres happen to fall inside
        Stage 1's region -- a post-hoc filter, not a re-detection."""
        all_boxes = self(img)
        in_road = [b for b in all_boxes if box_in_region(b, road_boxes)]
        return all_boxes, in_road


OBJECT_PROMPT = ("car. truck. bus. tire. wheel. cardboard box. suitcase. "
                  "traffic cone. barrel. wooden pallet. mattress. trash can. "
                  "wooden plank. ladder. refrigerator.")


class Dinov2Multiblob:
    """Class-agnostic: Stage 2 does not need to know WHAT an object is, only
    THAT something is there -- a named-class prompt (GdinoObjects above)
    violates that, since it can only ever find objects whose name was listed.
    This reuses benchmark.py's Prowl (DINOv2 prototype-distance) exactly as
    already validated on the real benchmark, only changing what happens
    AFTER scoring: the first version of this evaluation (and the old
    inference demo script, since removed) took just the SINGLE largest connected component
    above threshold as "the detection", which is what produced 0% recall --
    a real car elsewhere in frame, correctly ranked as anomalous by DINOv2,
    is almost always a bigger blob than the small debris object, so it
    always won that single slot even when the debris was *also* correctly
    ranked highly. Taking every component above threshold as its own
    candidate, instead of only the largest, is the direct fix -- still
    exactly the same class-agnostic scoring, just not throwing away every
    detection but one."""
    name = "dinov2-multiblob"

    def __init__(self, road_frac: float = 0.03, min_component: int = 20, side: int = 896):
        # Tried side=1568 (vs. the benchmark's own 896 default) on the
        # theory that these UAV frames, up to 3840px wide, downsample more
        # than the ground-level B1/B2 photos 896 was tuned on -- measured
        # 0.99GB peak VRAM, well inside budget, but recall did not improve
        # (if anything slightly worse on a 5-image check) -- reverted.
        # Resolution was not the bottleneck here.
        from benchmark import Prowl, calibrate
        self.prowl = Prowl("dinov2-large", side=side)
        self.calibrate = calibrate
        self.road_frac = road_frac
        self.min_component = min_component

    def score(self, img: Image.Image) -> np.ndarray:
        return self.prowl(img)

    @staticmethod
    def components(mask: np.ndarray, min_component: int) -> list[tuple[int, int, int, int]]:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        out = []
        for sl in ndimage.find_objects(lab):
            m = lab[sl] > 0
            if int(m.sum()) >= min_component:
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                out.append((x0, y0, x1 - x0, y1 - y0))
        return out

    def detect_in_region(self, scores: np.ndarray, region_mask: np.ndarray) -> list:
        if not region_mask.any():
            return []
        tau = self.calibrate(scores[region_mask], self.road_frac)
        detected = (scores >= tau) & region_mask
        return self.components(detected, self.min_component)

    def predict_both(self, img: Image.Image, road_boxes: list):
        """Two independent detections, not one filtered by the other: the
        whole-image one is thresholded against the WHOLE image's own score
        distribution (self-calibrated), the road one against just the
        road region's -- a different, tighter threshold, since restricting
        the search area first is the whole point of the two-stage design,
        not just a filter applied after the fact."""
        scores = self.score(img)
        whole_mask = np.ones(scores.shape, dtype=bool)
        road_mask = road_mask_from_boxes(road_boxes, img.size)
        if road_mask.shape != scores.shape:
            road_mask = np.array(Image.fromarray(road_mask).resize(
                scores.shape[::-1], Image.NEAREST)).astype(bool)
        unrestricted = self.detect_in_region(scores, whole_mask)
        in_road = self.detect_in_region(scores, road_mask)
        return unrestricted, in_road


class Dinov2Tiled:
    """Root cause found by directly looking at SAM's and DINOv2's actual
    output (not just the recall number): these frames run up to 3840px, and
    every method so far has had to downscale the WHOLE frame to fit either
    the model's native input size or this 4GB GPU's budget -- SAM's masks
    on a downscaled 1024px frame turned out to be a handful of huge
    background-region blobs (the whole pavement, the whole gantry), not
    individual vehicles; DINOv2's patch grid has the same problem at any
    single global resolution. Splitting the image into overlapping tiles
    and running Prowl on each tile at its OWN native-ish resolution -- no
    single whole-frame downscale -- is the direct fix, and a bonus: each
    tile gets its own local prototype instead of one global one, which is
    arguably more faithful to "self-calibrated, locally normal" than a
    single prototype for a frame that mixes pavement, trees and buildings.
    Still fully class-agnostic -- no prompt, no object names."""
    name = "dinov2-tiled"

    def __init__(self, tile: int = 896, overlap: int = 128,
                 road_frac: float = 0.25, min_component: int = 20):
        # road_frac raised from Dinov2Multiblob's 0.03: swept 0.03-0.40 on a
        # 5-image check (5.0/11.2/13.8/22.5/18.8% recall at
        # 0.03/0.08/0.15/0.25/0.40) -- 0.25 was the peak before components
        # start re-merging into oversized blobs at higher fractions, same
        # failure mode as the untiled version at any threshold.
        from benchmark import Prowl, calibrate
        self.prowl = Prowl("dinov2-large", side=tile)
        self.calibrate = calibrate
        self.tile = tile
        self.overlap = overlap
        self.road_frac = road_frac
        self.min_component = min_component

    def _tile_origins(self, size: int, tile: int, overlap: int) -> list[int]:
        if size <= tile:
            return [0]
        step = tile - overlap
        origins = list(range(0, size - tile + 1, step))
        if origins[-1] + tile < size:
            origins.append(size - tile)
        return origins

    def score_full(self, img: Image.Image) -> np.ndarray:
        """Per-tile score map stitched back to full-image coordinates,
        overlaps averaged. This is the shared expensive step; road- and
        whole-image detection both threshold this same map."""
        w, h = img.size
        xs = self._tile_origins(w, self.tile, self.overlap)
        ys = self._tile_origins(h, self.tile, self.overlap)
        acc = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        for y0 in ys:
            for x0 in xs:
                tw, th = min(self.tile, w - x0), min(self.tile, h - y0)
                crop = img.crop((x0, y0, x0 + tw, y0 + th))
                s = self.prowl(crop)
                if s.shape != (th, tw):
                    s = np.array(Image.fromarray(s).resize((tw, th), Image.BILINEAR))
                acc[y0:y0 + th, x0:x0 + tw] += s
                cnt[y0:y0 + th, x0:x0 + tw] += 1
        return acc / np.maximum(cnt, 1)

    @staticmethod
    def components(mask: np.ndarray, min_component: int) -> list[tuple[int, int, int, int]]:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        out = []
        for sl in ndimage.find_objects(lab):
            m = lab[sl] > 0
            if int(m.sum()) >= min_component:
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                out.append((x0, y0, x1 - x0, y1 - y0))
        return out

    def predict_both(self, img: Image.Image, road_boxes: list):
        scores = self.score_full(img)
        whole_mask = np.ones(scores.shape, dtype=bool)
        road_mask = road_mask_from_boxes(road_boxes, img.size)

        def detect(region_mask):
            if not region_mask.any():
                return []
            tau = self.calibrate(scores[region_mask], self.road_frac)
            detected = (scores >= tau) & region_mask
            return self.components(detected, self.min_component)

        return detect(whole_mask), detect(road_mask)


class SamEverything:
    """Also class-agnostic, but a different mechanism than PROWL's single-
    prototype distance: SAM is trained to propose EVERY distinct object in
    an image via a dense grid of point prompts, with no assumption that
    "normal" is the majority of the frame -- unlike PROWL, which measures
    distance from one prototype built out of "whatever dominates the
    frame", so a scene with dozens of similar-looking cars can partly
    absorb them into "normal" itself (see Dinov2Multiblob's docstring / the
    project's own note that PROWL assumes anomalies are a frame minority).
    SAM has no such assumption -- every mask competes on its own "is this a
    coherent object" score, not on how different it looks from the rest.

    Every synthetic image here can run past 3840px, but SAM2 was trained at
    1024px and the automatic mask generator's own postprocessing scales
    with resolution -- OOM'd outright on a 3840x2160 frame on this 4GB GPU
    before being resized down first; resizing to the model's own native
    1024 side fixed both the OOM and kept per-image time ~25s regardless of
    source resolution. Predicted boxes are scaled back to source-image
    pixel coordinates before scoring."""
    name = "sam-everything"

    def __init__(self, side: int = 1024, points_per_batch: int = 64):
        import torch
        from transformers import pipeline
        self.gen = pipeline("mask-generation", model=str(REPO_ROOT / "models" / "sam2.1-hiera-base-plus"),
                            device=0, dtype=torch.float32)
        self.side = side
        self.points_per_batch = points_per_batch

    def masks_to_boxes(self, masks, scale: float) -> list[tuple[int, int, int, int]]:
        boxes = []
        for mask in masks:
            arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
            ys, xs = np.where(arr)
            if xs.size == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            boxes.append((int(x0 / scale), int(y0 / scale),
                          int((x1 - x0 + 1) / scale), int((y1 - y0 + 1) / scale)))
        return boxes

    def predict_both(self, img: Image.Image, road_boxes: list):
        sc = min(1.0, self.side / max(img.size))
        small = img.resize((int(img.width * sc), int(img.height * sc))) if sc < 1 else img
        out = self.gen(small, points_per_batch=self.points_per_batch)
        all_boxes = self.masks_to_boxes(out["masks"], sc)
        in_road = [b for b in all_boxes if box_in_region(b, road_boxes)]
        return all_boxes, in_road


class Owlv2Objectness:
    """Class-agnostic by construction: OWLv2's objectness head scores "is
    this box a whole object" with no class list and no prompt, which is
    exactly Stage 2's job (the project spec: the class is never needed).

    Tiled for the same root cause Dinov2Tiled found: the frames are up to
    3840 px and the targets are 5-150 px, so one whole-frame pass at the
    model's 960 px input shrinks a small object below one 16 px patch.
    Each square tile is `tile_frac` of the image's short side, padded to a
    square and resized to 960, so small frames are upsampled and large ones
    are seen near native resolution. Tiles overlap, so an object cut by one
    tile's edge is whole in a neighbour; the merge step then drops the
    fragment."""
    name = "owlv2-objectness"

    def __init__(self, tile_fracs=(0.5,), overlap: float = 0.25,
                 max_per_tile: int = 300, max_box_frac: float = 0.35):
        import torch
        import transformers
        p = REPO_ROOT / "models" / "owlv2-base-ensemble"
        self.torch = torch
        self.m = transformers.Owlv2ForObjectDetection.from_pretrained(
            str(p), dtype=torch.float16).to("cuda").eval()
        cfg = json.loads((p / "preprocessor_config.json").read_text())
        self.mean = torch.tensor(cfg["image_mean"], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["image_std"], device="cuda").view(1, 3, 1, 1)
        self.side = cfg["size"]["height"]
        self.tile_fracs = tile_fracs
        self.overlap = overlap
        self.max_per_tile = max_per_tile
        self.max_box_frac = max_box_frac

    @staticmethod
    def _origins(size: int, tile: int, step: int) -> list[int]:
        if size <= tile:
            return [0]
        o = list(range(0, size - tile + 1, step))
        if o[-1] + tile < size:
            o.append(size - tile)
        return o

    def _tile(self, crop: Image.Image, tile: int):
        torch = self.torch
        sq = Image.new("RGB", (tile, tile), (128, 128, 128))
        sq.paste(crop, (0, 0))
        arr = np.asarray(sq.resize((self.side, self.side), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].cuda()
        x = ((x - self.mean) / self.std).half()
        with torch.inference_mode():
            fmap, _ = self.m.image_embedder(pixel_values=x)
            b, h, w, d = fmap.shape
            feats = fmap.reshape(b, h * w, d)
            obj = self.m.objectness_predictor(feats)[0].float()
            box = self.m.box_predictor(feats, fmap)[0].float()
        score = obj.sigmoid()
        k = min(self.max_per_tile, score.numel())
        top = score.topk(k).indices
        cx, cy, bw, bh = (box[top] * tile).unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        return xyxy.cpu().numpy(), score[top].cpu().numpy()

    def detect(self, img: Image.Image) -> list[tuple[int, int, int, int, float]]:
        import torchvision
        torch = self.torch
        W, H = img.size
        allb, alls = [], []
        for frac in self.tile_fracs:
            tile = max(64, int(min(W, H) * frac))
            step = max(1, int(tile * (1 - self.overlap)))
            for y0 in self._origins(H, tile, step):
                for x0 in self._origins(W, tile, step):
                    crop = img.crop((x0, y0, min(W, x0 + tile), min(H, y0 + tile)))
                    xyxy, s = self._tile(crop, tile)
                    cw, ch = crop.size
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, cw)
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, ch)
                    bw = xyxy[:, 2] - xyxy[:, 0]
                    bh = xyxy[:, 3] - xyxy[:, 1]
                    # A box covering a large part of the tile is a lane, a
                    # verge or a gantry, not a vehicle or a piece of debris.
                    ok = (bw > 2) & (bh > 2) & (bw < self.max_box_frac * tile) & \
                         (bh < self.max_box_frac * tile)
                    xyxy = xyxy[ok] + np.array([x0, y0, x0, y0], dtype=np.float32)
                    allb.append(xyxy)
                    alls.append(s[ok])
        boxes = torch.from_numpy(np.concatenate(allb))
        scores = torch.from_numpy(np.concatenate(alls))
        keep = torchvision.ops.nms(boxes, scores, 0.4)
        boxes, scores = boxes[keep].numpy(), scores[keep].numpy()
        # Drop a box mostly inside a higher-scoring one: tile-edge fragments,
        # and a wheel or a windscreen proposed separately from its vehicle.
        out = []
        kept_boxes: list[np.ndarray] = []
        for b, s in zip(boxes, scores):
            ab = (b[2] - b[0]) * (b[3] - b[1])
            frag = False
            for k in kept_boxes:
                ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
                iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
                if ix * iy > 0.7 * ab:
                    frag = True
                    break
            if frag:
                continue
            kept_boxes.append(b)
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            out.append((x0, y0, x1 - x0, y1 - y0, float(s)))
            if len(out) >= 1500:
                break
        return out

    def predict_both(self, img: Image.Image, road_boxes: list):
        """Detection runs on the whole frame; "in road" keeps the boxes whose
        centre falls in Stage 1's road region."""
        all_boxes = self.detect(img)
        return all_boxes, [b for b in all_boxes if box_in_region(b[:4], road_boxes)]


class Owlv2Scored:
    """OWLv2 objectness, v2: three changes, each from the failure analysis of
    the first version (Owlv2Objectness above).

    1. Size-routed scales. Each scale keeps only boxes in its own size band,
       as a fraction of its tile side. The half-short-side tiles handle
       small and medium objects; a coarse pass over the whole short side
       handles what the fine tiles cannot hold. The first version dropped
       every box longer than 0.35 of a tile (to drop lanes), which silently
       discarded long trucks and car transporters: a 1400 px transporter
       never had a box at all.
    2. Known-background penalty. OWLv2's class head scores each box against
       text queries for things that are NOT targets: street lamps, signs,
       gantries, traffic lights, cabinets, markings, vegetation. The target
       side stays class-agnostic -- no debris or vehicle name is used -- the
       model is only told what to ignore. A person is deliberately not on
       the list: a pedestrian on a motorway is a real hazard.
       score = objectness * (1 - background probability) ** gamma
    3. Every box keeps its raw parts: [x, y, w, h, score, objectness,
       background probability, index of the background query, scale index],
       so the penalty can be re-tuned offline without re-running the model.
    """
    name = "owlv2-scored"

    BACKGROUND = [
        "a street lamp", "a lamp post", "a traffic sign", "a road sign",
        "an overhead sign gantry", "a traffic light", "an electrical cabinet",
        "a road marking", "a painted arrow on the road", "a manhole cover",
        "a guard rail", "a fence", "a tree", "a bush", "grass", "a pole",
    ]

    def __init__(self, scales=((0.5, 0.0, 0.35), (1.0, 0.15, 0.9)), overlap: float = 0.25,
                 max_per_tile: int = 300, gamma: float = 0.0):
        import torch
        import transformers
        p = REPO_ROOT / "models" / "owlv2-base-ensemble"
        self.torch = torch
        self.m = transformers.Owlv2ForObjectDetection.from_pretrained(
            str(p), dtype=torch.float16).to("cuda").eval()
        cfg = json.loads((p / "preprocessor_config.json").read_text())
        self.mean = torch.tensor(cfg["image_mean"], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["image_std"], device="cuda").view(1, 3, 1, 1)
        self.side = cfg["size"]["height"]
        self.scales = scales
        self.overlap = overlap
        self.max_per_tile = max_per_tile
        self.gamma = gamma
        proc = transformers.Owlv2Processor.from_pretrained(str(p))
        tok = proc(text=self.BACKGROUND, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = self.m.owlv2.text_model(input_ids=tok["input_ids"],
                                          attention_mask=tok["attention_mask"])
            self.bg_embeds = self.m.owlv2.text_projection(out.pooler_output)[None]

    _origins = staticmethod(Owlv2Objectness._origins)

    def _tile(self, crop: Image.Image, tile: int):
        torch = self.torch
        sq = Image.new("RGB", (tile, tile), (128, 128, 128))
        sq.paste(crop, (0, 0))
        arr = np.asarray(sq.resize((self.side, self.side), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].cuda()
        x = ((x - self.mean) / self.std).half()
        with torch.inference_mode():
            fmap, _ = self.m.image_embedder(pixel_values=x)
            b, h, w, d = fmap.shape
            feats = fmap.reshape(b, h * w, d)
            obj = self.m.objectness_predictor(feats)[0].float().sigmoid()
            box = self.m.box_predictor(feats, fmap)[0].float()
            k = min(self.max_per_tile, obj.numel())
            top = obj.topk(k).indices
            logits, cls_emb = self.m.class_predictor(feats[:, top], self.bg_embeds.to(feats.dtype))
            bgp = logits[0].float().sigmoid()
            bg_max, bg_idx = bgp.max(-1)
        cx, cy, bw, bh = (box[top] * tile).unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        return (xyxy.cpu().numpy(), obj[top].cpu().numpy(), bg_max.cpu().numpy(),
                bg_idx.cpu().numpy(), cls_emb[0].float().cpu().numpy())

    def detect(self, img: Image.Image, with_embeddings: bool = False):
        """Boxes as [x, y, w, h, score, objectness, bg prob, bg query, scale].
        With `with_embeddings`, also returns each box's 512-d OWLv2 class
        embedding (the input of the re-scorer, train_rescorer.py)."""
        import torchvision
        torch = self.torch
        W, H = img.size
        rows, embs = [], []
        for si, (frac, lo, hi) in enumerate(self.scales):
            tile = max(64, int(min(W, H) * frac))
            step = max(1, int(tile * (1 - self.overlap)))
            for y0 in self._origins(H, tile, step):
                for x0 in self._origins(W, tile, step):
                    crop = img.crop((x0, y0, min(W, x0 + tile), min(H, y0 + tile)))
                    xyxy, obj, bgm, bgi, emb = self._tile(crop, tile)
                    cw, ch = crop.size
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, cw)
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, ch)
                    bw = xyxy[:, 2] - xyxy[:, 0]
                    bh = xyxy[:, 3] - xyxy[:, 1]
                    long_side = np.maximum(bw, bh)
                    ok = (bw > 2) & (bh > 2) & (long_side >= lo * tile) & (long_side < hi * tile)
                    xyxy = xyxy[ok] + np.array([x0, y0, x0, y0], dtype=np.float32)
                    score = obj[ok] * (1 - bgm[ok]) ** self.gamma
                    for b, s, o, g, gi in zip(xyxy, score, obj[ok], bgm[ok], bgi[ok]):
                        rows.append([*b.tolist(), float(s), float(o), float(g), int(gi), si])
                    embs.append(emb[ok].astype(np.float16))
        if not rows:
            return ([], np.zeros((0, 512), np.float16)) if with_embeddings else []
        arr = np.array(rows, dtype=np.float64)
        emb_all = np.concatenate(embs)
        keep = torchvision.ops.nms(torch.from_numpy(arr[:, :4]).float(),
                                   torch.from_numpy(arr[:, 4]).float(), 0.4).numpy()
        arr, emb_all = arr[keep], emb_all[keep]
        out, kept, out_emb = [], [], []
        for r, e in zip(arr, emb_all):
            b = r[:4]
            ab = (b[2] - b[0]) * (b[3] - b[1])
            # Drop a fragment of an object already kept (a tile-edge cut, a
            # windscreen): mostly inside a kept box of similar size. Without
            # the size cap a coarse-scale box around a stretch of road
            # swallowed the debris lying in it.
            if any(max(0.0, min(b[2], k[2]) - max(b[0], k[0])) *
                   max(0.0, min(b[3], k[3]) - max(b[1], k[1])) > 0.7 * ab and
                   (k[2] - k[0]) * (k[3] - k[1]) < 6 * ab for k in kept):
                continue
            kept.append(b)
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            out.append([x0, y0, x1 - x0, y1 - y0, float(r[4]), float(r[5]), float(r[6]),
                        int(r[7]), int(r[8])])
            out_emb.append(e)
            if len(out) >= 1500:
                break
        if with_embeddings:
            return out, np.array(out_emb, dtype=np.float16).reshape(-1, emb_all.shape[1])
        return out

    def predict_both(self, img: Image.Image, road_boxes: list):
        all_boxes = self.detect(img)
        return all_boxes, [b for b in all_boxes if box_in_region(b[:4], road_boxes)]


METHODS = {
    "gdino-objects": lambda: GdinoObjects(OBJECT_PROMPT),
    "dinov2-multiblob": Dinov2Multiblob,
    "dinov2-tiled": Dinov2Tiled,
    "sam-everything": SamEverything,
    "owlv2-objectness": Owlv2Objectness,
    "owlv2-objectness-2scale": lambda: Owlv2Objectness(tile_fracs=(0.5, 0.25)),
    "owlv2-objectness-fine": lambda: Owlv2Objectness(tile_fracs=(0.25,)),
    "owlv2-scored": Owlv2Scored,
    # Produced by scripts/train_rescorer.py, scored here with --rescore.
    "owlv2-rescored": lambda: (_ for _ in ()).throw(
        SystemExit("owlv2-rescored comes from scripts/train_rescorer.py")),
    "owlv2-fused": lambda: (_ for _ in ()).throw(
        SystemExit("owlv2-fused comes from scripts/train_rescorer.py")),
}


# ---------------------------------------------------------------- scoring
def centre_in_any(box, regions) -> bool:
    x, y, w, h = box[:4]
    cx, cy = x + w / 2, y + h / 2
    return any(rx <= cx < rx + rw and ry <= cy < ry + rh for rx, ry, rw, rh in regions)


def match(gts: list, preds: list, thr: float) -> tuple[list[bool], int]:
    """One-to-one greedy matching, highest score first, IoU > HIT_IOU.
    Returns (hit flag per ground-truth box, number of predictions used).
    One-to-one matters for precision: with "any overlap counts" three
    boxes on one car would all be scored correct."""
    use = sorted((p for p in preds if (p[4] if len(p) > 4 else 1.0) >= thr),
                 key=lambda p: -(p[4] if len(p) > 4 else 1.0))
    hit = [False] * len(gts)
    for p in use:
        best, best_i = HIT_IOU, -1
        for i, g in enumerate(gts):
            if hit[i]:
                continue
            v = iou(g["bbox"], p[:4])
            if v > best:
                best, best_i = v, i
        if best_i >= 0:
            hit[best_i] = True
    return hit, len(use)


def score_at(img_ids, gt_by_img, preds_by_img, ignore_by_img, thr: float,
             which: int) -> dict:
    """Recall and precision at one score threshold. `which` picks the
    prediction list: 0 = unrestricted, 1 = inside Stage 1's road region.
    Predictions centred in an ignore region are neither hits nor false alarms."""
    n = {"all": 0, "vehicle": 0, "debris": 0}
    h = {"all": 0, "vehicle": 0, "debris": 0}
    tp = fp = 0
    for img_id in img_ids:
        gts = gt_by_img.get(img_id, [])
        ign = ignore_by_img.get(img_id, [])
        preds = [p for p in preds_by_img[img_id][which] if not centre_in_any(p, ign)]
        hit, used = match(gts, preds, thr)
        tp += sum(hit)
        fp += used - sum(hit)
        for g, ok in zip(gts, hit):
            grp = "vehicle" if g["category"] == "vehicle" else "debris"
            for k in ("all", grp):
                n[k] += 1
                h[k] += int(ok)
    out = {f"recall_{k}": (h[k] / n[k] if n[k] else None) for k in n}
    out.update({f"n_{k}": n[k] for k in n})
    out["precision"] = tp / (tp + fp) if tp + fp else None
    out["predictions_per_image"] = (tp + fp) / len(img_ids)
    return out


THRESHOLD_MAX = None  # set from a run's saved "threshold_max" (fused runs)


def sweep(img_ids, gt_by_img, preds_by_img, ignore_by_img, which: int) -> list[dict]:
    scores = sorted({round(p[4], 4) for i in img_ids for p in preds_by_img[i][which] if len(p) > 4
                     and (THRESHOLD_MAX is None or p[4] <= THRESHOLD_MAX)})
    if not scores:
        return [dict(threshold=0.0, **score_at(img_ids, gt_by_img, preds_by_img,
                                                ignore_by_img, 0.0, which))]
    grid = sorted(set(np.quantile(scores, np.linspace(0, 0.995, 60)).round(4).tolist()))
    return [dict(threshold=t, **score_at(img_ids, gt_by_img, preds_by_img, ignore_by_img, t, which))
            for t in grid]


def best_f1(rows: list[dict]) -> dict:
    """Operating point: best F1 of precision and the MEAN of vehicle and
    debris recall. Plain all-object recall is 92% vehicles here (342 of
    372), so maximising it picks a threshold that ignores debris."""
    def f1(r):
        p = r["precision"] or 0
        rc = ((r["recall_vehicle"] or 0) + (r["recall_debris"] or 0)) / 2
        return 2 * p * rc / (p + rc) if p + rc else 0
    return max(rows, key=f1)


def stage1_roads(img_ids, images, fallback: bool) -> dict:
    """Stage 1 road boxes per image, cached: Stage 1 does not change between
    Stage 2 experiments, and it is the slow step. The cache key includes each
    image's size and mtime, so a re-rendered sample is recomputed."""
    cache_f = OUT / "stage1_roads.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.is_file() else {}

    def key(img_id):
        f = DATASET / "images" / images[img_id]["file_name"]
        st = f.stat()
        return f"{f.name}:{st.st_size}:{int(st.st_mtime)}:{int(fallback)}"

    todo = [i for i in img_ids if key(i) not in cache]
    if todo:
        import torch
        print(f"Stage 1: Grounding DINO road localization on {len(todo)} images ...")
        from segment_benchmark import GDinoRoad
        stage1 = GDinoRoad()
        for img_id in todo:
            img = Image.open(DATASET / "images" / images[img_id]["file_name"]).convert("RGB")
            cache[key(img_id)] = [list(b) for b, _ in stage1.detect(img)]
        del stage1
        torch.cuda.empty_cache()
        zero = [i for i in todo if not cache[key(i)]]
        if fallback and zero:
            # GDINO returning zero boxes on an obvious highway is a known
            # failure mode; SegFormer-B2 fills in for those frames only,
            # loaded after GDINO is freed so the two never share the 4 GB.
            print(f"Stage 1 fallback: SegFormer-B2 on {len(zero)} zero-detection frame(s)")
            from segment_benchmark import SegformerRoad
            from scipy import ndimage
            seg = SegformerRoad("segformer-b2-cityscapes")
            for img_id in zero:
                img = Image.open(DATASET / "images" / images[img_id]["file_name"]).convert("RGB")
                lab, _ = ndimage.label(seg(img))
                boxes = []
                for sl in ndimage.find_objects(lab):
                    if (lab[sl] > 0).sum() >= 200:
                        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
                cache[key(img_id)] = boxes
            del seg
            torch.cuda.empty_cache()
        OUT.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache), encoding="utf-8")
    return {i: cache[key(i)] for i in img_ids}


STAGE1_MODEL = REPO_ROOT / "models" / "segformer-b2-aeroscapes-road"
STAGE1_CHOICES = ("aeroscapes-mask", "gdino-boxes")
MASK_DOWNSAMPLE = 4
MASK_VERSION = "v3"  # v2: no border erosion; v3: morphology at stored resolution


def road_mask(prob: np.ndarray) -> np.ndarray:
    """Drivable-surface probability (full resolution) -> Stage 1 region at
    1/MASK_DOWNSAMPLE resolution.

    1. prob > 0.5.
    2. Closing, then fill holes: a vehicle or a piece of debris the
       segmenter calls "not road" is a hole inside the road, and the whole
       point of Stage 1 is to keep it.
    3. Dilate by 1.5% of the long side, so an object on the hard shoulder or
       half over a lane edge still has its centre inside."""
    from scipy import ndimage
    # Work at 1/MASK_DOWNSAMPLE, the resolution the mask is stored and used
    # at. At full 3840x2160 the morphology took 40 s per frame on the CPU,
    # ten times the detector's own time.
    prob = prob[::MASK_DOWNSAMPLE, ::MASK_DOWNSAMPLE]
    h, w = prob.shape
    k = max(3, int(0.015 * max(h, w)))
    # Edge-replicate before the closing. scipy's erosion treats outside the
    # frame as background, so the closing ate 2k px of road along every
    # border: vehicles cut by the top edge of the frame fell outside Stage 1.
    pad = 3 * k
    m = np.pad(prob > 0.5, pad, mode="edge")
    m = ndimage.binary_closing(m, structure=np.ones((k, k)), iterations=2)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_dilation(m, structure=np.ones((k, k)))
    return m[pad:-pad, pad:-pad]


def stage1_masks(img_ids, images, dataset: Path = DATASET) -> dict[int, np.ndarray]:
    """Stage 1 region per image from the drone-view road segmenter
    (scripts/train_road_segmenter.py), stored at 1/MASK_DOWNSAMPLE
    resolution. Cached like stage1_roads()."""
    cache_dir = OUT / ("stage1_masks" if dataset == DATASET else f"stage1_masks_{dataset.name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, model = {}, None
    for img_id in img_ids:
        f = dataset / "images" / images[img_id]["file_name"]
        st = f.stat()
        cf = cache_dir / f"{f.stem}_{st.st_size}_{int(st.st_mtime)}_{MASK_VERSION}.npz"
        if cf.is_file():
            out[img_id] = np.load(cf)["mask"]
            continue
        if model is None:
            import torch
            import transformers
            from train_road_segmenter import predict
            model = transformers.SegformerForSemanticSegmentation.from_pretrained(
                str(STAGE1_MODEL)).cuda().eval()
            print("Stage 1: drone-view road segmenter ...")
        img = Image.open(f).convert("RGB")
        m = road_mask(predict(model, img))
        np.savez_compressed(cf, mask=m)
        out[img_id] = m
    if model is not None:
        import torch
        del model
        torch.cuda.empty_cache()
    return out


def centre_in_mask(box, mask: np.ndarray) -> bool:
    x, y, w, h = box[:4]
    cy = int((y + h / 2) / MASK_DOWNSAMPLE)
    cx = int((x + w / 2) / MASK_DOWNSAMPLE)
    return 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and bool(mask[cy, cx])


RING_MIN_ROAD = 0.5


def on_road(box, mask: np.ndarray) -> bool:
    """Centre on the road, or at least RING_MIN_ROAD of a ring around the box
    (the box grown by a quarter of its size on each side, minus the box) is
    road. The ring catches what the centre test misses: the segmenter was
    trained on AeroScapes, whose vehicles are cars, so a truck's body is a
    hole in the road mask, and a truck cut by the frame edge is a hole that
    fill-holes cannot close."""
    if centre_in_mask(box, mask):
        return True
    x, y, w, h = (v / MASK_DOWNSAMPLE for v in box[:4])
    H, W = mask.shape
    gx, gy = max(1.0, w / 4), max(1.0, h / 4)
    X0, Y0 = int(max(0, x - gx)), int(max(0, y - gy))
    X1, Y1 = int(min(W, x + w + gx + 1)), int(min(H, y + h + gy + 1))
    if X1 <= X0 or Y1 <= Y0:
        return False
    outer = mask[Y0:Y1, X0:X1]
    inner = mask[int(max(0, y)):int(min(H, y + h + 1)), int(max(0, x)):int(min(W, x + w + 1))]
    ring_n = outer.size - inner.size
    if ring_n <= 0:
        return False
    return (int(outer.sum()) - int(inner.sum())) / ring_n >= RING_MIN_ROAD


def stage1_report(img_ids, gt_by_img, ignore_by_img, inside) -> dict:
    """Stage 1 on its own: share of labelled objects whose centre is inside
    the region (anything outside can never be found by the two-stage
    pipeline), and the share of the frame the region covers (smaller means
    fewer places for false alarms)."""
    n = {"vehicle": 0, "debris": 0}
    k = {"vehicle": 0, "debris": 0}
    for img_id in img_ids:
        for g in gt_by_img.get(img_id, []):
            if centre_in_any(g["bbox"], ignore_by_img.get(img_id, [])):
                continue
            grp = "vehicle" if g["category"] == "vehicle" else "debris"
            n[grp] += 1
            k[grp] += int(inside(img_id, g["bbox"]))
    return {f"coverage_{g}": k[g] / n[g] for g in n if n[g]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="owlv2-objectness", choices=list(METHODS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage1-fallback", action="store_true",
                    help="SegFormer-B2 fallback for frames GDINO finds zero road boxes on")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse saved predictions, only recompute the metrics")
    ap.add_argument("--stage1", default="aeroscapes-mask", choices=STAGE1_CHOICES,
                    help="aeroscapes-mask: pixel mask from the drone-view road "
                         "segmenter (default). gdino-boxes: the original Grounding "
                         "DINO road boxes, kept for comparison")
    args = ap.parse_args()
    run = args.method if args.stage1 == STAGE1_CHOICES[0] else f"{args.method}__{args.stage1}"

    anns = json.loads((DATASET / "annotations.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in anns["categories"]}
    images = {im["id"]: im for im in anns["images"]}
    ignore_by_img = {im["id"]: im.get("ignore_regions", []) for im in anns["images"]}
    gt_by_img: dict[int, list] = {}
    for a in anns["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(
            {"bbox": a["bbox"], "category": cats[a["category_id"]]})
    img_ids = sorted(images)
    if args.limit:
        img_ids = img_ids[:args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    pred_f = OUT / f"preds_{run}.json"
    if args.stage1 == "aeroscapes-mask":
        masks = stage1_masks(img_ids, images)
        inside = lambda i, b: on_road(b, masks[i])  # noqa: E731
    else:
        boxes1 = stage1_roads(img_ids, images, args.stage1_fallback)
        inside = lambda i, b: box_in_region(b[:4], boxes1[i])  # noqa: E731
    s1 = stage1_report(img_ids, gt_by_img, ignore_by_img, inside)
    s1["area_fraction"] = (float(np.mean([masks[i].mean() for i in img_ids]))
                           if args.stage1 == "aeroscapes-mask" else float(np.mean([
                               road_mask_from_boxes(boxes1[i], (images[i]["width"],
                                                                images[i]["height"])).mean()
                               for i in img_ids])))
    if args.rescore and pred_f.is_file():
        saved = json.loads(pred_f.read_text(encoding="utf-8"))
        pred_by_img = {int(k): v for k, v in saved["predictions"].items()}
        sec_per_img = saved["sec_per_image"]
        extra = {k: v for k, v in saved.items() if k not in ("predictions", "sec_per_image")}
        global THRESHOLD_MAX
        THRESHOLD_MAX = extra.get("threshold_max")
    else:
        roads = (boxes1 if args.stage1 == "gdino-boxes"
                 else stage1_roads(img_ids, images, args.stage1_fallback))
        import torch
        print(f"Stage 2: {args.method} on {len(img_ids)} images ...")
        method = METHODS[args.method]()
        t0 = time.time()
        pred_by_img = {}
        for i, img_id in enumerate(img_ids):
            img = Image.open(DATASET / "images" / images[img_id]["file_name"]).convert("RGB")
            u, r = method.predict_both(img, roads[img_id])
            pred_by_img[img_id] = [[list(p) for p in u], [list(p) for p in r]]
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(img_ids)}", flush=True)
        sec_per_img = (time.time() - t0) / len(img_ids)
        del method
        torch.cuda.empty_cache()
        extra = {}
        pred_f.write_text(json.dumps({"sec_per_image": sec_per_img,
                                      "predictions": pred_by_img}), encoding="utf-8")
    if args.stage1 == "aeroscapes-mask":
        # Two-stage = the unrestricted detections that are on the road (on_road()).
        for img_id in img_ids:
            pred_by_img[img_id][1] = [p for p in pred_by_img[img_id][0] if inside(img_id, p)]
        pred_f.write_text(json.dumps({"sec_per_image": sec_per_img, **extra,
                                      "predictions": pred_by_img}), encoding="utf-8")

    result = {"method": args.method, "stage1": args.stage1, "images": len(img_ids),
              "hit_iou": HIT_IOU, "sec_per_image": sec_per_img, "stage1_alone": s1}
    for which, label in ((0, "unrestricted"), (1, "two_stage")):
        rows = sweep(img_ids, gt_by_img, pred_by_img, ignore_by_img, which)
        best = best_f1(rows)
        # Threshold picked on half the images and applied to the other half,
        # both ways: shows whether the best-F1 threshold is overfitted to
        # these 30 images.
        halves = (img_ids[0::2], img_ids[1::2])
        held = []
        for tune, test in (halves, halves[::-1]):
            t = best_f1(sweep(tune, gt_by_img, pred_by_img, ignore_by_img, which))["threshold"]
            held.append(dict(threshold=t, **score_at(test, gt_by_img, pred_by_img,
                                                    ignore_by_img, t, which)))
        result[label] = {"best_f1": best, "held_out": held, "curve": rows,
                         "max_recall": rows[0]}

    (OUT / f"{run}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def line(r):
        pct = lambda v: "  n/a" if v is None else f"{v * 100:5.1f}%"
        return (f"thr={r['threshold']:.3f}  all {pct(r['recall_all'])}  "
                f"vehicle {pct(r['recall_vehicle'])}  debris {pct(r['recall_debris'])}  "
                f"precision {pct(r['precision'])}  boxes/img {r['predictions_per_image']:.0f}")

    speed = f"{sec_per_img:.2f} s/image" if sec_per_img else "speed not measured"
    print(f"\n=== {run}  ({speed}) ===")
    print("stage 1 alone: " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in s1.items()))
    b = result["unrestricted"]["best_f1"]
    print(f"objects: {b['n_all']} ({b['n_vehicle']} vehicle, {b['n_debris']} debris)")
    for label in ("unrestricted", "two_stage"):
        r = result[label]
        print(f"[{label}]")
        print(f"  balanced    {line(r['best_f1'])}")
        for h in r["held_out"]:
            print(f"  held-out    {line(h)}")
        print(f"  all boxes   {line(r['max_recall'])}")
    print(f"saved: {(OUT / f'{run}.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
