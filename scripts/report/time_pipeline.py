"""End-to-end timing of the full chain, from the raw frame to risk levels.

    python scripts/report/time_pipeline.py
    python scripts/report/time_pipeline.py --co-resident

Architecture a (local Qwen, context gate), staged. b and c run their VLM steps
on the API; their Objective 1 and gate times are the same as a's.

    python scripts/report/time_pipeline.py --arch d

Architecture d, live per frame, split into tier 1 (every frame, on the device:
road, detection, re-scoring, the detector's screen) and tier 2 (only frames
with a candidate: identification, gate, risk; API calls in parallel, no
answer cache).

Two ways to fit the 4 GB card:
  staged (default)  pass 1 runs Objective 1 on every frame with only its
                    models loaded; they are then unloaded and pass 2 runs the
                    VLM stages. Suits batch processing of a flight's frames.
  --co-resident     every model loaded once, one frame at a time through all
                    stages. What a live system needs; measured to peak above
                    4 GB here (Windows spills to shared memory instead of
                    failing), so it is reported but not recommended.

Stages timed per frame: road (segmenter + mask), detect (OWLv2 tiles + box
features), rescore (probe + fusion + road filter), identify (VLM, incl.
physics re-asks), context (the gate incl. part questions), risk (VLM).

Uses the deployable re-scorer (models/owlv2-rescorer, trained on all four
locations), so boxes can differ slightly from the evaluation runs, which use
leave-one-location-out probes. Accuracy comes from those runs; this script is
for time and memory. Output: results/risk-assessment/a_qwen_local/timing.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
import assess_risk as A  # noqa: E402
import eval_road_objects as E  # noqa: E402
import train_rescorer as T  # noqa: E402

STAGES = ["road", "detect", "rescore", "identify", "context", "risk"]
# Architecture d: tier 1 runs on every frame; tier 2 only when tier 1 flags a candidate.
TIER1 = ["road_detect", "rescore", "screen"]  # road and detection run in parallel
TIER2 = ["identify", "context", "risk"]
HEDGE_S = 4.0  # re-send an API request not answered within this (see api_vlm.ApiVLM)


def run_fast(args) -> int:
    """Architecture d, live: every frame goes through tier 1 (road, detection,
    re-scoring, the detector's screen) on the device; tier 2 (identification,
    context gate, risk) runs only when the screen finds a candidate, with the
    API calls for a frame's candidates made in parallel. No answer cache, so
    the times include the network."""
    from concurrent.futures import ThreadPoolExecutor

    import torch
    import transformers
    from api_vlm import ApiVLM
    from train_road_segmenter import OUT as SEG_DIR, predict

    images, gt, loc = A.load(A.TEST)
    ids = sorted(images)[: args.limit or None]
    ppm = A.location_ppm()
    thr = A.operating_threshold()
    sel = json.loads(A.SELECTION.read_text(encoding="utf-8"))
    variant = json.loads(max(sel, key=lambda r: r["score"])["variant"])
    opinion = A.DetectorOpinion()
    z = np.load(T.PROBE_DIR / "probe.npz")
    load = {}
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    seg = transformers.SegformerForSemanticSegmentation.from_pretrained(str(SEG_DIR)).cuda().eval()
    load["road_segmenter"] = round(time.time() - t, 1)
    t = time.time()
    det = E.Owlv2Scored()
    load["owlv2"] = round(time.time() - t, 1)
    pool_warm = ThreadPoolExecutor(4)
    vlm = ApiVLM(A.API_PROVIDER, A.API_MODEL, min_interval_s=0.0, cache=False,
                 hedge_after_s=HEDGE_S)
    # Connect once at start-up, as a deployed system would; the lookup and the
    # connections are then reused (see api_vlm.ApiVLM).
    t = time.time()
    list(pool_warm.map(lambda _: vlm.client.models.retrieve(A.API_MODEL), range(4)))
    load["api_connect"] = round(time.time() - t, 1)
    pool = ThreadPoolExecutor(8)
    per = {}
    for i in ids:
        img = Image.open(A.TEST / "images" / images[i]["file_name"]).convert("RGB")
        ign = images[i].get("ignore_regions", [])
        g = ppm[loc[i]]
        tt = {}
        # ---------------- tier 1: on the device, every frame
        # Road segmentation and detection do not depend on each other: the
        # road mask is only needed afterwards, by the on-road filter. They run
        # side by side; "road" and "detect" are each one's own duration.
        torch.cuda.synchronize()
        t0 = time.time()

        def road_job():
            t = time.time()
            m = E.road_mask(predict(seg, img))
            torch.cuda.synchronize()
            return m, time.time() - t
        fut = pool.submit(road_job)
        t1 = time.time()
        rows, emb = det.detect(img, with_embeddings=True)
        torch.cuda.synchronize()
        tt["detect"] = time.time() - t1
        mask, tt["road"] = fut.result()
        tt["road_detect"] = time.time() - t0
        t0 = time.time()
        r = np.array(rows, dtype=np.float32).reshape(-1, 9)
        keep = r[:, 5] >= T.MIN_OBJ
        r, e = r[keep], emb[keep].astype(np.float32)
        o = np.clip(r[:, 5], 1e-4, 1 - 1e-4)
        x = np.hstack([np.log(o / (1 - o))[:, None], e])
        p = 1 / (1 + np.exp(-(((x - z["mean"]) / z["scale"]) @ z["coef"] + z["intercept"])))
        boxes = []
        for k, (b, s) in enumerate(zip(r, p)):
            score = (1 + s) if (b[5] >= T.OBJ_GATE and s >= T.PROBE_GATE) else s
            bb = [int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(score), float(b[5])]
            if score >= thr and E.on_road(bb, mask) and not E.centre_in_any(bb, ign):
                boxes.append((bb, x[k]))
        tt["rescore"] = time.time() - t0
        t0 = time.time()
        P = (opinion.probes[loc[i]].predict_proba(np.stack([b[1] for b in boxes]))
             if boxes else np.zeros((0, 3)))
        objs = []
        for k, ((bb, _), p3) in enumerate(zip(boxes, P)):
            cand = p3.argmax() == 2 and p3[2] >= A.VETO_T
            objs.append({"id": k + 1, "box": bb[:4], "p3": p3, "cand": bool(cand),
                         "vlm_category": "debris?" if cand else A.DETECTOR_CLASS[int(p3.argmax())]})
        cands = [o for o in objs if o["cand"]]
        tt["screen"] = time.time() - t0
        # ---------------- tier 2: only when the screen found a candidate
        tt.update({"identify": 0.0, "context": 0.0, "risk": 0.0})
        survivors = []
        if cands:
            t0 = time.time()
            futs = [pool.submit(A.identify, vlm, img, o["box"], g, variant) for o in cands]
            for o, f in zip(cands, futs):
                ident = f.result()
                o.update({"vlm_category": ident["category"], "vlm_name": ident["name"]})
            tt["identify"] = time.time() - t0
            t0 = time.time()
            rec = {"width": images[i]["width"], "height": images[i]["height"], "px_per_metre": g}
            u = A.road_direction(mask)

            def ask_part(o):
                raw = vlm.ask([A.context_view(img, o["box"], 6.0, 200), A.tight_view(img, o["box"])],
                              A.PART_PROMPT, 40, prefix='{"part_of_vehicle": ')
                head = raw.split(",")[0].split(":")[-1].strip().lower()
                return head.startswith("true") or head.startswith("1")
            survivors = [o for o in cands if o["vlm_category"] in A.ASSESSED
                         and not A.context_reason(o, objs, rec, u, o["p3"], ask_part)]
            tt["context"] = time.time() - t0
            t0 = time.time()
            vehicles = [v for v in objs if v["vlm_category"] == "vehicle"]

            def nearest(o):
                c = np.array([o["box"][0] + o["box"][2] / 2, o["box"][1] + o["box"][3] / 2])
                d = [np.hypot(*(c - [v["box"][0] + v["box"][2] / 2, v["box"][1] + v["box"][3] / 2])) / g
                     for v in vehicles]
                return min(d) if d else None
            futs = [pool.submit(A.assess, vlm, img, o["box"],
                                {"name": o["vlm_name"], "category": o["vlm_category"]}, g,
                                nearest(o), len(vehicles)) for o in survivors]
            for f in futs:
                f.result()
            tt["risk"] = time.time() - t0
        tt.update({"boxes": len(boxes), "candidates": len(cands), "risk_calls": len(survivors)})
        tt["tier1"] = sum(tt[k] for k in TIER1)
        tt["tier2"] = sum(tt[k] for k in TIER2)
        per[i] = tt
        print(f"  {i:02d}: tier1 {tt['tier1']:.2f} s  tier2 {tt['tier2']:.2f} s  "
              f"boxes {len(boxes)}  candidates {len(cands)}", flush=True)
    rows_ = list(per.values())
    trig = [x for x in rows_ if x["candidates"]]
    out = {
        "arch": A.ARCH["d"], "mode": "live, per frame", "frames": len(rows_),
        "model_load_s": load, "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "mean_s_per_frame": {k: round(float(np.mean([x[k] for x in rows_])), 3)
                             for k in ["road", "detect"] + TIER1 + TIER2},
        "tier1_mean_s": round(float(np.mean([x["tier1"] for x in rows_])), 2),
        "tier1_max_s": round(float(np.max([x["tier1"] for x in rows_])), 2),
        "tier2_triggered_frames": len(trig),
        "tier2_mean_s_when_triggered": round(float(np.mean([x["tier2"] for x in trig])), 2) if trig else 0,
        "tier2_mean_s_per_frame": round(float(np.mean([x["tier2"] for x in rows_])), 2),
        "mean_total_s_per_frame": round(float(np.mean([x["tier1"] + x["tier2"] for x in rows_])), 2),
        "max_total_s_per_frame": round(float(np.max([x["tier1"] + x["tier2"] for x in rows_])), 2),
        "mean_candidates_per_frame": round(float(np.mean([x["candidates"] for x in rows_])), 2),
        "api_calls": vlm.calls, "api_tokens": vlm.usage, "api_retried_errors": vlm.errors,
        "api_call_seconds": sorted(vlm.durations), "api_hedged_requests": vlm.hedged,
        "per_frame": {str(i): {k: round(v, 3) for k, v in x.items()} for i, x in per.items()},
    }
    d = A.out_dir("d", "test")
    (d / "timing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--co-resident", action="store_true")
    ap.add_argument("--arch", default="a", choices=["a", "d"],
                    help="a: staged, local Qwen; d: live per frame, detector screen + gpt-5.4")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.arch == "d":
        return run_fast(args)
    import torch
    import transformers
    from train_road_segmenter import OUT as SEG_DIR, predict

    images, gt, loc = A.load(A.TEST)
    ids = sorted(images)[: args.limit or None]
    ppm = A.location_ppm()
    thr = A.operating_threshold()
    sel = json.loads(A.SELECTION.read_text(encoding="utf-8"))
    variant = json.loads(max(sel, key=lambda r: r["score"])["variant"])
    opinion = A.DetectorOpinion()
    z = np.load(T.PROBE_DIR / "probe.npz")
    load, peaks, per = {}, {}, {i: {} for i in ids}

    def gb():
        return torch.cuda.max_memory_allocated() / 1024 ** 3

    def load_vlm():
        t = time.time()
        v = A.VLM(variant["model"], variant["bits"], variant.get("quant_vision", False))
        load["vlm"] = round(time.time() - t, 1)
        return v

    # ---- Objective 1
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    seg = transformers.SegformerForSemanticSegmentation.from_pretrained(str(SEG_DIR)).cuda().eval()
    load["road_segmenter"] = round(time.time() - t, 1)
    t = time.time()
    det = E.Owlv2Scored()
    load["owlv2"] = round(time.time() - t, 1)
    vlm = load_vlm() if args.co_resident else None
    peaks["resident_after_load"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)

    def objective1(i):
        img = Image.open(A.TEST / "images" / images[i]["file_name"]).convert("RGB")
        ign = images[i].get("ignore_regions", [])
        tt = per[i]
        torch.cuda.synchronize()
        t0 = time.time()
        mask = E.road_mask(predict(seg, img))
        torch.cuda.synchronize()
        tt["road"] = time.time() - t0
        t0 = time.time()
        rows, emb = det.detect(img, with_embeddings=True)
        torch.cuda.synchronize()
        tt["detect"] = time.time() - t0
        t0 = time.time()
        r = np.array(rows, dtype=np.float32).reshape(-1, 9)
        keep = r[:, 5] >= T.MIN_OBJ
        r, e = r[keep], emb[keep].astype(np.float32)
        o = np.clip(r[:, 5], 1e-4, 1 - 1e-4)
        x = np.hstack([np.log(o / (1 - o))[:, None], e])
        p = 1 / (1 + np.exp(-(((x - z["mean"]) / z["scale"]) @ z["coef"] + z["intercept"])))
        boxes = []
        for k, (b, s) in enumerate(zip(r, p)):
            score = (1 + s) if (b[5] >= T.OBJ_GATE and s >= T.PROBE_GATE) else s
            bb = [int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(score), float(b[5])]
            if score >= thr and E.on_road(bb, mask) and not E.centre_in_any(bb, ign):
                boxes.append((bb, x[k]))
        tt["rescore"] = time.time() - t0
        return img, mask, boxes

    def vlm_stages(i, img, mask, boxes):
        tt = per[i]
        g = ppm[loc[i]]
        t0 = time.time()
        objs = []
        for k, (b, _) in enumerate(boxes):
            ident = A.identify(vlm, img, b, g, variant)
            objs.append({"id": k + 1, "box": b[:4], "detection_score": b[4] - 1 if b[4] > 1 else b[4],
                         "vlm_name": ident["name"], "vlm_category": ident["category"], "k": k})
        tt["identify"] = time.time() - t0
        t0 = time.time()
        rec = {"width": images[i]["width"], "height": images[i]["height"], "px_per_metre": g}
        cands = [c for c in objs if c["vlm_category"] in A.ASSESSED]
        n_part = 0
        u = A.road_direction(mask)

        def ask_part(o):
            nonlocal n_part
            n_part += 1
            raw = vlm.ask([A.context_view(img, o["box"], 6.0, 200), A.tight_view(img, o["box"])],
                          A.PART_PROMPT, 40, prefix='{"part_of_vehicle": ')
            head = raw.split(",")[0].split(":")[-1].strip().lower()
            return head.startswith("true") or head.startswith("1")
        kept = []
        for c in cands:
            p3 = opinion.probes[loc[i]].predict_proba(boxes[c["k"]][1][None])[0]
            if not A.context_reason(c, objs, rec, u, p3, ask_part):
                kept.append(c)
        cands = kept
        tt["context"] = time.time() - t0
        t0 = time.time()
        vehicles = [v for v in objs if v["vlm_category"] == "vehicle"]
        for c in cands:
            A.assess(vlm, img, c["box"], {"name": c["vlm_name"], "category": c["vlm_category"]},
                     g, None, len(vehicles))
        tt["risk"] = time.time() - t0
        tt.update({"boxes": len(boxes), "risk_calls": len(cands), "part_questions": n_part})

    if args.co_resident:
        torch.cuda.reset_peak_memory_stats()
        for i in ids:
            img, mask, boxes = objective1(i)
            vlm_stages(i, img, mask, boxes)
            print(f"  {i:02d}: " + "  ".join(f"{k} {v:.1f}" for k, v in per[i].items()), flush=True)
        peaks["all_stages"] = round(gb(), 2)
    else:
        torch.cuda.reset_peak_memory_stats()
        stash = {}
        for i in ids:
            stash[i] = objective1(i)
        peaks["objective1_pass"] = round(gb(), 2)
        del seg, det
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vlm = load_vlm()
        for i in ids:
            vlm_stages(i, *stash[i])
            print(f"  {i:02d}: " + "  ".join(f"{k} {v:.1f}" for k, v in per[i].items()), flush=True)
        peaks["vlm_pass"] = round(gb(), 2)

    rows = list(per.values())
    out = {
        "arch": A.ARCH["a"], "mode": "co-resident" if args.co_resident else "staged",
        "frames": len(rows), "model_load_s": load, "vram_gb": peaks,
        "mean_s_per_frame": {s: round(float(np.mean([x[s] for x in rows])), 2) for s in STAGES},
        "mean_total_s_per_frame": round(float(np.mean([sum(x[s] for s in STAGES) for x in rows])), 2),
        "max_total_s_per_frame": round(float(np.max([sum(x[s] for s in STAGES) for x in rows])), 2),
        "total_s_all_frames": round(float(np.sum([sum(x[s] for s in STAGES) for x in rows])), 1),
        "mean_boxes_per_frame": round(float(np.mean([x["boxes"] for x in rows])), 1),
        "mean_risk_calls_per_frame": round(float(np.mean([x["risk_calls"] for x in rows])), 2),
        "mean_part_questions_per_frame": round(float(np.mean([x["part_questions"] for x in rows])), 2),
        "per_frame": {str(i): {k: round(v, 2) for k, v in x.items()} for i, x in per.items()},
    }
    d = A.out_dir("a", "test")
    name = "timing_co_resident.json" if args.co_resident else "timing.json"
    (d / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
