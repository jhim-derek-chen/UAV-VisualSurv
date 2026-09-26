"""End-to-end timing of the full chain, from the raw frame to risk levels.

    python scripts/time_pipeline.py --path qwen
    python scripts/time_pipeline.py --path qwen-relations
    python scripts/time_pipeline.py --path qwen --co-resident

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
physics re-asks), context (path 2 gate incl. part questions), risk (VLM).

Uses the deployable re-scorer (models/owlv2-rescorer, trained on all four
locations), so boxes can differ slightly from the evaluation runs, which use
leave-one-location-out probes. Accuracy comes from those runs; this script is
for time and memory. Output: results/risk-assessment/<path dir>/timing.json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import assess_risk as A  # noqa: E402
import eval_road_objects as E  # noqa: E402
import train_rescorer as T  # noqa: E402

STAGES = ["road", "detect", "rescore", "identify", "context", "risk"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="qwen", choices=["qwen", "qwen-relations"])
    ap.add_argument("--co-resident", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    import transformers
    from train_road_segmenter import OUT as SEG_DIR, predict

    images, gt, loc = A.load(A.TEST)
    ids = sorted(images)[: args.limit or None]
    ppm = A.location_ppm()
    thr = A.operating_threshold()
    sel = json.loads(A.SELECTION.read_text(encoding="utf-8"))
    variant = json.loads(max(sel, key=lambda r: r["score"])["variant"])
    opinion = A.DetectorOpinion() if args.path == "qwen-relations" else None
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
        if args.path == "qwen-relations":
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
        "path": A.PATH_DIRS[args.path], "mode": "co-resident" if args.co_resident else "staged",
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
    d = A.out_dir(args.path, "test")
    name = "timing_co_resident.json" if args.co_resident else "timing.json"
    (d / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
