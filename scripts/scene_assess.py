"""Architecture C: one scene-level analysis per frame, after B's per-box steps.

A and B judge each candidate on its own. Here the VLM sees the frame as a
whole: an overview cropped to the road with every candidate hazard boxed in
red and numbered (#) and every vehicle boxed in blue (V), a close view of
each candidate, and a table of measured facts, including each candidate's
first-pass risk. It returns, for each candidate, where it lies (running
lane, hard shoulder, off the carriageway) and a revised risk; which
candidates belong to one event (a spilled load) and which vehicle it came
from; how many running lanes are blocked; an overall scene risk; and one
instruction for the traffic control room.

The context gate still runs first: removing false alarms stays its job,
since the VLM alone let 23 of them through on the test set. A frame with no
candidate left is "none" by rule, without a call.

score() compares the answers with the expected ones written into the
scene-relation set's manifest (scripts/build_scene_relations.py). B's
answers are in the same records (the per-box risk), so B is scored
alongside: its scene risk is its highest per-box risk.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RISKS = ["none", "low", "medium", "high"]
OVERVIEW_MAX = 2048
VIEW = 448
CLOSE_M = 20.0      # side of each close view, metres: a few lanes either side
MAX_CLOSE = 6       # close views per call; beyond that, the overview only
MARGIN_M = 25.0     # road context kept around the boxes in the overview

PROMPT = """You are the hazard analyst for a motorway safety patrol drone.
Image 1 is the whole scene seen from straight above, cropped to the road. Every \
candidate hazard is boxed in red and numbered (#); every vehicle is boxed in blue (V).
{close_line}

Facts measured by the system:
{facts}

Each candidate was first rated on its own, from a close view. Now judge the scene as \
a whole, using the lane markings, the traffic and the positions of the objects \
relative to each other:
- where each candidate lies: in a running lane (where traffic drives), on the hard \
shoulder (the strip outside the solid edge line), or off the carriageway;
- whether several candidates belong to one event, such as a load spilled from one \
vehicle, and which vehicle;
- how many running lanes are obstructed.
Risk levels (the same as in the first pass):
- high: a rigid or heavy object in a traffic lane that a car could hit at speed or \
swerve to avoid (tyre, appliance, ladder, pallet, lumber, barrel), or a person in a lane
- medium: a smaller or softer object in a lane (suitcase, mattress, cone, bin), or a \
large object on the hard shoulder
- low: a light, soft, small item unlikely to cause loss of control (empty box, bag)
- none: not a hazard, for example not a real object on the road
Write the reasoning first, about the scene and not about these instructions; then \
the answers. Answer with JSON only:
{{"reasoning": "<two to four sentences about the scene>",
 "objects": [{{"id": <candidate number>, "where": "lane|shoulder|off", "risk": "high|medium|low|none"}}],
 "groups": [{{"ids": [<candidate numbers>], "what": "<a few words>", "source_vehicle": "<V number or none>"}}],
 "lanes_blocked": <number of running lanes obstructed>,
 "scene_risk": "high|medium|low|none",
 "action": "<one short instruction for the traffic control room>"}}
List a group only for two or more candidates that belong together."""


def _font(n):
    return ImageFont.load_default(n)


def _centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def candidates(rec):
    """Boxes that reach the risk step: identified as debris or a person and
    not removed by the context gate."""
    return [o for o in rec["objects"] if o.get("risk_source") == "vlm"]


def overview(img: Image.Image, rec: dict, cands: list, vehicles: list, ppm: float):
    boxes = [o["box"] for o in cands + vehicles]  # ignore regions are already left out
    m = MARGIN_M * ppm
    x0 = max(0, min(b[0] for b in boxes) - m)
    y0 = max(0, min(b[1] for b in boxes) - m)
    x1 = min(img.width, max(b[0] + b[2] for b in boxes) + m)
    y1 = min(img.height, max(b[1] + b[3] for b in boxes) + m)
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    s = min(1.0, OVERVIEW_MAX / max(crop.size))
    crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    f = _font(max(14, round(18 * max(crop.size) / 2048)))
    for group, col, tag, wd in ((vehicles, (60, 170, 255), "V", 2), (cands, (255, 0, 0), "#", 3)):
        for o in group:
            x, y, w, h = o["box"]
            X0, Y0 = (x - x0) * s - 3, (y - y0) * s - 3
            d.rectangle([X0, Y0, (x + w - x0) * s + 3, (y + h - y0) * s + 3], outline=col, width=wd)
            d.text((X0, Y0 - f.size - 2), f"{tag}{o['id']}", fill=col, font=f,
                   stroke_width=2, stroke_fill=(0, 0, 0))
    return crop


def close_view(img, o, ppm):
    side = int(min(max(CLOSE_M * ppm, 160), img.width, img.height))
    cx, cy = _centre(o["box"])
    x0 = int(min(max(0, cx - side / 2), img.width - side))
    y0 = int(min(max(0, cy - side / 2), img.height - side))
    crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((VIEW, VIEW), Image.LANCZOS)
    s = VIEW / side
    x, y, w, h = o["box"]
    d = ImageDraw.Draw(crop)
    d.rectangle([(x - x0) * s - 4, (y - y0) * s - 4, (x + w - x0) * s + 4, (y + h - y0) * s + 4],
                outline=(255, 0, 0), width=3)
    d.text(((x - x0) * s - 4, (y - y0) * s - 26), f"#{o['id']}", fill=(255, 0, 0), font=_font(20),
           stroke_width=2, stroke_fill=(0, 0, 0))
    return crop


def facts(rec, cands, vehicles, ppm):
    lines = [f"- the frame covers about {rec['width'] / ppm:.0f} x {rec['height'] / ppm:.0f} m"
             f" ({ppm:.1f} pixels per metre)",
             f"- vehicles in view: {len(vehicles)}"]
    long_v = [(v, max(v["box"][2], v["box"][3]) / ppm) for v in vehicles]
    long_v = [f"V{v['id']} ({m:.0f} m)" for v, m in long_v if m >= 8]
    if long_v:
        lines.append("- long vehicles (lorries, buses): " + ", ".join(long_v))
    lines.append("- candidate hazards:")
    for o in cands:
        c = _centre(o["box"])
        near = min(((np.hypot(c[0] - _centre(v["box"])[0], c[1] - _centre(v["box"])[1]) / ppm, v)
                    for v in vehicles), key=lambda t: t[0], default=None)
        sw, sh = o["size_m"]
        lines.append(f"  #{o['id']}: {o['vlm_name']} (category: {o['vlm_category']}), about "
                     f"{max(sw, sh):.1f} x {min(sw, sh):.1f} m; "
                     + (f"nearest vehicle V{near[1]['id']}, {near[0]:.0f} m away; " if near else "")
                     + f"first-pass risk on its own: {o['risk']}")
    return "\n".join(lines)


def assess_scene(vlm, img: Image.Image, rec: dict, mask: np.ndarray) -> dict:
    cands = candidates(rec)
    if not cands:
        return {"called": False, "scene_risk": "none", "objects": [], "groups": [],
                "lanes_blocked": 0, "reasoning": "No candidate hazard passed the per-box steps.",
                "action": "none"}
    vehicles = [o for o in rec["objects"] if o["vlm_category"] == "vehicle"]
    ppm = rec["px_per_metre"]
    shown = cands[:MAX_CLOSE]
    ims = [overview(img, rec, cands, vehicles, ppm)] + [close_view(img, o, ppm) for o in shown]
    names = ", ".join(f"#{o['id']}" for o in shown)
    close_line = (f"Image 2 is a close view, about {CLOSE_M:.0f} m across, of candidate {names}."
                  if len(shown) == 1 else
                  f"Images 2 to {len(ims)} are close views, about {CLOSE_M:.0f} m across, of "
                  f"candidates {names}, in that order.")
    prompt = PROMPT.format(close_line=close_line, facts=facts(rec, cands, vehicles, ppm))
    raw = vlm.ask(ims, prompt, 900, prefix='{"reasoning": "')
    try:
        j = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        j = {}
    objs = []
    for x in j.get("objects", []):
        try:
            objs.append({"id": int(str(x.get("id")).lstrip("#")),
                         "where": str(x.get("where", "")).lower(),
                         "risk": str(x.get("risk", "")).lower()})
        except ValueError:
            continue
    groups = []
    for g in j.get("groups", []):
        try:
            ids = [int(str(t).lstrip("#")) for t in g.get("ids", [])]
        except ValueError:
            continue
        src = str(g.get("source_vehicle", "none")).upper().lstrip("V")
        groups.append({"ids": ids, "what": g.get("what", ""),
                       "source_vehicle": int(src) if src.isdigit() else None})
    try:
        lanes = int(j.get("lanes_blocked"))
    except (TypeError, ValueError):
        lanes = None
    risk = str(j.get("scene_risk", "")).lower()
    return {"called": True, "scene_risk": risk if risk in RISKS else None, "objects": objs,
            "groups": groups, "lanes_blocked": lanes, "reasoning": j.get("reasoning", raw),
            "action": j.get("action", ""), "raw": raw, "n_close_views": len(shown)}


# ------------------------------------------------------------------ scoring
def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u else 0.0


def match(rec, ann_box):
    best = max(rec["objects"], key=lambda o: _iou(o["box"], ann_box), default=None)
    return best if best is not None and _iou(best["box"], ann_box) > 0.1 else None


def frame_result(rec, sample, anns):
    exp = sample["expected"]
    scene = rec.get("scene", {})
    c_obj = {o["id"]: o for o in scene.get("objects", [])}
    b_scene = max((o["risk"] for o in rec["objects"]), key=RISKS.index, default="none")
    out = {"image_id": rec["image_id"], "scenario": sample["scenario"],
           "expected_scene": exp["scene_risk"], "B_scene": b_scene,
           "C_scene": scene.get("scene_risk"), "expected_lanes": exp["lanes_blocked"],
           "C_lanes": scene.get("lanes_blocked"), "objects": []}
    for eo in exp["objects"]:
        box = match(rec, anns[eo["ann_id"]])
        r = {"ann_id": eo["ann_id"], "category": eo["category"], "zone": eo["zone"],
             "expected": eo["risk"], "detected": box is not None}
        if box is not None:
            r.update({"id": box["id"], "vlm_category": box["vlm_category"],
                      "reached_risk_step": box.get("risk_source") == "vlm",
                      "gated": bool(box.get("context")), "B": box["risk"],
                      "C": c_obj.get(box["id"], {}).get("risk", box["risk"]),
                      "C_where": c_obj.get(box["id"], {}).get("where")})
        out["objects"].append(r)
    # A false alarm: a box on no debris (truth_any, as on the test set: a box
    # covering at least 10% of a debris object counts as that debris).
    false = [o for o in rec["objects"] if o["truth_any"] in ("vehicle", "none")]
    out["false_high"] = {
        "B": sum(o["risk"] == "high" for o in false),
        "C": sum(c_obj.get(o["id"], {}).get("risk", o["risk"]) == "high" for o in false)}
    if exp["groups"]:
        ids = {r["id"] for r in out["objects"] if r["detected"] and r["reached_risk_step"]}
        g = [set(x["ids"]) for x in scene.get("groups", [])]
        out["spill_items_reaching_scene"] = len(ids)
        out["spill_grouped"] = bool(ids) and len(ids) >= 2 and any(ids <= s for s in g)
        src = exp.get("spill_source_vehicle")
        src_box = anns.get(src) if src else None
        said = [x["source_vehicle"] for x in scene.get("groups", []) if ids & set(x["ids"])]
        said_box = [next((o["box"] for o in rec["objects"] if o["id"] == v), None) for v in said]
        out["spill_source_correct"] = bool(src_box and any(
            b is not None and _iou(b, src_box) > 0.3 for b in said_box))
    return out


def score(recs, root: Path) -> dict:
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = {s["image_id"]: s for s in man["samples"] if "expected" in s}
    if not samples:
        return {}
    anns = {a["id"]: a["bbox"] for a in
            json.loads((root / "annotations.json").read_text(encoding="utf-8"))["annotations"]}
    rows = [frame_result(r, samples[r["image_id"]], anns) for r in recs if r["image_id"] in samples]
    objs = [o for r in rows for o in r["objects"]]
    det = [o for o in objs if o["detected"]]

    def frac(xs):
        xs = list(xs)
        return {"correct": sum(xs), "n": len(xs)}
    by = {}
    for sc in ("clear", "lane", "shoulder", "spill", "blockage"):
        rs = [r for r in rows if r["scenario"] == sc]
        if rs:
            by[sc] = {"frames": len(rs),
                      "B_scene_correct": sum(r["B_scene"] == r["expected_scene"] for r in rs),
                      "C_scene_correct": sum(r["C_scene"] == r["expected_scene"] for r in rs)}
    return {
        "frames": len(rows),
        "scene_risk": {"B": frac(r["B_scene"] == r["expected_scene"] for r in rows),
                       "C": frac(r["C_scene"] == r["expected_scene"] for r in rows)},
        "by_scenario": by,
        "placed_objects": len(objs), "detected": len(det),
        "reached_risk_step": sum(o["reached_risk_step"] for o in det),
        "object_risk": {"B": frac(o["B"] == o["expected"] for o in det),
                        "C": frac(o["C"] == o["expected"] for o in det)},
        # Only objects that reached the risk step: the scene step's own effect,
        # without detection and identification misses.
        "object_risk_reached": {
            zone: {"n": len(xs),
                   "B": {r: sum(o["B"] == r for o in xs) for r in ("high", "medium", "low")},
                   "C": {r: sum(o["C"] == r for o in xs) for r in ("high", "medium", "low")}}
            for zone in ("lane", "shoulder")
            for xs in [[o for o in det if o["reached_risk_step"] and o["zone"] == zone]]},
        "zone_C": frac(o.get("C_where") == o["zone"] for o in det if o.get("C_where")),
        "lanes_blocked_C": frac(r["C_lanes"] == r["expected_lanes"] for r in rows
                                if r["C_scene"] is not None),
        "spill_grouped_C": frac(r["spill_grouped"] for r in rows if "spill_grouped" in r
                                and r["spill_items_reaching_scene"] >= 2),
        "spill_source_C": frac(r["spill_source_correct"] for r in rows if "spill_grouped" in r
                               and r["spill_items_reaching_scene"] >= 2),
        "false_high": {"B": sum(r["false_high"]["B"] for r in rows),
                       "C": sum(r["false_high"]["C"] for r in rows)},
        "rows": rows,
    }


# ------------------------------------------------------------------ figure
RISK_COL = {"high": (230, 40, 40), "medium": (255, 150, 0), "low": (240, 220, 0),
            "none": (150, 150, 150), None: (150, 150, 150)}


def add_scene_panel(fig: Image.Image, img: Image.Image, rec: dict, sample: dict | None = None):
    """Adds panel 4 under the chain figure: the scene analysis, and, on the
    scene-relation set, the expected answer next to it."""
    sc = rec["scene"]
    H = 330
    out = Image.new("RGB", (fig.width, fig.height + H), (18, 18, 18))
    out.paste(fig, (0, 0))
    d = ImageDraw.Draw(out)
    y = fig.height + 6
    d.text((14, y), "4  Scene analysis (architecture C): the frame judged as a whole",
           fill=(255, 255, 255), font=_font(24))
    y += 40
    risk = sc.get("scene_risk")
    d.text((14, y), f"SCENE RISK: {str(risk).upper()}", fill=RISK_COL.get(risk, (200, 200, 200)),
           font=_font(26))
    d.text((360, y + 4), f"running lanes blocked: {sc.get('lanes_blocked')}", fill=(230, 230, 230),
           font=_font(20))
    if sample:
        e = sample["expected"]
        ok = risk == e["scene_risk"] and (not sc.get("called") or
                                          sc.get("lanes_blocked") == e["lanes_blocked"])
        d.text((700, y + 4), f"answer key ({sample['scenario']}): scene {e['scene_risk'].upper()}, "
               f"lanes blocked {e['lanes_blocked']}  ->  {'matches' if ok else 'does not match'}",
               fill=(200, 200, 200), font=_font(20))
    y += 40
    per = "   ".join(f"#{o['id']} {o['where']} {o['risk'].upper()}" for o in sc.get("objects", []))
    first = {o["id"]: o["risk"] for o in rec["objects"]}
    firsts = "   ".join(f"#{o['id']} {first.get(o['id'], '?').upper()}" for o in sc.get("objects", []))
    if per:
        d.text((14, y), "per candidate, in the scene: " + per, fill=(230, 230, 230), font=_font(18))
        d.text((14, y + 24), "per candidate, judged alone (first pass): " + firsts,
               fill=(170, 170, 170), font=_font(18))
        y += 52
    for g in sc.get("groups", []):
        src = f", from V{g['source_vehicle']}" if g.get("source_vehicle") else ""
        d.text((14, y), f"group: {', '.join('#%d' % i for i in g['ids'])}: {g.get('what', '')}{src}",
               fill=(255, 200, 120), font=_font(18))
        y += 24
    for line in textwrap.wrap(str(sc.get("reasoning", "")), 190)[:3]:
        d.text((14, y), line, fill=(220, 220, 220), font=_font(17))
        y += 22
    d.text((14, y + 4), "action: " + str(sc.get("action", "")), fill=(120, 230, 120), font=_font(19))
    return out
