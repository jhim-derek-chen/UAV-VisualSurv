"""Build the milestone deck: figures, then report/milestone1.md -> 16:9 slides
(HTML) -> report/milestone1.pdf, plus report/milestone1_notes.md.

    python scripts/report/build_report.py            # figures + PDF + speaker notes
    python scripts/report/build_report.py --no-figs  # PDF + notes only, after editing the markdown

Markdown format: slides are separated by a line `---`. A first line
`<!-- class: name -->` gives the slide a CSS class. Text after a line `???`
is the presenter's note: it is left off the slide and collected into
report/milestone1_notes.md. Diagrams and tables are HTML inside the markdown, so
they stay editable as text.

Figures drawn here go to report/figures/: the annotated dataset example and
one demo tile per architecture and frame (a crop
around the objects that matter, verdicts written on the image). The PDF is
printed by Microsoft Edge (or Chrome) in headless mode.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "report"
FIGS = REPORT / "figures"
DECK = "milestone1"  # report/<DECK>.md -> report/<DECK>.pdf + report/<DECK>_notes.md
RES = REPO_ROOT / "results" / "risk-assessment"
TEST = REPO_ROOT / "datasets" / "synthetic-highway-debris"
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]
FONT_DIR = Path(r"C:\Windows\Fonts")

# Demo frames: the same nine test frames for A, B and D, in the proportions of
# the whole test set (both find the debris in 18 of 30 frames, only A in 4,
# only B / D in 3). Row 1: all find it, D with a fraction of the VLM calls.
# Row 2: all find it; B / D name or rate it right, A does not. Row 3: frames
# where they differ: A's false alarm, a fridge only B / D find, a tyre only A
# finds. Focus = object ids the crop keeps.
DEMO = [(23, [4, 3]), (20, [23, 20]), (27, [3, 11]),
        (19, [8]), (14, [5, 2]), (4, [2]),
        (18, [6, 5]), (2, [3]), (26, [23])]
TIER1_S = 2.4          # shared Objective 1 code, measured live in D's run
RISK_CALL_S = 1.85     # gpt-5.4 risk call, mean in D's live run (B's own risk times include ungated calls)
DEMO_ARCHS = ("a_qwen_local", "b_gpt", "d_fast")

RISK_COL = {"high": (227, 60, 60), "medium": (240, 140, 20), "low": (225, 190, 0)}
GATED, MISSED, VEH = (30, 160, 110), (215, 60, 160), (60, 140, 230)
SHORT = {"refrigerator or appliance": "fridge", "wooden plank or lumber": "planks",
         "wooden pallet": "pallet", "suitcase or bag": "suitcase", "barrel or drum": "barrel",
         "traffic cone": "cone", "trash can": "bin", "cardboard box": "box",
         "vegetation or shadow": "shadow", "other debris": "debris", "tire": "tyre",
         "person": "person", "vehicle": "vehicle"}


def short(cat: str) -> str:
    if cat.startswith("roadside structure"):
        return "roadside structure"
    return SHORT.get(cat, cat)


def font(n, bold=False):
    f = FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(f), n) if f.is_file() else ImageFont.load_default(n)


def symbol_font(n):
    f = FONT_DIR / "seguisym.ttf"
    return ImageFont.truetype(str(f), n) if f.is_file() else ImageFont.load_default(n)


# ------------------------------------------------------------------ dataset example
def fig_dataset(image_id: int = 23) -> Path:
    """One test frame with every element's source written next to it."""
    anns = json.loads((TEST / "annotations.json").read_text(encoding="utf-8"))
    man = json.loads((TEST / "manifest.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in anns["categories"]}
    im_meta = next(i for i in anns["images"] if i["id"] == image_id)
    sample = next(s for s in man["samples"] if s["image_id"] == image_id)
    img = Image.open(TEST / "images" / im_meta["file_name"]).convert("RGB")
    W = 1500
    s = W / img.width
    img = img.resize((W, round(img.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    deb, veh = None, []
    for a in anns["annotations"]:
        if a["image_id"] != image_id:
            continue
        x, y, w, h = [v * s for v in a["bbox"]]
        if cats[a["category_id"]] == "vehicle":
            veh.append((x, y, w, h))
            d.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], outline=VEH, width=4)
        else:
            deb = (x, y, w, h)
            d.rectangle([x - 5, y - 5, x + w + 5, y + h + 5], outline=(227, 60, 60), width=5)
    pad_r = 1000
    canvas = Image.new("RGB", (W + pad_r, img.height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    clip = Path(sample["background_source"]).parent.name
    title, rest = sample["model_credit"].split(" by ", 1)
    author = rest.split(",")[0]
    veh_t = max(veh, key=lambda v: v[1]) if veh else None
    notes = [
        ("Background", [f"real UAV video, Pexels clip {clip}", "Pexels License · open motorway only"],
         None),
        ("Debris (red)", [f"3D model {title}", f"by {author} · CC BY · Sketchfab",
                          "Blender 5.2.2 · random pose",
                          "real-world scale from calibrated", "ground resolution (px per m)"],
         (deb[0] + deb[2] + 6, deb[1] + deb[3] / 2) if deb else None),
        ("Debris label", ["pixel-exact, from the render's alpha"], None),
        ("Vehicle labels (blue)", ["Grounding DINO proposals", "every box checked by hand"],
         (veh_t[0] + veh_t[2] + 4, veh_t[1] + veh_t[3] / 2) if veh_t else None),
    ]
    y = 40
    for head, lines, target in notes:
        d.text((W + 44, y), head, fill=(20, 20, 20), font=font(40, True))
        for k, line in enumerate(lines):
            d.text((W + 44, y + 54 + 40 * k), line, fill=(80, 80, 80), font=font(33))
        if target:
            d.line([(W + 34, y + 24), target], fill=(40, 40, 40), width=3)
            d.ellipse([target[0] - 7, target[1] - 7, target[0] + 7, target[1] + 7],
                      fill=(40, 40, 40))
        y += 54 + 40 * len(lines) + 34
    out = FIGS / "fig_dataset.png"
    canvas.save(out)
    return out


# ------------------------------------------------------------------ demo tiles
def _crop_box(boxes, W, H, aspect=2.4, min_frac=0.35, margin=80):
    x0 = min(b[0] for b in boxes) - margin
    y0 = min(b[1] for b in boxes) - margin
    x1 = max(b[0] + b[2] for b in boxes) + margin
    y1 = max(b[1] + b[3] for b in boxes) + margin
    w = max(x1 - x0, min_frac * W, (y1 - y0) * aspect)
    h = w / aspect
    if h > H:
        h, w = H, H * aspect
    w = min(w, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(0, cx - w / 2), W - w)
    top = min(max(0, cy - h / 2), H - h)
    return int(left), int(top), int(left + w), int(top + h)


def _tag(d, x, y, text, colour, mark=None, size=40, above=True):
    """A filled label tag at (x, y) with an optional ✓/✗ glyph."""
    f, fs = font(size, True), symbol_font(size)
    tw = d.textlength(text, font=f)
    mw = d.textlength(" " + mark, font=fs) if mark else 0
    h = size + 12
    top = y - h - 4 if above else y + 4
    d.rounded_rectangle([x, top, x + tw + mw + 16, top + h], radius=6, fill=colour)
    ink = (0, 0, 0) if sum(colour) > 520 else (255, 255, 255)
    d.text((x + 8, top + 4), text, fill=ink, font=f)
    if mark:
        d.text((x + 8 + tw, top + 2), " " + mark, fill=ink, font=fs)


def _dashed_rect(d, box, colour, width, dash=16, gap=12):
    x0, y0, x1, y1 = box
    for a0, a1, fixed, horizontal in ((x0, x1, y0, True), (x0, x1, y1, True),
                                      (y0, y1, x0, False), (y0, y1, x1, False)):
        k = a0
        while k < a1:
            e = min(k + dash, a1)
            d.line([(k, fixed), (e, fixed)] if horizontal else [(fixed, k), (fixed, e)],
                   fill=colour, width=width)
            k += dash + gap


def _draw_tile(img, crop, rec, labels, out, banner=None):
    x0, y0, x1, y1 = crop
    scale = 1600 / (x1 - x0)
    tile = img.crop(crop).resize((1600, round((y1 - y0) * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(tile)

    def tb(b):
        return [(b[0] - x0) * scale, (b[1] - y0) * scale,
                (b[0] + b[2] - x0) * scale, (b[1] + b[3] - y0) * scale]
    inside = [o for o in rec["objects"]
              if o["box"][0] + o["box"][2] > x0 and o["box"][0] < x1
              and o["box"][1] + o["box"][3] > y0 and o["box"][1] < y1]
    for o in inside:
        if o["id"] not in labels and o["vlm_category"] == "vehicle":
            r = tb(o["box"])
            d.rectangle([r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3], outline=VEH, width=3)
    for o in inside:
        if o["id"] not in labels:
            continue
        text, colour, mark = labels[o["id"]]
        r = tb(o["box"])
        d.rectangle([r[0] - 6, r[1] - 6, r[2] + 6, r[3] + 6], outline=colour, width=6)
        if text is None:  # a group member: the group's tag speaks for it
            continue
        lx = min(max(0, r[0] - 6), 1600 - 20 - d.textlength(text, font=font(40, True)) - 60)
        _tag(d, lx, r[1] - 6, text, colour, mark, above=r[1] > 70)
    if banner:  # a strip above the image, so it never hides an object near the top edge
        text, mark = banner
        strip = Image.new("RGB", (tile.width, tile.height + 70), (30, 30, 30))
        strip.paste(tile, (0, 70))
        ImageDraw.Draw(strip).text((14, 10), text, fill=(255, 255, 255), font=font(46, True))
        tile = strip
    tile.save(out, quality=90)
    return out


def tile_ab(arch: str, image_id: int, focus: list) -> Path:
    rec = json.loads((RES / arch / "per_image" / f"{image_id:02d}.json").read_text(encoding="utf-8"))
    img = Image.open(TEST / "images" / rec["file"]).convert("RGB")
    by = {o["id"]: o for o in rec["objects"]}
    crop = _crop_box([by[i]["box"] for i in focus], img.width, img.height)
    labels = {}
    rules = {"frame edge": "frame edge", "sideways": "far from traffic",
             "detector": "detector", "abuts": "cab or trailer",
             "part of vehicle": "vehicle part"}
    for o in rec["objects"]:
        truth, cat = o["truth_any"], o["vlm_category"]
        debris = truth not in ("vehicle", "none")
        if o.get("context"):
            why = next((v for k, v in rules.items() if k in o["context"]), "context")
            labels[o["id"]] = (f"removed · {why}", GATED, "✗" if debris else "✓")
        elif o["risk"] != "none":
            name = short(cat)
            if not debris:
                labels[o["id"]] = (f"'{name}' · {o['risk'].upper()} (false alarm)",
                                   RISK_COL[o["risk"]], "✗")
            else:
                real = "" if short(truth) == name else f" (really {short(truth)})"
                labels[o["id"]] = (f"{name}{real} · {o['risk'].upper()}", RISK_COL[o["risk"]], "✓")
        elif debris and o.get("identified_by") == "detector":
            labels[o["id"]] = (f"{short(truth)} screened out as {cat}: missed", MISSED, "✗")
        elif debris:
            labels[o["id"]] = (f"{short(truth)} named '{short(cat)}': missed", MISSED, "✗")
    # Banner: this frame's two tier times and how many boxes the VLM saw.
    n_all = len(rec["objects"])
    n_vlm = sum(o.get("identified_by", "vlm") == "vlm" for o in rec["objects"])
    if arch == "d_fast":  # measured live
        t = json.loads((RES / arch / "timing.json").read_text(encoding="utf-8"))["per_frame"][str(image_id)]
        t1, t2, approx = t["tier1"], t["tier2"], ""
    elif arch == "a_qwen_local":  # measured, staged run
        t = json.loads((RES / arch / "timing.json").read_text(encoding="utf-8"))["per_frame"][str(image_id)]
        t1, t2, approx = TIER1_S, t["identify"] + t["context"] + t["risk"], ""
    else:  # B: measured identification + risk calls at D's measured rate
        tm = rec["timing_s"]
        t1, t2, approx = TIER1_S, tm["identify"] + tm.get("context", 0) + tm["n_risk"] * RISK_CALL_S, "≈ "
    banner = (f"tier 1 {t1:.1f} s · tier 2 {approx}{t2:.1f} s · VLM saw {n_vlm} of {n_all} boxes", None)
    out = FIGS / f"demo_{arch.split('_')[0]}_{image_id:02d}.jpg"
    return _draw_tile(img, crop, rec, labels, out, banner=banner)


# ------------------------------------------------------------------ slides -> PDF
CSS = """
@page { size: 338.7mm 190.5mm; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; color: #1b1b1b; font-size: 12.5pt; }
section.slide { width: 338.7mm; height: 190.5mm; padding: 10mm 15mm 9mm 15mm; position: relative;
  overflow: hidden; page-break-after: always; }
section.slide:last-child { page-break-after: auto; }
.kicker { font-size: 10.5pt; letter-spacing: .08em; text-transform: uppercase; color: #7a7a7a;
  margin-bottom: 1mm; }
h1 { font-size: 24pt; margin: 0 0 5mm 0; font-weight: 600; }
h2 { font-size: 14pt; margin: 3mm 0 2mm 0; font-weight: 600; }
p { margin: 1.5mm 0; }
ul { margin: 1mm 0; padding-left: 6mm; } li { margin: 1mm 0; }
.foot { position: absolute; left: 15mm; right: 15mm; bottom: 4mm; font-size: 8.5pt; color: #9a9a9a;
  display: flex; justify-content: space-between; }
img { display: block; max-width: 100%; }
.muted { color: #6b6b6b; } .small { font-size: 10pt; } .tiny { font-size: 8.5pt; color: #777; }
.row { display: flex; gap: 6mm; align-items: flex-start; }
.col { flex: 1; }
.chips span { display: inline-block; border: 1px solid #cfcfcf; border-radius: 10mm; padding: .6mm 3mm;
  margin: 0 1.5mm 1.5mm 0; font-size: 10.5pt; background: #fafafa; }
/* title slide */
section.title { display: flex; flex-direction: column; justify-content: center; padding-left: 22mm; }
section.title h1 { font-size: 34pt; margin-bottom: 4mm; }
section.title .sub { font-size: 16pt; color: #555; }
/* flow diagrams */
.flow { display: flex; align-items: stretch; gap: 0; }
.flow .arrow { align-self: center; font-size: 24pt; color: #777; padding: 0 1.5mm; }
.blk { border: 1.4px solid #555; border-radius: 3mm; padding: 2.6mm 3mm; flex: 1; background: #fff;
  display: flex; flex-direction: column; }
.blk .bt { font-weight: 700; font-size: 15pt; margin-bottom: 1.2mm; }
.blk .fn { font-size: 11.5pt; color: #555; margin-bottom: 2mm; }
.blk .out { margin-top: auto; padding-top: 2mm; font-size: 11pt; color: #333; border-top: 1px dashed #bbb; }
.io { align-self: center; border: 1.4px dashed #888; border-radius: 3mm; padding: 2.5mm; font-size: 12pt;
  text-align: center; width: 27mm; }
.next { opacity: .45; }
.addon { border-style: dashed; }
.sub { display: flex; flex-direction: column; align-items: stretch; }
.sb { border: 1px solid #9a9a9a; border-radius: 2mm; padding: 1.4mm 2mm; font-size: 11pt; background: #fff;
  margin-bottom: 1mm; }
.sb i { color: #555; font-style: normal; font-size: 10pt; }
.dn { text-align: center; color: #888; font-size: 11pt; line-height: 1; margin: -.6mm 0 .4mm 0; }
.gpu { background: #e1f3eb; } .cpu { background: #efefef; } .api { background: #fde7dc; }
.vlm { background: #f3eefc; }
.kpi { font-size: 12.5pt; margin-top: 1.5mm; line-height: 1.4; }
.kpi b { font-size: 15pt; }
.stats { display: flex; gap: 8mm; margin-top: 7mm; }
.stat { flex: 1; border-left: 4px solid #8a8a8a; padding: .5mm 0 .5mm 4mm; }
.stat .v { font-size: 30pt; font-weight: 600; line-height: 1.05; }
.stat .l { font-size: 12pt; color: #555; }
section h2 { margin-top: 6mm; }
/* matrix + evaluation tables */
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #d0d0d0; padding: 1mm 2mm; text-align: left; vertical-align: middle; }
th { background: #f2f2f2; font-weight: 600; }
section.arch .blk .bt { font-size: 13.5pt; }
section.arch .blk .fn { font-size: 10.5pt; margin-bottom: 1.5mm; }
section.arch .blk .out { font-size: 10pt; }
section.arch .sb { font-size: 10pt; padding: .7mm 1.8mm; margin-bottom: .6mm; }
section.arch .dn { margin: -1mm 0 -.4mm 0; font-size: 9pt; }
section.arch h1 { margin-bottom: 3mm; }
section.arch .sb i { font-size: 9pt; }
section.arch .io { font-size: 11pt; width: 24mm; }
section.arch table.mx { font-size: 11pt; margin-top: 3mm; }
section.arch table.mx td, section.arch table.mx th { padding: 1.1mm 2.2mm; }
table.mx { font-size: 11.5pt; margin-top: 6mm; }
table.mx td, table.mx th { padding: 1.8mm 2.5mm; }
table.mx td.gpu, table.mx td.cpu, table.mx td.api { text-align: center; }
table.ev { font-size: 10.2pt; }
table.ev td { padding: .55mm 2mm; }
table.ev td.n { text-align: center; font-variant-numeric: tabular-nums; }
section.big h2 { font-size: 19pt; margin: 4mm 0 4mm 0; }
section.big ul { font-size: 16pt; padding-left: 7mm; }
section.big li { margin: 3.5mm 0; }
table.ev tr.g td { background: #f2f2f2; font-weight: 700; font-size: 9.5pt; }
table.ev td.bn { background: #fbd9d6; font-weight: 700; }
table.ev td.best { font-weight: 700; }
table.ev td.span { text-align: center; color: #333; }
table.ev tr.t1tot td { background: #d9ebfb; font-weight: 700; font-size: 11pt; color: #0d3c66; }
table.ev tr.tot td { font-weight: 700; background: #fafafa; }
.tierrow { margin-bottom: 1.5mm; align-items: center; }
.tier { font-size: 10.5pt; font-weight: 700; text-align: center; padding: 1.2mm; border-radius: 2mm; }
.tier.t1 { background: #d9ebfb; color: #0d3c66; }
.tier.t2 { background: #efefef; color: #444; }
/* demo grid */
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5mm 6mm; }
.tile img { width: 100%; height: 63mm; object-fit: contain; background: #f3f3f3; border-radius: 1.5mm; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2mm 4mm; }
.grid3 .tile img { height: 41mm; }
.grid3 .cue { font-size: 9pt; margin-top: .4mm; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lead { font-size: 11pt; color: #444; margin: -3mm 0 2mm 0; }
.cue { font-size: 11pt; font-weight: 600; margin-top: 1mm; }
.cue .m { font-weight: 400; color: #555; }
.legend { font-size: 10pt; color: #555; margin: -3mm 0 2mm 0; }
.legend span { display: inline-block; width: 3.2mm; height: 3.2mm; border-radius: .8mm; margin: 0 1mm -.5mm 3mm; }
"""


def parse_slides(src: str):
    slides = []
    for chunk in re.split(r"^---\s*$", src, flags=re.M):
        if not chunk.strip():
            continue
        body, _, notes = chunk.partition("\n???")
        m = re.match(r"\s*<!--\s*class:\s*([\w -]+?)\s*-->", body)
        cls = m.group(1) if m else ""
        slides.append((cls, body[m.end():] if m else body, notes.strip()))
    return slides


def build_pdf() -> Path:
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    slides = parse_slides((REPORT / f"{DECK}.md").read_text(encoding="utf-8"))
    parts, notes = [], ["# Speaker notes\n"]
    for k, (cls, body, note) in enumerate(slides, 1):
        foot = ("" if "title" in cls else
                f"<div class='foot'><span>UAV-VisualSurv · stage report</span><span>{k}</span></div>")
        parts.append(f"<section class='slide {cls}'>{md.render(body)}{foot}</section>")
        title = re.search(r"^#\s+(.+)$", body, flags=re.M)
        notes.append(f"## {k}. {title.group(1) if title else ''}\n\n{note or '(no note)'}\n")
    (REPORT / f"{DECK}_notes.md").write_text("\n".join(notes), encoding="utf-8")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Milestone 1</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    page = REPORT / f"{DECK}.html"
    page.write_text(html, encoding="utf-8")
    pdf = REPORT / f"{DECK}.pdf"
    browser = next((b for b in BROWSERS if b.is_file()), None)
    if browser is None:
        raise SystemExit(f"no Edge or Chrome found; open report/{DECK}.html and print it")
    tmp = REPO_ROOT / ".cache" / "tmp" / "edge_profile"
    pdf.unlink(missing_ok=True)
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--user-data-dir={tmp}",
                    f"--print-to-pdf={pdf}", page.as_uri()],
                   capture_output=True, timeout=180)
    # The browser can hand the job to a background process and return early:
    # wait until the PDF exists and has stopped growing.
    last, t0 = -1, time.time()
    while time.time() - t0 < 120:
        size = pdf.stat().st_size if pdf.is_file() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1.5)
    if not pdf.is_file():
        raise SystemExit(f"the browser wrote no PDF; open {page} and print it by hand")
    shutil.rmtree(tmp, ignore_errors=True)
    page.unlink()
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    if not args.no_figs:
        for old in FIGS.glob("*"):
            old.unlink()
        made = [fig_dataset()]
        for arch in DEMO_ARCHS:
            made += [tile_ab(arch, i, focus) for i, focus in DEMO]
        for f in made:
            print("figure:", f.relative_to(REPO_ROOT))
    print("pdf:", build_pdf().relative_to(REPO_ROOT))
    print("notes:", (REPORT / f"{DECK}_notes.md").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
