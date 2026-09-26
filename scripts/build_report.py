"""Build the stage report: its figures, then report/stage_report.md -> HTML -> PDF.

    python scripts/build_report.py            # figures + PDF
    python scripts/build_report.py --no-figs  # PDF only, after editing the markdown

The markdown is the source to edit. Figures drawn here go to report/figures/;
the demo pages use the chain figures in results/ directly. The PDF is printed
by Microsoft Edge (or Chrome) in headless mode, so nothing beyond the
project's Python environment is needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "report"
FIGS = REPORT / "figures"
TEST = REPO_ROOT / "datasets" / "synthetic-highway-debris"
SCENE = REPO_ROOT / "datasets" / "scene-relations"
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]


def font(n, bold=False):
    return ImageFont.load_default(n)


# ------------------------------------------------------------------ figure 1: dataset example
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
            d.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], outline=(60, 170, 255), width=3)
        else:
            deb = (x, y, w, h)
            d.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline=(255, 60, 60), width=4)
    pad_r = 1020
    canvas = Image.new("RGB", (W + pad_r, img.height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    clip = Path(sample["background_source"]).parent.name
    title, rest = sample["model_credit"].split(" by ", 1)
    author = rest.split(",")[0]
    veh_t = max(veh, key=lambda v: v[1]) if veh else None  # the lowest vehicle, clear of the edge
    notes = [
        ("Background frame", [f"Real UAV video: Pexels clip {clip}", "(Pexels License). Open-motorway",
                              "frames only, 4 shoot locations."], None),
        ("Debris object (red box)", [f"3D model {title} by {author}", "(CC BY, Sketchfab).",
                                     "Rendered in Blender 5.2.2, random pose.",
                                     "Scaled from the location's ground",
                                     "resolution, calibrated on real cars;",
                                     "tyres are drawn up to 2x real size."],
         (deb[0] + deb[2] + 4, deb[1] + deb[3] / 2) if deb else None),
        ("Debris label", ["Box and mask taken pixel-exact from", "the render's alpha channel."],
         None),
        ("Vehicle labels (blue boxes)", ["Grounding DINO proposals, every box",
                                         "then checked by hand: 342 vehicles", "in 30 frames."],
         (veh_t[0] + veh_t[2] + 3, veh_t[1] + veh_t[3] / 2) if veh_t else None),
    ]
    y = 34
    for head, lines, target in notes:
        d.text((W + 40, y), head, fill=(20, 20, 20), font=font(38))
        for k, line in enumerate(lines):
            d.text((W + 40, y + 50 + 36 * k), line, fill=(70, 70, 70), font=font(30))
        if target:
            d.line([(W + 30, y + 20), target], fill=(40, 40, 40), width=3)
            d.ellipse([target[0] - 6, target[1] - 6, target[0] + 6, target[1] + 6],
                      fill=(40, 40, 40))
        y += 50 + 36 * len(lines) + 34
    out = FIGS / "fig1_dataset_example.png"
    canvas.save(out)
    return out


# ------------------------------------------------------------------ figure 2: scene variants
def fig_scene_variants(first_id: int = 10) -> Path:
    """The five variants of one scene-relation background, cropped around
    the placed objects, each with its expected answer."""
    man = json.loads((SCENE / "manifest.json").read_text(encoding="utf-8"))
    anns = {a["id"]: a for a in json.loads(
        (SCENE / "annotations.json").read_text(encoding="utf-8"))["annotations"]}
    samples = [s for s in man["samples"]
               if s["background_source"] == man["samples"][first_id]["background_source"]]
    tiles = []
    win_w, y0, y1 = 1300, 840, 1320  # about 70 m of both carriageways of this camera
    last_cx = None
    for s in samples:
        im = Image.open(SCENE / "images" / f"{s['image_id']:05d}.jpg").convert("RGB")
        boxes = [anns[o["ann_id"]]["bbox"] for o in s["objects"]]
        cx = (min(b[0] for b in boxes) + max(b[0] + b[2] for b in boxes)) / 2 if boxes else None
        cx = cx or last_cx or im.width / 2
        last_cx = cx
        x0 = int(min(max(0, cx - win_w / 2), im.width - win_w))
        d = ImageDraw.Draw(im)
        for x, y, w, h in boxes:
            d.rectangle([x - 8, y - 8, x + w + 8, y + h + 8], outline=(255, 40, 40), width=4)
        c = im.crop((x0, y0, x0 + win_w, y1)).resize((700, round(700 * (y1 - y0) / win_w)),
                                                      Image.LANCZOS)
        e = s["expected"]
        band = Image.new("RGB", (c.width, c.height + 64), (255, 255, 255))
        band.paste(c, (0, 64))
        dd = ImageDraw.Draw(band)
        dd.text((4, 2), s["scenario"], fill=(20, 20, 20), font=font(28))
        dd.text((4, 34), f"expected: scene {e['scene_risk']}, lanes blocked {e['lanes_blocked']}"
                + (", one event" if e["groups"] else ""), fill=(80, 80, 80), font=font(22))
        tiles.append(band)
    cols, gap = 2, 16
    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    out_im = Image.new("RGB", (cols * 700 + gap, rows * (th + gap)), (255, 255, 255))
    for k, t in enumerate(tiles):
        out_im.paste(t, ((k % cols) * (700 + gap), (k // cols) * (th + gap)))
    out = FIGS / "fig2_scene_variants.png"
    out_im.save(out)
    return out


# ------------------------------------------------------------------ figure 3: architectures
def fig_architectures() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    LOCAL, CPU, CLOUD = "#d8efd9", "#e8e8e8", "#fde2c4"
    stages = [
        ("Road region", "SegFormer-B2, fine-tuned\non AeroScapes (drone view)", LOCAL,
         "RGB frame (up to 4K)", "road mask"),
        ("Object detection", "OWLv2 objectness, class-agnostic,\ntiled at two scales", LOCAL,
         "frame + road mask", "boxes, objectness,\n512-d box features"),
        ("Re-scoring", "logistic-regression probe on\nbox features; on-road filter", CPU,
         "boxes + features", "candidate boxes"),
        ("Identification", None, None, "two crops + size (m)", "name, category;\nphysics check"),
        ("Context gate", "4 rules: frame edge, far from\ntraffic, detector opinion,\n"
                         "part of a vehicle", CPU, "candidates + scene\ngeometry",
         "hazard candidates"),
        ("Per-object risk", None, None, "wide view + measured\nfacts", "risk level +\nreasoning"),
        ("Scene analysis", None, None, "numbered overview,\nclose views, facts",
         "lane / shoulder, events,\nlanes blocked, action"),
    ]
    vlm = {"A": ("Qwen3.5-2B, 4-bit\n(local GPU)", LOCAL), "B": ("GPT-5.4 (OpenAI API)", CLOUD),
           "C": ("GPT-5.4 (OpenAI API)", CLOUD)}
    import textwrap
    fig, ax = plt.subplots(figsize=(12, 10.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 10.2)
    ax.axis("off")
    cols = {"A": 4.55, "B": 7.15, "C": 9.75}
    bw, bh, step = 2.4, 0.92, 1.12
    ys = [8.55 - k * step - (0.5 if k >= 3 else 0) for k in range(len(stages))]
    for name, x in cols.items():
        ax.text(x, 9.95, f"Architecture {name}", ha="center", va="center", fontsize=13,
                weight="bold")
    for label, y in (("Objective 1: see (shared by A, B and C)", ys[0] + 0.72),
                     ("Objective 2: identify and assess risk", ys[3] + 0.72)):
        ax.text(0.1, y, label, fontsize=10.5, weight="bold", color="#333", va="center")
        ax.plot([0.1, 11.0], [y - 0.2, y - 0.2], color="#bbb", lw=0.8, ls="--")
    for k, (title, model, colour, inp, out) in enumerate(stages):
        y = ys[k]
        io = ("in:  " + textwrap.fill(inp.replace("\n", " "), 26, subsequent_indent="     ")
              + "\nout: " + textwrap.fill(out.replace("\n", " "), 26, subsequent_indent="     "))
        ax.text(0.1, y, io, fontsize=7.6, va="center", color="#333", family="monospace")
        for name, x in cols.items():
            if k == 6 and name != "C":
                ax.text(x, y, "—", ha="center", va="center", color="#999", fontsize=12)
                continue
            m, c = (model, colour) if model else vlm[name]
            if title == "Scene analysis":
                m, c = "GPT-5.4: the whole frame\njudged at once", CLOUD
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=c, ec="#555", lw=0.9))
            ax.text(x, y + 0.26, title, ha="center", va="center", fontsize=9.5, weight="bold")
            ax.text(x, y - 0.12, m, ha="center", va="center", fontsize=7.6)
            if k < len(stages) - 1 and not (k == 5 and name != "C"):
                ax.annotate("", (x, ys[k + 1] + bh / 2 + 0.02), (x, y - bh / 2 - 0.01),
                            arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    for lab, c, x in (("local GPU", LOCAL, 4.0), ("CPU", CPU, 5.9), ("cloud API", CLOUD, 7.4)):
        ax.add_patch(FancyBboxPatch((x, -0.1), 0.3, 0.22, boxstyle="round,pad=0.01", fc=c,
                                    ec="#555", lw=0.8))
        ax.text(x + 0.4, 0.01, lab, va="center", fontsize=9)
    out = FIGS / "fig3_architectures.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ markdown -> PDF
CSS = """
@page { size: A4; margin: 14mm 14mm 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42;
       color: #1a1a1a; max-width: 182mm; margin: auto; }
h1 { font-size: 19pt; margin: 0 0 2mm 0; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #bbb; padding-bottom: 1mm; margin-top: 6mm; }
h3 { font-size: 11pt; margin: 4mm 0 1.5mm 0; }
p, li { margin: 1.2mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm 0; font-size: 9pt; }
th, td { border: 1px solid #c8c8c8; padding: 1.2mm 2mm; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 1.5mm auto; }
.page { page-break-before: always; }
table, img, .cap { page-break-inside: avoid; }
.demo h3 { margin: 1mm 0 1mm 0; }
.demo img { height: 46mm; width: auto; max-width: 100%; margin: 1mm auto 0.5mm auto; }
.keep { page-break-inside: avoid; }
.demo .cap { font-size: 8pt; margin: 0 0 1.6mm 0; }
img[alt="Figure 3"] { max-height: 122mm; width: auto; }
.cap { font-size: 8.6pt; color: #444; margin: 0 0 2.5mm 0; }
.sub { color: #555; font-size: 9.5pt; }
code { font-size: 8.8pt; }
"""


def build_pdf() -> Path:
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    src = (REPORT / "stage_report.md").read_text(encoding="utf-8")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Stage report</title>"
            f"<style>{CSS}</style></head><body>{md.render(src)}</body></html>")
    page = REPORT / "stage_report.html"
    page.write_text(html, encoding="utf-8")
    pdf = REPORT / "stage_report.pdf"
    browser = next((b for b in BROWSERS if b.is_file()), None)
    if browser is None:
        raise SystemExit("no Edge or Chrome found; open report/stage_report.html and print it")
    tmp = REPO_ROOT / ".cache" / "tmp" / "edge_profile"
    pdf.unlink(missing_ok=True)
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--user-data-dir={tmp}",
                    f"--print-to-pdf={pdf}", page.as_uri()],
                   capture_output=True, timeout=180)
    # The browser can hand the job to a background process and return early:
    # wait until the PDF exists and has stopped growing.
    import time
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
        for f in (fig_dataset, fig_scene_variants, fig_architectures):
            print("figure:", f().relative_to(REPO_ROOT))
    print("pdf:", build_pdf().relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
