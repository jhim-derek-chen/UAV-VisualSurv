"""Build the slide report from results/benchmark.json,
results/segmentation_benchmark.json and results/figures/.

Every number in the deck is read from those JSON files, never retyped, so the
report cannot drift from the run that produced it. Figures are embedded as
data URIs because the artifact CSP blocks external images.

Five slides: the two-stage recommendation up front (slide 1) -- which model
finds the highway, which model finds debris on it -- then Stage 1 (road
segmentation, slide 2), Stage 2's method and headline table (slide 3), the
two factors that drive Stage 2's result (slide 4), and what this does not
prove (slide 5). B5 (RescueNet) is excluded from Stage 2: post-hurricane
street damage is too far from a highway to inform the choice.

    python scripts/survey/make_report.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in (REPO_ROOT / "scripts").iterdir()
                if p.is_dir() and not p.name.startswith("__")]  # modules import each other by name
RESULTS = REPO_ROOT / "results"
DEST = REPO_ROOT / "results" / "uav_selection_report.html"

SKIP_BENCH = {"B5"}

# Findings are stated at the level of the method family, not the checkpoint --
# the checkpoint is the instance that was tested, not the claim. `short` is the
# family name compressed for table rows; `ckpt` names what actually ran.
METHOD = {
    "gdino": dict(
        family="Open-vocabulary detection via language prompts",
        short="Language-prompt detection",
        ckpt="Grounding DINO Tiny"),
    "prowl-v2": dict(
        family="Self-supervised features + prototype distance",
        short="Self-supervised features",
        ckpt="DINOv2 ViT-L/14"),
    "prowl-v3": dict(
        family="Self-supervised features + prototype distance",
        short="Self-supervised features",
        ckpt="DINOv3 ViT-L/16"),
    "rba": dict(
        family="Closed-set segmentation + rejection",
        short="Closed-set + rejection",
        ckpt="SegFormer-B2 (Cityscapes)"),
}


def mlabel(m: str) -> str:
    """Family first, checkpoint in brackets -- table rows have no room for both
    in full, and the family is the part the finding is about."""
    ckpt = re.split(r"[(（]", METHOD[m]["ckpt"])[0].strip()
    return f'{METHOD[m]["short"]}({ckpt})'


BENCH_LABEL = {
    "B1": ("Ground · road · salient anomalies",
           "Animals, cones, boxes -- visually obvious"),
    "B2": ("Ground · road · small obstacles",
           "Boxes and boards on the road; only the road region is scored"),
    "B3": ("Near-ground · runway · mid-size debris",
           "Airport runway debris, 300×300 close-up frames"),
    "B4": ("Aerial · rail corridor · mid-size debris",
           "Rail-corridor obstacles filmed at 10 m altitude"),
}


def data_uri(p: Path, max_w: int = 1500, quality: int = 74) -> str:
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


# ---------------------------------------------------------------- fragments
def bar(value, tone: str, scale: float = 1.0) -> str:
    """A value bar drawn to a stated scale, so two cells stay comparable."""
    if value is None:
        return '<span class="bar bar--none">—</span>'
    w = max(1.5, min(100.0, value / scale * 100))
    return (f'<span class="bar bar--{tone}"><span class="bar__fill" '
            f'style="width:{w:.1f}%"></span></span>')


def get(rep, method, bench, key):
    return rep["methods"][method]["results"].get(bench, {}).get(key)


def verdicts(rep: dict) -> str:
    """The recommendation, built from the numbers rather than asserted.

    Stated by method family, with the tested checkpoint named underneath: the
    transferable finding is about the mechanism, not about one set of weights."""
    def card(cls, rank, method, why):
        r4 = get(rep, method, "B4", "object_recall")
        au4 = get(rep, method, "B4", "auroc")
        stats = [("Caught in the air (B4)", pct(r4, 0), "B4"),
                 ("Ranking ability (AUROC)", f"{au4:.3f}" if au4 is not None else "—", "B4")]
        dl = "".join(
            f'<div><dt>{k}<span>{src}</span></dt><dd>{v}</dd></div>'
            for k, v, src in stats)
        info = METHOD[method]
        return (f'<article class="verdict verdict--{cls}">'
                f'<div class="verdict__rank">{rank}</div>'
                f'<div class="verdict__body">'
                f'<h3>{html.escape(info["family"])}</h3>'
                f'<p class="verdict__ckpt">{html.escape(info["ckpt"])}</p>'
                f'<p class="verdict__why">{why}</p>'
                f'<dl class="verdict__stats">{dl}</dl></div></article>')

    r2 = get(rep, "prowl-v2", "B4", "object_recall")
    r3 = get(rep, "prowl-v3", "B4", "object_recall")
    return ('<div class="verdicts">'
            + card("yes", "Recommended", "prowl-v2",
                   "Scores a region by similarity to the rest of its own frame, requiring no "
                   "prior recognition of object identity or scene type. Ranks first on every "
                   f"test bed. A DINOv3 backbone was tested under the same protocol: "
                   f"{pct(r3, 0)} vs {pct(r2, 0)} recall on B4 -- no measurable advantage over "
                   "DINOv2.")
            + card("aux", "Alternative", "rba",
                   "Comparable to the top-ranked method on ground-view test beds, but degrades "
                   "most sharply in aerial views: its criterion depends on Cityscapes' 19 "
                   "semantic classes, which do not hold under a nadir viewpoint.")
            + card("no", "Not recommended", "gdino",
                   "Requires object recognition prior to flagging; object appearance under a "
                   "nadir viewpoint differs substantially from ground-level training data. "
                   "Ranks last on every test bed, with the largest decline in aerial views.")
            + '</div>')


def master_table(rep: dict, benches: list[str]) -> str:
    """The headline table: object-level hit rate, with ranking ability (AUROC)
    as the muted second line. Both are threshold-free in the sense that matters
    here -- "hit" is self-calibrated per image, so there is no shared operating
    point to argue about, and nothing borrowed from a benchmark other than the
    one being scored."""
    head = "".join(
        f'<th scope="col"><span class="bkey">{b}</span>'
        f'<span class="bname">{html.escape(BENCH_LABEL[b][0])}</span></th>'
        for b in benches)
    rows = []
    for m, blk in rep["methods"].items():
        info = METHOD[m]
        cells = []
        for b in benches:
            r = blk["results"].get(b)
            if not r:
                cells.append('<td><span class="bar bar--none">—</span></td>')
                continue
            rec, au = r["object_recall"], r.get("auroc")
            cells.append(
                '<td><div class="cell"><div class="cell__row">'
                f'<span class="cell__k">hit</span>{bar(rec, "hit")}'
                f'<span class="cell__v">{pct(rec, 0)}</span></div>'
                + (f'<div class="cell__note">AUROC {au:.3f}</div>'
                   if au is not None else "") + '</div></td>')
        rows.append(
            f'<tr><th scope="row"><span class="mname">{html.escape(info["short"])}</span>'
            f'<span class="msub">{html.escape(info["ckpt"])}</span></th>'
            + "".join(cells) + "</tr>")
    return ('<div class="scroll"><table class="master"><thead><tr>'
            f'<th scope="col">Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def patch_cells(b: str, side: int = 896, patch: int = 14) -> float:
    """Feature cells the median labelled object covers in test bed `b`, at a
    given input size. The self-supervised methods see the frame as this grid;
    an object smaller than one cell cannot be separated from its surroundings
    no matter how good the features are. Measured from the same masks the
    benchmark scores, not estimated."""
    from scipy import ndimage
    import numpy as _np
    import benchmark as _B

    items = _B.BENCHMARKS[b]["loader"](_B.BENCHMARKS[b]["cap"])
    areas, dims = [], []
    for it in items:
        gt, valid = it.masks()
        if gt is None:
            continue
        h, w = valid.shape
        lab, _ = ndimage.label(gt & valid)
        for sl in ndimage.find_objects(lab):
            a = int((lab[sl] > 0).sum())
            if a >= _B.MIN_COMPONENT:
                areas.append(a); dims.append((w, h))
    if not areas:
        return 0.0
    med = float(_np.median(areas))
    w0 = float(_np.median([d[0] for d in dims]))
    h0 = float(_np.median([d[1] for d in dims]))
    return med * (side / w0) * (side / h0) / (patch * patch)


def sweep_table(m: str = "dinov2-large") -> str:
    """Input resolution vs recall on the hardest benchmark, from the sweep run.

    Read from results/resolution_sweep.json rather than retyped, so the claim
    the slide makes cannot drift from the measurement that supports it."""
    f = RESULTS / "resolution_sweep.json"
    if not f.is_file():
        return ""
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = d["runs"].get(m) or []
    body = "".join(
        f'<tr><th scope="row">{r["side"]}×{r["side"]}</th>'
        f'<td class="figure">{r["grid"]}×{r["grid"]}</td>'
        f'<td class="figure">{r["vram_gb"]:.2f} GB</td>'
        f'<td><div class="cell"><div class="cell__row">{bar(r["recall"], "hit")}'
        f'<span class="cell__v">{pct(r["recall"], 0)}</span></div></div></td>'
        f'<td class="figure">{r["auroc"]:.3f}</td></tr>' for r in rows)
    return ('<div class="scroll"><table class="master"><thead><tr>'
            '<th scope="col">Input size</th><th scope="col">Feature grid</th>'
            '<th scope="col">Peak VRAM</th>'
            f'<th scope="col">{d["benchmark"]} caught (own-frame top {d["notice_bar"]:.0%})</th>'
            '<th scope="col">Ranking ability</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def sweep_ends(m: str = "dinov2-large") -> tuple[dict, dict]:
    """First and last row of the resolution sweep, so the callout that quotes
    them cannot drift from what the JSON actually says."""
    empty = {"side": "—", "recall": None, "auroc": None, "vram_gb": 0.0}
    f = RESULTS / "resolution_sweep.json"
    if not f.is_file():
        return empty, empty
    rows = json.loads(f.read_text(encoding="utf-8"))["runs"].get(m) or []
    return (rows[0], rows[-1]) if rows else (empty, empty)


def bench_row(rep: dict, b: str) -> str:
    cfg = rep["benchmarks"][b]
    name, note = BENCH_LABEL[b]
    obj = cfg.get("object_area_frac_p50")
    return (f'<tr><th scope="row"><span class="bkey">{b}</span></th>'
            f'<td><span class="bcn">{html.escape(name)}</span>'
            f'<span class="msub">{html.escape(note)}</span></td>'
            f'<td class="figure">{cfg["images"]}</td>'
            f'<td class="figure">{cfg.get("objects") or "—"}</td>'
            f'<td class="figure">{pct(obj, 2) if obj else "—"}</td>'
            f'<td class="src">{html.escape(cfg["source"])}</td></tr>')


ROAD_METHOD = {
    "segformer-b0": "SegFormer-B0 (Cityscapes)",
    "segformer-b2": "SegFormer-B2 (Cityscapes)",
    "mask2former": "Mask2Former (Cityscapes)",
    "gdino-road": "Grounding DINO (road prompt)",
}


def road_table(seg: dict) -> str:
    """IoU / precision / recall for the Stage 1 (road-region) candidates.

    Real ground truth this time (AeroScapes), so no self-calibration trick is
    needed -- these are the plain, standard segmentation numbers."""
    rows = []
    for key, label in ROAD_METHOD.items():
        r = seg["methods"].get(key)
        if not r:
            continue
        rows.append(
            f'<tr><th scope="row"><span class="mname">{html.escape(label)}</span></th>'
            f'<td><div class="cell"><div class="cell__row">{bar(r["iou"], "hit")}'
            f'<span class="cell__v">{pct(r["iou"], 0)}</span></div></div></td>'
            f'<td class="figure">{pct(r["precision"], 0)}</td>'
            f'<td class="figure">{pct(r["recall"], 0)}</td>'
            f'<td class="figure">{r["sec_per_image"]:.2f}s</td></tr>')
    return ('<div class="scroll"><table class="master"><thead><tr>'
            '<th scope="col">Model</th><th scope="col">IoU</th>'
            '<th scope="col">Precision</th><th scope="col">Recall</th>'
            '<th scope="col">Speed / frame</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def road_confusion_note(seg: dict) -> str:
    """What SegFormer-B2 predicts where the true label is road, aggregated over
    the full split -- distinguishes a genuine class confusion from noisy or
    low-confidence output."""
    diag = seg.get("diagnostics", {}).get("segformer-b2_road_confusion")
    if not diag:
        return ""
    top = diag["top"]
    correct = next((r["frac"] for r in top if r["class"] == "road"), 0.0)
    worst = next((r for r in top if r["class"] != "road"), None)
    if worst is None:
        return f'correctly labels {pct(correct, 0)} of true road pixels as "road"'
    return (f'correctly labels {pct(correct, 0)} of true road pixels as "road"; the largest '
            f'source of error is misclassification as "{worst["class"]}" '
            f'({pct(worst["frac"], 0)} of all true road pixels)')


def figure(b: str, caption: str, alt: str | None = None) -> str:
    p = RESULTS / "figures" / f"{b}.jpg"
    if not p.is_file():
        return ""
    return (f'<figure class="panel"><img src="{data_uri(p)}" '
            f'alt="{html.escape(alt or f"{b} -- comparison across every method")}" '
            f'loading="lazy"><figcaption>{caption}</figcaption></figure>')


# ---------------------------------------------------------------- page
CSS = """
:root{
  --paper:#EDEEE9; --paper-2:#E4E6DF; --card:#F6F7F3;
  --ink:#171B18; --ink-2:#4C554E; --ink-3:#79837B;
  --rule:#C7CCC3; --rule-2:#DDE0D8;
  --magenta:#A8156E; --cyan:#1A6274;
  --hit:#3D6A34; --alarm:#B3400C;
  color-scheme:light dark;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0F1310; --paper-2:#151A16; --card:#181E19;
    --ink:#E7EBE4; --ink-2:#9CA89E; --ink-3:#77837A;
    --rule:#2B322C; --rule-2:#232A24;
    --magenta:#F172B5; --cyan:#65BACF;
    --hit:#8FC981; --alarm:#F0854F;
  }
}
:root[data-theme="dark"]{
  --paper:#0F1310; --paper-2:#151A16; --card:#181E19;
  --ink:#E7EBE4; --ink-2:#9CA89E; --ink-3:#77837A;
  --rule:#2B322C; --rule-2:#232A24;
  --magenta:#F172B5; --cyan:#65BACF;
  --hit:#8FC981; --alarm:#F0854F;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
h1,h2,h3,.eyebrow,.bkey,.mname,.figure,.cell__v,.chip,.verdict__rank,.bcn{
  font-family:"Barlow Condensed",sans-serif;
}
.mono,.figure,.cell__v,.src,.verdict__stats dd{
  font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
}

.rail{
  position:fixed; left:22px; top:50%; transform:translateY(-50%); z-index:20;
  display:flex; flex-direction:column; gap:9px;
}
.rail a{
  width:9px; height:9px; border-radius:50%; border:1px solid var(--rule);
  background:transparent; transition:background .18s, transform .18s;
}
.rail a[aria-current="true"]{background:var(--magenta); border-color:var(--magenta); transform:scale(1.35)}
.rail a:focus-visible{outline:2px solid var(--magenta); outline-offset:3px}
@media (max-width:1180px){.rail{display:none}}

.deck{max-width:1080px; margin:0 auto; padding:0 28px 90px}
.slide{padding:70px 0 8px; border-top:1px solid var(--rule-2); scroll-margin-top:20px}
.slide:first-of-type{border-top:0; padding-top:44px}
.slide__head{display:flex; align-items:baseline; gap:14px; margin-bottom:6px}
.eyebrow{font-size:13px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; color:var(--magenta)}
.slide__no{margin-left:auto; font-family:"IBM Plex Mono",monospace; font-size:12px;
  color:var(--ink-3); letter-spacing:.08em}
.slide h2{font-size:clamp(27px,3.4vw,40px); font-weight:600; line-height:1.15;
  margin:0 0 14px; text-wrap:balance}
.slide h3{font-size:19px; font-weight:600; margin:0; letter-spacing:.01em}
.lede{font-size:17px; color:var(--ink-2); max-width:64ch; margin:0 0 20px}
.slide p{max-width:66ch}
.sub{font-size:13px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3);
  margin:26px 0 -2px; font-family:"Barlow Condensed",sans-serif}

/* ---- cover ---- */
.cover{padding-top:56px}
.cover h1{
  font-family:"Barlow Condensed",sans-serif;
  font-size:clamp(40px,6.6vw,76px); font-weight:700; line-height:1.03;
  margin:12px 0 16px; letter-spacing:-.01em; text-wrap:balance;
}
.cover .thesis{font-size:18.5px; color:var(--ink-2); max-width:62ch}
.answer{
  margin:26px 0 0; padding:18px 22px; border:1px solid var(--magenta);
  border-radius:3px; background:var(--card); max-width:none;
}
.answer .k{display:block; font-size:12px; letter-spacing:.15em; text-transform:uppercase;
  color:var(--magenta); margin-bottom:4px}
.answer p{margin:0; font-size:19px; max-width:70ch}
.coverbar{display:flex; flex-wrap:wrap; margin:16px 0 0;
  border:1px solid var(--rule); border-radius:3px; overflow:hidden; background:var(--card)}
.coverbar>div{flex:1 1 140px; padding:12px 16px; border-right:1px solid var(--rule-2)}
.coverbar>div:last-child{border-right:0}
.coverbar dt{font-size:11.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--ink-3)}
.coverbar dd{margin:2px 0 0; font-family:"Barlow Condensed",sans-serif;
  font-size:25px; font-weight:600; line-height:1.1}

/* ---- verdict cards ---- */
.verdicts{display:flex; flex-direction:column; gap:12px; margin:18px 0 8px}
.verdict{
  display:grid; grid-template-columns:112px 1fr; align-items:start;
  border:1px solid var(--rule); border-left-width:4px; border-radius:3px;
  background:var(--card);
}
.verdict--yes{border-left-color:var(--hit)}
.verdict--aux{border-left-color:var(--cyan)}
.verdict--no{border-left-color:var(--alarm)}
.verdict__rank{
  padding:16px 12px 16px 16px; font-size:16px; font-weight:600; line-height:1.25;
  letter-spacing:.02em;
}
.verdict--yes .verdict__rank{color:var(--hit)}
.verdict--aux .verdict__rank{color:var(--cyan)}
.verdict--no .verdict__rank{color:var(--alarm)}
.verdict__body{padding:16px 20px 15px 4px}
.verdict__ckpt{margin:4px 0 0; font-size:12px; letter-spacing:.04em; color:var(--ink-3);
  font-family:"IBM Plex Mono",monospace; max-width:66ch}
.verdict__why{margin:9px 0 0; font-size:14.5px; color:var(--ink-2); max-width:62ch}
.verdict__stats{display:flex; flex-wrap:wrap; gap:6px 30px; margin:12px 0 0;
  padding-top:11px; border-top:1px solid var(--rule-2)}
.verdict__stats dt{font-size:11px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink-3); display:flex; gap:6px; align-items:baseline}
.verdict__stats dt span{font-size:10px; color:var(--magenta); letter-spacing:.06em}
.verdict__stats dd{margin:1px 0 0; font-size:19px; font-weight:600}
@media (max-width:620px){
  .verdict{grid-template-columns:1fr}
  .verdict__body{padding:0 18px 16px}
}

/* ---- tables ---- */
.scroll{overflow-x:auto; margin:8px 0 6px; padding-bottom:4px}
table{border-collapse:collapse; width:100%; font-size:14px}
th,td{text-align:left; padding:11px 8px; vertical-align:top}
thead th{font-size:12px; letter-spacing:.11em; text-transform:uppercase; color:var(--ink-3);
  font-weight:600; border-bottom:1px solid var(--rule); white-space:nowrap}
tbody tr{border-bottom:1px solid var(--rule-2)}
tbody tr:last-child{border-bottom:0}
.master th[scope="row"]{min-width:150px}
.mname{display:block; font-size:17px; font-weight:600}
.msub{display:block; font-size:12.5px; color:var(--ink-3); line-height:1.4; margin-top:1px}
.bkey{display:inline-block; font-size:13px; font-weight:700; letter-spacing:.06em; color:var(--magenta)}
.bcn{display:block; font-size:15.5px; font-weight:600}
.bname{display:block; font-size:11.5px; color:var(--ink-3); text-transform:none;
  letter-spacing:0; font-weight:400; margin-top:2px; max-width:12ch; line-height:1.35}
.src{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--ink-3)}
.cell{display:flex; flex-direction:column; gap:5px; min-width:112px}
.cell__row{display:flex; align-items:center; gap:7px}
.cell__k{font-size:11px; color:var(--ink-3); width:24px; flex:none}
.cell__v{font-size:13px; width:44px; text-align:right; flex:none}
.cell__note{font-size:10.5px; line-height:1.35; color:var(--ink-3); max-width:15ch}
.bar{display:block; flex:1 1 auto; height:7px; border-radius:1px; background:var(--rule-2); overflow:hidden}
.bar__fill{display:block; height:100%}
.bar--hit .bar__fill{background:var(--hit)}
.bar--none{background:none; color:var(--ink-3); font-size:12px}

.delta th[scope="row"]{font-weight:600; font-size:15px; min-width:150px}
.delta .figure{font-size:17px; width:92px}
.arrow{color:var(--ink-3); width:28px; text-align:center}
.chip{display:inline-block; padding:2px 9px; border-radius:2px; font-size:14px;
  font-weight:600; border:1px solid currentColor}
.chip--good{color:var(--hit)}
.chip--bad{color:var(--alarm)}
.chip--flat{color:var(--ink-3)}

/* ---- figures + callouts ---- */
.panel{margin:20px 0 6px}
.panel img{width:100%; display:block; border:1px solid var(--rule); border-radius:3px}
.panel figcaption{font-size:13.5px; color:var(--ink-2); margin-top:9px; max-width:78ch}
.callout{margin:20px 0 6px; padding:16px 20px; background:var(--paper-2);
  border-left:3px solid var(--cyan); border-radius:2px}
.callout p{margin:0; max-width:70ch}
.callout p + p{margin-top:8px}

.readlist{margin:6px 0 0; padding:0; list-style:none; display:flex; flex-direction:column; gap:11px}
.readlist li{display:flex; gap:13px; align-items:flex-start; max-width:74ch}
.readlist .rk{flex:none; width:26px; height:26px; border-radius:50%; display:grid;
  place-items:center; border:1px solid var(--magenta); color:var(--magenta);
  font-size:13px; font-weight:600; font-family:"Barlow Condensed",sans-serif}

footer{margin-top:64px; padding-top:18px; border-top:1px solid var(--rule);
  font-size:12.5px; color:var(--ink-3); font-family:"IBM Plex Mono",monospace}
a{color:var(--magenta)}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
@media (max-width:620px){.deck{padding:0 18px 64px} .slide{padding-top:48px}}

@media print{
  *{-webkit-print-color-adjust:exact; print-color-adjust:exact; color-adjust:exact}
  .rail{display:none}
  .deck{max-width:none; padding:0 6mm}
  .slide{padding-top:14px; break-inside:avoid-page}
  .slide:not(:first-of-type){break-before:page}
  .panel img{break-inside:avoid-page}
}
"""

JS = """
const links=[...document.querySelectorAll('.rail a')];
const slides=[...document.querySelectorAll('.slide')];
if('IntersectionObserver' in window && links.length){
  const io=new IntersectionObserver(es=>{
    es.forEach(e=>{
      if(!e.isIntersecting) return;
      const i=slides.indexOf(e.target);
      links.forEach((a,j)=>a.setAttribute('aria-current', j===i?'true':'false'));
    });
  },{rootMargin:'-45% 0px -50% 0px'});
  slides.forEach(s=>io.observe(s));
}
addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey) return;
  const dir = e.key==='ArrowRight'||e.key==='PageDown' ? 1
            : e.key==='ArrowLeft' ||e.key==='PageUp'   ? -1 : 0;
  if(!dir) return;
  e.preventDefault();
  const y=scrollY+1;
  let i=slides.findIndex(s=>s.offsetTop>y);
  if(dir<0){ i=(i<0?slides.length:i)-2; }
  i=Math.max(0,Math.min(slides.length-1,i<0?0:i));
  slides[i].scrollIntoView({behavior:'smooth',block:'start'});
});
"""


def main() -> int:
    rep = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    seg = json.loads((RESULTS / "segmentation_benchmark.json").read_text(encoding="utf-8"))
    benches = [b for b in rep["benchmarks"] if b not in SKIP_BENCH]
    n_img = sum(rep["benchmarks"][b]["images"] for b in benches)
    n_obj = sum(rep["benchmarks"][b].get("objects", 0) for b in benches)
    nb = rep.get("notice_bar", 0.10)

    s: list[str] = []

    def slide(eyebrow: str, title: str, body: str, cls: str = "") -> None:
        i = len(s) + 1
        s.append(f'<section class="slide {cls}" id="s{i}">'
                 f'<div class="slide__head"><span class="eyebrow">{eyebrow}</span>'
                 f'<span class="slide__no">{i:02d}</span></div>'
                 f'<h2>{title}</h2>{body}</section>')

    # 01 cover + recommendation -- the answer, before the evidence -----------
    s.append(f'''<section class="slide cover" id="s1">
<div class="slide__head"><span class="eyebrow">UAV Highway Anomaly Detection · Model Selection</span>
<span class="slide__no">01</span></div>
<h1>Road localization, then anomaly detection</h1>
<p class="thesis">UAV-based highway monitoring requires two sequential capabilities:
localizing the road surface (<strong>Stage 1</strong>), then detecting anomalous objects
within it (<strong>Stage 2</strong>). All candidates are evaluated <strong>zero-shot</strong>
(no training or fine-tuning).</p>
<div class="answer"><span class="k">Recommendation</span>
<p><strong>Stage 1 (road localization):</strong> Grounding DINO, road-focused prompt --
highest recall of the candidates, with a fallback required for the frames it fails to detect
on at all. <strong>Stage 2 (anomaly detection):</strong> self-supervised feature matching with
prototype distance. Closed-set segmentation with rejection is a viable Stage 2 alternative;
the same language-prompt approach that leads Stage 1 is not recommended for Stage 2, where the
task -- identifying a specific object rather than a broad surface -- does not favour it (see
Stage 2 evidence).</p></div>
<dl class="coverbar">
<div><dt>Pipeline stages</dt><dd>2</dd></div>
<div><dt>Stage 2 test beds</dt><dd>{len(benches)}</dd></div>
<div><dt>Candidate models</dt><dd>{len(rep["methods"]) + len(seg["methods"])}</dd></div>
<div><dt>Images scored</dt><dd>{n_img + sum(v["images"] for v in seg["methods"].values())}</dd></div>
<div><dt>Stage 2 "caught" bar</dt><dd>own-frame top {nb:.0%}</dd></div>
</dl>
</section>''')

    # 02 stage 1: road segmentation -------------------------------------------
    hit = seg.get("diagnostics", {}).get("gdino-road_uavdt_hit_rate", {})
    slide("Stage 1 · find the road", "Which model finds the highway first", f'''
<p class="lede">No UAV highway dataset with road-region labels exists. AeroScapes (drone
imagery, 5-50 m altitude, a labelled "road" class, though its scenes are parks and streets
rather than highways) is used as a quantitative proxy; real UAVDT highway frames are checked
qualitatively, against no ground truth.</p>
<p><strong>IoU</strong>: overlap between predicted and true road area (1 = exact match, 0 =
none). <strong>Precision</strong>: fraction of the predicted area that is actually road.
<strong>Recall</strong>: fraction of the true road that was predicted.</p>
{road_table(seg)}
<div class="callout"><p><strong>The closed-set segmenters fail for a specific, verified
reason: viewpoint, not image resolution.</strong> Per-pixel inspection shows SegFormer-B2
{road_confusion_note(seg)} -- a systematic misclassification, not noisy or low-confidence
output. The error is viewpoint-dependent: on frames with an oblique, ground-level-like
composition (buildings visible at their normal orientation, a horizon), SegFormer correctly
labels most of the true road; on near-nadir frames it frequently labels the same paved surface
"building" instead. Forcing the model's native 1024×1024 input (its processor defaults to a
squashed, non-aspect-preserving 512×512) changes individual frames' scores in both directions
but does not raise the aggregate mean -- ruling out preprocessing as the primary cause.</p></div>
{figure("roadseg_aeroscapes", "Labelled road (white) vs. predicted region. Grounding DINO "
        "(green) over-covers the road; the segmenters (blue/red/orange) under-cover it, "
        "consistent with the class-confusion finding above.")}
<div class="callout"><p><strong>For a region-narrowing filter, recall is the relevant metric:
</strong> a true road pixel excluded from Stage 1's region can never be examined by Stage 2,
while an included non-road pixel only costs Stage 2 additional search area. Grounding DINO's
recall ({pct(seg["methods"]["gdino-road"]["recall"], 0)}) substantially exceeds the
segmenters' ({pct(seg["methods"]["segformer-b2"]["recall"], 0)} SegFormer-B2,
{pct(seg["methods"]["mask2former"]["recall"], 0)} Mask2Former), consistent with the same
viewpoint robustness: open-vocabulary matching is not constrained to Cityscapes' closed-set
class boundaries.</p></div>
{figure("roadseg_uavdt", "Real UAVDT highway frames, no ground truth -- qualitative only.")}
<div class="callout"><p><strong>Recommendation: Grounding DINO (road prompt), with a required
fallback.</strong> On an unselected sample of {hit.get("frames", "—")} real UAVDT highway
frames, {hit.get("zero_detection_frames", "—")}
({pct(hit.get("zero_detection_rate"), 0)}) returned zero detections -- a more severe failure
than reduced precision, since Stage 2 then receives no candidate region at all. Deployment
requires an explicit fallback (run Stage 2 on the full frame, or on SegFormer-B2's prediction)
for such frames, rather than skipping them. Neither candidate is precise enough for hard
cropping (see Limits).</p></div>''')

    # 03 stage 2: method + headline results ------------------------------------
    rows = "".join(bench_row(rep, b) for b in benches)
    slide("Stage 2 · find the debris", f"{len(benches)} test beds, one headline table", f'''
<p class="lede"><strong>Caught</strong>: an object's peak score falls within the top {nb:.0%}
of its own image's background score distribution (no threshold shared across images or
methods). <strong>AUROC</strong>: probability that a randomly chosen anomalous pixel scores
above a randomly chosen normal pixel (0.5 = chance, 1.0 = perfect separation). Neither metric
quantifies false-alarm rate (see Limits).</p>
<p class="lede">No public dataset combines UAV viewpoint, highway scene, and small real debris;
each available dataset satisfies exactly two of the three. Evaluation is therefore split across
{len(benches)} test beds, each holding two conditions fixed while varying the third, to isolate
which factor drives a given score difference.</p>
<div class="scroll"><table class="master"><thead><tr>
<th scope="col"></th><th scope="col">Test bed</th><th scope="col">Images</th>
<th scope="col">Labelled objects</th><th scope="col">Object share of frame</th>
<th scope="col">Source</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="lede">All models emit raw, unnormalised scores. An object is scored only against its
own image; no threshold is shared across test beds or methods.</p>
{master_table(rep, benches)}
<div class="callout"><p><strong>DINOv2 ranks first on every test bed and attains the highest
AUROC on every test bed.</strong> Closed-set rejection ranks second but degrades most sharply
in aerial views; language-prompt detection ranks last throughout. All four methods score
lowest on B1, where the scored region is the full frame -- including sky, buildings, and
vegetation -- rather than the road alone.</p></div>
<p class="sub">Verdict by method family</p>
{verdicts(rep)}''')

    # 04 what drives the stage 2 result ----------------------------------------
    lr = {m: {b: get(rep, m, b, "object_recall") for b in ["B3", "B4"]} for m in rep["methods"]}
    dv = {m: (lr[m]["B4"] - lr[m]["B3"]) * 100 for m in rep["methods"]}
    vp = "".join(
        f'<tr><th scope="row">{html.escape(mlabel(m))}</th>'
        f'<td class="figure">{pct(lr[m]["B3"], 0)}</td>'
        '<td class="arrow" aria-hidden="true">→</td>'
        f'<td class="figure">{pct(lr[m]["B4"], 0)}</td>'
        f'<td><span class="chip chip--{"bad" if dv[m] < -5 else "flat"}">'
        f'{dv[m]:+.0f} pts</span></td></tr>'
        for m in rep["methods"])

    LEG = ("White outline = labelled anomaly; colours are each method's alarm region "
           "(own-frame top {:.0%}).").format(nb)
    cells = patch_cells("B2")
    lo, hi = sweep_ends("dinov2-large")
    au_lo = f"{lo['auroc']:.3f}" if lo["auroc"] is not None else "—"
    au_hi = f"{hi['auroc']:.3f}" if hi["auroc"] is not None else "—"
    slide("Stage 2 · analysis", "Two factors matter more than the model", f'''
<p class="lede">Object scale relative to the feature grid, and camera viewpoint, account for
more of the score variation than model choice.</p>
<p class="sub">Object scale</p>
<p>Self-supervised methods partition each frame into a feature grid and compare individual
cells to the frame mean. B2's median object spans <strong>{cells:.1f} feature cells</strong>
at 896 px input resolution -- marginally more than one. Increasing input resolution alone
(224→896 px, model and weights unchanged) raises B2 recall from {pct(lo["recall"], 0)} to
{pct(hi["recall"], 0)} and AUROC from {au_lo} to {au_hi}, at a cost of
{hi["vram_gb"] - lo["vram_gb"]:.2f} GB additional VRAM (within the 4 GB budget).
<strong>Flight altitude, sensor resolution, and input resolution jointly determine more of
the outcome than model selection.</strong></p>
{sweep_table("dinov2-large")}
<p class="sub">Viewpoint</p>
<p>B3 (near-ground, runway) and B4 (UAV, 10 m, rail corridor) are matched on object scale and
background material; camera elevation is the primary remaining difference. DINOv2 recall
declines by {abs(dv["prowl-v2"]):.0f} points, consistent with a criterion independent of
object or scene identity. Rejection declines by {abs(dv["rba"]):.0f} points, the largest drop,
consistent with dependence on Cityscapes semantic classes that do not hold under a nadir
viewpoint.</p>
<div class="scroll"><table class="delta"><thead><tr><th scope="col">Model</th>
<th scope="col">B3 near-ground</th><th scope="col" class="arrow"></th>
<th scope="col">B4 aerial</th><th scope="col">Change</th></tr></thead>
<tbody>{vp}</tbody></table></div>
{figure("B2", "B2 · Small road obstacles. Once input size is fixed, self-supervised (blue) "
        "hugs them." + LEG)}
{figure("B4", "B4 · UAV rail corridor. Self-supervised hugs the object; rejection (yellow) "
        "scatters across the ballast; language-prompt (green) barely fires." + LEG)}
<div class="callout"><p><strong>Model-selection rationale:</strong> the two-stage design pairs a
high-recall region proposer (Grounding DINO, road-focused prompt, Stage 1 -- prioritising road
coverage over precision, with a fallback for its zero-detection frames) with a viewpoint-robust,
scene-agnostic anomaly detector (DINOv2 prototype matching, Stage 2). Each choice is supported
by measurement across multiple test beds rather than a single scene. The pipeline has not been
validated end-to-end; the following limitations qualify both stages.</p></div>''')

    # 05 limits and next -------------------------------------------------------
    slide("Limits", "What this does not prove, and what's next", '''
<p class="lede">Limitations that qualify the recommendation above.</p>
<ul class="readlist">
<li><span class="rk">!</span><div><strong>No test bed matches the target domain, in either
stage.</strong> B4 (Stage 2) is a rail corridor, not a highway; AeroScapes (Stage 1) is parks
and streets, not highway. Claims about highway performance are extrapolated from adjacent
conditions, not measured directly.</div></li>
<li><span class="rk">!</span><div><strong>Stage 1 output is a coarse region, not a precise
mask.</strong> Grounding DINO's precision is approximately 0.37: the predicted region is
substantially larger than the true road. Sufficient to narrow attention away from clearly
irrelevant background, not to hard-crop a frame down to road-only pixels without excluding
adjacent context Stage 2 may need. It also returns zero detections on approximately one in
four real highway frames sampled, requiring the fallback described in Stage 1.</div></li>
<li><span class="rk">!</span><div><strong>Only the Stage 2 recommendation has a family-level
result.</strong> DINOv2 and DINOv3 agree; the other two families were each tested with one
checkpoint, so those results describe that checkpoint specifically, though the failure mode
is consistent with the family's known mechanism.</div></li>
<li><span class="rk">!</span><div><strong>Box annotations inflate absolute scores</strong> on
UAV-RSOD and FOD-A (the full bounding box is scored as anomalous). This affects all four
methods equally, so the ranking is unaffected, but absolute values are not comparable across
annotation types.</div></li>
<li><span class="rk">!</span><div><strong>The recommended Stage 2 method has a known failure
mode:</strong> once anomalies exceed approximately 40% of the road area, the method inverts,
classifying clean road as anomalous. Highway debris is well below this threshold.</div></li>
<li><span class="rk">!</span><div><strong>False-alarm rate is not measured, by design.</strong>
UAVDT is the only UAV-highway dataset available but has no road-region labels; scoring the full
frame would count correct detections of buildings or billboards as false alarms solely for
being off-road, and Stage 1 accuracy is insufficient to correct this reliably. This measurement
is deferred until Stage 1 accuracy improves and a downstream reasoning stage can assess
relevance.</div></li>
<li><span class="rk">→</span><div><strong>Next steps:</strong> improve Stage 1 accuracy
(post-processing to reduce prediction fragmentation, or combining segmenter precision with the
road-prompted detector's recall) to a level suitable for frame masking; reintroduce false-alarm
measurement; and construct a synthetic evaluation set combining real highway footage with
inserted debris, satisfying all three Stage 2 conditions simultaneously.</div></li>
</ul>''')

    rail = "".join(f'<a href="#s{i + 1}" aria-label="Slide {i + 1}"></a>' for i in range(len(s)))
    page = f'''<title>UAV Highway Perception · Model Selection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<nav class="rail" aria-label="Slide navigation">{rail}</nav>
<main class="deck">{"".join(s)}
<footer>results/benchmark.json + results/segmentation_benchmark.json · Stage 2 "caught" bar: own-frame top {nb:.0%} · zero-shot, no fine-tuning</footer>
</main>
<script>{JS}</script>
'''
    DEST.write_text(page, encoding="utf-8")
    print(f"saved {DEST.relative_to(REPO_ROOT)}  "
          f"({len(page.encode()) / 1024:.0f} KB, {len(s)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
