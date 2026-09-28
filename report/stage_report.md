<!-- class: title -->
<div class="kicker">Stage report · 27 September 2026</div>

# UAV-VisualSurv

<div class="sub">Highway hazard perception from a drone:<br>connecting the chain from seeing to risk, and comparing three architectures</div>

???
This stage had two aims: first, connect the whole chain from a drone frame to a risk judgement; second, compare three ways of running it. The deck follows that order: what we built, what it looks like on real frames, and how the three compare in time, accuracy and hardware.

---

<div class="kicker">Overview</div>

# This stage: connect the chain, compare the architectures

<h2>① Chain connected end to end · all 30 test frames</h2>
<div class="flow">
<div class="io">UAV frame</div>
<div class="arrow">→</div>
<div class="blk gpu"><div class="bt">See</div><div class="fn">road + every object</div><div class="kpi">vehicles <b>95%</b><br>debris <b>90%</b></div></div>
<div class="arrow">→</div>
<div class="blk vlm"><div class="bt">Identify</div><div class="fn">what is it</div><div class="kpi">debris named<br>A <b>56%</b><br>B <b>64%</b></div></div>
<div class="arrow">→</div>
<div class="blk cpu"><div class="bt">Context gate</div><div class="fn">drop false alarms</div><div class="kpi">false alarms to risk<br>A 34 → <b>1</b><br>B 23 → <b>0</b></div></div>
<div class="arrow">→</div>
<div class="blk vlm"><div class="bt">Assess risk</div><div class="fn">level + reason</div><div class="kpi">false alarms rated high<br>A <b>1</b><br>B <b>0</b></div></div>
<div class="arrow">→</div>
<div class="blk api addon"><div class="bt">Scene analysis</div><div class="fn">add-on (C)</div><div class="kpi">lane vs shoulder <b>24/25</b><br>spills grouped <b>4/4</b></div></div>
<div class="arrow">→</div>
<div class="io next">Decide · Act<br><span class="tiny">next stage</span></div>
</div>

<h2>② Three architectures compared</h2>
<div class="row">
<div class="blk gpu"><div class="bt">A · local Qwen3.5-2B</div><div class="kpi">offline · free · <b>58 s</b> / frame</div></div>
<div class="blk api"><div class="bt">B · GPT-5.4 (API)</div><div class="kpi">best accuracy · <b>28 s</b> / frame · $0.036</div></div>
<div class="blk api addon"><div class="bt">C · B + scene analysis</div><div class="kpi">relational judgement · +2.4 s · +$0.007</div></div>
</div>

<div class="stats">
<div class="stat"><div class="v">30 / 30</div><div class="l">test frames run end to end:<br>frame → risk level + reason</div></div>
<div class="stat"><div class="v">0</div><div class="l">false alarms rated high risk<br>with the gate (B, C)</div></div>
<div class="stat"><div class="v">84%</div><div class="l">of A's time spent identifying boxes:<br>the bottleneck</div></div>
</div>

???
The first goal is met: every stage runs on all 30 test frames, from the raw frame to a risk level with a written reason. The numbers under each block are the headline result of that stage. Detection finds 95% of vehicles and 90% of debris. The VLM names 56% of debris correctly locally and 64% with GPT-5.4. The context gate is what makes the output trustworthy: false alarms reaching the risk step drop from 34 to 1 locally and from 23 to 0 with GPT-5.4. For the second goal, A runs fully offline but takes about a minute per frame; B is twice as fast and more accurate but needs the API; C adds a scene-level judgement for a small extra cost. The next stage, decision and execution, is greyed out on the right.

---

<div class="kicker">Setup · dataset</div>

# Real UAV motorway frames + rendered 3D debris

<img src="figures/fig_dataset.png" style="width:100%; max-height:122mm; object-fit:contain">

<div class="chips" style="margin-top:3mm">
<span>4 open-motorway locations</span><span>30 test frames · fixed</span><span>30 dev frames · all tuning</span><span>342 vehicles · hand-checked</span><span>1 debris object per test frame</span><span>leave-one-location-out scoring</span>
</div>

???
No public dataset has UAV imagery of open motorways with labelled debris, so we built one. The background is a real drone video frame. The debris is a 3D model from Sketchfab, rendered in Blender and placed at its real-world size using a per-location ground resolution calibrated on real cars. Debris labels come from the render itself, so they are pixel-exact; the 342 vehicle labels were proposed by Grounding DINO and every one was checked by hand. All settings are chosen on a separate development set, and learned parts are always tested on a location they never saw.

---
<!-- class: arch -->
<div class="kicker">Setup · architectures</div>

# One chain, three architectures

<div class="flow">
<div class="io">UAV frame<br><span class="tiny">RGB · up to 4K</span></div>
<div class="arrow">→</div>
<div class="blk" style="flex:1.2"><div class="bt">① See</div><div class="fn">Objective 1 · class-agnostic</div>
<div class="sub"><div class="sb gpu">Road segmentation<br><i>SegFormer-B2 · AeroScapes</i></div><div class="dn">↓</div><div class="sb gpu">Object detection<br><i>OWLv2 objectness · tiled</i></div><div class="dn">↓</div><div class="sb cpu">Re-scoring + road filter<br><i>logistic probe</i></div></div>
<div class="out">out: boxes + features</div></div>
<div class="arrow">→</div>
<div class="blk"><div class="bt">② Identify</div><div class="fn">what each box is</div>
<div class="sub"><div class="sb vlm">VLM: describe → 1 of 16 categories<br><i>two crops + size in metres</i></div><div class="dn">↓</div><div class="sb cpu">Physics check<br><i>category must fit the size</i></div></div>
<div class="out">out: name + category</div></div>
<div class="arrow">→</div>
<div class="blk"><div class="bt">③ Context gate</div><div class="fn">remove false alarms</div>
<div class="sub"><div class="sb cpu">cut by frame edge</div><div class="sb cpu">&gt; 12.5 m from traffic</div><div class="sb cpu">detector: not debris<br><i>3-class probe</i></div><div class="sb cpu">part of a vehicle<br><i>geometry · VLM check</i></div></div>
<div class="out">out: hazard candidates</div></div>
<div class="arrow">→</div>
<div class="blk"><div class="bt">④ Assess risk</div><div class="fn">per object</div>
<div class="sub"><div class="sb vlm">VLM: wide view + facts<br><i>size · distance to traffic</i></div><div class="dn">↓</div><div class="sb vlm">reason → level<br><i>high · medium · low</i></div></div>
<div class="out">out: risk + reason</div></div>
<div class="arrow">→</div>
<div class="blk addon"><div class="bt">⑤ Scene analysis</div><div class="fn">add-on · C only</div>
<div class="sub"><div class="sb api">whole frame, numbered<br><i>overview + close views + facts</i></div><div class="dn">↓</div><div class="sb api">lane / shoulder · events<br><i>lanes blocked · action</i></div></div>
<div class="out">out: scene risk + action</div></div>
</div>

<table class="mx">
<tr><th style="width:15%"></th><th>① See</th><th>② Identify</th><th>③ Context gate</th><th>④ Assess risk</th><th>⑤ Scene analysis</th></tr>
<tr><td><b>A</b> · local</td><td class="gpu">local GPU</td><td class="gpu">Qwen3.5-2B 4-bit · local</td><td class="cpu">CPU (+ Qwen check)</td><td class="gpu">Qwen3.5-2B · local</td><td style="text-align:center">–</td></tr>
<tr><td><b>B</b> · API</td><td class="gpu">local GPU</td><td class="api">GPT-5.4 · API</td><td class="cpu">CPU (+ GPT check)</td><td class="api">GPT-5.4 · API</td><td style="text-align:center">–</td></tr>
<tr><td><b>C</b> · B + add-on</td><td class="gpu">local GPU</td><td class="api">GPT-5.4 · API</td><td class="cpu">CPU (+ GPT check)</td><td class="api">GPT-5.4 · API</td><td class="api">GPT-5.4 · API</td></tr>
</table>
<div class="legend" style="margin-top:2mm"><span class="gpu" style="border:1px solid #999"></span>local GPU <span class="cpu" style="border:1px solid #999"></span>CPU <span class="api" style="border:1px solid #999"></span>cloud API <span class="vlm" style="border:1px solid #999"></span>VLM, per architecture (table)</div>

???
The main flow runs left to right; inside each block is the sub-flow and the model it uses. Objective 1 is shared by all three architectures: a drone-view road segmenter, then OWLv2's objectness score, which finds objects without being told their class, then a small learned re-scorer. Identification asks a vision-language model to describe each box and choose a category, and a physics check rejects answers that cannot match the measured size. The context gate applies four checks that only ever remove candidates. The risk step writes its reason before the level. The table underneath shows the only differences: A runs the language model locally, B and C call GPT-5.4, and only C adds the scene-analysis block, which is an add-on.

---

<div class="kicker">Demo · architecture A</div>

# A · local Qwen3.5-2B

<div class="legend"><span style="background:#e33c3c"></span>high <span style="background:#f08c14"></span>medium <span style="background:#1ea06e"></span>removed by gate <span style="background:#d73ca0"></span>missed <span style="background:#3c8ce6"></span>vehicle</div>
<div class="grid2">
<div class="tile"><img src="figures/demo_a_23.jpg"><div class="cue">✓ tyre found · HIGH <span class="m">· truck cut by frame edge removed</span></div></div>
<div class="tile"><img src="figures/demo_a_26.jpg"><div class="cue">✓ small tyre found · HIGH</div></div>
<div class="tile"><img src="figures/demo_a_18.jpg"><div class="cue">✗ lorry cab → 'barrel' · HIGH <span class="m">· false alarm reaches the alert</span></div></div>
<div class="tile"><img src="figures/demo_a_02.jpg"><div class="cue">✗ fridge → 'roadside structure' <span class="m">· hazard missed</span></div></div>
</div>

???
Four test frames, the same four for B on the next slide. Top row: what works. The tyre is found and rated high, and a truck cut by the frame edge is removed by the gate instead of raising an alarm. Even a small tyre in a busy lane is found. Bottom row: the weaknesses of a 2-billion-parameter model. It calls the red cab of an articulated lorry a barrel, and when the gate asks whether it is part of the lorry, it says no, so a false alarm is rated high. It also misreads a fridge seen from above as a roadside structure, so a real hazard is missed.

---

<div class="kicker">Demo · architecture B</div>

# B · GPT-5.4

<div class="legend"><span style="background:#e33c3c"></span>high <span style="background:#f08c14"></span>medium <span style="background:#1ea06e"></span>removed by gate <span style="background:#d73ca0"></span>missed <span style="background:#3c8ce6"></span>vehicle</div>
<div class="grid2">
<div class="tile"><img src="figures/demo_b_23.jpg"><div class="cue">✓ tyre found · HIGH</div></div>
<div class="tile"><img src="figures/demo_b_26.jpg"><div class="cue">✗ small tyre → 'roadside structure' <span class="m">· missed</span></div></div>
<div class="tile"><img src="figures/demo_b_18.jpg"><div class="cue">✓ lorry cab removed <span class="m">· GPT: "part of the vehicle"</span></div></div>
<div class="tile"><img src="figures/demo_b_02.jpg"><div class="cue">✓ fridge found · HIGH <span class="m">· but it lies on the shoulder</span></div></div>
</div>

???
Same frames with GPT-5.4. It fixes both of A's failures: asked by the gate, it confirms the red box matches the lorry's cab, so no false alarm; and it names the fridge correctly. But it is not uniformly better: it misses the small tyre that A found, calling it a roadside post. And it judges each object on its own crop, so the fridge is rated high although it actually lies across the edge line on the hard shoulder. That last point is what the scene-analysis add-on addresses.

---

<div class="kicker">Demo · add-on</div>

# C · scene analysis: the frame judged as a whole

<p class="small muted" style="margin:-3mm 0 2mm 0">23 scene-relation frames · lane / shoulder / spill / blockage / clear · answers fixed before running</p>
<div class="grid2">
<div class="tile"><img src="figures/demo_c_12.jpg"><div class="cue">✓ shoulder object: HIGH alone → MEDIUM <span class="m">· "shoulder clearance"</span></div></div>
<div class="tile"><img src="figures/demo_c_13.jpg"><div class="cue">✓ 4 planks → one spill from lorry V9 <span class="m">· 2 lanes · "close lanes"</span></div></div>
<div class="tile"><img src="figures/demo_c_22.jpg"><div class="cue">✓ tyre + fridge side by side → 2 lanes blocked</div></div>
<div class="tile"><img src="figures/demo_c_07.jpg"><div class="cue">✗ planks on the shoulder judged "in lane" <span class="m">· HIGH, expected MEDIUM</span></div></div>
</div>

???
This is the add-on. To test it we built a small scene-relation set where the right answer depends on relations: the same object in a lane or on the hard shoulder, a load strewn behind a lorry, or two objects blocking adjacent lanes; the expected answers were written down before any model ran. C sees the whole frame with every candidate numbered. It moves a ladder on the shoulder from high to medium and asks for a shoulder clearance instead of a lane closure. It groups four planks into one spilled load, names the open-load lorry as the likely source and counts two blocked lanes. It is not always right: here it puts planks on the shoulder into the lane.

---

<div class="kicker">Evaluation</div>

# Time, accuracy and GPU in one table

<table class="ev">
<tr><th style="width:44%"></th><th>A · local Qwen</th><th>B · GPT-5.4</th><th>C · B + scene add-on</th></tr>
<tr class="g"><td colspan="4">Inference time · s per frame · 30 test frames · RTX 3050 Ti (4 GB)</td></tr>
<tr><td>road segmentation</td><td class="n">0.32</td><td class="n">0.32</td><td class="n">0.32</td></tr>
<tr><td>object detection (OWLv2)</td><td class="n">4.33</td><td class="n">4.33</td><td class="n">4.33</td></tr>
<tr><td>re-scoring</td><td class="n">&lt; 0.01</td><td class="n">&lt; 0.01</td><td class="n">&lt; 0.01</td></tr>
<tr><td><b>identification (VLM, per box) · bottleneck</b></td><td class="n bn">49.2</td><td class="n bn">22.0</td><td class="n bn">22.0</td></tr>
<tr><td>context gate</td><td class="n">0.09</td><td class="n">0.13</td><td class="n">0.13</td></tr>
<tr><td>risk assessment (VLM)</td><td class="n">4.45</td><td class="n">≈ 1.3</td><td class="n">≈ 1.3</td></tr>
<tr><td>scene analysis (add-on)</td><td class="n">–</td><td class="n">–</td><td class="n">2.4</td></tr>
<tr><td><b>total</b></td><td class="n"><b>58.4</b></td><td class="n best">≈ 28</td><td class="n"><b>≈ 30.5</b></td></tr>
<tr class="g"><td colspan="4">Accuracy · 30 test frames</td></tr>
<tr><td>detection: vehicles · debris · precision</td><td class="span" colspan="3">95.2% · 90.0% · 84.1% (shared)</td></tr>
<tr><td>debris named correctly</td><td class="n">56%</td><td class="n best">64%</td><td class="n best">64%</td></tr>
<tr><td>vehicles named correctly</td><td class="n">96.0%</td><td class="n">96.4%</td><td class="n">96.4%</td></tr>
<tr><td>false alarms removed by the gate · real debris lost</td><td class="n">33 · 0</td><td class="n">24 · 0</td><td class="n">24 · 0</td></tr>
<tr><td>false alarms rated high</td><td class="n">1</td><td class="n best">0</td><td class="n best">0</td></tr>
<tr class="g"><td colspan="4">Accuracy · 23 scene-relation frames (add-on test)</td></tr>
<tr><td>scene risk correct</td><td class="n">16 / 23</td><td class="n">16 / 23</td><td class="n best">17 / 23</td></tr>
<tr><td>lane vs shoulder · spill grouped · lanes blocked</td><td class="n">–</td><td class="n">–</td><td class="n">24/25 · 4/4 · 16/23</td></tr>
<tr class="g"><td colspan="4">GPU memory · peak</td></tr>
<tr><td>Objective 1 pass · VLM pass</td><td class="n">2.19 GB · 1.97 GB</td><td class="n">2.19 GB · cloud</td><td class="n">2.19 GB · cloud</td></tr>
<tr><td>all models loaded at once</td><td class="n bn">4.05 GB &gt; 4 GB card</td><td class="n">2.19 GB</td><td class="n">2.19 GB</td></tr>
<tr class="g"><td colspan="4">Deployment</td></tr>
<tr><td>API cost per frame · runs offline</td><td class="n best">$0 · yes</td><td class="n">$0.036 · no</td><td class="n">$0.043 · no</td></tr>
</table>

???
Everything in one table. Time: the bottleneck is identification, one VLM call per box and about thirteen boxes per frame, mostly ordinary vehicles; it is 84% of A's time and 78% of B's. Detection plus the gate costs under five seconds. Accuracy: detection is shared; B names debris best and, with C, is the only one with no false alarm rated high on the test set. On the scene-relation set C is only one frame better than A and B on the overall level, but it is the only one that can say lane or shoulder, group a spill and count blocked lanes. GPU: A only fits the 4 GB card if models are loaded one stage at a time; B and C need only the detector on the device. The price is an API bill of about four cents per frame and no offline operation.

---
<!-- class: big -->
<div class="kicker">Takeaways</div>

# Where we stand, what comes next

<div class="row" style="margin-top:4mm">
<div class="col">
<h2>This stage</h2>
<ul>
<li>✓ chain connected: frame → objects → identity → gate → risk + reason</li>
<li>✓ "no false alarm rated high": met by B and C on the test set</li>
<li>B: best accuracy and speed today</li>
<li>A: offline and free, but slow</li>
<li>C: add-on for relational decisions</li>
</ul>
</div>
<div class="col">
<h2>Next stage</h2>
<ul>
<li>decide + act: lane signals, message signs</li>
<li>bottleneck: identify only unclear boxes, in parallel</li>
<li>small-object detection (pallets ≈ 22 px)</li>
<li>object size from altitude + focal length</li>
</ul>
</div>
</div>

???
To close: the chain now runs end to end, and the gate makes its alerts trustworthy with GPT-5.4. For a demonstration, B is the best choice today; A matters for an offline, private product; C is a useful add-on when the control room needs to know which lanes to close and whether several objects are one incident. The next stage is the decision and execution step. On the perception side, the clear target is identification time, for example by skipping boxes the detector is already sure are vehicles and by running calls in parallel, and better detection of small objects.
