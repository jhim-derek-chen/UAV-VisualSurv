<!-- class: title -->
<div class="kicker">Stage report · 28 September 2026</div>

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
<div class="blk vlm"><div class="bt">Identify</div><div class="fn">what is it</div><div class="kpi">debris named<br>A <b>56%</b><br>B, D <b>64%</b></div></div>
<div class="arrow">→</div>
<div class="blk cpu"><div class="bt">Context gate</div><div class="fn">drop false alarms</div><div class="kpi">false alarms to risk<br>A 34 → <b>1</b><br>B 23 → <b>0</b></div></div>
<div class="arrow">→</div>
<div class="blk vlm"><div class="bt">Assess risk</div><div class="fn">level + reason</div><div class="kpi">false alarms rated high<br>A <b>1</b><br>B, D <b>0</b></div></div>
<div class="arrow">→</div>
<div class="io next">Decide · Act<br><span class="tiny">next stage</span></div>
</div>

<h2>② Three architectures compared</h2>
<div class="row">
<div class="blk gpu"><div class="bt">A · local Qwen3.5-2B</div><div class="kpi">offline · free · ≈ <b>56 s</b> / frame</div></div>
<div class="blk api"><div class="bt">B · GPT-5.4 (API)</div><div class="kpi">asks the VLM about every box · ≈ <b>26 s</b> / frame</div></div>
<div class="blk api"><div class="bt">D · detector screens, GPT-5.4 on flagged boxes</div><div class="kpi">B's accuracy · <b>6.3 s</b> / frame · $0.006</div></div>
</div>

<div class="stats">
<div class="stat"><div class="v">30 / 30</div><div class="l">test frames run end to end:<br>frame → risk level + reason</div></div>
<div class="stat"><div class="v">0</div><div class="l">false alarms rated high risk<br>with the gate (B, D)</div></div>
<div class="stat"><div class="v">2.4 s</div><div class="l">tier 1 per frame, the drone's constant load;<br>6.3 s end to end with D</div></div>
</div>

???
The first goal is met: every stage runs on all 30 test frames, from the raw frame to a risk level with a written reason. The numbers under each block are the headline result of that stage. Detection finds 95% of vehicles and 90% of debris. The VLM names 56% of debris correctly locally and 64% with GPT-5.4. The context gate is what makes the output trustworthy: false alarms reaching the risk step drop from 34 to 1 locally and from 23 to 0 with GPT-5.4. For the second goal we compare three architectures. A runs fully offline but takes close to a minute per frame. B is more accurate but still asks the VLM about every box, mostly ordinary cars. D lets the detector screen first and asks GPT-5.4 only about boxes that may be debris: the same answers as B at 6.3 seconds per frame. The chain now has two tiers: tier 1 looks at every frame on the drone's GPU in 2.4 seconds; tier 2, the language model, runs only when tier 1 flags a candidate.

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

<div class="flow tierrow">
<div class="io" style="visibility:hidden">UAV frame<br><span class="tiny">RGB · up to 4K</span></div>
<div class="arrow" style="visibility:hidden">→</div>
<div class="tier t1" style="flex:1.25">Tier 1 · look · every frame, on the device</div>
<div class="arrow" style="visibility:hidden">→</div>
<div class="tier t2" style="flex:3.35">Tier 2 · assess · A, B: every frame · D: only when tier 1 flags a candidate</div>
</div>
<div class="flow">
<div class="io">UAV frame<br><span class="tiny">RGB · up to 4K</span></div>
<div class="arrow">→</div>
<div class="blk" style="flex:1.25"><div class="bt">① See</div><div class="fn">Objective 1 · class-agnostic</div>
<div class="sub"><div class="sb gpu">Road segmentation<br><i>SegFormer-B2 · AeroScapes</i></div><div class="dn">↓</div><div class="sb gpu">Object detection<br><i>OWLv2 objectness · tiled</i></div><div class="dn">↓</div><div class="sb cpu">Re-scoring + road filter<br><i>logistic probe</i></div><div class="dn">↓</div><div class="sb cpu"><b>D:</b> screen<br><i>3-class probe: only "debris" goes on</i></div></div>
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
</div>

<table class="mx">
<tr><th style="width:15%"></th><th>① See · tier 1</th><th>② Identify</th><th>③ Context gate</th><th>④ Assess risk</th></tr>
<tr><td><b>A</b> · local</td><td class="gpu">local GPU</td><td class="gpu">Qwen3.5-2B 4-bit · local · every box</td><td class="cpu">CPU (+ Qwen check)</td><td class="gpu">Qwen3.5-2B · local</td></tr>
<tr><td><b>B</b> · API</td><td class="gpu">local GPU</td><td class="api">GPT-5.4 · API · every box</td><td class="cpu">CPU (+ GPT check)</td><td class="api">GPT-5.4 · API</td></tr>
<tr><td><b>D</b> · screen + API</td><td class="gpu">local GPU + screen</td><td class="api">GPT-5.4 · API · flagged boxes only</td><td class="cpu">CPU (+ GPT check)</td><td class="api">GPT-5.4 · API</td></tr>
</table>
<div class="legend" style="margin-top:2mm"><span class="gpu" style="border:1px solid #999"></span>local GPU <span class="cpu" style="border:1px solid #999"></span>CPU <span class="api" style="border:1px solid #999"></span>cloud API <span class="vlm" style="border:1px solid #999"></span>VLM, per architecture (table)</div>

???
The main flow runs left to right; inside each block is the sub-flow and the model it uses. The band on top splits it into two tiers. Tier 1 is Objective 1 on the drone's GPU and must keep up with every frame: a drone-view road segmenter, OWLv2's objectness score, which finds objects without being told their class, and a small learned re-scorer. Tier 2 is the language model: identification with a physics check, the context gate, and the risk step. The table shows the differences. A runs the language model locally, B calls GPT-5.4; both ask it about every box. D adds one cheap step at the end of tier 1: the detector's own background, vehicle or debris probe screens the boxes, and only those it rates most likely debris go to tier 2. In B only such boxes can pass the gate anyway, so nothing that could reach the risk step is lost.

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
Same frames with GPT-5.4. It fixes both of A's failures: asked by the gate, it confirms the red box matches the lorry's cab, so no false alarm; and it names the fridge correctly. But it is not uniformly better: it misses the small tyre that A found, calling it a roadside post. And it judges each object on its own crop, so the fridge is rated high although it actually lies across the edge line on the hard shoulder. It also asks GPT-5.4 about every box, most of them ordinary cars, which is where its time goes.

---

<div class="kicker">Demo · architecture D</div>

# D · detector screens first, GPT-5.4 on flagged boxes

<div class="legend"><span style="background:#e33c3c"></span>high <span style="background:#1ea06e"></span>removed by gate <span style="background:#d73ca0"></span>missed <span style="background:#3c8ce6"></span>vehicle (by the detector) · banner: this frame's tier times</div>
<div class="grid2">
<div class="tile"><img src="figures/demo_d_23.jpg"><div class="cue">✓ tyre found · HIGH <span class="m">· 1 of 5 boxes sent to the VLM</span></div></div>
<div class="tile"><img src="figures/demo_d_26.jpg"><div class="cue">✗ small tyre → 'roadside structure' <span class="m">· as B · 1 of 30 boxes sent</span></div></div>
<div class="tile"><img src="figures/demo_d_18.jpg"><div class="cue">✓ lorry cab removed <span class="m">· as B · 2 of 9 boxes sent</span></div></div>
<div class="tile"><img src="figures/demo_d_02.jpg"><div class="cue">✓ fridge found · HIGH <span class="m">· 1 of 3 boxes sent</span></div></div>
</div>

???
The same four frames with D. The answers are B's, by construction: the boxes that go to GPT-5.4 are exactly the ones that could reach B's risk step, so the tyre, the lorry cab and the fridge come out as with B, and the small tyre is missed as with B. What changes is the work: on the busy frame on the right, GPT-5.4 is asked about one box instead of thirty. The banner on each frame shows its two tier times: about 2.4 seconds to look, then 2 to 5 seconds of language model when something was flagged. The risk to keep in mind is that the screen is a probe trained on our renders: a kind of debris it has never seen could be taken for a vehicle and never reach the language model. None of the 27 debris boxes in the test set were lost this way.

---

<div class="kicker">Evaluation</div>

# Time, accuracy and GPU in one table

<table class="ev">
<tr><th style="width:40%"></th><th>A · local Qwen</th><th>B · GPT-5.4</th><th>D · screen + GPT-5.4</th></tr>
<tr class="g"><td colspan="4">Tier 1 · look · every frame, on the drone's GPU · s per frame, 30 test frames, RTX 3050 Ti (4 GB)</td></tr>
<tr><td>road segmentation ∥ object detection (OWLv2), run side by side</td><td class="n">2.36</td><td class="n">2.36</td><td class="n">2.36</td></tr>
<tr><td>re-scoring (+ screen in D)</td><td class="n">&lt; 0.01</td><td class="n">&lt; 0.01</td><td class="n">&lt; 0.01</td></tr>
<tr class="t1tot"><td>TIER 1 TOTAL · can the drone keep up?</td><td class="n">2.4</td><td class="n">2.4</td><td class="n">2.4</td></tr>
<tr class="g"><td colspan="4">Tier 2 · assess · A, B: every frame · D: only frames tier 1 flags (27 of 30 here, all hold debris)</td></tr>
<tr><td>identification (VLM) · <b>bottleneck</b></td><td class="n bn">49.2</td><td class="n bn">22.0</td><td class="n">2.1</td></tr>
<tr><td>context gate</td><td class="n">0.09</td><td class="n">0.13</td><td class="n">0.05</td></tr>
<tr><td>risk assessment (VLM)</td><td class="n">4.45</td><td class="n">≈ 1.3</td><td class="n">1.8</td></tr>
<tr class="tot"><td>tier 2 total</td><td class="n">53.7</td><td class="n">≈ 23.4</td><td class="n best">4.0</td></tr>
<tr><td>VLM calls per frame</td><td class="n">16.2</td><td class="n">14.9</td><td class="n best">2.3</td></tr>
<tr class="tot"><td>END TO END, s per frame</td><td class="n">≈ 56</td><td class="n">≈ 26</td><td class="n best">6.3</td></tr>
<tr class="g"><td colspan="4">Accuracy · 30 test frames</td></tr>
<tr><td>detection: vehicles · debris · precision</td><td class="span" colspan="3">95.2% · 90.0% · 84.1% (shared)</td></tr>
<tr><td>debris named correctly</td><td class="n">56%</td><td class="n best">64%</td><td class="n best">64%</td></tr>
<tr><td>vehicles recognised</td><td class="n">96.0% (VLM)</td><td class="n">96.4% (VLM)</td><td class="n best">98.5% (detector)</td></tr>
<tr><td>debris boxes assessed for risk (of 27) · real debris lost by gate or screen</td><td class="n">22 · 0</td><td class="n">21 · 0</td><td class="n">21 · 0</td></tr>
<tr><td>false alarms rated high</td><td class="n">1</td><td class="n best">0</td><td class="n best">0</td></tr>
<tr class="g"><td colspan="4">GPU memory · peak</td></tr>
<tr><td>on the device</td><td class="n">2.19 GB tier 1 · 1.97 GB VLM</td><td class="n">2.19 GB</td><td class="n">2.31 GB</td></tr>
<tr><td>all models loaded at once</td><td class="n bn">4.05 GB &gt; 4 GB card</td><td class="n">2.19 GB</td><td class="n">2.31 GB</td></tr>
<tr class="g"><td colspan="4">Deployment</td></tr>
<tr><td>API cost per frame · runs offline</td><td class="n best">$0 · yes</td><td class="n">$0.036 · no</td><td class="n">$0.0056 · no</td></tr>
</table>

???
Everything in one table, split into the two tiers. Tier 1 is what the drone must do for every frame: 2.4 seconds, the same for all three, since Objective 1 is shared; we sped its code up from 4.7 seconds with bit-identical output, by vectorising a duplicate filter and overlapping CPU and GPU work. Tier 2 is the language model. In A and B it runs on every frame and asks about every box, which makes identification the bottleneck: 49 and 22 seconds. D asks about 2 boxes per frame instead of 15 and runs tier 2 only when tier 1 flags something: 4 seconds, 6.3 end to end. On this test set every frame holds debris, so tier 2 ran on 27 of 30 frames; on a patrol it would run far less often. Accuracy: D matches B on every test-set measure and recognises vehicles slightly better. GPU: A only fits the 4 GB card if its models are loaded one stage at a time; B and D need only the detector on the device. D also cuts the API bill to about half a cent per frame.

---
<!-- class: big -->
<div class="kicker">Takeaways</div>

# Where we stand, what comes next

<div class="row" style="margin-top:4mm">
<div class="col">
<h2>This stage</h2>
<ul>
<li>✓ chain connected: frame → objects → identity → gate → risk + reason</li>
<li>✓ "no false alarm rated high": B and D on the test set</li>
<li>✓ D: B's accuracy at 6.3 s per frame · tier 1 2.4 s</li>
<li>A: offline and free, but ≈ 56 s per frame</li>
</ul>
</div>
<div class="col">
<h2>Next stage</h2>
<ul>
<li>decide + act: lane signals, message signs</li>
<li>tier 1 below 1 s, without losing detections</li>
<li>small-object detection (pallets ≈ 22 px)</li>
<li>object size from altitude + focal length</li>
</ul>
</div>
</div>

???
To close: the chain now runs end to end, the gate makes its alerts trustworthy, and D brings it to 6.3 seconds per frame with B's accuracy: 2.4 seconds of looking on the drone for every frame, and a few seconds of language model only when something is flagged. A remains the option for an offline, private product. The next stage is the decision and execution step. On the perception side, the target is tier 1 below one second without losing detections; LeVJEPA, a new video encoder we tried for this, was fast but missed too much, so that work continues.
