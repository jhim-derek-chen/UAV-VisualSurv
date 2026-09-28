# Speaker notes

## 1. UAV-VisualSurv

This stage had two aims: first, connect the whole chain from a drone frame to a risk judgement; second, compare three ways of running it. The deck follows that order: what we built, what it looks like on real frames, and how the three compare in time, accuracy and hardware.

## 2. This stage: connect the chain, compare the architectures

The first goal is met: every stage runs on all 30 test frames, from the raw frame to a risk level with a written reason. The numbers under each block are the headline result of that stage. Detection finds 95% of vehicles and 90% of debris. The VLM names 56% of debris correctly locally and 64% with GPT-5.4. The context gate is what makes the output trustworthy: false alarms reaching the risk step drop from 34 to 1 locally and from 23 to 0 with GPT-5.4. For the second goal, A runs fully offline but takes about a minute per frame; B is twice as fast and more accurate but needs the API; C adds a scene-level judgement for a small extra cost. The next stage, decision and execution, is greyed out on the right.

## 3. Real UAV motorway frames + rendered 3D debris

No public dataset has UAV imagery of open motorways with labelled debris, so we built one. The background is a real drone video frame. The debris is a 3D model from Sketchfab, rendered in Blender and placed at its real-world size using a per-location ground resolution calibrated on real cars. Debris labels come from the render itself, so they are pixel-exact; the 342 vehicle labels were proposed by Grounding DINO and every one was checked by hand. All settings are chosen on a separate development set, and learned parts are always tested on a location they never saw.

## 4. One chain, three architectures

The main flow runs left to right; inside each block is the sub-flow and the model it uses. Objective 1 is shared by all three architectures: a drone-view road segmenter, then OWLv2's objectness score, which finds objects without being told their class, then a small learned re-scorer. Identification asks a vision-language model to describe each box and choose a category, and a physics check rejects answers that cannot match the measured size. The context gate applies four checks that only ever remove candidates. The risk step writes its reason before the level. The table underneath shows the only differences: A runs the language model locally, B and C call GPT-5.4, and only C adds the scene-analysis block, which is an add-on.

## 5. A · local Qwen3.5-2B

Four test frames, the same four for B on the next slide. Top row: what works. The tyre is found and rated high, and a truck cut by the frame edge is removed by the gate instead of raising an alarm. Even a small tyre in a busy lane is found. Bottom row: the weaknesses of a 2-billion-parameter model. It calls the red cab of an articulated lorry a barrel, and when the gate asks whether it is part of the lorry, it says no, so a false alarm is rated high. It also misreads a fridge seen from above as a roadside structure, so a real hazard is missed.

## 6. B · GPT-5.4

Same frames with GPT-5.4. It fixes both of A's failures: asked by the gate, it confirms the red box matches the lorry's cab, so no false alarm; and it names the fridge correctly. But it is not uniformly better: it misses the small tyre that A found, calling it a roadside post. And it judges each object on its own crop, so the fridge is rated high although it actually lies across the edge line on the hard shoulder. That last point is what the scene-analysis add-on addresses.

## 7. C · scene analysis: the frame judged as a whole

This is the add-on. To test it we built a small scene-relation set where the right answer depends on relations: the same object in a lane or on the hard shoulder, a load strewn behind a lorry, or two objects blocking adjacent lanes; the expected answers were written down before any model ran. C sees the whole frame with every candidate numbered. It moves a ladder on the shoulder from high to medium and asks for a shoulder clearance instead of a lane closure. It groups four planks into one spilled load, names the open-load lorry as the likely source and counts two blocked lanes. It is not always right: here it puts planks on the shoulder into the lane.

## 8. Time, accuracy and GPU in one table

Everything in one table. Time: the bottleneck is identification, one VLM call per box and about thirteen boxes per frame, mostly ordinary vehicles; it is 84% of A's time and 78% of B's. Detection plus the gate costs under five seconds. Accuracy: detection is shared; B names debris best and, with C, is the only one with no false alarm rated high on the test set. On the scene-relation set C is only one frame better than A and B on the overall level, but it is the only one that can say lane or shoulder, group a spill and count blocked lanes. GPU: A only fits the 4 GB card if models are loaded one stage at a time; B and C need only the detector on the device. The price is an API bill of about four cents per frame and no offline operation.

## 9. Where we stand, what comes next

To close: the chain now runs end to end, and the gate makes its alerts trustworthy with GPT-5.4. For a demonstration, B is the best choice today; A matters for an offline, private product; C is a useful add-on when the control room needs to know which lanes to close and whether several objects are one incident. The next stage is the decision and execution step. On the perception side, the clear target is identification time, for example by skipping boxes the detector is already sure are vehicles and by running calls in parallel, and better detection of small objects.
