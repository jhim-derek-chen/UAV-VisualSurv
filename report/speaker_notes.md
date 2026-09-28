# Speaker notes

## 1. UAV-VisualSurv

This stage had two aims: first, connect the whole chain from a drone frame to a risk judgement; second, compare three ways of running it. The deck follows that order: what we built, what it looks like on real frames, and how the three compare in time, accuracy and hardware.

## 2. This stage: connect the chain, compare the architectures

The first goal is met: every stage runs on all 30 test frames, from the raw frame to a risk level with a written reason. The numbers under each block are the headline result of that stage. Detection finds 95% of vehicles and 90% of debris. The VLM names 56% of debris correctly locally and 64% with GPT-5.4. The context gate is what makes the output trustworthy: false alarms reaching the risk step drop from 34 to 1 locally and from 23 to 0 with GPT-5.4. For the second goal we compare three architectures. A runs fully offline but takes close to a minute per frame. B is more accurate but still asks the VLM about every box, mostly ordinary cars. D lets the detector screen first and asks GPT-5.4 only about boxes that may be debris: the same answers as B at 6.3 seconds per frame. The chain now has two tiers: tier 1 looks at every frame on the drone's GPU in 2.4 seconds; tier 2, the language model, runs only when tier 1 flags a candidate.

## 3. Real UAV motorway frames + rendered 3D debris

No public dataset has UAV imagery of open motorways with labelled debris, so we built one. The background is a real drone video frame. The debris is a 3D model from Sketchfab, rendered in Blender and placed at its real-world size using a per-location ground resolution calibrated on real cars. Debris labels come from the render itself, so they are pixel-exact; the 342 vehicle labels were proposed by Grounding DINO and every one was checked by hand. All settings are chosen on a separate development set, and learned parts are always tested on a location they never saw.

## 4. One chain, three architectures

The main flow runs left to right; inside each block is the sub-flow and the model it uses. The band on top splits it into two tiers. Tier 1 is Objective 1 on the drone's GPU and must keep up with every frame: a drone-view road segmenter, OWLv2's objectness score, which finds objects without being told their class, and a small learned re-scorer. Tier 2 is the language model: identification with a physics check, the context gate, and the risk step. The table shows the differences. A runs the language model locally, B calls GPT-5.4; both ask it about every box. D adds one cheap step at the end of tier 1: the detector's own background, vehicle or debris probe screens the boxes, and only those it rates most likely debris go to tier 2. In B only such boxes can pass the gate anyway, so nothing that could reach the risk step is lost.

## 5. A · local Qwen3.5-2B

Nine test frames, the same nine on the next two slides; the banner on each gives that frame's tier times and how many boxes the language model had to look at. Top row: A gets the tyre and the fridge, but it asks the model about every box, and on the busy middle frame the model misnames six car parts that the gate then has to remove; on the right it calls a tyre a car and misses it. Middle row: its weaknesses. The cab of a lorry becomes a barrel rated high, a false alarm; a fridge becomes a roadside structure and is missed; an empty cardboard box is rated high where the rubric says low. Bottom row: its strength. A finds three objects that GPT-5.4 misses: a small tyre, a tyre in shadow and a fridge. Tier 2 takes between 12 seconds and two minutes per frame, because every box costs a local model call.

## 6. B · GPT-5.4

The same nine frames with GPT-5.4. Top row: it finds all three, including the tyre A called a car, though it names that one a ladder. The busy frame shows its cost: 27 boxes, 27 calls, and still five misnamed car parts that only the gate removes. Middle row: it fixes A's weaknesses. It confirms the red box is part of the lorry, so no false alarm; it names the fridge; and it rates the empty box low, as the rubric says. Bottom row: its weakness. It misses the three objects A found, calling a small tyre a roadside structure and a tyre and a fridge in shadow a shadow. Tier 2 runs on every frame and asks about every box, mostly ordinary cars.

## 7. D · detector screens first, GPT-5.4 on flagged boxes

D on the same nine frames. Every answer is B's: the boxes sent to GPT-5.4 are exactly those that could reach B's risk step, so the hits and the misses are the same. What changes is the work. The model sees one to three boxes per frame instead of all of them: 2 of 27 on the busy frame, 1 of 31 on the last. On the busy frame the cars never reach the model, so there is nothing misnamed for the gate to clean up. Tier 2 falls to 2 to 7 seconds, and tier 1 stays at about 2.4 seconds on every frame. The price is reliance on the detector's screen, a probe trained on our renders: a kind of debris it has never seen could be screened out as a car. None of the 27 debris boxes in the test set were.

## 8. Time, accuracy and GPU in one table

Everything in one table, split into the two tiers. Tier 1 is what the drone must do for every frame: 2.4 seconds, the same for all three, since Objective 1 is shared; we sped its code up from 4.7 seconds with bit-identical output, by vectorising a duplicate filter and overlapping CPU and GPU work. Tier 2 is the language model. In A and B it runs on every frame and asks about every box, which makes identification the bottleneck: 49 and 22 seconds. D asks about 2 boxes per frame instead of 15 and runs tier 2 only when tier 1 flags something: 4 seconds, 6.3 end to end. On this test set every frame holds debris, so tier 2 ran on 27 of 30 frames; on a patrol it would run far less often. Accuracy: D matches B on every test-set measure and recognises vehicles slightly better. GPU: A only fits the 4 GB card if its models are loaded one stage at a time; B and D need only the detector on the device. D also cuts the API bill to about half a cent per frame.

## 9. Where we stand, what comes next

To close: the chain now runs end to end, the gate makes its alerts trustworthy, and D brings it to 6.3 seconds per frame with B's accuracy: 2.4 seconds of looking on the drone for every frame, and a few seconds of language model only when something is flagged. A remains the option for an offline, private product. The next stage is the decision and execution step. On the perception side, the target is tier 1 below one second without losing detections; LeVJEPA, a new video encoder we tried for this, was fast but missed too much, so that work continues.
