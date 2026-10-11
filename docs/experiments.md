# Comprehensive Experiment Index

This document provides a systematic index of all experimental trials, failed hypotheses, failure diagnoses, quantitative metrics, and architectural decisions across **Task 1 (Pretrained Foundation Segmentation)**, **Task 2 (Classical Image-Processing Segmentation)**, and **Task 4 (Temporal Video Tracking)**.

---

## 1. Task 1: Pretrained Foundation Segmentation Experiments

### Objective
Automatically detect and segment every egg across the dataset using a pretrained vision foundation model, producing clean instance masks to serve as pseudo-ground-truth for subsequent tasks.

| Exp ID | Approach / Architecture | Evaluated Subset | Observed Outcome & Metrics | Failure Cause & Diagnosis | Decision & Action |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **01** | **SAM 2.1 Base** (`sam2.1_b.pt`, ~350M params) | 1 image test | **CUDA Out Of Memory (OOM)** crash. 0 masks generated. | Base transformer encoder allocated >3.85 GB on 4GB VRAM GPU. | ❌ **Failed.** Adopt lightweight SAM 2.1 Tiny architecture. |
| **02** | **SAM 2.1 Tiny** (`sam2.1_t.pt`) + Manual Point Prompt | Representative test set | High-fidelity instance masks; near-perfect egg boundary alignment. | Manual point prompting requires $O(N)$ human interaction; non-scalable for autonomous pipeline. | 🟡 **Proof-of-Concept.** Confirmed SAM Tiny boundary capacity; search for automated localizer. |
| **03** | **YOLO-World v2 Small** + SAM 2.1 Tiny | 12-image stress test | Detections on 9/12 images (75%); missed eggs on 3/12 images (25%). | Open-vocabulary text embeddings for `"egg"` were brittle under shadows and non-uniform lighting. | 🔴 **Rejected.** Detector miss rate too high to serve as dataset-wide pseudo-ground-truth. |
| **04** | **SAM 2.1 Automatic** Dense Point-Grid Generator | Representative test set | Generated 45–120 arbitrary masks per image (countertops, plates, hands). | SAM dense grids possess zero semantic object priors; cannot differentiate an egg from background tiles. | 🔴 **Rejected.** Dense prompting requires an upstream semantic object detector. |
| **05** | **Classical Candidate Points** + SAM 2.1 Tiny | 88-image development set | **GOOD = 4 images (4.5%)**, **REVIEW_NO_CANDIDATES = 84 images (95.5%)**. | Classical color heuristics failed to reliably find single interior points across diverse kitchen scenes. | 🔴 **Failed.** Revealed that zero-shot semantic localization was the true core bottleneck, not SAM. |
| **Final** | **Grounding DINO Tiny + SAM 2.1 Tiny** | 88 development + 22 evaluation | **88/88 GOOD on dev (100%)**; **22/22 GOOD on eval (100%)**. 0 mismatches, 0 misses. | None. Cross-attention vision-language grounding provided robust spatial bounds for SAM prompts. | 🟢 **SELECTED & FROZEN.** Deployed as frozen pseudo-ground-truth reference masks. |

### Grounding DINO + SAM Configuration
- **Model:** `IDEA-Research/grounding-dino-tiny` + `sam2.1_t.pt`
- **Text Prompt:** `"egg."`
- **Thresholds:** `BOX_THRESHOLD = 0.25`, `TEXT_THRESHOLD = 0.20`
- **Confidence Scores Observed:** 0.789, 0.752, 0.520, 0.737, 0.735, 0.721, 0.874
- **Instance Count Distribution (Dev):** 1 egg: 60 images | 2 eggs: 14 images | 3 eggs: 13 images | 4 eggs: 1 image

---

## 2. Task 2: Classical Image-Processing Experiments (E1–E8)

### Constraint
Strict classical computer vision: **zero neural networks**, no learned embeddings, deterministic mathematical image processing only.

> **Important Sample-Size Notice:** Experiments **E2–E6** and **E7-pilot** were evaluated on a controlled 10-image development pilot. Experiments **E1**, **E7-full**, and **E8** were evaluated across the entire 88-image development set.

| Exp ID | Algorithmic Paradigm | Evaluated Subset | N | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Status & Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1** | **Dual HSV/LAB Threshold Baseline** | 10-image dev subset | 10 | 0.3273 | 0.2926 | 0.4641 | 0.3545 | 0.7728 | 🟢 **Kept Baseline** |
| **E1-Full** | **Dual HSV/LAB Threshold Baseline** | **Full 88-image dev** | **88** | **0.3392** | **0.3015** | **0.4700** | **0.3820** | **0.7315** | 🟢 **FINAL CLASSICAL WINNER** |
| **E2** | Local Adaptive Gaussian Threshold | 10-image dev subset | 10 | 0.2556 | 0.1874 | 0.3858 | 0.2689 | **0.9673** | 🔴 Rejected (Background noise flood) |
| **E3** | Canny Edge Contour Closing | 10-image dev subset | 10 | 0.1913 | 0.0000 | 0.2559 | 0.3981 | 0.1917 | 🔴 Rejected (Broken shadow contours) |
| **E4** | E1 + Geometric Elliptical Gating | 10-image dev subset | 10 | 0.2420 | 0.0908 | 0.3068 | 0.3972 | 0.2761 | 🔴 Rejected (Discarded partial eggs) |
| **E5** | Hybrid GrabCut Energy Minimization | 10-image dev subset | 10 | 0.2670 | 0.2374 | 0.3611 | 0.4489 | 0.4050 | 🔴 Rejected (Shadowed egg halves cut) |
| **E6** | Aggressive Shape / Fill-Ratio Gating | 10-image dev subset | 10 | 0.1000 | 0.0000 | 0.1000 | 0.0000 | 0.1000 | 🔴 Rejected (Catastrophic egg pruning) |
| **E7-Pilot** | E1 + Spatial Coherence Decay | 10-image dev subset | 10 | **0.3833** | **0.3438** | **0.5411** | **0.4455** | **0.8350** | 🟡 **Appeared superior in pilot...** |
| **E7-Full** | E1 + Spatial Coherence Decay | **Full 88-image dev** | **88** | 0.3277 | 0.2703 | 0.4580 | 0.3674 | 0.7345 | 🔴 **...Degraded on full development** |
| **E8** | Direct Algebraic Ellipse Fitting | **Full 88-image dev** | **88** | 0.2046 | 0.0000 | 0.2476 | 0.3030 | 0.2588 | 🔴 Rejected (Median collapsed to 0) |

### Detailed Failure Analysis for Task 2 Trials

#### Experiment E1 (Selected Baseline)
- **Parameters:** LAB $L \ge 65$\%tile, HSV $S \le 100$, morphological kernel $7\times 7$ ellipse, area $\ge 500$, max 6 components.
- **Why Kept:** Highest generalization across the full 88-image development set. Captures egg bodies reliably (recall 0.7315) across all kitchen environments.
- **Limitation:** Predicts pale background pixels (countertops, bowls) as foreground, keeping precision moderate (0.3820).

#### Experiment E2 (Adaptive Thresholding — The High Recall Trap)
- **Failure:** While recall reached a near-perfect 0.9673, IoU dropped to 0.2556. Local adaptive Gaussian thresholding responded to micro-surface textures, flooding the prediction with background noise.
- **Lesson:** High recall without precision control is destructive to segmentation overlap.

#### Experiment E3 (Canny Edge Contour Closing)
- **Failure:** Natural kitchen scenes lack continuous boundary gradients against light-colored plates. Shadows broke contour continuity, causing morphological closing to fail and collapsing recall to 0.1917.

#### Experiment E4 (Baseline + Geometric Filtering)
- **Failure:** Partially shadowed or touching eggs produced non-ideal aspect ratios and circularity scores, causing rigid geometric filters to reject valid egg pixels. Recall dropped from 0.77 to 0.27.

#### Experiment E5 (GrabCut Energy Minimization)
- **Failure:** Graph cuts improved precision (0.4489) along bright edges, but energy minimization repeatedly clipped away darker, shadowed halves of eggs, depressing recall to 0.4050.

#### Experiment E6 (Aggressive Multi-Shape Gating)
- **Failure:** Total algorithmic collapse. Enforcing strict solidity ($\ge 0.85$), extent ($\ge 0.60$), and circularity ($\ge 0.65$) pruned valid eggs in almost every test image.

#### Experiment E7 (Spatial Coherence — The Pilot Generalization Fallacy)
- **Failure:** On the 10-image pilot, distance decay gave a +0.056 IoU gain (0.3833). But on the full 88-image development set, mean IoU dropped to 0.3277 (inferior to E1's 0.3392).
- **Lesson:** Tuning heuristics on a small pilot frequently overfits. Full-dataset validation is mandatory.

#### Experiment E8 (Direct Algebraic Ellipse Fitting)
- **Failure:** Median IoU collapsed to exactly 0.0000 on over 50% of images. Non-convex specular glares and contact points produce concave contours that fail direct algebraic ellipse fitting.

---

## 3. Task 4: Video Tracking Experiments (V1–V8 & Final)

### Video Properties
- **Resolution:** 1920 × 1080 | **FPS:** 24.0 | **Duration:** 60.04s | **Frames:** 1,441 frames
- **Target:** Track individual eggs under spinning motion, hand occlusions, and table reflections.

| Version | Architecture & Mechanism | Observed Tracking Behavior | Failure Cause & Diagnosis | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **V1** | Hough Circle Init + Local E1 Search | Clean initial tracking; lost eggs during high-speed spinning. | Centroid association jumped onto experimenter's hand during interaction. | Foundational Prototype |
| **V2** | E1 Candidates + Joint 2-Track Assignment | Rigid geometric gates rejected spinning eggs; tracks dropped to LOST. | Eggs deformed during spinning, violating rigid bounding-box aspect gates. | 🔴 Rejected |
| **V3** | Local Search + Forced 2 Persistent IDs | Both tracks collapsed onto a single egg during close proximity. | Forcing exactly two active tracks forces the tracker to hijack clutter or merge tracks during occlusion. | 🔴 Rejected (Rigid constraint trap) |
| **V4** | Detection-Validated Gate Tracker | Tracks frequently switched to LOST; failed to recover eggs after occlusion. | Overly conservative gating caused permanent track termination. | 🔴 Rejected |
| **V5** | Hybrid Confidence-Weighted Tracker | Tracker drifted onto moving hand and tracked knuckles across the table. | Confidence metric gave excessive weight to high brightness, latching onto pale skin. | 🔴 Rejected (Hand hijacking) |
| **V6** | Lucas-Kanade Optical Flow + Local E1 | Optical flow drifted when E1 merged egg and hand into one candidate component. | Upstream candidate generator failure: classical color thresholding cannot separate hands from eggs. | 🔴 Rejected |
| **V7** | Multi-Scale Template Matching + Flow | Egg 1 tracked stably; Egg 2 lost entirely during high angular velocity. | High spin speed destroyed appearance correlation against static template. | 🔴 Rejected |
| **V8** | Protected Template + Global Reacquisition | After occlusion, global template search falsely reacquired onto moving hand. | Normalized cross-correlation found high matching score on palm surfaces. | 🔴 Rejected |
| **Final** | **Grounding DINO Tiny + SORT** | 1,441 frames processed; 19 track IDs; zero hand hijacking. | Identity fragmentation across long occlusions (>12 frames). | 🟢 **SELECTED FINAL** |

### Final Task 4 Grounding DINO + SORT Telemetry
- **Processed Frames:** 1,441 / 1,441 frames (100% full-video coverage)
- **Stride:** Detector evaluated every 3 frames; Kalman filter interpolates intermediate steps.
- **Detector Downscaling:** 960px width inference for $4\times$ speedup; boxes mapped back to 1080p.
- **Track IDs Created:** 19 unique tracks (fragmentation across prolonged occlusions).
- **Sum of Matched Frames:** 618 track-frames.
- **Sum of Predicted Frames:** 1,435 track-frames.
- **Dominant Track Lifespans:** Track 3 (22.25s), Track 4 (12.13s), Track 11 (11.88s).
- **MOTA/HOTA Disclaimer:** These are tracker operational diagnostics, *not* ground-truth tracking accuracy. No human frame-by-frame annotations exist for this video.
