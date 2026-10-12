# Comprehensive Engineering Report — Nosh Robotics Assignment

## 1. Executive Summary

This repository presents the end-to-end technical deliverables for the **Nosh Robotics Computer Vision Intern Assignment (Egg Detection, Classical & Deep Segmentation, Evaluation, and Temporal Video Tracking)**.

- **Candidate:** Saksham Chauhan
- **Public Report PDF:** [Google Drive Report Document](https://drive.google.com/file/d/1z3NqJekPECgqXZN-NxWknRqwoE1C_gnp/view?usp=sharing)
- **Rendered Output Video:** [Google Drive Tracking Video](https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing)

---

## 2. Core Technical Findings by Task

### Task 1: Pretrained Foundation Segmentation
- **Zero-Shot Pseudo-Ground-Truth:** Because the source dataset lacked human instance segmentation masks, an open-vocabulary detection and promptable segmentation pipeline (**Grounding DINO Tiny + SAM 2.1 Tiny**) was deployed to generate 131 instance reference masks across 88 development images and 23 instance reference masks across 22 evaluation images.
- **Reliability:** Achieved $100\%$ detection and segmentation success across all 110 dataset frames without human intervention.

### Task 2: Classical Image Processing (Experiments E1–E8)
- **Constraint:** Strictly classical computer vision techniques (zero learned neural weights).
- **Evaluation:** Evaluated 8 algorithmic hypotheses. **Experiment E01 (Dual HSV / CIE LAB Percentile Thresholding)** emerged as the winning classical baseline:
  - **Full Development Set ($N = 88$):** Mean IoU **0.3392**, Median IoU **0.3015**, Mean Dice **0.4700**, Precision **0.3820**, Recall **0.7315**.
- **Key Empirical Insight:** Complex shape gating (E4, E6) and direct ellipse fitting (E8) severely degraded on touching or shadowed eggs, while adaptive thresholding (E2) suffered from catastrophic false-positive background noise. Simple, adaptive color-space percentile gating proved most resilient.

### Task 3: Quantitative and Qualitative Benchmarks
- **Evaluation Split Scorecard ($N = 22$ Held-Out Images):**
  - **Mean IoU:** **0.3507**
  - **Median IoU:** **0.3401**
  - **Mean Dice:** **0.4936**
  - **Mean Precision:** **0.3822**
  - **Mean Recall:** **0.8223**
- The high recall ($>82\%$) confirms that the classical pipeline robustly isolates true egg foreground mass, while precision ($~38\%$) reflects mild over-segmentation on pale marble and laminate countertops.

### Task 4: Video Processing and Identity Tracking
- **Input Footage:** 60.04s, 1,441 frames of high-speed spinning eggs with severe human hand occlusions.
- **Pipeline Architecture:** **Grounding DINO Tiny + SORT (Kalman Filtering + Hungarian Association)** with a 3-frame detection stride and 12-frame occlusion memory.
- **Results:** 100% full-video coverage (1,441 frames processed), **zero hand-hijacking occurrences**, and persistent identity maintenance throughout active spinning cycles.

---

## 3. Dataset Integrity & Leakage Mitigation

- **Source:** 110 images from Roboflow Universe (`CC BY 4.0`).
- **Forensic Discovery:** Pairwise ResNet-18 embedding analysis uncovered a high-similarity duplicate pair ($0.9667$ cosine similarity) spanning the original Roboflow train and test partitions.
- **Mitigation:** Applied graph-based connected component clustering ($\tau = 0.82$) to group related scenes into 21 indivisible candidate clusters.
- **Leakage-Aware Split:** Established an atomic $88$-image development and $22$-image evaluation partition with **zero cross-split scene overlap**.
