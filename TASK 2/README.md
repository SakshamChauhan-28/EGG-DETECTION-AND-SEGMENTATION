# TASK 2: Classical Image Processing & Experiments E1–E8

Task 2 explores deterministic classical computer vision algorithms to segment eggs without relying on deep neural networks or learned weights.

---

## 1. Problem Statement & Constraints

- **Core Rule:** Strictly classical computer vision techniques (color-space transforms, adaptive thresholding, edge filtering, morphological operations, contour geometry, energy minimization).
- **Zero Learned Weights:** No neural network inference or fine-tuning permitted at this stage.
- **Evaluation Target:** Segmentation predictions are benchmarked against the Task 1 foundation reference masks.

---

## 2. Experimental Progression: Summary of E1–E8

Eight distinct algorithmic hypotheses were implemented, tuned, and evaluated:

| Exp ID | Algorithm / Paradigm | Test Scope | Mean IoU | Mean Dice | Precision | Recall | Key Finding & Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E01** | **Dual HSV + LAB Color Gating** | **Full Dev (88)** | **0.3392** | **0.4700** | **0.3820** | **0.7315** | ✅ **Selected Winner.** Highest robustness and boundary recall across varied kitchen surfaces. |
| **E02** | Local Adaptive Gaussian Thresholding | 10-img pilot | 0.2556 | 0.3858 | 0.2689 | **0.9673** | ❌ **High Recall Trap.** Flooded masks with countertop micro-textures; precision collapsed. |
| **E03** | Canny Multi-Scale Contour Closing | 10-img pilot | 0.1913 | 0.2559 | 0.3981 | 0.1917 | ❌ **Broken Boundaries.** Specular highlights and weak contrast broke edge loops. |
| **E04** | E01 + Geometric Aspect/Solidity Gating | 10-img pilot | 0.2420 | 0.3068 | 0.3972 | 0.2761 | ❌ **Rigid Filter Failure.** Discarded valid touching and partially occluded eggs. |
| **E05** | Hybrid GrabCut Energy Minimization | 10-img pilot | 0.2670 | 0.3611 | 0.4489 | 0.4050 | ❌ **Shadow Clipping.** Energy minimization cut off dark egg sides; reduced recall. |
| **E06** | Multi-Feature Morphological Filtering | 10-img pilot | 0.1000 | 0.1000 | 0.0000 | 0.1000 | ❌ **Over-Pruning.** Stricter geometry pruned eggs in nearly every test frame. |
| **E07** | E01 + Spatial Distance Transform Decay | Full Dev (88) | 0.3277 | 0.4580 | 0.3674 | 0.7345 | ❌ **Pilot Overfitting.** Gained +0.056 IoU on pilot but degraded on full 88-image set. |
| **E08** | Direct Algebraic Direct Ellipse Fitting | Full Dev (88) | 0.2046 | 0.2476 | 0.3030 | 0.2588 | ❌ **Non-Convex Collapse.** Specular glares caused contour concavities; median IoU dropped to 0.0. |

---

## 3. The Winning Method: Baseline E01

### Algorithmic Pipeline
1. **Color Decomposition:** Convert RGB input to CIE LAB and HSV color representations.
2. **Dynamic L-Channel Percentile Threshold:** Compute foreground luminance mask where $L \ge P_{65}$ to adapt to varied scene brightness.
3. **HSV Saturation Suppressor:** Suppress vividly colored objects using saturation ceiling $S \le 100$.
4. **Morphological Filtering:** Apply elliptical structuring element ($7 \times 7$) with opening to remove salt-and-pepper noise, followed by dilation to bridge minor discontinuities.
5. **Connected Component Analysis:** Retain components with pixel area $\ge 500$, selecting up to 6 largest candidate regions.

---

## 4. Directory Organization

```text
TASK 2/
├── README.md
├── experiments/
│   ├── E01/ # Winning Dual HSV/LAB Baseline
│   │   ├── README.md
│   │   ├── scripts/ (22_task2_baseline.py)
│   │   └── results/ (masks/, visualizations/, experiment_01_report.csv)
│   ├── E02/ # Adaptive Gaussian Thresholding
│   ├── E03/ # Canny Edge Closing
│   ├── E04/ # Geometric Elliptical Gating
│   ├── E05/ # GrabCut Energy Minimization
│   ├── E06/ # Multi-Feature Morphological Gating
│   ├── E07/ # Spatial Coherence Decay
│   └── E08/ # Direct Algebraic Ellipse Fitting
├── final_evaluation/
│   ├── 29_task2_final_evaluation.py
│   ├── masks/          # 22 evaluation split prediction masks
│   ├── visualizations/ # Visual comparison overlays
│   └── reports/        # final_e1_evaluation_report.csv
└── reports/
```

---

## 5. Execution Commands

To reproduce the winning baseline E01 on the development images:
```bash
python "TASK 2/experiments/E01/scripts/22_task2_baseline.py"
```

To run final evaluation of E01 on the unseen evaluation partition:
```bash
python "TASK 2/final_evaluation/29_task2_final_evaluation.py"
```
