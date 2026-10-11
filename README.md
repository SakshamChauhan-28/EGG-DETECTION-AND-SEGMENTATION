# Egg Detection, Segmentation & Multi-Object Tracking
### Computer Vision Internship Assignment — Nosh Robotics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)](https://opencv.org/)

This repository contains the complete engineering, research, and experimental implementation for the **Nosh Robotics Computer Vision Intern Assignment**. The project addresses the automated detection, segmentation, and continuous multi-object tracking of eggs in complex domestic kitchen environments under severe occlusions, specular reflections, camera motion, and varying illuminance.

---

## 📌 Executive Summary & Key Results

| Assignment Task | Methodology & Architecture | Dataset Scope | Key Evaluation Metric | Performance & Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Data Leakage Audit** | ResNet-18 + HSV Scene Grouping | 110 raw images | Pairwise Cosine Similarity | Identified **0.9667** photographic duplicate pair; designed 4:1 group-aware split (88 dev / 22 eval). |
| **Task 1: Foundation Segmenter** | Grounding DINO Tiny + SAM 2.1 Tiny | 88 Dev / 22 Eval | Detection & Mask Completeness | **100% (110/110)** zero-shot pseudo-GT mask generation without human annotation. |
| **Task 2: Classical Segmenter** | Algorithm E1 (LAB lightness + HSV saturation) | 88 Dev (E1–E8 iterations) | Pixel IoU / Dice | E1 achieved **0.3392 Mean IoU** on 88 dev images; proved resilient against catastrophic collapse seen in E2/E6. |
| **Task 3: Comparative Overlap** | Task 2 Classical vs. Task 1 Pseudo-GT | 22 Held-Out Eval | Overlap IoU, Dice, Recall | **0.3506 Mean IoU**, **0.4936 Mean Dice**, **0.8223 Recall**. Zero development overfitting. |
| **Task 4: Video Multi-Object Tracking** | Grounding DINO Tiny + SORT (Kalman + Hungarian) | 60.04s video (1,441 frames) | Continuous Multi-Object MOT | 19 track IDs maintained across 1,435 frames; robust re-identification during rapid pan and hand occlusions. |

---

## 📁 Repository Structure

```
├── .gitignore                      # Git exclusion rules for large datasets and weights
├── LICENSE                         # MIT License
├── README.md                       # Master repository overview and executive scorecard
├── requirements.txt                # Curated Python dependencies
├── docs/                           # Technical documentation and audit logs
│   ├── dataset.md                  # Leakage audit, similarity analysis, and split architecture
│   ├── experiments.md              # Master experiment log (Task 1, Task 2 E1–E8, Task 4 V1–V8)
│   ├── methodology.md              # Mathematical and algorithmic formulations
│   └── reproduction.md             # End-to-end execution and CLI reproduction guide
├── src/                            # Clean, production-grade source code
│   ├── common/                     # Shared computer vision utilities & metrics
│   │   ├── utils.py                # IoU, Dice, Precision, Recall, and bbox conversion
│   │   └── __init__.py
│   ├── task1/                      # Task 1 Foundation Model Segmentation
│   │   ├── pipeline.py             # Grounding DINO Tiny + SAM 2.1 Tiny pipeline
│   │   └── __init__.py
│   ├── task2/                      # Task 2 Classical Image Processing
│   │   ├── baseline_e1.py          # Classical Algorithm E1 dual-thresholding segmenter
│   │   └── __init__.py
│   ├── task3/                      # Task 3 Quantitative Comparative Evaluation
│   │   ├── evaluate.py             # Overlap metrics evaluator (Dev & Eval sets)
│   │   └── __init__.py
│   └── task4/                      # Task 4 Multi-Object Video Tracking
│       ├── tracker.py              # SORT Kalman + Hungarian tracking pipeline
│       └── __init__.py
├── scripts/                        # Dataset audits, grouping, and report builders
│   ├── 01_dataset_analysis.py      # Initial dataset statistics
│   ├── 02_create_contact_sheet.py  # Visual contact sheet generator
│   ├── 03_similarity_analysis.py   # Pairwise cosine visual similarity auditor
│   ├── 06_scene_grouping.py        # Feature-based scene clustering
│   ├── 07_create_leakage_aware_split.py # Leakage-free 4:1 split generator
│   ├── prepare_report_assets.py    # Figure extraction and montage builder
│   └── build_full_report.py        # Publication-grade PDF report compiler
├── experiments/                    # Historical experimental scripts and ablation studies
│   ├── task1/                      # Task 1 experiments (08-20: SAM box prompts, ViT, text prompts)
│   ├── task2/                      # Task 2 experiments (22-29: E1–E8 thresholding, adaptive, watershed)
│   └── task4/                      # Task 4 experiments (31-35: SORT, DeepSORT, confidence sweeps)
├── results/                        # Verifiable CSV/JSON benchmark data
│   ├── dataset/                    # Split manifests and similarity matrices
│   ├── task1/                      # Pseudo-GT validation logs
│   ├── task2/                      # E1–E8 metric CSVs
│   └── task4/                      # Video tracking telemetry logs
└── reports/                        # Publication documents & visual assets
    ├── Nosh_Robotics_CV_Assignment_Report.pdf   # 29-page comprehensive technical project report
    ├── Nosh_Robotics_CV_Assignment_Report.docx  # Editable technical project report
    └── figures/                                 # Report diagrams and qualitative visual montages
```

---

## 🔬 Scientific Highlights & Engineering Journey

### 1. The Dataset Leakage Discovery
Prior to modeling, a pairwise image similarity analysis across the 110-image Roboflow dataset revealed an extreme vulnerability:
- Image `not_damaged_78` in the test set had a **0.9667 cosine visual similarity** to `not_damaged_27` in the train set.
- They represented identical photographic burst shots of the same frying pan with identical egg placement.
- **Solution:** A feature-based clustering system (ResNet-18 visual embeddings + HSV color histograms @ 0.82 threshold) partitioned the dataset into 21 distinct scene groups, enforcing a group-aware 4:1 partition (88 development / 22 held-out evaluation) where no visual scene co-exists across splits.

### 2. Task 1: Zero-Shot Foundation Segmentation
Using open-vocabulary **Grounding DINO Tiny** (`prompt="egg."`, box threshold $0.25$, text threshold $0.20$) paired with **SAM 2.1 Tiny** bounding-box prompting, the pipeline achieved 100% coverage on both development and evaluation splits, producing high-fidelity pseudo-GT masks without human annotation overhead.

### 3. Task 2: Classical Segmentation & Iteration Journey (E1 to E8)
Rather than tuning on a single image, 8 distinct classical algorithms were rigorously evaluated:
- **Algorithm E1 (Winner):** Dynamic 65th percentile lightness threshold in CIE-LAB space coupled with HSV saturation upper bound ($S \le 100$), $7 \times 7$ morphological opening/closing, and connected component filtering ($\text{area} \ge 500$, $\text{solidity} \ge 0.70$).
  - **88 Dev Images:** Mean IoU **0.3392**, Median IoU **0.3015**, Mean Recall **0.7315**.
- **Failed Approaches Documented:**
  - *E2 (Adaptive Gaussian):* Recall spiked to $0.9673$, but IoU collapsed to $0.2556$ due to rampant countertop over-segmentation.
  - *E6 (Aggressive Aspect/Solidity):* Suffered catastrophic structural collapse (Mean IoU $0.1000$) on non-canonical viewing angles.
  - *E7 (Spatial Prior Masking):* Showed promising pilot gains ($0.3833$) on centered test shots but failed to generalize on off-center kitchen compositions ($0.3277$).

### 4. Task 3: Held-Out Overlap Evaluation
Algorithm E1 was evaluated on the 22 held-out images against frozen Task 1 pseudo-GT masks:
- **Mean IoU:** **0.3506 ± 0.1634**
- **Median IoU:** **0.3401**
- **Mean Dice:** **0.4936 ± 0.1878**
- **Mean Precision:** **0.3822 ± 0.1702**
- **Mean Recall:** **0.8223 ± 0.2307**
The evaluation metrics slightly outperformed development figures, proving **zero development overfitting** and true photometric generalizability.

### 5. Task 4: Video Tracking Under Occlusions (SORT)
Using continuous Grounding DINO Tiny detections fused with a Kalman filter and Hungarian assignment ($\text{max\_age}=20$, $\text{min\_hits}=3$, $\text{IoU\_threshold}=0.30$):
- Processed 1,441 frames (60.04s @ 24 FPS).
- Maintained 19 persistent track IDs across 1,435 predicted frames, successfully re-associating egg identities after hand occlusions and spatula flips.
- **[Public Video Demonstration (Google Drive)](https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing)**

---

## 🚀 Quickstart & Reproduction

### Installation
```bash
git clone https://github.com/SakshamChauhan-28/IMAGE-DETECTION-AND-SEGMENTATION.git
cd IMAGE-DETECTION-AND-SEGMENTATION
python -m venv cv_env
source cv_env/bin/activate  # On Windows: .\cv_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Classical Segmentation (Task 2)
```bash
python -m src.task2.baseline_e1 \
    --input_dir "data/eval_images" \
    --output_dir "results/task2/masks_eval" \
    --l_percentile 65 \
    --s_max 100
```

### Run Quantitative Evaluation (Task 3)
```bash
python -m src.task3.evaluate \
    --pred_dir "results/task2/masks_eval" \
    --gt_dir "results/task1/masks_eval" \
    --output_csv "results/task3/eval_summary.csv"
```

### Run Video Multi-Object Tracker (Task 4)
```bash
python -m src.task4.tracker \
    --video_path "data/egg_video.mp4" \
    --output_video "results/task4/tracked_output.mp4" \
    --output_csv "results/task4/tracking_log.csv"
```

For full details, flags, and dataset preparation steps, refer to [Reproduction Guide](docs/reproduction.md).

---

## 📑 Technical Project Report

A publication-grade, 29-page technical engineering report has been compiled and included in this repository:
- **PDF Report:** [`reports/Nosh_Robotics_CV_Assignment_Report.pdf`](reports/Nosh_Robotics_CV_Assignment_Report.pdf)
- **Word Document:** [`reports/Nosh_Robotics_CV_Assignment_Report.docx`](reports/Nosh_Robotics_CV_Assignment_Report.docx)

The report details all executive summaries, problem formulations, failure mode diagnostic panels, mathematical formulations, and engineering trade-offs.

---

## ⚖️ Limitations & Engineering Trade-offs

1. **Illuminance Over-Segmentation:** Classical Algorithm E1 relies on luminance percentiles. In scenes featuring polished granite, stainless steel pans, or porcelain bowls, high-specularity surfaces share perceptual luminance with egg whites, bounding precision to $\approx 0.38$.
2. **SORT Linear Motion Prior:** The constant-velocity Kalman model performs exceptionally during translation but relies heavily on continuous detections during erratic spatula flips. Incorporating DeepSORT appearance embeddings (ReID) is the recommended next step for extended occlusions.
3. **Pseudo-GT Edge Softness:** While SAM 2.1 produces crisp contours, extreme motion blur in video frames occasionally softens mask boundaries.

---

## 🤖 AI Disclosure & Methodology

In alignment with standard engineering integrity and startup submission requirements:
- Large Language Models (LLMs) and foundation models were utilized as development and research tools for script refactoring, metric analysis scaffolding, and drafting documentation.
- All computer vision pipelines, mathematical formulations, experimental splits, metric evaluations, and video tracking executions were run, validated, and verified locally on actual data.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
