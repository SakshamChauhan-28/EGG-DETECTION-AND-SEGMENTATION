# Reproduction & Execution Guide

This guide provides end-to-end instructions for reproducing the entire pipeline: environment setup, dataset acquisition, scene-grouping split generation, Task 1 pseudo-GT mask generation, Task 2 classical segmentation, Task 3 quantitative evaluation, and Task 4 multi-object video tracking.

---

## 1. Prerequisites & Environment Setup

### Hardware Recommendations
- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+).
- **RAM:** Minimum 8 GB (16 GB recommended).
- **GPU:** NVIDIA CUDA-compatible GPU with $\ge 4$ GB VRAM recommended for Task 1 and Task 4 (CPU execution is fully supported via PyTorch fallback).

### Python Virtual Environment
We recommend Python 3.10+:

```bash
# Clone the repository
git clone https://github.com/SakshamChauhan-28/IMAGE-DETECTION-AND-SEGMENTATION.git
cd IMAGE-DETECTION-AND-SEGMENTATION

# Create and activate virtual environment
python -m venv cv_env

# On Windows (PowerShell):
.\cv_env\Scripts\Activate.ps1
# On Linux/macOS:
source cv_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

The original Roboflow dataset (`egg.v1i.folder`) contains 110 raw egg images across varying kitchen counter environments.

### Option A: Using Local Directory
Set the environment variable or pass `--dataset_root`:
```bash
# Set environment variable (Windows PowerShell)
$env:DATASET_ROOT = "C:\path\to\egg.v1i.folder"

# Set environment variable (Linux/macOS)
export DATASET_ROOT="/path/to/egg.v1i.folder"
```

### Option B: Automatic Discovery
If `--dataset_root` is not supplied, the scripts automatically inspect `DATASET_ROOT` in the environment, fallback to sibling directories, or run in benchmark mode using existing split metadata in `results/dataset/`.

---

## 3. Leakage Analysis & Split Reproduction

To reproduce the leakage audit and generate the clean 4:1 (88 dev / 22 eval) split:

```bash
# Step 1: Compute pairwise visual similarity across the dataset
python scripts/03_similarity_analysis.py --dataset_dir "$DATASET_ROOT"

# Step 2: Perform feature-based scene clustering (ResNet-18 + HSV features)
python scripts/06_scene_grouping.py --dataset_dir "$DATASET_ROOT"

# Step 3: Generate the group-aware, leakage-free 4:1 split
python scripts/07_create_leakage_aware_split.py \
    --dataset_dir "$DATASET_ROOT" \
    --output_dir "results/dataset"
```

**Outputs Created:**
- `results/dataset/leakage_aware_split_summary.json`
- `results/dataset/dev_dataset.csv` (88 images)
- `results/dataset/eval_dataset.csv` (22 held-out images)

---

## 4. Task 1: Pretrained Foundation Segmentation (Pseudo-GT)

Generate zero-shot pseudo-ground-truth masks using Grounding DINO Tiny and SAM 2.1 Tiny:

```bash
# Generate pseudo-GT for development set
python -m src.task1.pipeline \
    --input_dir "data/dev_images" \
    --output_dir "results/task1/masks_dev" \
    --box_threshold 0.25 \
    --text_threshold 0.20 \
    --prompt "egg."

# Generate pseudo-GT for evaluation set
python -m src.task1.pipeline \
    --input_dir "data/eval_images" \
    --output_dir "results/task1/masks_eval" \
    --box_threshold 0.25 \
    --text_threshold 0.20 \
    --prompt "egg."
```

*(Note: If pre-downloaded weights are present, pass `--weights_path path/to/sam2.1.pt`. Otherwise, SAM weights can be loaded from HuggingFace/Meta checkpoints.)*

---

## 5. Task 2: Classical Computer Vision Segmentation (Algorithm E1)

Run the classical dual-space thresholding segmenter:

```bash
# Run on development set
python -m src.task2.baseline_e1 \
    --input_dir "data/dev_images" \
    --output_dir "results/task2/masks_dev" \
    --l_percentile 65 \
    --s_max 100 \
    --min_area 500 \
    --solidity_min 0.70

# Run on held-out evaluation set
python -m src.task2.baseline_e1 \
    --input_dir "data/eval_images" \
    --output_dir "results/task2/masks_eval" \
    --l_percentile 65 \
    --s_max 100 \
    --min_area 500 \
    --solidity_min 0.70
```

---

## 6. Task 3: Quantitative Overlap Evaluation

Evaluate Task 2 predictions against Task 1 pseudo-GT masks to compute pixel-level IoU, Dice, Precision, and Recall:

```bash
# Evaluate on Development Split (88 images)
python -m src.task3.evaluate \
    --pred_dir "results/task2/masks_dev" \
    --gt_dir "results/task1/masks_dev" \
    --output_csv "results/task3/eval_dev_metrics.csv"

# Evaluate on Held-Out Evaluation Split (22 images)
python -m src.task3.evaluate \
    --pred_dir "results/task2/masks_eval" \
    --gt_dir "results/task1/masks_eval" \
    --output_csv "results/task3/eval_test_metrics.csv"
```

Expected output on 22 Held-Out Evaluation images:
```text
Evaluation Metrics Summary:
  Total Evaluated: 22 images
  Mean IoU:        0.3506 ± 0.1634
  Median IoU:      0.3401
  Mean Dice:       0.4936 ± 0.1878
  Mean Precision:  0.3822 ± 0.1702
  Mean Recall:     0.8223 ± 0.2307
```

---

## 7. Task 4: Multi-Object Video Tracking (SORT)

Run the tracking pipeline on any target cooking/kitchen video:

```bash
python -m src.task4.tracker \
    --video_path "data/egg_video.mp4" \
    --output_video "results/task4/tracked_video_output.mp4" \
    --output_csv "results/task4/tracking_log.csv" \
    --max_age 20 \
    --min_hits 3 \
    --iou_threshold 0.30 \
    --conf_threshold 0.25
```

**Full Video Demonstration:**
The processed 60-second video demo (1,441 frames @ 24 FPS with Kalman filter state updates and active bounding box track IDs) is available on Google Drive:
- [Google Drive Video Output Link](https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing)

---

## 8. PDF Assignment Report Generation

The 29-page publication-grade PDF report can be recompiled directly from code:

```bash
python scripts/build_full_report.py
```
This produces `reports/Nosh_Robotics_CV_Assignment_Report.pdf` and `Nosh_Robotics_CV_Assignment_Report.docx`.
