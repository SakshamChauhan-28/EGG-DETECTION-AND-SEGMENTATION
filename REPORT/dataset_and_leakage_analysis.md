# Dataset Analysis & Leakage-Aware Partitioning

This document outlines the dataset exploration, forensic data-quality investigation, cross-split leakage discovery, and the scene-grouping methodology executed to establish a scientifically sound 4:1 development/evaluation split for the Nosh Robotics Computer Vision Assignment.

---

## 1. Dataset Modality & Initial Exploration

The source image corpus originates from a Roboflow computer vision collection formatted for object detection in egg-handling environments.

| Property | Value | Observation / Engineering Implication |
| :--- | :--- | :--- |
| **Total Images** | 110 images | Small sample size; mandates zero data waste and strict split sanitization. |
| **Image Resolution** | 640 × 640 pixels | Uniform square aspect ratio. Eggs span between ~40px and ~350px. |
| **Color Channels** | 3-channel RGB (JPEG) | 8-bit dynamic range per channel with standard JPEG compression artifacts. |
| **Original Split** | 77 train / 22 valid / 11 test | Default random Roboflow partition; investigated for data leakage below. |
| **Human Segmentation GT** | **None (Zero Masks)** | The dataset provided bounding boxes but **no human instance masks**. All segmentation benchmarks in Task 2 and Task 3 evaluate against Task 1 foundation model masks as **empirical pseudo-ground-truth**. |

---

## 2. Forensic Audit: Cross-Split Data Leakage Discovery

Before model development, an automated pairwise similarity audit was executed across all 5,995 unique image pairs in the 110-image corpus to test whether photographic frames from the same scene crossed split boundaries.

### The Leakage Pathology
The audit flagged a near-duplicate image pair spanning the original splits:
- **Test split:** `not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5.jpg`
- **Train split:** `not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad.jpg`
- **Pairwise Cosine Similarity:** **0.9667**

Visual inspection confirmed that these two files represent the **identical physical scene**: identical tabletop grain, identical egg position, and identical lighting, differing only by minor camera jitter.

```
Original Test Split: not_damaged_78 ──┐
                                      ├── Cosine Similarity: 0.9667 (Visual Scene Duplicate)
Original Train Split: not_damaged_27 ─┘
```

### Why Cross-Split Leakage Corrupts Validation
1. **Artificially Optimistic Evaluation:** If visually identical scenes exist in both development and evaluation splits, evaluation scores measure the model's ability to memorize specific lighting intensities and background surfaces rather than its capacity to generalize to unseen kitchens.
2. **Heuristic Overfitting:** Thresholds tuned on the development split inadvertently exploit background quirks shared with the test split, producing a false sense of accuracy.
3. **Research Integrity:** Publishing benchmark scores evaluated on contaminated test splits violates industrial research standards.

---

## 3. Scene Grouping Methodology

To permanently eliminate cross-split scene duplication, an automated clustering pipeline was designed to group related photographs into discrete scene clusters:

1. **Spatial & Compositional Embeddings:** A pretrained **ResNet-18** backbone (penultimate pooling layer, 512 dimensions) extracted structural and compositional features.
2. **Color Distribution Features:** 3D **HSV color histograms** (24 hue × 16 saturation × 16 value bins) captured ambient kitchen illumination and table surface tones.
3. **Combined Affinity Metric:** Pairwise similarity was computed via:
   $$\text{Affinity}(i, j) = 0.80 \times \text{CosineSim}(\text{ResNet}_i, \text{ResNet}_j) + 0.20 \times \text{HistCorr}(\text{HSV}_i, \text{HSV}_j)$$
4. **Graph-Based Scene Clustering:** An affinity threshold of **0.82** was applied to construct an adjacency graph. Connected components defined atomic candidate scene groups that were strictly forbidden from being partitioned across splits.

### Grouping Outcome
The clustering algorithm identified:
- **21 candidate multi-image scene groups**
- **14 singleton images**

> **Note on Conservative Over-Grouping:** Some candidate groups slightly over-clustered scenes that shared illumination despite minor angle changes. This conservative grouping was deliberately accepted: *guaranteeing zero leakage is vastly more important than artificially maximizing the training set size.*

---

## 4. Final Leakage-Aware 4:1 Split Architecture

To satisfy the assignment's mandate for a 4:1 split while guaranteeing zero leakage, atomic scene groups were assigned to either Development or Evaluation:

```
                            Total Dataset: 110 Images
                        (21 Scene Groups + 14 Singletons)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        Development Sandbox                      Held-Out Evaluation Split
         88 Images (80.0%)                           22 Images (20.0%)
                    │                                       │
     • 14 Multi-image groups + 14 singletons     • 7 Complete Held-Out Groups:
     • Used for all Task 1, 2, and 4 trials        candidate_004, candidate_005,
     • Heuristic calibration sandbox               candidate_006, candidate_010,
                                                   candidate_011, candidate_016,
                                                   candidate_017
                                                 • UNTOUCHED during development
```

### The Golden Rule of Development
The 22 held-out evaluation images were stored in `split_dataset/evaluation` and **kept completely untouched throughout development**. Not a single heuristic threshold or filter parameter was calibrated against evaluation images.

---

## 5. Metadata Files & Artifacts

All dataset analysis reports and split assignments are preserved in `results/dataset/`:
- `results/dataset/dataset_summary.json`: High-level resolution, format, and split counts.
- `results/dataset/image_metadata.csv`: Per-image dimensions, aspect ratios, and format logs.
- `results/dataset/similarity_pairs.csv`: Pairwise similarity scores documenting the 0.9667 leakage pair.
- `results/dataset/final_group_split_summary.csv`: Summary of the 21 candidate groups and their partition.
- `results/dataset/final_split_assignments.csv`: Complete image-by-image split assignment mapping.
- `results/dataset/final_split_report.json`: Verification report validating zero cross-split leakage.
