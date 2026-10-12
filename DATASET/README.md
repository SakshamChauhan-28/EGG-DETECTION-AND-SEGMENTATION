# DATASET: Leakage-Aware Partition & Forensic Audit

This directory houses the benchmark image corpus, split manifests, scene-grouping records, exploratory data analysis, and the official leakage-aware development/evaluation partition for the Nosh Robotics Computer Vision Assignment.

---

## 1. Provenance & Dataset Attribution

- **Original Source:** Exported from Roboflow Universe ([Egg Detection Dataset](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr)).
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Modality:** 110 JPEG photographic frames of raw/intact eggs in domestic kitchen setups, bowls, cartons, and countertops.
- **Native Resolution:** $640 \times 640$ pixels across all images.
- **Critical Notice on Annotations:** The source dataset provided only bounding box annotations. **Zero human pixel-level segmentation masks existed in the original corpus.** All segmentation ground-truth used across Task 2 and Task 3 are high-fidelity empirical reference masks (pseudo-ground-truth) generated in Task 1.

---

## 2. Forensic Discovery: Cross-Split Data Leakage

In standard machine learning workflows, random train/validation/test splits frequently introduce **near-duplicate photographic leakage**, inflating benchmark scores without true generalization.

### Pairwise Audit
An exhaustive pairwise visual similarity matrix ($N = 110$, $\frac{110 \times 109}{2} = 5,995$ unique image pairs) was evaluated using pretrained ResNet-18 feature embeddings:
- A near-identical pair spanning the original splits was uncovered:
  - **Test image:** `not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5.jpg`
  - **Train image:** `not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad.jpg`
  - **Pairwise Cosine Similarity:** **0.9667** (Identical physical kitchen countertop, lighting, camera angle, and egg arrangement with minimal camera shift).

---

## 3. Scene-Grouping Methodology & Leakage Mitigation

To eliminate scene contamination:
1. **Clustering:** Candidate scene clusters were formed using connected components over a similarity threshold of $\tau = 0.82$.
2. **Cluster Validation:** 21 candidate multi-image scene groups were identified alongside singleton scenes.
3. **Group Partitioning:** Entire scene clusters were assigned as atomic, indivisible blocks to either the **Development** split or the **Evaluation** split.
4. **Target Ratio:** Maintained a strict $4:1$ ratio:
   - **Development Split:** 88 images ($80.0\%$)
   - **Evaluation Split:** 22 images ($20.0\%$)
5. **Guarantee:** **Zero cross-split scene overlap.**

---

## 4. Directory Structure

```text
DATASET/
├── README.md
├── manifests/
│   ├── development.csv             # 88 image manifest with group assignments
│   ├── evaluation.csv              # 22 evaluation manifest with group assignments
│   ├── scene_groups.csv            # Cluster membership per scene group
│   ├── final_split_assignments.csv # Master split mapping across all 110 images
│   ├── final_group_split_summary.csv
│   ├── similarity_pairs.csv        # Cross-image similarity audit scores
│   └── image_metadata.csv          # Width, height, aspect ratio, file size
├── development/
│   ├── images/                     # 88 development JPEG images (640x640)
│   └── reference_masks/            # Task 1 pseudo-ground-truth instance masks
├── evaluation/
│   ├── images/                     # 22 evaluation JPEG images (640x640)
│   └── reference_masks/            # Task 1 pseudo-ground-truth instance masks
├── analysis/
│   ├── reports/                    # JSON summary records (final_split_report.json)
│   ├── visualizations/             # PCA projection plots & candidate groups
│   ├── contact_sheets/             # Full 110-image mosaic overview
│   └── logs/
└── scripts/
    ├── 01_dataset_analysis.py       # Metadata extraction and sizing audit
    ├── 02_create_contact_sheet.py  # Mosaic generation
    ├── 03_similarity_analysis.py   # Pairwise feature embedding similarity
    ├── 06_scene_grouping.py        # Graph connected-component clustering
    ├── 07_create_leakage_aware_split.py # Split optimization solver
    └── validate_dataset.py         # Automated integrity & leakage verification test
```

---

## 5. Verification Command

Execute the self-contained verification test to audit split integrity:
```bash
python DATASET/scripts/validate_dataset.py
```
**Expected Output:**
```text
[INFO] Validating dataset integrity...
[INFO] Manifest entries: Development=88, Evaluation=22
[PASS] Scene-group isolation verified. Zero overlapping groups between splits.
[INFO] Disk image counts: Development=88, Evaluation=22
[PASS] Zero duplicate image hashes across development and evaluation partitions.
[PASS] All images verified at 640x640 resolution.
[INFO] Reference masks: Development=131, Evaluation=23
[SUCCESS] All dataset integrity checks PASSED successfully.
```
