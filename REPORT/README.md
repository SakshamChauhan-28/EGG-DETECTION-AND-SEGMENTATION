# Nosh Robotics Computer Vision Assignment — Technical Report Documentation

Welcome to the comprehensive assignment report and research documentation for the **Nosh Robotics Computer Vision Assignment (Egg Detection, Segmentation, Evaluation, and Tracking)**.

---

## 1. Quick Access to Deliverables

- **Complete Technical Assignment Report (PDF on Google Drive):**  
  👉 **[https://drive.google.com/file/d/1z3NqJekPECgqXZN-NxWknRqwoE1C_gnp/view?usp=sharing](https://drive.google.com/file/d/1z3NqJekPECgqXZN-NxWknRqwoE1C_gnp/view?usp=sharing)**
- **Full Video Output Demonstration (MP4 on Google Drive):**  
  👉 **[https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing](https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing)**

---

## 2. Report Index & Documentation Architecture

The complete documentation suite is organized across dedicated engineering chapters:

1. **[Assignment Executive Report](assignment_report.md)**  
   *High-level summary of the engineering objectives, findings, results scorecard, and conclusions.*
2. **[Methodology & Architectural Design](methodology.md)**  
   *Deep mathematical and algorithmic formulations for Task 1 (Foundation Models), Task 2 (Classical Vision), Task 3 (Evaluation Overlap), and Task 4 (SORT Tracking).*
3. **[Experimental History & Benchmarks](experimental_results.md)**  
   *Systematic tabular breakdown of all experimental hypotheses across Task 1 (Trials 1–5 + Final), Task 2 (E1–E8), and Task 4 (V1–V8 + Final SORT).*
4. **[Dataset Forensic Audit & Leakage Analysis](dataset_and_leakage_analysis.md)**  
   *Investigation into cross-split image duplication, pairwise feature similarity analysis, and graph-based scene grouping.*
5. **[Engineering Limitations & Edge Cases](limitations.md)**  
   *Honest disclosure of algorithmic failure modes, specular reflection challenges, and tracking ID fragmentation under severe occlusion.*
6. **[Reproducibility & Execution Guide](reproducibility.md)**  
   *Step-by-step instructions to verify dataset integrity, run classical segmentations, evaluate overlap metrics, and track video sequences.*

---

## 3. Included Report Assets & Figures

The `REPORT/figures/` subdirectory contains qualitative and diagnostic assets:
- `cover_montage.jpg`: Overview montage of assignment tasks.
- `leakage_pair.jpg`: Cross-split leakage forensic evidence pair (similarity: $0.9667$).
- `t1_gdino_sam_grid.jpg`: Grounding DINO + SAM 2 zero-shot segmentation outputs.
- `t2_exp_01.jpg` through `t2_exp_08.jpg`: Qualitative segmentation results across classical experiments E1–E8.
- `t3_sample_best.jpg` to `t3_sample_worst.jpg`: High- and low-performing evaluation split cases.
- `t4_spinning_motion.jpg`, `t4_hand_interaction.jpg`, `t4_occlusion_prediction.jpg`: Video tracking telemetry snapshots.
