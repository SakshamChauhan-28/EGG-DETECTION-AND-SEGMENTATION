# TASK 1: Pretrained Foundation Segmentation & Reference Mask Generation

Task 1 establishes high-precision instance segmentation masks across the 110-image dataset using vision-language foundation models to serve as empirical pseudo-ground-truth for classical algorithm benchmarking.

---

## 1. Motivation & Challenge

The original dataset contained bounding box labels but zero human pixel-level instance masks. Because training a supervised segmentation model requires dense ground truth, we leveraged foundation models in a zero-shot configuration to extract high-fidelity instance masks without human-in-the-loop annotation bottlenecks.

---

## 2. Model Evolution & Architectural Exploration

Multiple foundation model architectures and prompting paradigms were tested before freezing the final pipeline:

| Trial ID | Approach | Outcome | Diagnosis & Architectural Decision |
| :---: | :--- | :--- | :--- |
| **Trial 01** | **SAM 2.1 Base** (`sam2.1_b.pt`, ~350M params) | **CUDA Out Of Memory (OOM)** crash | Memory footprint exceeded 4GB GPU budget. Switched to SAM 2.1 Tiny. |
| **Trial 02** | **SAM 2.1 Tiny** (`sam2.1_t.pt`) + Manual Point Prompt | High quality masks | Requires manual interaction per image; non-viable for automated pipelines. |
| **Trial 03** | **YOLO-World v2 Small** + SAM 2.1 Tiny | Missed eggs in 3/12 test images (25% miss rate) | Open-vocabulary text embeddings for `"egg"` were brittle under shadows. |
| **Trial 04** | **SAM 2.1 Automatic** Dense Point Grid | 45–120 false positives per image | Automatic dense grids possess zero semantic object priors (segmented plates, counters). |
| **Trial 05** | **Classical Candidate Points** + SAM 2.1 Tiny | 4/88 images succeeded (95.5% failure rate) | Classical color heuristics failed to reliably find interior points. |
| **FINAL** | **Grounding DINO Tiny + SAM 2.1 Tiny** | **100% Success across all 110 images** | Cross-attention vision-language transformer generated robust box prompts. |

---

## 3. The Production Foundation Pipeline

$$\text{Image } (640 \times 640) \xrightarrow[\text{"egg."}]{\text{Grounding DINO Tiny}} \text{Bounding Boxes } \mathcal{B} \xrightarrow{\text{Box Prompts}} \text{SAM 2.1 Tiny} \xrightarrow{} \text{Binary Instance Masks } \mathcal{M}$$

### Model Specifications
- **Grounding DINO:** `IDEA-Research/grounding-dino-tiny`
  - Text prompt: `"egg."`
  - Box threshold: $0.25$
  - Text threshold: $0.20$
- **Segment Anything 2.1:** `sam2.1_hiera_tiny.pt`
  - Prompt mode: Bounding box prompt forwarding
  - Logit threshold: $0.0$ (sigmoid binarization $\ge 0.5$)
- **Outputs Produced:**
  - Individual instance masks: `<image_stem>_egg_<id>.png`
  - Consolidated multi-egg overlays: Visualizations in `TASK 1/visualizations/`
  - Tabular detection metadata: Detection confidence, bounding box coordinates, pixel area.

---

## 4. Directory Organization

```text
TASK 1/
├── README.md
├── scripts/
│   ├── 21_grounding_dino_sam.py             # Full dataset mask generator (Grounding DINO + SAM)
│   ├── 21b_grounding_dino_sam_evaluation.py # Standalone evaluation split generator
│   ├── pipeline_canonical.py               # Modular production pipeline entry point
│   ├── 08_test_sam.py                      # SAM initialization smoke test
│   ├── 09_sam2_baseline.py                 # SAM 2.1 base benchmark
│   ├── 10_sam2_point_test.py               # Interactive point prompt trial
│   ├── 11_yoloworld_egg_test.py            # YOLO-World open-vocabulary test
│   ├── 12_yolo_sam_pipeline_test.py        # YOLO + SAM coupling trial
│   ├── 13_task1_stress_test.py             # 12-image stress test
│   ├── 14_sam2_tiny_auto_test.py           # Automatic dense grid trial
│   └── 15-19_candidate_generator_*.py      # Classical candidate point exploration
├── masks/                                  # 131 instance masks for development set
├── visualizations/                         # Bounding box and segmentation overlay previews
├── reports/                                # CSV detection confidence and area logs
└── experiments/
    ├── task1_segmentation_history/         # Early exploratory checkpoints
    └── gdino_sam_evaluation/               # Evaluation split masks (23 instances)
```

---

## 5. Execution & Reproduction

To run the Grounding DINO + SAM 2 pipeline on any directory of images:
```bash
python "TASK 1/scripts/21_grounding_dino_sam.py" \
  --input-dir "DATASET/development/images" \
  --output-masks "TASK 1/masks" \
  --output-vis "TASK 1/visualizations"
```
*(Requires `torch>=2.0.0`, `transformers>=4.35.0`, and SAM 2.1 checkpoint).*
