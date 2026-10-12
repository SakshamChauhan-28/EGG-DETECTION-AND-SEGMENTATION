import os
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd


# ============================================================
# TASK 2 - FINAL EVALUATION
# EXPERIMENT 1 - FROZEN CLASSICAL BASELINE
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
)

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data" / "egg.v1i.folder"))
)

# Leakage-aware split
SPLIT_FILE = (
    PROJECT_ROOT
    / "dataset_report"
    / "final_split_assignments.csv"
)

# ------------------------------------------------------------
# IMPORTANT:
# These are the Task 1 masks generated specifically for the
# 22-image evaluation set.
# ------------------------------------------------------------

TASK1_MASK_DIR = (
    PROJECT_ROOT
    / "task1_gdino_sam"
    / "evaluation"
    / "masks"
)

# ------------------------------------------------------------
# Final evaluation outputs
# ------------------------------------------------------------

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "task2_final_evaluation"
)

VIS_DIR = (
    OUTPUT_ROOT
    / "visualizations"
)

MASK_DIR = (
    OUTPUT_ROOT
    / "masks"
)

REPORT_DIR = (
    OUTPUT_ROOT
    / "reports"
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FROZEN E1 PARAMETERS
# ============================================================

# Relative brightness threshold
BRIGHT_PERCENTILE = 65

# Saturation threshold
MAX_SATURATION = 100

# Minimum connected-component area
MIN_AREA = 500

# Morphological kernel
MORPH_KERNEL_SIZE = 7

# Maximum number of candidate components
MAX_COMPONENTS = 6


# ============================================================
# GET EXACT EVALUATION IMAGES
# ============================================================

def get_evaluation_images():

    df = pd.read_csv(
        SPLIT_FILE
    )

    print("Split assignment columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Identify split column
    # --------------------------------------------------------

    split_column = None

    for column in [
        "final_split",
        "split",
        "dataset_split",
        "assigned_split",
    ]:

        if column in df.columns:

            split_column = column
            break

    if split_column is None:

        raise ValueError(
            f"Could not find split column.\n"
            f"Columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Identify image column
    # --------------------------------------------------------

    image_column = None

    for column in [
        "image_path",
        "filepath",
        "file_path",
        "path",
        "image",
        "filename",
        "file_name",
    ]:

        if column in df.columns:

            image_column = column
            break

    if image_column is None:

        raise ValueError(
            f"Could not find image column.\n"
            f"Columns: {list(df.columns)}"
        )

    split_values = (
        df[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    development_df = df[
        split_values.isin(
            ["development", "dev"]
        )
    ].copy()

    evaluation_df = df[
        split_values.isin(
            ["evaluation", "eval"]
        )
    ].copy()

    # --------------------------------------------------------
    # HARD SAFETY CHECKS
    # --------------------------------------------------------

    if len(development_df) != 88:

        raise ValueError(
            f"Expected 88 development images, "
            f"found {len(development_df)}"
        )

    if len(evaluation_df) != 22:

        raise ValueError(
            f"Expected 22 evaluation images, "
            f"found {len(evaluation_df)}"
        )

    print()
    print("=" * 70)
    print("LEAKAGE-AWARE SPLIT")
    print("=" * 70)

    print(
        f"Development images: {len(development_df)}"
    )

    print(
        f"Evaluation images:  {len(evaluation_df)}"
    )

    # --------------------------------------------------------
    # Resolve evaluation paths
    # --------------------------------------------------------

    image_paths = []

    for raw_path in evaluation_df[
        image_column
    ]:

        raw_path = str(raw_path).strip()

        path = Path(raw_path)

        if path.is_absolute():

            image_path = path

        else:

            candidate = (
                DATASET_ROOT
                / raw_path
            )

            if candidate.exists():

                image_path = candidate

            else:

                matches = list(
                    DATASET_ROOT.rglob(
                        Path(raw_path).name
                    )
                )

                if len(matches) != 1:

                    raise FileNotFoundError(
                        f"Could not uniquely locate: "
                        f"{raw_path}"
                    )

                image_path = matches[0]

        if not image_path.exists():

            raise FileNotFoundError(
                f"Evaluation image does not exist:\n"
                f"{image_path}"
            )

        image_paths.append(
            image_path
        )

    # --------------------------------------------------------
    # FINAL LEAKAGE CHECK
    # --------------------------------------------------------

    development_filenames = {
        Path(str(p)).name
        for p in development_df[
            image_column
        ]
    }

    evaluation_filenames = {
        p.name
        for p in image_paths
    }

    overlap = (
        development_filenames
        &
        evaluation_filenames
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED!\n"
            f"Overlapping images: {overlap}"
        )

    print()
    print("Leakage check: PASSED")
    print("Development/evaluation overlap: 0")

    print("=" * 70)

    return sorted(image_paths)


# ============================================================
# TASK 2 SEGMENTATION
# ============================================================

def segment_egg(image):

    """
    FROZEN E1 CLASSICAL SEGMENTATION.

    No neural model.
    No Task 1 mask.
    No evaluation feedback.

    Pipeline:

        HSV/LAB brightness + saturation
        -> morphology
        -> connected components
        -> candidate filtering
    """

    # --------------------------------------------------------
    # Convert to color spaces
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    saturation = hsv[:, :, 1]

    lightness = lab[:, :, 0]

    # --------------------------------------------------------
    # Adaptive brightness threshold
    # --------------------------------------------------------

    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= brightness_threshold
    )

    # --------------------------------------------------------
    # Low saturation
    # --------------------------------------------------------

    low_saturation = (
        saturation <= MAX_SATURATION
    )

    # --------------------------------------------------------
    # Combine cues
    # --------------------------------------------------------

    mask = (
        bright_mask
        &
        low_saturation
    )

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_AREA:

            continue

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    # --------------------------------------------------------
    # Sort largest components first
    # --------------------------------------------------------

    components.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    components = components[
        :MAX_COMPONENTS
    ]

    # --------------------------------------------------------
    # Reconstruct final mask
    # --------------------------------------------------------

    final_mask = np.zeros_like(
        mask
    )

    for component in components:

        label = component["label"]

        final_mask[
            labels == label
        ] = 255

    return final_mask, components


# ============================================================
# LOAD TASK 1 EVALUATION MASK
# ============================================================

def load_task1_union_mask(
    image_name,
    shape,
):

    stem = Path(
        image_name
    ).stem

    mask_files = sorted(
        TASK1_MASK_DIR.glob(
            f"{stem}_egg_*.png"
        )
    )

    if not mask_files:

        return None

    union = np.zeros(
        shape,
        dtype=np.uint8
    )

    for mask_file in mask_files:

        mask = cv2.imread(
            str(mask_file),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            continue

        if mask.shape != shape:

            mask = cv2.resize(
                mask,
                (
                    shape[1],
                    shape[0],
                ),
                interpolation=cv2.INTER_NEAREST
            )

        union[
            mask > 127
        ] = 255

    return union


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
):

    pred = prediction > 127
    gt = target > 127

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_area = pred.sum()
    gt_area = gt.sum()

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:

        iou = (
            1.0
            if pred_area == 0
            and gt_area == 0
            else 0.0
        )

    else:

        iou = (
            intersection / union
        )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    denominator = (
        pred_area + gt_area
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * intersection
            / denominator
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if pred_area == 0:

        precision = 0.0

    else:

        precision = (
            intersection / pred_area
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if gt_area == 0:

        recall = 0.0

    else:

        recall = (
            intersection / gt_area
        )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image,
    prediction,
    target,
    output_path,
):

    result = image.copy()

    # Task 2 prediction
    pred = prediction > 127

    # Task 1 pseudo-GT
    gt = target > 127

    # True overlap
    overlap = pred & gt

    # False positive
    false_positive = pred & ~gt

    # False negative
    false_negative = gt & ~pred

    overlay = result.copy()

    # Yellow = overlap
    overlay[overlap] = (
        0,
        255,
        255
    )

    # Green = false positive
    overlay[false_positive] = (
        0,
        255,
        0
    )

    # Red = false negative
    overlay[false_negative] = (
        0,
        0,
        255
    )

    result = cv2.addWeighted(
        result,
        0.55,
        overlay,
        0.45,
        0
    )

    cv2.imwrite(
        str(output_path),
        result
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 2 - FINAL EVALUATION")
    print("EXPERIMENT 1 - FROZEN CLASSICAL BASELINE")
    print("=" * 70)

    print()
    print("This is the FINAL evaluation.")
    print("The 22 evaluation images are untouched.")
    print("E1 parameters are frozen.")
    print("No parameter tuning will be performed.")

    print()
    print("Frozen parameters:")
    print(
        f"  BRIGHT_PERCENTILE = "
        f"{BRIGHT_PERCENTILE}"
    )

    print(
        f"  MAX_SATURATION    = "
        f"{MAX_SATURATION}"
    )

    print(
        f"  MIN_AREA          = "
        f"{MIN_AREA}"
    )

    print(
        f"  MORPH_KERNEL_SIZE = "
        f"{MORPH_KERNEL_SIZE}"
    )

    print(
        f"  MAX_COMPONENTS    = "
        f"{MAX_COMPONENTS}"
    )

    # --------------------------------------------------------
    # Check Task 1 evaluation masks
    # --------------------------------------------------------

    if not TASK1_MASK_DIR.exists():

        raise FileNotFoundError(
            "Task 1 evaluation mask directory "
            "was not found:\n"
            f"{TASK1_MASK_DIR}"
        )

    # --------------------------------------------------------
    # Get EXACT 22 evaluation images
    # --------------------------------------------------------

    image_paths = (
        get_evaluation_images()
    )

    print()
    print(
        f"Evaluation images found: "
        f"{len(image_paths)}"
    )

    rows = []

    # --------------------------------------------------------
    # Process all 22 images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "  ERROR loading image"
            )

            continue

        # ----------------------------------------------------
        # Task 2 prediction
        # ----------------------------------------------------

        prediction, components = (
            segment_egg(image)
        )

        # ----------------------------------------------------
        # Task 1 frozen evaluation mask
        # ----------------------------------------------------

        target = load_task1_union_mask(
            image_path.name,
            prediction.shape
        )

        if target is None:

            raise FileNotFoundError(
                "Task 1 evaluation mask not found for:\n"
                f"{image_path.name}"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            prediction,
            target
        )

        # ----------------------------------------------------
        # Save final E1 mask
        # ----------------------------------------------------

        output_mask = (
            MASK_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_mask),
            prediction
        )

        # ----------------------------------------------------
        # Save comparison visualization
        # ----------------------------------------------------

        output_visualization = (
            VIS_DIR
            / image_path.name
        )

        save_visualization(
            image,
            prediction,
            target,
            output_visualization
        )

        # ----------------------------------------------------
        # Report row
        # ----------------------------------------------------

        row = {
            "image": image_path.name,
            "components": len(components),
            "predicted_area": int(
                (prediction > 127).sum()
            ),
            "target_area": int(
                (target > 127).sum()
            ),
            "iou": round(
                metrics["iou"],
                4
            ),
            "dice": round(
                metrics["dice"],
                4
            ),
            "precision": round(
                metrics["precision"],
                4
            ),
            "recall": round(
                metrics["recall"],
                4
            ),
        }

        rows.append(row)

        print(
            f"  Components: "
            f"{len(components)}"
        )

        print(
            f"  IoU: "
            f"{metrics['iou']:.3f}"
        )

        print(
            f"  Dice: "
            f"{metrics['dice']:.3f}"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.3f}"
        )

        print(
            f"  Recall: "
            f"{metrics['recall']:.3f}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    if not rows:

        raise RuntimeError(
            "No evaluation results were generated."
        )

    report_path = (
        REPORT_DIR
        / "final_e1_evaluation_report.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "components",
                "predicted_area",
                "target_area",
                "iou",
                "dice",
                "precision",
                "recall",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    print()
    print("=" * 70)
    print("FINAL E1 EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: "
        f"{len(results)}"
    )

    print(
        f"Mean IoU: "
        f"{results['iou'].mean():.4f}"
    )

    print(
        f"Median IoU: "
        f"{results['iou'].median():.4f}"
    )

    print(
        f"Mean Dice: "
        f"{results['dice'].mean():.4f}"
    )

    print(
        f"Mean Precision: "
        f"{results['precision'].mean():.4f}"
    )

    print(
        f"Mean Recall: "
        f"{results['recall'].mean():.4f}"
    )

    print()
    print(
        f"Report: "
        f"{report_path}"
    )

    print(
        f"Visualizations: "
        f"{VIS_DIR}"
    )

    print(
        f"Masks: "
        f"{MASK_DIR}"
    )

    print()
    print("=" * 70)
    print("FINAL E1 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()