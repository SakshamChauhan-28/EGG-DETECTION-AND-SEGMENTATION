import os
"""
TASK 2 - EXPERIMENT 1
Classical Image Processing Baseline

Method:
    HSV/LAB brightness + saturation threshold
    -> morphology
    -> connected components
    -> candidate filtering

IMPORTANT:
    Task 1 masks are used ONLY for evaluation.
    They are NOT used as input to the segmentation algorithm.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
)

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent.parent / "data" / "egg.v1i.folder"))
)

# Leakage-aware split
SPLIT_FILE = (
    PROJECT_ROOT
    / "dataset_report"
    / "final_split_assignments.csv"
)

# Frozen Task 1 masks
TASK1_MASK_DIR = (
    PROJECT_ROOT
    / "task1_gdino_sam"
    / "masks"
)

# Outputs for this experiment
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "task2_experiment_01"
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
# PARAMETERS
# ============================================================

# Relative brightness threshold.
# Pixel must be brighter than this percentile.
BRIGHT_PERCENTILE = 65

# Saturation threshold.
MAX_SATURATION = 100

# Minimum component size.
MIN_AREA = 500

# Morphology
MORPH_KERNEL_SIZE = 7

# Maximum number of candidate components to retain.
MAX_COMPONENTS = 6


# ============================================================
# FIND DEVELOPMENT IMAGES
# ============================================================

def get_development_images():

    df = pd.read_csv(
        SPLIT_FILE
    )

    # Identify columns
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

    dev_df = df[
        split_values.isin(
            ["development", "dev"]
        )
    ].copy()

    eval_df = df[
        split_values.isin(
            ["evaluation", "eval"]
        )
    ].copy()

    # Safety checks
    if len(dev_df) != 88:
        raise ValueError(
            f"Expected 88 development images, "
            f"found {len(dev_df)}"
        )

    if len(eval_df) != 22:
        raise ValueError(
            f"Expected 22 evaluation images, "
            f"found {len(eval_df)}"
        )

    image_paths = []

    for raw_path in dev_df[
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
                image_path
            )

        image_paths.append(
            image_path
        )

    print(
        f"Development images: {len(image_paths)}"
    )

    print(
        f"Evaluation images: {len(eval_df)}"
    )

    print(
        "Evaluation set is NOT processed."
    )

    return sorted(image_paths)


# ============================================================
# TASK 2 SEGMENTATION
# ============================================================

def segment_egg(image):

    """
    Classical segmentation.

    No neural model.
    No Task 1 mask.
    """

    # --------------------------------------------------------
    # Convert to different color spaces
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

    value = hsv[:, :, 2]

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
# LOAD TASK 1 UNION MASK
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

    # IoU
    if union == 0:

        iou = 1.0 if pred_area == 0 and gt_area == 0 else 0.0

    else:

        iou = (
            intersection / union
        )

    # Dice
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

    # Precision
    if pred_area == 0:

        precision = 0.0

    else:

        precision = (
            intersection / pred_area
        )

    # Recall
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

    # Task 2 prediction -> green
    pred = prediction > 127

    # Task 1 target -> red
    gt = target > 127

    # True overlap -> yellow-ish
    overlap = pred & gt

    # False positive -> green
    false_positive = pred & ~gt

    # False negative -> red
    false_negative = gt & ~pred

    # Draw semi-transparent overlays
    overlay = result.copy()

    overlay[overlap] = (
        0,
        255,
        255
    )

    overlay[false_positive] = (
        0,
        255,
        0
    )

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
    print("TASK 2 - EXPERIMENT 1")
    print("BRIGHTNESS + SATURATION BASELINE")
    print("=" * 70)

    image_paths = (
        get_development_images()
    )[:10]

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print("  ERROR loading image")

            continue

        # ----------------------------------------------------
        # Task 2 prediction
        # ----------------------------------------------------

        prediction, components = (
            segment_egg(image)
        )

        # ----------------------------------------------------
        # Frozen Task 1 target
        # ----------------------------------------------------

        target = load_task1_union_mask(
            image_path.name,
            prediction.shape
        )

        if target is None:

            print(
                "  WARNING: Task 1 mask not found"
            )

            continue

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            prediction,
            target
        )

        # ----------------------------------------------------
        # Save mask
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
        # Save visualization
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
        # Report
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
            f"  Components: {len(components)}"
        )

        print(
            f"  IoU: {metrics['iou']:.3f}"
        )

        print(
            f"  Dice: {metrics['dice']:.3f}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        REPORT_DIR
        / "experiment_01_report.csv"
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
    # SUMMARY
    # ========================================================

    if rows:

        results = pd.DataFrame(
            rows
        )

        print()
        print("=" * 70)
        print("EXPERIMENT 1 SUMMARY")
        print("=" * 70)

        print(
            f"Images evaluated: {len(results)}"
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

        print(
            f"\nReport: {report_path}"
        )

        print(
            f"Visualizations: {VIS_DIR}"
        )

        print(
            f"Masks: {MASK_DIR}"
        )


if __name__ == "__main__":
    main()