from pathlib import Path
"""
======================================================================
TASK 2 - EXPERIMENT 7
E1 + CONSERVATIVE SPATIAL COHERENCE FILTERING
======================================================================

Purpose:
    E1 achieved the best development performance so far:

        Mean IoU       = 0.3273
        Mean Dice      = 0.4641
        Precision      = 0.3545
        Recall         = 0.7728

    The main problem is excessive background segmentation.

    E7 therefore keeps the original E1 segmentation strategy and
    applies ONLY conservative spatial cleanup.

    Unlike E6:
        - No aggressive shape filtering
        - No circularity filtering
        - No solidity filtering
        - No GrabCut
        - No edge-based segmentation
        - No Task-1 information during segmentation

    Task 1 masks are used ONLY for evaluation.

======================================================================
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent.parent / "data" / "egg.v1i.folder"))

SPLIT_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset_report",
    "final_split_assignments.csv"
)

TASK1_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "task1_gdino_sam",
    "masks"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "task2_experiment_07"
)

MASK_DIR = os.path.join(
    OUTPUT_ROOT,
    "masks"
)

VIS_DIR = os.path.join(
    OUTPUT_ROOT,
    "visualizations"
)

REPORT_DIR = os.path.join(
    OUTPUT_ROOT,
    "reports"
)

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================================
# EXPERIMENT PARAMETERS
# =====================================================================

# These are intentionally the same core values as E1.

BRIGHT_PERCENTILE = 65

MAX_SATURATION = 100


# E1 morphology

E1_MORPH_KERNEL = 7


# ---------------------------------------------------------------------
# E7 conservative cleanup
# ---------------------------------------------------------------------

# Very small connected regions are usually texture/noise.
#
# We deliberately keep this low.
#
# E6 used aggressive component-level shape rejection.
# E7 does NOT do that.

MIN_COMPONENT_AREA = 150


# Fill relatively small holes inside candidate regions.

HOLE_CLOSE_KERNEL = 9


# Remove tiny isolated noise after closing.

NOISE_OPEN_KERNEL = 3


# ---------------------------------------------------------------------
# Spatial coherence
# ---------------------------------------------------------------------

# A component is preserved if its area is large enough.
#
# We DO NOT use:
#   circularity
#   solidity
#   aspect ratio
#   extent
#
# because E6 demonstrated that these assumptions are too restrictive.

LARGE_COMPONENT_AREA = 1500


# ---------------------------------------------------------------------
# Initial development subset
# ---------------------------------------------------------------------

# Use exactly the same 10 images used by E1-E6.

MAX_IMAGES = 88


# =====================================================================
# GET DEVELOPMENT IMAGES
# =====================================================================

def get_development_images():

    df = pd.read_csv(SPLIT_CSV)

    split_col = None

    for col in [
        "final_split",
        "split",
        "dataset_split",
        "assigned_split"
    ]:
        if col in df.columns:
            split_col = col
            break

    if split_col is None:
        raise ValueError(
            "Could not find development/evaluation split column."
        )

    path_col = None

    for col in [
        "image_path",
        "filepath",
        "file_path",
        "path",
        "image",
        "filename",
        "file_name"
    ]:
        if col in df.columns:
            path_col = col
            break

    if path_col is None:
        raise ValueError(
            "Could not find image path column."
        )

    split_values = (
        df[split_col]
        .astype(str)
        .str.lower()
    )

    dev_df = df[
        split_values.isin(
            [
                "development",
                "dev",
                "train"
            ]
        )
    ]

    eval_df = df[
        split_values.isin(
            [
                "evaluation",
                "eval",
                "test"
            ]
        )
    ]

    print(
        f"Development images: {len(dev_df)}"
    )

    print(
        f"Evaluation images: {len(eval_df)}"
    )

    image_paths = []

    for value in dev_df[path_col]:

        path = str(value)

        # Direct path

        if os.path.exists(path):
            image_paths.append(path)
            continue

        # Dataset-relative path

        candidate = os.path.join(
            DATASET_ROOT,
            path
        )

        if os.path.exists(candidate):
            image_paths.append(candidate)
            continue

        # Search by filename

        basename = os.path.basename(path)

        matches = glob.glob(
            os.path.join(
                DATASET_ROOT,
                "**",
                basename
            ),
            recursive=True
        )

        if matches:
            image_paths.append(matches[0])

    return image_paths


# =====================================================================
# LOAD TASK 1 UNION MASK
# =====================================================================

def load_task1_union_mask(image_path):

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    pattern = os.path.join(
        TASK1_MASK_DIR,
        f"{stem}_egg_*.png"
    )

    mask_paths = sorted(
        glob.glob(pattern)
    )

    if not mask_paths:
        return None

    union_mask = None

    for path in mask_paths:

        mask = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        mask = (
            mask > 127
        ).astype(np.uint8) * 255

        if union_mask is None:
            union_mask = mask

        else:
            union_mask = cv2.bitwise_or(
                union_mask,
                mask
            )

    return union_mask


# =====================================================================
# E1 BASELINE
# =====================================================================

def create_e1_mask(image):

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

    # ---------------------------------------------------------------
    # Brightness threshold
    # ---------------------------------------------------------------

    threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= threshold
    ).astype(np.uint8) * 255

    # ---------------------------------------------------------------
    # Saturation constraint
    # ---------------------------------------------------------------

    low_saturation = (
        saturation <= MAX_SATURATION
    ).astype(np.uint8) * 255

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    mask = cv2.bitwise_and(
        bright_mask,
        low_saturation
    )

    # ---------------------------------------------------------------
    # Same morphology used by E1
    # ---------------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            E1_MORPH_KERNEL,
            E1_MORPH_KERNEL
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

    return mask


# =====================================================================
# E7 SPATIAL COHERENCE FILTER
# =====================================================================

def spatial_coherence_filter(mask):

    # ---------------------------------------------------------------
    # STEP 1:
    # Remove only very small connected components.
    # ---------------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    component_count = 0
    removed_small = 0

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        component_count += 1

        if area >= MIN_COMPONENT_AREA:

            cleaned[
                labels == label
            ] = 255

        else:

            removed_small += 1

    # ---------------------------------------------------------------
    # STEP 2:
    # Close small gaps.
    #
    # This is particularly important because an egg may be split
    # by highlights, shadows or small texture variations.
    # ---------------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            HOLE_CLOSE_KERNEL,
            HOLE_CLOSE_KERNEL
        )
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # ---------------------------------------------------------------
    # STEP 3:
    # Very light opening to remove isolated pixels.
    # ---------------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            NOISE_OPEN_KERNEL,
            NOISE_OPEN_KERNEL
        )
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        open_kernel
    )

    return (
        cleaned,
        component_count,
        removed_small
    )


# =====================================================================
# FINAL SEGMENTATION
# =====================================================================

def segment_egg(image):

    # Start from E1.

    e1_mask = create_e1_mask(
        image
    )

    # Conservative spatial cleanup.

    final_mask, component_count, removed_small = (
        spatial_coherence_filter(
            e1_mask
        )
    )

    return (
        final_mask,
        component_count,
        removed_small
    )


# =====================================================================
# METRICS
# =====================================================================

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 127
    )

    ground_truth = (
        ground_truth > 127
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    prediction_pixels = (
        prediction.sum()
    )

    ground_truth_pixels = (
        ground_truth.sum()
    )

    # IoU

    if union > 0:

        iou = (
            intersection
            / union
        )

    else:

        iou = (
            1.0
            if ground_truth_pixels == 0
            else 0.0
        )

    # Dice

    denominator = (
        prediction_pixels
        + ground_truth_pixels
    )

    if denominator > 0:

        dice = (
            2.0
            * intersection
            / denominator
        )

    else:

        dice = (
            1.0
            if ground_truth_pixels == 0
            else 0.0
        )

    # Precision

    if prediction_pixels > 0:

        precision = (
            intersection
            / prediction_pixels
        )

    else:

        precision = 0.0

    # Recall

    if ground_truth_pixels > 0:

        recall = (
            intersection
            / ground_truth_pixels
        )

    else:

        recall = (
            1.0
            if prediction_pixels == 0
            else 0.0
        )

    return (
        iou,
        dice,
        precision,
        recall
    )


# =====================================================================
# VISUALIZATION
# =====================================================================

def create_visualization(
    image,
    prediction,
    ground_truth
):

    vis = image.copy()

    pred = (
        prediction > 127
    )

    gt = (
        ground_truth > 127
    )

    # Correct prediction / overlap

    overlap = (
        pred & gt
    )

    # Prediction but not Task 1

    false_positive = (
        pred & (~gt)
    )

    # Task 1 but not prediction

    false_negative = (
        gt & (~pred)
    )

    overlay = vis.copy()

    # Yellow = correct overlap

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

    vis = cv2.addWeighted(
        vis,
        0.55,
        overlay,
        0.45,
        0
    )

    return vis


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("TASK 2 - EXPERIMENT 7")
    print("E1 + CONSERVATIVE SPATIAL COHERENCE")
    print("=" * 70)

    print()
    print("E7 strategy:")
    print("  1. Start with E1")
    print("  2. Remove only tiny components")
    print("  3. Fill small gaps")
    print("  4. Apply very light cleanup")
    print()
    print("No shape filtering.")
    print("No circularity filtering.")
    print("No GrabCut.")
    print("No edge detection.")
    print("No Task-1 mask used during segmentation.")

    # ---------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------

    image_paths = get_development_images()

    if not image_paths:

        raise RuntimeError(
            "No development images found."
        )

    # Same subset as E1-E6

    image_paths = image_paths[
        :MAX_IMAGES
    ]

    print()
    print(
        f"Images evaluated: "
        f"{len(image_paths)}"
    )

    print(
        "Evaluation set is NOT processed."
    )

    # ---------------------------------------------------------------
    # Metric storage
    # ---------------------------------------------------------------

    results = []

    ious = []
    dices = []
    precisions = []
    recalls = []

    removed_counts = []

    # ---------------------------------------------------------------
    # Process
    # ---------------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )

        print()
        print(
            f"[{index}/{len(image_paths)}] "
            f"{filename}"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "  ERROR: Could not read image."
            )

            continue

        # -----------------------------------------------------------
        # Segmentation
        # -----------------------------------------------------------

        (
            prediction,
            component_count,
            removed_small
        ) = segment_egg(
            image
        )

        # -----------------------------------------------------------
        # Frozen Task 1 mask
        # -----------------------------------------------------------

        ground_truth = (
            load_task1_union_mask(
                image_path
            )
        )

        if ground_truth is None:

            print(
                "  WARNING: Task 1 mask not found."
            )

            continue

        if (
            ground_truth.shape
            != prediction.shape
        ):

            ground_truth = cv2.resize(
                ground_truth,
                (
                    prediction.shape[1],
                    prediction.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        (
            iou,
            dice,
            precision,
            recall
        ) = calculate_metrics(
            prediction,
            ground_truth
        )

        print(
            f"  Original components: "
            f"{component_count}"
        )

        print(
            f"  Removed small components: "
            f"{removed_small}"
        )

        print(
            f"  IoU: {iou:.3f}"
        )

        print(
            f"  Dice: {dice:.3f}"
        )

        print(
            f"  Precision: {precision:.3f}"
        )

        print(
            f"  Recall: {recall:.3f}"
        )

        # -----------------------------------------------------------
        # Save mask
        # -----------------------------------------------------------

        stem = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            MASK_DIR,
            f"{stem}_e7.png"
        )

        cv2.imwrite(
            mask_path,
            prediction
        )

        # -----------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------

        visualization = (
            create_visualization(
                image,
                prediction,
                ground_truth
            )
        )

        vis_path = os.path.join(
            VIS_DIR,
            f"{stem}_e7_comparison.png"
        )

        cv2.imwrite(
            vis_path,
            visualization
        )

        # -----------------------------------------------------------
        # Record
        # -----------------------------------------------------------

        results.append(
            {
                "image": filename,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "original_components": component_count,
                "removed_small_components": removed_small,
            }
        )

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        removed_counts.append(
            removed_small
        )

    # =================================================================
    # REPORT
    # =================================================================

    if not results:

        raise RuntimeError(
            "No images were successfully evaluated."
        )

    results_df = pd.DataFrame(
        results
    )

    report_path = os.path.join(
        REPORT_DIR,
        "experiment_07_report.csv"
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    # =================================================================
    # SUMMARY
    # =================================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 07 SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: "
        f"{len(results)}"
    )

    print(
        f"Mean IoU: "
        f"{np.mean(ious):.4f}"
    )

    print(
        f"Median IoU: "
        f"{np.median(ious):.4f}"
    )

    print(
        f"Mean Dice: "
        f"{np.mean(dices):.4f}"
    )

    print(
        f"Mean Precision: "
        f"{np.mean(precisions):.4f}"
    )

    print(
        f"Mean Recall: "
        f"{np.mean(recalls):.4f}"
    )

    print(
        f"Mean removed components/image: "
        f"{np.mean(removed_counts):.2f}"
    )

    print()
    print("Report:")
    print(report_path)

    print()
    print("Visualizations:")
    print(VIS_DIR)

    print()
    print("Masks:")
    print(MASK_DIR)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()