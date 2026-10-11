from pathlib import Path
"""
======================================================================
TASK 2 - EXPERIMENT 6
E1 + CONNECTED COMPONENT / SHAPE FILTERING
======================================================================

Goal:
    Improve Experiment 1 by reducing background false positives
    while preserving E1's relatively high recall.

E1:
    Brightness + Saturation thresholding
        ->
    Morphology
        ->
    Connected components

E6:
    Brightness + Saturation thresholding
        ->
    Morphology
        ->
    Connected components
        ->
    Shape filtering
        ->
    Morphology
        ->
    Final mask

IMPORTANT:
    Task 1 masks are used ONLY for evaluation.
    They are NOT used by the segmentation algorithm.
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
    "task2_experiment_06"
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

# ---- E1 threshold parameters ----

BRIGHT_PERCENTILE = 65

MAX_SATURATION = 100


# ---- Morphology ----

MORPH_KERNEL_SIZE = 7


# ---- Component filtering ----

MIN_AREA = 500

MAX_AREA_RATIO = 0.20
# Reject components larger than 20% of the complete image.
# This is intended to suppress large background regions.


MIN_SOLIDITY = 0.55

MIN_EXTENT = 0.25

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0


# Circularity:
# 4*pi*area / perimeter^2
#
# Eggs are generally reasonably compact/elliptical.
# Very irregular texture should have lower circularity.

MIN_CIRCULARITY = 0.18


# Components touching the image border are suspicious,
# but we do NOT automatically reject them because an egg
# can occasionally be close to an image boundary.

BORDER_AREA_RATIO_LIMIT = 0.08


# Final morphology

FINAL_CLOSE_KERNEL = 5
FINAL_OPEN_KERNEL = 3


# Evaluate only the first N development images.
# Set to 88 later if we want the complete development evaluation.

MAX_IMAGES = 10


# =====================================================================
# UTILITY FUNCTIONS
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
            "Could not find development/evaluation split column "
            "in final_split_assignments.csv"
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
            "Could not find image path column in split CSV."
        )

    dev_df = df[
        df[split_col]
        .astype(str)
        .str.lower()
        .isin(["development", "dev", "train"])
    ].copy()

    eval_df = df[
        df[split_col]
        .astype(str)
        .str.lower()
        .isin(["evaluation", "eval", "test"])
    ].copy()

    print(f"Development images: {len(dev_df)}")
    print(f"Evaluation images: {len(eval_df)}")

    if len(dev_df) != 88:
        print(
            f"WARNING: Expected 88 development images, "
            f"found {len(dev_df)}"
        )

    if len(eval_df) != 22:
        print(
            f"WARNING: Expected 22 evaluation images, "
            f"found {len(eval_df)}"
        )

    image_paths = []

    for value in dev_df[path_col]:

        path = str(value)

        if os.path.exists(path):
            image_paths.append(path)
            continue

        # Try relative to dataset root
        candidate = os.path.join(DATASET_ROOT, path)

        if os.path.exists(candidate):
            image_paths.append(candidate)
            continue

        # Search by basename
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
        else:
            print(
                f"WARNING: Image not found: {path}"
            )

    return image_paths


# =====================================================================
# TASK 1 MASK LOADER
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

    if len(mask_paths) == 0:
        return None

    union_mask = None

    for mask_path in mask_paths:

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        mask = (mask > 127).astype(np.uint8)

        if union_mask is None:
            union_mask = mask
        else:
            union_mask = np.maximum(
                union_mask,
                mask
            )

    return union_mask


# =====================================================================
# E1 BASE MASK
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

    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= brightness_threshold
    ).astype(np.uint8) * 255

    # ---------------------------------------------------------------
    # Low saturation
    # ---------------------------------------------------------------

    low_saturation_mask = (
        saturation <= MAX_SATURATION
    ).astype(np.uint8) * 255

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    mask = cv2.bitwise_and(
        bright_mask,
        low_saturation_mask
    )

    # ---------------------------------------------------------------
    # Morphological cleanup
    # ---------------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
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
# SHAPE FEATURES
# =====================================================================

def component_features(
    component_mask,
    component_area
):

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4.0
            * np.pi
            * area
            / (perimeter * perimeter)
        )

    else:
        circularity = 0.0

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area > 0:

        solidity = (
            area / hull_area
        )

    else:
        solidity = 0.0

    x, y, w, h = cv2.boundingRect(
        contour
    )

    bbox_area = w * h

    if bbox_area > 0:

        extent = (
            area / bbox_area
        )

    else:
        extent = 0.0

    aspect_ratio = (
        max(w, h)
        / max(1, min(w, h))
    )

    return {
        "area": float(area),
        "component_area": float(component_area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "extent": float(extent),
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "x": int(x),
        "y": int(y),
    }


# =====================================================================
# E6 SEGMENTATION
# =====================================================================

def segment_egg(image):

    # ---------------------------------------------------------------
    # STEP 1
    # Start directly from the strongest baseline: E1
    # ---------------------------------------------------------------

    e1_mask = create_e1_mask(image)

    height, width = e1_mask.shape

    image_area = height * width

    # ---------------------------------------------------------------
    # STEP 2
    # Connected components
    # ---------------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        e1_mask,
        connectivity=8
    )

    final_mask = np.zeros_like(
        e1_mask
    )

    candidate_count = 0
    kept_count = 0

    component_records = []

    # ---------------------------------------------------------------
    # STEP 3
    # Analyze each component
    # ---------------------------------------------------------------

    for label in range(
        1,
        num_labels
    ):

        component_area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if component_area < MIN_AREA:
            continue

        candidate_count += 1

        component = np.zeros_like(
            e1_mask
        )

        component[
            labels == label
        ] = 255

        features = component_features(
            component,
            component_area
        )

        if features is None:
            continue

        x = features["x"]
        y = features["y"]
        w = features["width"]
        h = features["height"]

        area_ratio = (
            component_area
            / image_area
        )

        touches_border = (
            x <= 0
            or y <= 0
            or (x + w) >= width
            or (y + h) >= height
        )

        # -----------------------------------------------------------
        # Shape checks
        # -----------------------------------------------------------

        area_ok = (
            area_ratio <= MAX_AREA_RATIO
        )

        solidity_ok = (
            features["solidity"]
            >= MIN_SOLIDITY
        )

        extent_ok = (
            features["extent"]
            >= MIN_EXTENT
        )

        aspect_ok = (
            MIN_ASPECT_RATIO
            <= (
                1.0 / features["aspect_ratio"]
                if features["aspect_ratio"] > 0
                else 0
            )
            <= MAX_ASPECT_RATIO
        )

        circularity_ok = (
            features["circularity"]
            >= MIN_CIRCULARITY
        )

        # -----------------------------------------------------------
        # Border handling
        #
        # Border components are allowed when they are relatively
        # small. Large border-connected regions are more likely
        # to represent background.
        # -----------------------------------------------------------

        if touches_border:

            border_ok = (
                area_ratio
                <= BORDER_AREA_RATIO_LIMIT
            )

        else:

            border_ok = True

        keep = (
            area_ok
            and solidity_ok
            and extent_ok
            and aspect_ok
            and circularity_ok
            and border_ok
        )

        component_records.append(
            {
                "label": label,
                "area": component_area,
                "area_ratio": area_ratio,
                "circularity": features["circularity"],
                "solidity": features["solidity"],
                "extent": features["extent"],
                "aspect_ratio": features["aspect_ratio"],
                "touches_border": touches_border,
                "kept": keep,
            }
        )

        if keep:

            final_mask[
                labels == label
            ] = 255

            kept_count += 1

    # ---------------------------------------------------------------
    # STEP 4
    # Final morphological cleanup
    # ---------------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            FINAL_CLOSE_KERNEL,
            FINAL_CLOSE_KERNEL
        )
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            FINAL_OPEN_KERNEL,
            FINAL_OPEN_KERNEL
        )
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    return (
        final_mask,
        candidate_count,
        kept_count,
        component_records
    )


# =====================================================================
# METRICS
# =====================================================================

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = prediction > 127
    ground_truth = ground_truth > 127

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    prediction_pixels = prediction.sum()
    gt_pixels = ground_truth.sum()

    # ---------------------------------------------------------------
    # IoU
    # ---------------------------------------------------------------

    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0 if gt_pixels == 0 else 0.0

    # ---------------------------------------------------------------
    # Dice
    # ---------------------------------------------------------------

    denominator = (
        prediction_pixels
        + gt_pixels
    )

    if denominator > 0:

        dice = (
            2.0
            * intersection
            / denominator
        )

    else:

        dice = 1.0 if gt_pixels == 0 else 0.0

    # ---------------------------------------------------------------
    # Precision
    # ---------------------------------------------------------------

    if prediction_pixels > 0:

        precision = (
            intersection
            / prediction_pixels
        )

    else:

        precision = 0.0

    # ---------------------------------------------------------------
    # Recall
    # ---------------------------------------------------------------

    if gt_pixels > 0:

        recall = (
            intersection
            / gt_pixels
        )

    else:

        recall = 1.0 if prediction_pixels == 0 else 0.0

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

    pred = prediction > 127
    gt = ground_truth > 127

    # ---------------------------------------------------------------
    # Correct overlap
    # Yellow
    # ---------------------------------------------------------------

    overlap = pred & gt

    # ---------------------------------------------------------------
    # False positive
    # Green
    # ---------------------------------------------------------------

    false_positive = pred & (~gt)

    # ---------------------------------------------------------------
    # False negative
    # Red
    # ---------------------------------------------------------------

    false_negative = gt & (~pred)

    overlay = vis.copy()

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
    print("TASK 2 - EXPERIMENT 6")
    print("E1 + CONNECTED COMPONENT / SHAPE FILTERING")
    print("=" * 70)

    print()
    print("Objective:")
    print(
        "Reduce E1 false positives while preserving "
        "its high recall."
    )

    print()
    print(f"Brightness percentile: {BRIGHT_PERCENTILE}")
    print(f"Maximum saturation: {MAX_SATURATION}")
    print(f"Minimum component area: {MIN_AREA}")
    print(f"Maximum area ratio: {MAX_AREA_RATIO}")
    print(f"Minimum solidity: {MIN_SOLIDITY}")
    print(f"Minimum extent: {MIN_EXTENT}")
    print(f"Minimum circularity: {MIN_CIRCULARITY}")

    # ---------------------------------------------------------------
    # Get development images
    # ---------------------------------------------------------------

    image_paths = get_development_images()

    if len(image_paths) == 0:

        raise RuntimeError(
            "No development images found."
        )

    # IMPORTANT:
    # Start with the same 10-image subset used in E1-E5.
    image_paths = image_paths[:MAX_IMAGES]

    print()
    print(
        f"Images selected for experiment: "
        f"{len(image_paths)}"
    )

    print()
    print("Evaluation set is NOT processed.")

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    results = []

    ious = []
    dices = []
    precisions = []
    recalls = []

    candidate_counts = []
    kept_counts = []

    # ---------------------------------------------------------------
    # Process images
    # ---------------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{filename}"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print("  ERROR: Could not read image.")
            continue

        # -----------------------------------------------------------
        # E6 segmentation
        # -----------------------------------------------------------

        (
            prediction,
            candidate_count,
            kept_count,
            component_records
        ) = segment_egg(image)

        # -----------------------------------------------------------
        # Load frozen Task 1 mask
        # -----------------------------------------------------------

        ground_truth = load_task1_union_mask(
            image_path
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
            f"  Candidates: {candidate_count}"
        )

        print(
            f"  Kept components: {kept_count}"
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
            f"{stem}_e6.png"
        )

        cv2.imwrite(
            mask_path,
            prediction
        )

        # -----------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------

        visualization = create_visualization(
            image,
            prediction,
            ground_truth
        )

        vis_path = os.path.join(
            VIS_DIR,
            f"{stem}_e6_comparison.png"
        )

        cv2.imwrite(
            vis_path,
            visualization
        )

        # -----------------------------------------------------------
        # Save results
        # -----------------------------------------------------------

        results.append(
            {
                "image": filename,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "candidate_components": candidate_count,
                "kept_components": kept_count,
            }
        )

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

        candidate_counts.append(
            candidate_count
        )

        kept_counts.append(
            kept_count
        )

    # =================================================================
    # SUMMARY
    # =================================================================

    if len(results) == 0:

        raise RuntimeError(
            "No images were successfully evaluated."
        )

    results_df = pd.DataFrame(
        results
    )

    report_path = os.path.join(
        REPORT_DIR,
        "experiment_06_report.csv"
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 06 SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: {len(results)}"
    )

    print(
        f"Mean IoU: {np.mean(ious):.4f}"
    )

    print(
        f"Median IoU: {np.median(ious):.4f}"
    )

    print(
        f"Mean Dice: {np.mean(dices):.4f}"
    )

    print(
        f"Mean Precision: {np.mean(precisions):.4f}"
    )

    print(
        f"Mean Recall: {np.mean(recalls):.4f}"
    )

    print(
        f"Mean candidates/image: "
        f"{np.mean(candidate_counts):.2f}"
    )

    print(
        f"Mean kept components/image: "
        f"{np.mean(kept_counts):.2f}"
    )

    print()
    print(f"Report:")
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