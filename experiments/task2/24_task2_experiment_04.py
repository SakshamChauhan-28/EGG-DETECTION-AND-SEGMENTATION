from pathlib import Path
import os
import cv2
import json
import glob
import numpy as np
import pandas as pd


# ============================================================
# TASK 2 - EXPERIMENT 4
# E1 FOREGROUND SEGMENTATION + EGG SHAPE FILTERING
# ============================================================

print("=" * 70)
print("TASK 2 - EXPERIMENT 4")
print("COLOR SEGMENTATION + GEOMETRIC EGG FILTERING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))

SPLIT_FILE = os.path.join(
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
    "task2_experiment_04"
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


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

# --- E1 foreground segmentation ---
BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100

# --- Morphological cleanup ---
MORPH_KERNEL_SIZE = 7

# --- Geometric filtering ---
MIN_AREA = 1500
MAX_AREA = 180000

MIN_ASPECT_RATIO = 0.40
MAX_ASPECT_RATIO = 2.50

MIN_EXTENT = 0.40
MAX_EXTENT = 0.90

MIN_SOLIDITY = 0.82
MIN_CIRCULARITY = 0.35

# Maximum number of eggs allowed in one image.
# Task 1 development results showed up to 4 detected eggs.
MAX_EGGS = 6


# ============================================================
# LOAD LEAKAGE-AWARE SPLIT
# ============================================================

def get_development_images():

    df = pd.read_csv(SPLIT_FILE)

    split_column = None

    for column in [
        "final_split",
        "split",
        "dataset_split",
        "assigned_split"
    ]:
        if column in df.columns:
            split_column = column
            break

    if split_column is None:
        raise ValueError(
            "Could not find split column in final_split_assignments.csv"
        )

    image_column = None

    for column in [
        "image_path",
        "filepath",
        "file_path",
        "path",
        "image",
        "filename",
        "file_name"
    ]:
        if column in df.columns:
            image_column = column
            break

    if image_column is None:
        raise ValueError(
            "Could not find image path column in final_split_assignments.csv"
        )

    development = df[
        df[split_column].astype(str).str.lower() == "development"
    ]

    evaluation = df[
        df[split_column].astype(str).str.lower() == "evaluation"
    ]

    print(f"Development images: {len(development)}")
    print(f"Evaluation images: {len(evaluation)}")

    if len(development) != 88:
        raise ValueError(
            f"Expected 88 development images, found {len(development)}"
        )

    if len(evaluation) != 22:
        raise ValueError(
            f"Expected 22 evaluation images, found {len(evaluation)}"
        )

    image_paths = []

    for value in development[image_column]:

        path = str(value)

        if os.path.exists(path):
            image_paths.append(path)
            continue

        candidate = os.path.join(PROJECT_ROOT, path)

        if os.path.exists(candidate):
            image_paths.append(candidate)
            continue

        matches = glob.glob(
            os.path.join(PROJECT_ROOT, "**", os.path.basename(path)),
            recursive=True
        )

        if matches:
            image_paths.append(matches[0])
        else:
            print(f"WARNING: image not found: {path}")

    return image_paths


# ============================================================
# TASK 1 UNION MASK
# ============================================================

def load_task1_union_mask(image_path, shape):

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    mask_paths = glob.glob(
        os.path.join(
            TASK1_MASK_DIR,
            f"{stem}_egg_*.png"
        )
    )

    union_mask = np.zeros(
        shape[:2],
        dtype=np.uint8
    )

    for mask_path in mask_paths:

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        if mask.shape != shape[:2]:
            mask = cv2.resize(
                mask,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        union_mask[mask > 127] = 255

    return union_mask


# ============================================================
# E1 FOREGROUND GENERATION
# ============================================================

def generate_foreground_mask(image):

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

    # Adaptive brightness threshold based on image percentile
    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= brightness_threshold
    ).astype(np.uint8) * 255

    # Low saturation regions
    low_saturation_mask = (
        saturation <= MAX_SATURATION
    ).astype(np.uint8) * 255

    # Combine
    foreground = cv2.bitwise_and(
        bright_mask,
        low_saturation_mask
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel
    )

    return foreground


# ============================================================
# GEOMETRIC FEATURES
# ============================================================

def calculate_component_features(
    contour,
    component_mask,
    image_area
):

    area = cv2.contourArea(contour)

    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = (
        max(w, h) / max(min(w, h), 1)
    )

    bounding_box_area = w * h

    extent = (
        area / max(bounding_box_area, 1)
    )

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = (
        area / max(hull_area, 1)
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4.0 * np.pi * area /
            (perimeter * perimeter)
        )

    else:

        circularity = 0.0

    area_ratio = area / image_area

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "area_ratio": area_ratio,
        "bbox": (x, y, w, h)
    }


# ============================================================
# EGG-LIKE COMPONENT FILTER
# ============================================================

def is_egg_like(features):

    if features is None:
        return False

    area = features["area"]
    aspect_ratio = features["aspect_ratio"]
    extent = features["extent"]
    solidity = features["solidity"]
    circularity = features["circularity"]

    # Area constraint
    if area < MIN_AREA:
        return False

    if area > MAX_AREA:
        return False

    # Egg should not be extremely elongated
    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    # Egg should occupy a reasonable portion
    # of its bounding rectangle
    if extent < MIN_EXTENT:
        return False

    if extent > MAX_EXTENT:
        return False

    # Eggs are approximately convex
    if solidity < MIN_SOLIDITY:
        return False

    # Reject very irregular thin structures
    if circularity < MIN_CIRCULARITY:
        return False

    return True


# ============================================================
# E4 SEGMENTATION
# ============================================================

def segment_eggs(image):

    foreground = generate_foreground_mask(image)

    height, width = foreground.shape
    image_area = height * width

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8
    )

    candidates = []

    for label in range(1, num_labels):

        component_mask = (
            labels == label
        ).astype(np.uint8) * 255

        component_area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if component_area < MIN_AREA:
            continue

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        features = calculate_component_features(
            contour,
            component_mask,
            image_area
        )

        if features is None:
            continue

        # Keep only egg-like components
        if is_egg_like(features):

            candidates.append({
                "label": label,
                "mask": component_mask,
                "features": features
            })

    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    # Larger, solid, reasonably circular objects receive
    # higher scores.
    for candidate in candidates:

        f = candidate["features"]

        area_score = min(
            f["area"] / 25000.0,
            1.0
        )

        solidity_score = min(
            max(f["solidity"], 0.0),
            1.0
        )

        circularity_score = min(
            f["circularity"] / 0.785,
            1.0
        )

        extent_score = min(
            f["extent"] / 0.75,
            1.0
        )

        candidate["score"] = (
            0.25 * area_score +
            0.30 * solidity_score +
            0.25 * circularity_score +
            0.20 * extent_score
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Limit maximum candidates
    candidates = candidates[:MAX_EGGS]

    # ========================================================
    # FINAL MASK
    # ========================================================

    final_mask = np.zeros_like(
        foreground
    )

    for candidate in candidates:

        final_mask = cv2.bitwise_or(
            final_mask,
            candidate["mask"]
        )

    return final_mask, candidates, foreground


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, ground_truth):

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

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    dice = (
        2.0 * intersection /
        (prediction_pixels + gt_pixels)
        if (prediction_pixels + gt_pixels) > 0
        else 1.0
    )

    precision = (
        intersection / prediction_pixels
        if prediction_pixels > 0
        else 0.0
    )

    recall = (
        intersection / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    return iou, dice, precision, recall


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image,
    prediction,
    ground_truth
):

    output = image.copy()

    pred = prediction > 127
    gt = ground_truth > 127

    true_positive = pred & gt
    false_positive = pred & (~gt)
    false_negative = (~pred) & gt

    # Yellow = overlap
    output[true_positive] = (
        0.5 * output[true_positive] +
        0.5 * np.array(
            [0, 255, 255]
        )
    ).astype(np.uint8)

    # Green = false positive
    output[false_positive] = (
        0.5 * output[false_positive] +
        0.5 * np.array(
            [0, 255, 0]
        )
    ).astype(np.uint8)

    # Red = false negative
    output[false_negative] = (
        0.5 * output[false_negative] +
        0.5 * np.array(
            [0, 0, 255]
        )
    ).astype(np.uint8)

    return output


# ============================================================
# DEVELOPMENT IMAGES
# ============================================================

image_paths = get_development_images()

# IMPORTANT:
# Use exactly the same first 10 development images
# used in E1/E2/E3.
image_paths = image_paths[:10]

print()
print("Controlled experiment subset: 10 images")
print("Evaluation set is NOT processed.")
print()


# ============================================================
# PROCESS
# ============================================================

results = []

for index, image_path in enumerate(
    image_paths,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print(
        f"[{index}/{len(image_paths)}] {filename}"
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print("  ERROR: Could not read image")
        continue

    # E4 segmentation
    prediction, candidates, foreground = segment_eggs(
        image
    )

    # Frozen Task 1 mask
    ground_truth = load_task1_union_mask(
        image_path,
        image.shape
    )

    # Metrics
    iou, dice, precision, recall = calculate_metrics(
        prediction,
        ground_truth
    )

    # Save mask
    stem = os.path.splitext(filename)[0]

    mask_path = os.path.join(
        MASK_DIR,
        f"{stem}.png"
    )

    cv2.imwrite(
        mask_path,
        prediction
    )

    # Visualization
    visualization = create_visualization(
        image,
        prediction,
        ground_truth
    )

    visualization_path = os.path.join(
        VIS_DIR,
        f"{stem}.jpg"
    )

    cv2.imwrite(
        visualization_path,
        visualization
    )

    # Candidate statistics
    candidate_areas = [
        c["features"]["area"]
        for c in candidates
    ]

    candidate_scores = [
        c["score"]
        for c in candidates
    ]

    result = {
        "image": filename,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "candidate_count": len(candidates),
        "candidate_areas": json.dumps(
            candidate_areas
        ),
        "candidate_scores": json.dumps(
            candidate_scores
        )
    }

    results.append(result)

    print(
        f"  Candidates: {len(candidates)}"
    )

    print(
        f"  IoU: {iou:.3f}"
    )

    print(
        f"  Dice: {dice:.3f}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_df = pd.DataFrame(
    results
)

report_path = os.path.join(
    REPORT_DIR,
    "experiment_04_report.csv"
)

report_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

if len(report_df) > 0:

    mean_iou = report_df["iou"].mean()
    median_iou = report_df["iou"].median()

    mean_dice = report_df["dice"].mean()

    mean_precision = (
        report_df["precision"].mean()
    )

    mean_recall = (
        report_df["recall"].mean()
    )

    mean_candidates = (
        report_df["candidate_count"].mean()
    )

else:

    mean_iou = 0
    median_iou = 0
    mean_dice = 0
    mean_precision = 0
    mean_recall = 0
    mean_candidates = 0


print()
print("=" * 70)
print("EXPERIMENT 04 SUMMARY")
print("=" * 70)

print(
    f"Images evaluated: {len(report_df)}"
)

print(
    f"Mean IoU: {mean_iou:.4f}"
)

print(
    f"Median IoU: {median_iou:.4f}"
)

print(
    f"Mean Dice: {mean_dice:.4f}"
)

print(
    f"Mean Precision: {mean_precision:.4f}"
)

print(
    f"Mean Recall: {mean_recall:.4f}"
)

print(
    f"Mean candidates/image: {mean_candidates:.2f}"
)

print()
print(
    f"Report:\n{report_path}"
)

print()
print(
    f"Visualizations:\n{VIS_DIR}"
)

print()
print(
    f"Masks:\n{MASK_DIR}"
)