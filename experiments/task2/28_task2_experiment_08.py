import os
"""
TASK 2 - EXPERIMENT 08
E1 BASELINE + GEOMETRIC EGG FILTERING

Purpose
-------
E8 keeps the successful E1 color-based segmentation and adds
geometric filtering to remove irregular/background components.

Pipeline:
    Image
      -> HSV/LAB color segmentation (E1)
      -> Morphological cleanup
      -> Connected components
      -> Geometric filtering
      -> Final egg mask

IMPORTANT
---------
- Task 1 masks are used ONLY for evaluation.
- Task 1 masks are NEVER used by segment_egg().
- Only the 88-image development set is processed.
- The 22-image evaluation set remains untouched.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent.parent / "data" / "egg.v1i.folder"))
)
SPLIT_FILE = PROJECT_ROOT / "dataset_report" / "final_split_assignments.csv"

TASK1_MASK_DIR = PROJECT_ROOT / "task1_gdino_sam" / "masks"

OUTPUT_ROOT = PROJECT_ROOT / "task2_experiment_08"
MASK_DIR = OUTPUT_ROOT / "masks"
VIS_DIR = OUTPUT_ROOT / "visualizations"
REPORT_DIR = OUTPUT_ROOT / "reports"


# ------------------------------------------------------------
# E1 PARAMETERS
# ------------------------------------------------------------

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100

MORPH_KERNEL_SIZE = 7
MIN_AREA = 500
MAX_COMPONENTS = 6


# ------------------------------------------------------------
# E8 GEOMETRIC PARAMETERS
# ------------------------------------------------------------

# Minimum/maximum component area
E8_MIN_AREA = 500
E8_MAX_AREA = 120000

# Width / height ratio.
# Eggs can be fairly elongated or almost round.
E8_MIN_ASPECT = 0.40
E8_MAX_ASPECT = 2.50

# Filled area / bounding-box area.
# Very thin structures should be rejected.
E8_MIN_EXTENT = 0.40

# Contour area / convex-hull area.
# Low solidity usually indicates irregular background regions.
E8_MIN_SOLIDITY = 0.78

# Circularity = 4*pi*A/P^2
# Eggs are not perfect circles, therefore this threshold is
# intentionally kept relatively low.
E8_MIN_CIRCULARITY = 0.30

# Maximum number of components retained after geometric filtering.
E8_MAX_COMPONENTS = 6

# ============================================================
# IMAGE / SPLIT HELPERS
# ============================================================


def find_column(df, candidates):
    """Find the first matching column name."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_image_path(value):
    """
    Resolve an image path from the split CSV.

    Supports:
    - absolute Windows paths
    - relative paths
    - filename-only entries
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Normalize separators
    value = value.replace("/", "\\")

    p = Path(value)

    # Absolute path
    if p.is_absolute() and p.exists():
        return p

    # Relative to project root
    candidate = PROJECT_ROOT / value
    if candidate.exists():
        return candidate

    # Relative to dataset root
    candidate = DATASET_ROOT / value
    if candidate.exists():
        return candidate

    # Search by filename
    filename = Path(value).name

    matches = list(DATASET_ROOT.rglob(filename))

    if len(matches) > 0:
        return matches[0]

    return None


def get_development_images():
    """
    Read the leakage-aware split and return exactly
    the 88 development images.

    Evaluation images are explicitly excluded.
    """

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    df = pd.read_csv(SPLIT_FILE)

    split_col = find_column(
        df,
        [
            "final_split",
            "split",
            "dataset_split",
            "assigned_split",
        ],
    )

    image_col = find_column(
        df,
        [
            "image_path",
            "filepath",
            "file_path",
            "path",
            "image",
            "filename",
            "file_name",
        ],
    )

    if split_col is None:
        raise ValueError(
            f"Could not find split column.\n"
            f"Available columns: {list(df.columns)}"
        )

    if image_col is None:
        raise ValueError(
            f"Could not find image-path column.\n"
            f"Available columns: {list(df.columns)}"
        )

    dev_df = df[
        df[split_col]
        .astype(str)
        .str.lower()
        .isin(["development", "dev"])
    ].copy()

    eval_df = df[
        df[split_col]
        .astype(str)
        .str.lower()
        .isin(["evaluation", "eval", "test"])
    ].copy()

    print(f"Development entries in split file: {len(dev_df)}")
    print(f"Evaluation entries in split file: {len(eval_df)}")

    image_paths = []

    for _, row in dev_df.iterrows():

        path = resolve_image_path(row[image_col])

        if path is not None:
            image_paths.append(path)

    # Remove duplicates while preserving order
    unique_paths = []
    seen = set()

    for path in image_paths:

        key = str(path.resolve()).lower()

        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    image_paths = unique_paths

    if len(image_paths) != 88:
        raise RuntimeError(
            f"Expected exactly 88 development images, "
            f"but resolved {len(image_paths)}."
        )

    if len(eval_df) != 22:
        raise RuntimeError(
            f"Expected exactly 22 evaluation images, "
            f"but split file contains {len(eval_df)}."
        )

    print(f"Resolved development images: {len(image_paths)}")
    print("Evaluation images will NOT be processed.")

    return image_paths


# ============================================================
# TASK 1 MASK LOADING
# ============================================================


def load_task1_union_mask(image_path):
    """
    Load all Task 1 masks corresponding to an image and
    combine them into a single binary pseudo-ground-truth mask.
    """

    stem = image_path.stem

    mask_files = sorted(
        TASK1_MASK_DIR.glob(f"{stem}_egg_*.png")
    )

    if len(mask_files) == 0:
        raise FileNotFoundError(
            f"No Task 1 masks found for:\n{image_path.name}"
        )

    union = None

    for mask_file in mask_files:

        mask = cv2.imread(
            str(mask_file),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            continue

        mask = (mask > 0).astype(np.uint8)

        if union is None:
            union = mask
        else:
            union = np.maximum(union, mask)

    if union is None:
        raise RuntimeError(
            f"Could not load Task 1 masks for {image_path.name}"
        )

    return union


# ============================================================
# E1 BASE SEGMENTATION
# ============================================================


def e1_color_segmentation(image):
    """
    Original E1 color-based segmentation.

    Uses:
        - HSV saturation constraint
        - LAB brightness constraint
        - morphology

    Task 1 masks are NOT used here.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]

    # Adaptive brightness threshold
    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE,
    )

    brightness_mask = (
        lightness >= brightness_threshold
    ).astype(np.uint8) * 255

    # Low/moderate saturation region
    saturation_mask = (
        saturation <= MAX_SATURATION
    ).astype(np.uint8) * 255

    # Combine E1 conditions
    mask = cv2.bitwise_and(
        brightness_mask,
        saturation_mask,
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


# ============================================================
# E8 GEOMETRIC FEATURES
# ============================================================


def calculate_component_features(contour):

    area = cv2.contourArea(contour)

    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return None

    bbox_area = float(w * h)

    # Extent:
    # contour area relative to bounding-box area
    extent = area / bbox_area

    # Convex hull / solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0.0

    # Aspect ratio
    aspect_ratio = w / float(h)

    # Circularity
    perimeter = cv2.arcLength(
        contour,
        True,
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

    # Equivalent diameter
    equivalent_diameter = np.sqrt(
        4.0 * area / np.pi
    )

    # Ellipse information if enough contour points
    ellipse_ratio = None

    if len(contour) >= 5:

        try:
            (_, _), (major_axis, minor_axis), _ = (
                cv2.fitEllipse(contour)
            )

            major = max(
                major_axis,
                minor_axis,
            )

            minor = min(
                major_axis,
                minor_axis,
            )

            if major > 0:
                ellipse_ratio = (
                    minor / major
                )

        except cv2.error:
            ellipse_ratio = None

    return {
        "area": area,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "equivalent_diameter": equivalent_diameter,
        "ellipse_ratio": ellipse_ratio,
    }


# ============================================================
# E8 GEOMETRIC FILTER
# ============================================================


def passes_geometric_filter(features):
    """
    Decide whether a connected component has reasonable
    egg-like geometry.

    The thresholds are deliberately broad because the dataset
    contains eggs at different orientations.
    """

    if features is None:
        return False

    area = features["area"]
    aspect = features["aspect_ratio"]
    extent = features["extent"]
    solidity = features["solidity"]
    circularity = features["circularity"]

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    if area < E8_MIN_AREA:
        return False

    if area > E8_MAX_AREA:
        return False

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    if aspect < E8_MIN_ASPECT:
        return False

    if aspect > E8_MAX_ASPECT:
        return False

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    if extent < E8_MIN_EXTENT:
        return False

    # --------------------------------------------------------
    # Solidity
    # --------------------------------------------------------

    if solidity < E8_MIN_SOLIDITY:
        return False

    # --------------------------------------------------------
    # Circularity
    # --------------------------------------------------------

    if circularity < E8_MIN_CIRCULARITY:
        return False

    return True


# ============================================================
# E8 FINAL SEGMENTATION
# ============================================================


def segment_egg(image):

    # --------------------------------------------------------
    # STEP 1: E1 segmentation
    # --------------------------------------------------------

    base_mask = e1_color_segmentation(image)

    # --------------------------------------------------------
    # STEP 2: Connected components
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        base_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        features = calculate_component_features(
            contour
        )

        if features is None:
            continue

        if features["area"] < MIN_AREA:
            continue

        candidates.append(
            (
                contour,
                features,
            )
        )

    # Sort by area
    candidates.sort(
        key=lambda item: item[1]["area"],
        reverse=True,
    )

    # Keep the largest E1 components before E8 filtering
    candidates = candidates[
        :MAX_COMPONENTS
    ]

    # --------------------------------------------------------
    # STEP 3: Geometric filtering
    # --------------------------------------------------------

    kept = []

    for contour, features in candidates:

        if passes_geometric_filter(features):
            kept.append(
                (
                    contour,
                    features,
                )
            )

    # --------------------------------------------------------
    # STEP 4: Limit final components
    # --------------------------------------------------------

    kept.sort(
        key=lambda item: item[1]["area"],
        reverse=True,
    )

    kept = kept[
        :E8_MAX_COMPONENTS
    ]

    # --------------------------------------------------------
    # STEP 5: Construct final mask
    # --------------------------------------------------------

    final_mask = np.zeros(
        base_mask.shape,
        dtype=np.uint8,
    )

    for contour, _ in kept:

        cv2.drawContours(
            final_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

    return final_mask, len(candidates), len(kept)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(pred_mask, gt_mask):

    pred = pred_mask > 0
    gt = gt_mask > 0

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    union = np.logical_or(
        pred,
        gt,
    ).sum()

    pred_area = pred.sum()
    gt_area = gt.sum()

    # IoU
    if union == 0:
        iou = 1.0 if pred_area == 0 else 0.0
    else:
        iou = intersection / union

    # Dice
    denominator = pred_area + gt_area

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0
            * intersection
            / denominator
        )

    # Precision
    if pred_area == 0:
        precision = 0.0
    else:
        precision = (
            intersection
            / pred_area
        )

    # Recall
    if gt_area == 0:
        recall = 1.0 if pred_area == 0 else 0.0
    else:
        recall = (
            intersection
            / gt_area
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


def create_comparison_visualization(
    image,
    pred_mask,
    gt_mask,
):

    visualization = image.copy()

    pred = pred_mask > 0
    gt = gt_mask > 0

    overlap = pred & gt
    false_positive = pred & (~gt)
    false_negative = gt & (~pred)

    # --------------------------------------------------------
    # Colors in BGR
    # --------------------------------------------------------

    # Overlap -> yellow
    visualization[overlap] = (
        0.55 * visualization[overlap]
        + 0.45 * np.array(
            [0, 255, 255],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    # False positive -> green
    visualization[false_positive] = (
        0.45 * visualization[false_positive]
        + 0.55 * np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    # False negative -> red
    visualization[false_negative] = (
        0.45 * visualization[false_negative]
        + 0.55 * np.array(
            [0, 0, 255],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    return visualization


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("TASK 2 - EXPERIMENT 08")
    print("E1 + GEOMETRIC EGG FILTERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load exact development split
    # --------------------------------------------------------

    image_paths = get_development_images()

    print()
    print(
        f"Images to evaluate: {len(image_paths)}"
    )
    print(
        "Evaluation set: 22 images "
        "(NOT PROCESSED)"
    )

    # --------------------------------------------------------
    # Print parameters
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("E1 PARAMETERS")
    print("-" * 70)

    print(
        f"Brightness percentile: {BRIGHT_PERCENTILE}"
    )

    print(
        f"Maximum saturation: {MAX_SATURATION}"
    )

    print(
        f"Morphology kernel: {MORPH_KERNEL_SIZE}"
    )

    print(
        f"Minimum area: {MIN_AREA}"
    )

    print()
    print("-" * 70)
    print("E8 GEOMETRIC PARAMETERS")
    print("-" * 70)

    print(
        f"Area range: "
        f"{E8_MIN_AREA} - {E8_MAX_AREA}"
    )

    print(
        f"Aspect ratio: "
        f"{E8_MIN_ASPECT} - {E8_MAX_ASPECT}"
    )

    print(
        f"Minimum extent: {E8_MIN_EXTENT}"
    )

    print(
        f"Minimum solidity: {E8_MIN_SOLIDITY}"
    )

    print(
        f"Minimum circularity: "
        f"{E8_MIN_CIRCULARITY}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    total_candidates = 0
    total_kept = 0

    # --------------------------------------------------------
    # Process development images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print("  ERROR: Could not read image.")
            continue

        # E8 prediction
        pred_mask, candidate_count, kept_count = (
            segment_egg(image)
        )

        # Frozen Task 1 pseudo-ground-truth
        gt_mask = load_task1_union_mask(
            image_path
        )

        # Metrics
        metrics = calculate_metrics(
            pred_mask,
            gt_mask,
        )

        total_candidates += candidate_count
        total_kept += kept_count

        print(
            f"  Candidates: {candidate_count}"
        )

        print(
            f"  Kept components: {kept_count}"
        )

        print(
            f"  IoU: {metrics['iou']:.3f}"
        )

        print(
            f"  Dice: {metrics['dice']:.3f}"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.3f}"
        )

        print(
            f"  Recall: "
            f"{metrics['recall']:.3f}"
        )

        # ----------------------------------------------------
        # Save predicted mask
        # ----------------------------------------------------

        mask_path = (
            MASK_DIR
            / f"{image_path.stem}_e8.png"
        )

        cv2.imwrite(
            str(mask_path),
            pred_mask,
        )

        # ----------------------------------------------------
        # Save comparison visualization
        # ----------------------------------------------------

        comparison = (
            create_comparison_visualization(
                image,
                pred_mask,
                gt_mask,
            )
        )

        vis_path = (
            VIS_DIR
            / f"{image_path.stem}_e8_comparison.png"
        )

        cv2.imwrite(
            str(vis_path),
            comparison,
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        results.append(
            {
                "image": image_path.name,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "candidate_components": candidate_count,
                "kept_components": kept_count,
            }
        )

    # ========================================================
    # REPORT
    # ========================================================

    if len(results) == 0:
        raise RuntimeError(
            "No images were successfully processed."
        )

    results_df = pd.DataFrame(results)

    report_path = (
        REPORT_DIR
        / "experiment_08_report.csv"
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    mean_iou = results_df["iou"].mean()
    median_iou = results_df["iou"].median()

    mean_dice = results_df["dice"].mean()

    mean_precision = (
        results_df["precision"].mean()
    )

    mean_recall = (
        results_df["recall"].mean()
    )

    mean_candidates = (
        results_df[
            "candidate_components"
        ].mean()
    )

    mean_kept = (
        results_df[
            "kept_components"
        ].mean()
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 08 SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: {len(results_df)}"
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
        f"Mean candidates/image: "
        f"{mean_candidates:.2f}"
    )

    print(
        f"Mean kept components/image: "
        f"{mean_kept:.2f}"
    )

    print()
    print(
        f"Report:\n{report_path}"
    )

    print(
        f"Visualizations:\n{VIS_DIR}"
    )

    print(
        f"Masks:\n{MASK_DIR}"
    )

    print()
    print("=" * 70)
    print("EVALUATION SET WAS NOT PROCESSED")
    print("=" * 70)


if __name__ == "__main__":
    main()