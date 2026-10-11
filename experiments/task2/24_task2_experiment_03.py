from pathlib import Path
"""
Task 2 - Experiment 03
Classical Image Processing Segmentation

Experiment:
    Canny Edge Detection
        +
    Morphological Closing
        +
    Contour Extraction
        +
    Geometric Filtering:
        - Area
        - Aspect Ratio
        - Solidity
        - Extent
        +
    Contour Filling

Purpose:
    Determine whether geometric properties of egg-shaped regions
    can reduce the background false positives observed in E1 and E2.

IMPORTANT:
    - Only the 88-image development split is used.
    - First run evaluates 10 development images.
    - The 22-image evaluation split remains untouched.
    - Task 1 masks are used ONLY for evaluation.
    - Task 1 masks are NEVER used by the segmentation algorithm.
"""


import os
import glob
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
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
    "task2_experiment_03"
)

MASK_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "masks"
)

VIS_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "visualizations"
)

REPORT_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "reports"
)


# ============================================================
# EXPERIMENT 03 PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Canny edge detection
# ------------------------------------------------------------

CANNY_LOW = 50
CANNY_HIGH = 150


# ------------------------------------------------------------
# Morphological processing
# ------------------------------------------------------------

MORPH_KERNEL_SIZE = 7


# ------------------------------------------------------------
# Contour filtering
# ------------------------------------------------------------

# Minimum and maximum contour area
MIN_AREA = 1500
MAX_AREA = 150000

# Egg-like aspect ratio
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20

# Solidity:
# contour area / convex hull area
MIN_SOLIDITY = 0.80

# Extent:
# contour area / bounding rectangle area
MIN_EXTENT = 0.45

# Maximum number of candidates retained
MAX_COMPONENTS = 8


# ============================================================
# QUICK TEST CONTROL
# ============================================================

# Start with exactly the same 10 development images
# used for Experiment 02.

RUN_QUICK_TEST = True

QUICK_TEST_IMAGES = 10


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MASK_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    VIS_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD LEAKAGE-AWARE SPLIT
# ============================================================

def load_split_assignments():

    if not os.path.exists(SPLIT_FILE):

        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    df = pd.read_csv(
        SPLIT_FILE
    )

    return df


# ============================================================
# FIND IMAGE COLUMN
# ============================================================

def find_image_column(df):

    possible_columns = [
        "image_path",
        "filepath",
        "file_path",
        "path",
        "image",
        "filename",
        "file_name"
    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not find image path column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# FIND SPLIT COLUMN
# ============================================================

def find_split_column(df):

    possible_columns = [
        "final_split",
        "split",
        "dataset_split",
        "assigned_split"
    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not find split column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# GET DEVELOPMENT IMAGES
# ============================================================

def get_development_images():

    df = load_split_assignments()

    image_column = find_image_column(
        df
    )

    split_column = find_split_column(
        df
    )

    df[split_column] = (
        df[split_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    development_df = df[
        df[split_column].isin(
            ["development", "dev"]
        )
    ].copy()

    evaluation_df = df[
        df[split_column].isin(
            ["evaluation", "eval"]
        )
    ].copy()

    print("\n" + "=" * 70)
    print("LEAKAGE-AWARE DATASET CHECK")
    print("=" * 70)

    print(
        f"Development images: {len(development_df)}"
    )

    print(
        f"Evaluation images : {len(evaluation_df)}"
    )

    # --------------------------------------------------------
    # STRICT SPLIT CHECK
    # --------------------------------------------------------

    if len(development_df) != 88:

        raise RuntimeError(
            "Expected exactly 88 development images, "
            f"found {len(development_df)}."
        )

    if len(evaluation_df) != 22:

        raise RuntimeError(
            "Expected exactly 22 evaluation images, "
            f"found {len(evaluation_df)}."
        )

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    image_paths = []

    for _, row in development_df.iterrows():

        raw_path = str(
            row[image_column]
        )

        # Direct path
        if os.path.exists(raw_path):

            image_paths.append(
                os.path.abspath(raw_path)
            )

            continue

        # Project-relative path
        project_path = os.path.join(
            PROJECT_ROOT,
            raw_path
        )

        if os.path.exists(project_path):

            image_paths.append(
                os.path.abspath(project_path)
            )

            continue

        # Search by filename
        filename = os.path.basename(
            raw_path
        )

        matches = []

        for root, dirs, files in os.walk(
            PROJECT_ROOT
        ):

            if filename in files:

                matches.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

        if len(matches) == 1:

            image_paths.append(
                os.path.abspath(matches[0])
            )

        elif len(matches) > 1:

            # Prefer original dataset
            dataset_matches = [
                path
                for path in matches
                if "egg.v1i.folder" in path.lower()
            ]

            if len(dataset_matches) == 1:

                image_paths.append(
                    os.path.abspath(
                        dataset_matches[0]
                    )
                )

            else:

                image_paths.append(
                    os.path.abspath(
                        matches[0]
                    )
                )

        else:

            print(
                f"WARNING: Image not found:\n"
                f"{raw_path}"
            )

    # Remove duplicate paths
    image_paths = list(
        dict.fromkeys(
            image_paths
        )
    )

    if len(image_paths) != 88:

        raise RuntimeError(
            "Expected 88 resolved development images, "
            f"found {len(image_paths)}."
        )

    return image_paths


# ============================================================
# LOAD TASK 1 UNION MASK
# ============================================================

def load_task1_union_mask(
    image_path,
    image_shape
):

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

    union_mask = np.zeros(
        image_shape[:2],
        dtype=np.uint8
    )

    if not mask_paths:

        print(
            "WARNING: No Task 1 masks found for "
            f"{stem}"
        )

        return union_mask

    for mask_path in mask_paths:

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        if mask.shape != image_shape[:2]:

            mask = cv2.resize(
                mask,
                (
                    image_shape[1],
                    image_shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

        union_mask[
            mask > 127
        ] = 255

    return union_mask


# ============================================================
# CONTOUR FEATURES
# ============================================================

def calculate_contour_features(
    contour
):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    if h == 0:

        aspect_ratio = 0.0

    else:

        aspect_ratio = (
            float(w) /
            float(h)
        )

    # --------------------------------------------------------
    # Convex hull
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area > 0:

        solidity = (
            float(area) /
            float(hull_area)
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    bbox_area = w * h

    if bbox_area > 0:

        extent = (
            float(area) /
            float(bbox_area)
        )

    else:

        extent = 0.0

    # --------------------------------------------------------
    # Circularity
    #
    # 4*pi*area / perimeter^2
    #
    # Not used as a hard filter in E3,
    # but recorded for analysis.
    # --------------------------------------------------------

    if perimeter > 0:

        circularity = (
            4.0 *
            np.pi *
            area /
            (perimeter * perimeter)
        )

    else:

        circularity = 0.0

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    moments = cv2.moments(
        contour
    )

    if moments["m00"] != 0:

        cx = (
            moments["m10"] /
            moments["m00"]
        )

        cy = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        cx = 0.0
        cy = 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "circularity": circularity,
        "cx": cx,
        "cy": cy
    }


# ============================================================
# EXPERIMENT 03 SEGMENTATION
# ============================================================

def segment_egg(image):

    """
    Classical contour-based segmentation.

    Pipeline:

        Image
          ↓
        Grayscale
          ↓
        Gaussian Blur
          ↓
        Canny
          ↓
        Morphological Closing
          ↓
        Contour Extraction
          ↓
        Area filtering
          ↓
        Aspect ratio filtering
          ↓
        Solidity filtering
          ↓
        Extent filtering
          ↓
        Fill accepted contours
          ↓
        Final mask

    Task 1 masks are NOT used here.
    """

    # --------------------------------------------------------
    # 1. Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 2. Slight smoothing
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # 3. Canny edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH
    )

    # --------------------------------------------------------
    # 4. Morphological closing
    #
    # Connect nearby boundary fragments.
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # 5. Find contours
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    # --------------------------------------------------------
    # 6. Calculate features and filter
    # --------------------------------------------------------

    for contour in contours:

        features = calculate_contour_features(
            contour
        )

        area = features["area"]
        aspect_ratio = features["aspect_ratio"]
        solidity = features["solidity"]
        extent = features["extent"]

        # ----------------------------------------------------
        # Area filter
        # ----------------------------------------------------

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        # ----------------------------------------------------
        # Aspect ratio filter
        # ----------------------------------------------------

        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # ----------------------------------------------------
        # Solidity filter
        # ----------------------------------------------------

        if solidity < MIN_SOLIDITY:
            continue

        # ----------------------------------------------------
        # Extent filter
        # ----------------------------------------------------

        if extent < MIN_EXTENT:
            continue

        candidates.append(
            (
                contour,
                features
            )
        )

    # --------------------------------------------------------
    # 7. Sort candidates by area
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[1]["area"],
        reverse=True
    )

    # Keep only largest candidates
    candidates = candidates[
        :MAX_COMPONENTS
    ]

    # --------------------------------------------------------
    # 8. Create final mask
    # --------------------------------------------------------

    final_mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    for contour, features in candidates:

        cv2.drawContours(
            final_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return (
        final_mask,
        candidates,
        edges,
        closed_edges
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    pred_mask,
    gt_mask
):

    pred = pred_mask > 0
    gt = gt_mask > 0

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
            if pred_area == 0 and gt_area == 0
            else 0.0
        )

    else:

        iou = (
            intersection /
            union
        )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    denominator = (
        pred_area +
        gt_area
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 *
            intersection /
            denominator
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if pred_area == 0:

        precision = 0.0

    else:

        precision = (
            intersection /
            pred_area
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if gt_area == 0:

        recall = 0.0

    else:

        recall = (
            intersection /
            gt_area
        )

    return (
        iou,
        dice,
        precision,
        recall
    )


# ============================================================
# CREATE COMPARISON VISUALIZATION
# ============================================================

def create_visualization(
    image,
    pred_mask,
    gt_mask
):

    output = image.copy()

    pred = pred_mask > 0
    gt = gt_mask > 0

    # --------------------------------------------------------
    # True positive
    # --------------------------------------------------------

    true_positive = (
        pred &
        gt
    )

    # --------------------------------------------------------
    # False positive
    # --------------------------------------------------------

    false_positive = (
        pred &
        ~gt
    )

    # --------------------------------------------------------
    # False negative
    # --------------------------------------------------------

    false_negative = (
        ~pred &
        gt
    )

    overlay = output.copy()

    # BGR
    # Yellow = overlap
    overlay[
        true_positive
    ] = (
        0,
        255,
        255
    )

    # Green = false positive
    overlay[
        false_positive
    ] = (
        0,
        255,
        0
    )

    # Red = false negative
    overlay[
        false_negative
    ] = (
        0,
        0,
        255
    )

    visualization = cv2.addWeighted(
        output,
        0.55,
        overlay,
        0.45,
        0
    )

    return visualization


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    image_path
):

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        MASK_OUTPUT_DIR,
        f"{stem}_mask.png"
    )

    cv2.imwrite(
        output_path,
        mask
    )

    return output_path


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    visualization,
    image_path
):

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        VIS_OUTPUT_DIR,
        f"{stem}_comparison.jpg"
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    return output_path


# ============================================================
# SAVE EDGE VISUALIZATION
# ============================================================

def save_edge_visualization(
    edges,
    closed_edges,
    image_path
):

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # Side-by-side diagnostic image
    diagnostic = np.hstack(
        [
            edges,
            closed_edges
        ]
    )

    output_path = os.path.join(
        VIS_OUTPUT_DIR,
        f"{stem}_edges.jpg"
    )

    cv2.imwrite(
        output_path,
        diagnostic
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("TASK 2 - EXPERIMENT 03")
    print("CANNY + CONTOURS + GEOMETRIC FILTERING")
    print("=" * 70)

    print("\nParameters:")
    print(
        f"Canny low threshold  : "
        f"{CANNY_LOW}"
    )

    print(
        f"Canny high threshold : "
        f"{CANNY_HIGH}"
    )

    print(
        f"Morphology kernel    : "
        f"{MORPH_KERNEL_SIZE}"
    )

    print(
        f"Minimum area         : "
        f"{MIN_AREA}"
    )

    print(
        f"Maximum area         : "
        f"{MAX_AREA}"
    )

    print(
        f"Aspect ratio range   : "
        f"{MIN_ASPECT_RATIO} - "
        f"{MAX_ASPECT_RATIO}"
    )

    print(
        f"Minimum solidity     : "
        f"{MIN_SOLIDITY}"
    )

    print(
        f"Minimum extent       : "
        f"{MIN_EXTENT}"
    )

    print(
        f"Maximum candidates   : "
        f"{MAX_COMPONENTS}"
    )

    # --------------------------------------------------------
    # Development images
    # --------------------------------------------------------

    image_paths = get_development_images()

    # --------------------------------------------------------
    # Quick test
    # --------------------------------------------------------

    if RUN_QUICK_TEST:

        image_paths = image_paths[
            :QUICK_TEST_IMAGES
        ]

        print(
            "\nQUICK TEST ENABLED"
        )

        print(
            f"Images to process: "
            f"{len(image_paths)}"
        )

    else:

        print(
            "\nFULL DEVELOPMENT RUN"
        )

        print(
            f"Images to process: "
            f"{len(image_paths)}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "ERROR: Could not read image."
            )

            continue

        # ----------------------------------------------------
        # Segment
        # ----------------------------------------------------

        (
            pred_mask,
            candidates,
            edges,
            closed_edges
        ) = segment_egg(
            image
        )

        # ----------------------------------------------------
        # Frozen Task 1 mask
        # ----------------------------------------------------

        task1_mask = load_task1_union_mask(
            image_path,
            image.shape
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        (
            iou,
            dice,
            precision,
            recall
        ) = calculate_metrics(
            pred_mask,
            task1_mask
        )

        # ----------------------------------------------------
        # Candidate statistics
        # ----------------------------------------------------

        candidate_count = len(
            candidates
        )

        predicted_area = int(
            np.count_nonzero(
                pred_mask
            )
        )

        task1_area = int(
            np.count_nonzero(
                task1_mask
            )
        )

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        mask_path = save_mask(
            pred_mask,
            image_path
        )

        # ----------------------------------------------------
        # Save comparison
        # ----------------------------------------------------

        visualization = create_visualization(
            image,
            pred_mask,
            task1_mask
        )

        visualization_path = (
            save_visualization(
                visualization,
                image_path
            )
        )

        # ----------------------------------------------------
        # Save edge diagnostic
        # ----------------------------------------------------

        edge_path = save_edge_visualization(
            edges,
            closed_edges,
            image_path
        )

        # ----------------------------------------------------
        # Candidate information
        # ----------------------------------------------------

        candidate_areas = [
            round(
                item[1]["area"],
                2
            )
            for item in candidates
        ]

        candidate_aspect_ratios = [
            round(
                item[1]["aspect_ratio"],
                3
            )
            for item in candidates
        ]

        candidate_solidities = [
            round(
                item[1]["solidity"],
                3
            )
            for item in candidates
        ]

        candidate_extents = [
            round(
                item[1]["extent"],
                3
            )
            for item in candidates
        ]

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Candidates : "
            f"{candidate_count}"
        )

        print(
            f"Pred area  : "
            f"{predicted_area}"
        )

        print(
            f"Task1 area : "
            f"{task1_area}"
        )

        print(
            f"IoU        : "
            f"{iou:.4f}"
        )

        print(
            f"Dice       : "
            f"{dice:.4f}"
        )

        print(
            f"Precision  : "
            f"{precision:.4f}"
        )

        print(
            f"Recall     : "
            f"{recall:.4f}"
        )

        if candidate_count > 0:

            print(
                f"Candidate areas: "
                f"{candidate_areas}"
            )

            print(
                f"Aspect ratios: "
                f"{candidate_aspect_ratios}"
            )

            print(
                f"Solidities: "
                f"{candidate_solidities}"
            )

            print(
                f"Extents: "
                f"{candidate_extents}"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "image": filename,

            "image_path": image_path,

            "candidate_count":
                candidate_count,

            "predicted_area":
                predicted_area,

            "task1_area":
                task1_area,

            "iou":
                iou,

            "dice":
                dice,

            "precision":
                precision,

            "recall":
                recall,

            "candidate_areas":
                str(candidate_areas),

            "candidate_aspect_ratios":
                str(candidate_aspect_ratios),

            "candidate_solidities":
                str(candidate_solidities),

            "candidate_extents":
                str(candidate_extents),

            "mask_path":
                mask_path,

            "visualization_path":
                visualization_path,

            "edge_visualization_path":
                edge_path
        })

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_df = pd.DataFrame(
        results
    )

    report_path = os.path.join(
        REPORT_OUTPUT_DIR,
        "experiment_03_report.csv"
    )

    report_df.to_csv(
        report_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if len(report_df) > 0:

        mean_iou = report_df[
            "iou"
        ].mean()

        median_iou = report_df[
            "iou"
        ].median()

        mean_dice = report_df[
            "dice"
        ].mean()

        mean_precision = report_df[
            "precision"
        ].mean()

        mean_recall = report_df[
            "recall"
        ].mean()

        mean_candidates = report_df[
            "candidate_count"
        ].mean()

    else:

        mean_iou = 0.0
        median_iou = 0.0
        mean_dice = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        mean_candidates = 0.0

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 03 SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: "
        f"{len(report_df)}"
    )

    print(
        f"Mean IoU: "
        f"{mean_iou:.4f}"
    )

    print(
        f"Median IoU: "
        f"{median_iou:.4f}"
    )

    print(
        f"Mean Dice: "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean Precision: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Mean Recall: "
        f"{mean_recall:.4f}"
    )

    print(
        f"Mean candidates/image: "
        f"{mean_candidates:.2f}"
    )

    print("\nReport:")
    print(
        report_path
    )

    print("\nVisualizations:")
    print(
        VIS_OUTPUT_DIR
    )

    print("\nMasks:")
    print(
        MASK_OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 03 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()