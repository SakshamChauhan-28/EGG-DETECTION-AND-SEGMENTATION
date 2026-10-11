from pathlib import Path
"""
TASK 2 - EXPERIMENT 5
HYBRID REGION + EDGE + GRABCUT + SHAPE SEGMENTATION

Goal:
    Improve upon E1-E4 by combining:
        - Color/brightness region proposals
        - Edge information
        - Classical GrabCut refinement
        - Geometric/shape filtering

IMPORTANT:
    Task 1 masks are used ONLY for evaluation.
    They are never provided to the segmentation algorithm.

Evaluation:
    Development subset only.
    Evaluation set is never processed.

Metrics:
    IoU
    Dice
    Precision
    Recall
"""

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
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
    "task2_experiment_05"
)

VIS_DIR = os.path.join(
    OUTPUT_ROOT,
    "visualizations"
)

MASK_DIR = os.path.join(
    OUTPUT_ROOT,
    "masks"
)

REPORT_DIR = os.path.join(
    OUTPUT_ROOT,
    "reports"
)


# ------------------------------------------------------------
# Run only the same 10 images used for E1-E4 initially
# ------------------------------------------------------------

RUN_LIMIT = 10


# ============================================================
# E5 PARAMETERS
# ============================================================

# ---- E1-style color/region proposal ----

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100

# ---- Morphological cleanup ----

COLOR_CLOSE_KERNEL = 7
COLOR_OPEN_KERNEL = 5

# ---- Edge detection ----

CANNY_LOW = 50
CANNY_HIGH = 150

EDGE_CLOSE_KERNEL = 7
EDGE_DILATE_KERNEL = 3

# ---- Candidate component ----

MIN_COMPONENT_AREA = 500
MAX_COMPONENTS = 8

# ---- GrabCut ----

GRABCUT_ITERATIONS = 5

# Erosion used to obtain confident foreground
SURE_FOREGROUND_KERNEL = 7

# ---- Shape filtering ----

MIN_FINAL_AREA = 500

MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 3.5

MIN_SOLIDITY = 0.70
MIN_EXTENT = 0.35

MIN_CIRCULARITY = 0.25

# Slight final smoothing
FINAL_CLOSE_KERNEL = 5


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_ROOT,
    VIS_DIR,
    MASK_DIR,
    REPORT_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# DATASET SPLIT
# ============================================================

def get_development_images():
    """
    Read the leakage-aware split created previously.

    Expected:
        Development = 88
        Evaluation  = 22
    """

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

    path_column = None

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
            path_column = column
            break

    if path_column is None:
        raise ValueError(
            "Could not find image path column in split CSV"
        )

    development = df[
        df[split_column].astype(str).str.lower().isin(
            ["development", "dev", "train"]
        )
    ].copy()

    evaluation = df[
        df[split_column].astype(str).str.lower().isin(
            ["evaluation", "eval", "test"]
        )
    ].copy()

    print(f"Development images: {len(development)}")
    print(f"Evaluation images: {len(evaluation)}")

    if len(development) != 88:
        print(
            f"WARNING: Expected 88 development images, "
            f"found {len(development)}"
        )

    if len(evaluation) != 22:
        print(
            f"WARNING: Expected 22 evaluation images, "
            f"found {len(evaluation)}"
        )

    image_paths = []

    for path in development[path_column]:

        path = str(path)

        if os.path.exists(path):
            image_paths.append(path)
            continue

        # If CSV contains only filename/path relative to dataset,
        # search recursively from Downloads dataset location.
        dataset_root = Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent.parent / "data" / "egg.v1i.folder"))

        filename = os.path.basename(path)

        found = None

        for root, dirs, files in os.walk(dataset_root):

            if filename in files:
                found = os.path.join(root, filename)
                break

        if found is not None:
            image_paths.append(found)
        else:
            print(
                f"WARNING: Could not locate image:\n{path}"
            )

    return image_paths


# ============================================================
# TASK 1 MASK LOADING
# ============================================================

def load_task1_union_mask(image_path):
    """
    Load all Task 1 masks belonging to an image
    and combine them into one binary mask.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    pattern = os.path.join(
        TASK1_MASK_DIR,
        f"{stem}_egg_*.png"
    )

    import glob

    mask_files = glob.glob(pattern)

    union_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for mask_file in mask_files:

        mask = cv2.imread(
            mask_file,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        union_mask[
            mask > 127
        ] = 255

    return union_mask


# ============================================================
# E5 STEP 1
# COLOR / REGION PROPOSAL
# ============================================================

def create_color_proposal(image):
    """
    E1-inspired region proposal.

    Uses:
        HSV brightness
        HSV saturation
        LAB lightness

    This is only a proposal, not the final segmentation.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    # Bright pixels
    brightness_threshold = np.percentile(
        V,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        V >= brightness_threshold
    ).astype(np.uint8) * 255

    # Low/moderate saturation
    saturation_mask = (
        S <= MAX_SATURATION
    ).astype(np.uint8) * 255

    # LAB lightness support
    lab_threshold = np.percentile(
        L,
        50
    )

    lab_mask = (
        L >= lab_threshold
    ).astype(np.uint8) * 255

    # Combine region cues
    proposal = cv2.bitwise_and(
        bright_mask,
        saturation_mask
    )

    proposal = cv2.bitwise_and(
        proposal,
        lab_mask
    )

    # Morphological cleanup
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            COLOR_CLOSE_KERNEL,
            COLOR_CLOSE_KERNEL
        )
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            COLOR_OPEN_KERNEL,
            COLOR_OPEN_KERNEL
        )
    )

    proposal = cv2.morphologyEx(
        proposal,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    proposal = cv2.morphologyEx(
        proposal,
        cv2.MORPH_OPEN,
        open_kernel
    )

    return proposal


# ============================================================
# E5 STEP 2
# EDGE INFORMATION
# ============================================================

def create_edge_map(image):
    """
    Generate an edge map used to understand object boundaries.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Slight smoothing prevents texture noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            EDGE_CLOSE_KERNEL,
            EDGE_CLOSE_KERNEL
        )
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            EDGE_DILATE_KERNEL,
            EDGE_DILATE_KERNEL
        )
    )

    edges = cv2.dilate(
        edges,
        dilate_kernel,
        iterations=1
    )

    return edges


# ============================================================
# E5 STEP 3
# SHAPE FILTER
# ============================================================

def shape_is_egg_like(contour):

    area = cv2.contourArea(contour)

    if area < MIN_FINAL_AREA:
        return False

    x, y, w, h = cv2.boundingRect(contour)

    if h == 0:
        return False

    aspect_ratio = w / float(h)

    if not (
        MIN_ASPECT_RATIO
        <= aspect_ratio
        <= MAX_ASPECT_RATIO
    ):
        return False

    contour_area = area

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area <= 0:
        return False

    solidity = (
        contour_area /
        float(hull_area)
    )

    if solidity < MIN_SOLIDITY:
        return False

    bounding_area = w * h

    if bounding_area <= 0:
        return False

    extent = (
        contour_area /
        float(bounding_area)
    )

    if extent < MIN_EXTENT:
        return False

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return False

    circularity = (
        4.0 * np.pi * contour_area /
        (perimeter * perimeter)
    )

    if circularity < MIN_CIRCULARITY:
        return False

    return True


# ============================================================
# E5 STEP 4
# GRABCUT REFINEMENT
# ============================================================

def refine_with_grabcut(
    image,
    component_mask,
    edge_map
):
    """
    Refine one candidate region using classical GrabCut.

    No neural model is used.

    Initialization:
        - sure foreground = eroded candidate
        - probable foreground = candidate region
        - probable background = outside candidate
        - sure background = ROI border

    Edge information is used to weaken foreground
    confidence around strong boundaries.
    """

    ys, xs = np.where(
        component_mask > 0
    )

    if len(xs) == 0:
        return np.zeros_like(
            component_mask
        )

    x1 = max(
        0,
        int(xs.min()) - 12
    )

    y1 = max(
        0,
        int(ys.min()) - 12
    )

    x2 = min(
        image.shape[1],
        int(xs.max()) + 13
    )

    y2 = min(
        image.shape[0],
        int(ys.max()) + 13
    )

    roi_image = image[
        y1:y2,
        x1:x2
    ]

    roi_candidate = component_mask[
        y1:y2,
        x1:x2
    ]

    roi_edges = edge_map[
        y1:y2,
        x1:x2
    ]

    if roi_image.size == 0:
        return np.zeros_like(
            component_mask
        )

    # --------------------------------------------------------
    # GrabCut initialization
    # --------------------------------------------------------

    gc_mask = np.full(
        roi_candidate.shape,
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Candidate = probable foreground
    gc_mask[
        roi_candidate > 0
    ] = cv2.GC_PR_FGD

    # Strong foreground = eroded candidate
    fg_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            SURE_FOREGROUND_KERNEL,
            SURE_FOREGROUND_KERNEL
        )
    )

    sure_fg = cv2.erode(
        roi_candidate,
        fg_kernel,
        iterations=1
    )

    gc_mask[
        sure_fg > 0
    ] = cv2.GC_FGD

    # --------------------------------------------------------
    # Use edge information around candidate boundary
    # as probable background.
    # --------------------------------------------------------

    candidate_boundary = cv2.morphologyEx(
        roi_candidate,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    boundary_edges = cv2.bitwise_and(
        roi_edges,
        candidate_boundary
    )

    # Only change probable foreground.
    edge_pixels = (
        boundary_edges > 0
    ) & (
        gc_mask == cv2.GC_PR_FGD
    )

    gc_mask[edge_pixels] = cv2.GC_PR_BGD

    # --------------------------------------------------------
    # Protect sure foreground
    # --------------------------------------------------------

    gc_mask[
        sure_fg > 0
    ] = cv2.GC_FGD

    # --------------------------------------------------------
    # Run GrabCut
    # --------------------------------------------------------

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            roi_image,
            gc_mask,
            None,
            bg_model,
            fg_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        # If GrabCut fails, return the original
        # candidate instead of crashing.
        result = np.zeros_like(
            component_mask
        )

        result[
            y1:y2,
            x1:x2
        ] = roi_candidate

        return result

    refined_roi = np.where(
        (
            (gc_mask == cv2.GC_FGD)
            |
            (gc_mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(np.uint8)

    result = np.zeros_like(
        component_mask
    )

    result[
        y1:y2,
        x1:x2
    ] = refined_roi

    return result


# ============================================================
# E5 STEP 5
# FINAL SHAPE FILTERING
# ============================================================

def apply_shape_filter(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    final_mask = np.zeros_like(mask)

    kept = 0

    for contour in contours:

        if kept >= MAX_COMPONENTS:
            break

        if not shape_is_egg_like(
            contour
        ):
            continue

        cv2.drawContours(
            final_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

        kept += 1

    # Final smoothing
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            FINAL_CLOSE_KERNEL,
            FINAL_CLOSE_KERNEL
        )
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return final_mask, kept


# ============================================================
# E5 MAIN SEGMENTATION
# ============================================================

def segment_egg_hybrid(image):

    # --------------------------------------------------------
    # STEP 1: Region proposal
    # --------------------------------------------------------

    color_proposal = create_color_proposal(
        image
    )

    # --------------------------------------------------------
    # STEP 2: Edge detection
    # --------------------------------------------------------

    edge_map = create_edge_map(
        image
    )

    # --------------------------------------------------------
    # STEP 3: Find candidate components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_proposal,
        connectivity=8
    )

    candidates = []

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_COMPONENT_AREA:
            continue

        component = np.zeros_like(
            color_proposal
        )

        component[
            labels == label
        ] = 255

        candidates.append(
            (
                area,
                component
            )
        )

    # Largest components first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    candidates = candidates[
        :MAX_COMPONENTS
    ]

    # --------------------------------------------------------
    # STEP 4: Hybrid refinement
    # --------------------------------------------------------

    refined_union = np.zeros_like(
        color_proposal
    )

    for area, component in candidates:

        refined = refine_with_grabcut(
            image,
            component,
            edge_map
        )

        refined_union = cv2.bitwise_or(
            refined_union,
            refined
        )

    # --------------------------------------------------------
    # STEP 5: Shape constraints
    # --------------------------------------------------------

    final_mask, kept = apply_shape_filter(
        refined_union
    )

    return final_mask, len(candidates), kept


# ============================================================
# METRICS
# ============================================================

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

    prediction_area = prediction.sum()
    ground_truth_area = ground_truth.sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    true_positive = intersection

    false_positive = np.logical_and(
        prediction,
        ~ground_truth
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        ground_truth
    ).sum()

    # IoU
    if union == 0:
        iou = 1.0 if prediction_area == 0 and ground_truth_area == 0 else 0.0
    else:
        iou = intersection / union

    # Dice
    denominator = (
        prediction_area +
        ground_truth_area
    )

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection /
            denominator
        )

    # Precision
    precision_denominator = (
        true_positive +
        false_positive
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive /
            precision_denominator
        )

    # Recall
    recall_denominator = (
        true_positive +
        false_negative
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive /
            recall_denominator
        )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall)
    }


# ============================================================
# VISUALIZATION
# ============================================================

def create_comparison_visualization(
    image,
    prediction,
    ground_truth
):
    """
    Color convention:

        YELLOW = overlap / true positive
        GREEN  = false positive
        RED    = false negative
    """

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pred = prediction > 127
    gt = ground_truth > 127

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    overlay = image_rgb.copy()

    # alpha blending
    alpha = 0.45

    # Yellow = TP
    overlay[tp] = (
        alpha * np.array(
            [255, 255, 0]
        )
        +
        (1 - alpha) * overlay[tp]
    ).astype(np.uint8)

    # Green = FP
    overlay[fp] = (
        alpha * np.array(
            [0, 255, 0]
        )
        +
        (1 - alpha) * overlay[fp]
    ).astype(np.uint8)

    # Red = FN
    overlay[fn] = (
        alpha * np.array(
            [255, 0, 0]
        )
        +
        (1 - alpha) * overlay[fn]
    ).astype(np.uint8)

    return cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# SAVE BINARY MASK
# ============================================================

def save_mask(mask, path):
    cv2.imwrite(
        path,
        mask
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 2 - EXPERIMENT 5")
    print("HYBRID REGION + EDGE + GRABCUT + SHAPE")
    print("=" * 70)

    image_paths = get_development_images()

    print(
        f"Development images available: "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Run only the same 10 images as E1-E4 initially.
    # --------------------------------------------------------

    image_paths = image_paths[:RUN_LIMIT]

    print(
        f"Images evaluated in E5: "
        f"{len(image_paths)}"
    )

    print(
        "Evaluation set is NOT processed."
    )

    results = []

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
                "  ERROR: Could not read image"
            )

            continue

        # ----------------------------------------------------
        # E5 segmentation
        # ----------------------------------------------------

        prediction, candidate_count, kept_count = (
            segment_egg_hybrid(image)
        )

        # ----------------------------------------------------
        # Load frozen Task 1 mask
        # ----------------------------------------------------

        ground_truth = load_task1_union_mask(
            image_path
        )

        if ground_truth is None:

            print(
                "  ERROR: Could not load Task 1 mask"
            )

            continue

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
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
        # Save mask
        # ----------------------------------------------------

        stem = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            MASK_DIR,
            f"{stem}_mask.png"
        )

        save_mask(
            prediction,
            mask_path
        )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        visualization = (
            create_comparison_visualization(
                image,
                prediction,
                ground_truth
            )
        )

        visualization_path = os.path.join(
            VIS_DIR,
            f"{stem}_comparison.png"
        )

        cv2.imwrite(
            visualization_path,
            visualization
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "image": filename,

            "candidate_components":
                candidate_count,

            "kept_components":
                kept_count,

            "iou":
                metrics["iou"],

            "dice":
                metrics["dice"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"]
        })

    # ========================================================
    # REPORT
    # ========================================================

    if len(results) == 0:

        print()
        print(
            "No valid results generated."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    report_path = os.path.join(
        REPORT_DIR,
        "experiment_05_report.csv"
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 70)

    print(
        f"Images evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Mean IoU: "
        f"{results_df['iou'].mean():.4f}"
    )

    print(
        f"Median IoU: "
        f"{results_df['iou'].median():.4f}"
    )

    print(
        f"Mean Dice: "
        f"{results_df['dice'].mean():.4f}"
    )

    print(
        f"Mean Precision: "
        f"{results_df['precision'].mean():.4f}"
    )

    print(
        f"Mean Recall: "
        f"{results_df['recall'].mean():.4f}"
    )

    print(
        f"Mean candidates/image: "
        f"{results_df['candidate_components'].mean():.2f}"
    )

    print(
        f"Mean kept components/image: "
        f"{results_df['kept_components'].mean():.2f}"
    )

    print()
    print(
        f"Report:\n{report_path}"
    )

    print(
        f"\nVisualizations:\n{VIS_DIR}"
    )

    print(
        f"\nMasks:\n{MASK_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()