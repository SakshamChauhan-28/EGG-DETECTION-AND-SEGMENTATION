from pathlib import Path
"""
Task 2 - Experiment 02
Classical Image Processing Segmentation

Experiment:
    Local Contrast Enhancement (CLAHE)
    +
    Adaptive Gaussian Thresholding
    +
    Morphological Cleanup
    +
    Connected Component Filtering

IMPORTANT:
- Uses ONLY the 88-image development split for experimentation.
- The 22-image evaluation split is never used for tuning.
- Task 1 masks are used ONLY as frozen pseudo-ground-truth for evaluation.
- Task 1 masks are NEVER passed into the segmentation algorithm.
"""

import os
import glob
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
    "task2_experiment_02"
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
# EXPERIMENT 02 PARAMETERS
# ============================================================

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Adaptive threshold parameters
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5

# Morphological cleanup
MORPH_KERNEL_SIZE = 5

# Connected-component filtering
MIN_AREA = 500
MAX_COMPONENTS = 6


# ============================================================
# EXPERIMENT CONTROL
# ============================================================

# Start with 10 development images.
#
# After inspecting the results, change this to False
# for the complete 88-image development run.
RUN_QUICK_TEST = True

QUICK_TEST_IMAGES = 10


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD LEAKAGE-AWARE SPLIT
# ============================================================

def load_split_assignments():

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    df = pd.read_csv(SPLIT_FILE)

    print("\n" + "=" * 70)
    print("LOADING LEAKAGE-AWARE SPLIT")
    print("=" * 70)

    print("Columns:")
    print(list(df.columns))

    return df


# ============================================================
# FIND IMAGE PATH COLUMN
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
        "Could not find image path column in split CSV.\n"
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
        "Could not find split column in split CSV.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# GET DEVELOPMENT IMAGES
# ============================================================

def get_development_images():

    df = load_split_assignments()

    image_column = find_image_column(df)
    split_column = find_split_column(df)

    print(f"\nImage column : {image_column}")
    print(f"Split column : {split_column}")

    # Normalize split names
    df[split_column] = (
        df[split_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Accept development/dev naming
    development_df = df[
        df[split_column].isin([
            "development",
            "dev"
        ])
    ].copy()

    evaluation_df = df[
        df[split_column].isin([
            "evaluation",
            "eval"
        ])
    ].copy()

    print(f"\nDevelopment images in CSV : {len(development_df)}")
    print(f"Evaluation images in CSV   : {len(evaluation_df)}")

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(development_df) != 88:
        raise RuntimeError(
            f"Expected exactly 88 development images, "
            f"but found {len(development_df)}."
        )

    if len(evaluation_df) != 22:
        raise RuntimeError(
            f"Expected exactly 22 evaluation images, "
            f"but found {len(evaluation_df)}."
        )

    # --------------------------------------------------------
    # RESOLVE IMAGE PATHS
    # --------------------------------------------------------

    image_paths = []

    for _, row in development_df.iterrows():

        raw_path = str(row[image_column])

        # Direct path
        if os.path.exists(raw_path):
            image_paths.append(os.path.abspath(raw_path))
            continue

        # Relative to project root
        project_relative = os.path.join(
            PROJECT_ROOT,
            raw_path
        )

        if os.path.exists(project_relative):
            image_paths.append(
                os.path.abspath(project_relative)
            )
            continue

        # Search by filename
        filename = os.path.basename(raw_path)

        matches = []

        for root, dirs, files in os.walk(PROJECT_ROOT):

            if filename in files:
                matches.append(
                    os.path.join(root, filename)
                )

        if len(matches) == 1:
            image_paths.append(
                os.path.abspath(matches[0])
            )

        elif len(matches) > 1:

            # Prefer dataset paths
            dataset_matches = [
                p for p in matches
                if "egg.v1i.folder" in p.lower()
            ]

            if len(dataset_matches) == 1:
                image_paths.append(
                    os.path.abspath(dataset_matches[0])
                )
            else:
                image_paths.append(
                    os.path.abspath(matches[0])
                )

        else:
            print(
                f"WARNING: Could not locate image:\n{raw_path}"
            )

    # Remove duplicates while preserving order
    image_paths = list(dict.fromkeys(image_paths))

    print(
        f"\nResolved development images: {len(image_paths)}"
    )

    if len(image_paths) != 88:
        raise RuntimeError(
            f"Expected 88 resolved development images, "
            f"but found {len(image_paths)}."
        )

    return image_paths


# ============================================================
# TASK 1 UNION MASK
# ============================================================

def load_task1_union_mask(image_path, shape):

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

        print(
            f"WARNING: No Task 1 masks found for:\n"
            f"{os.path.basename(image_path)}"
        )

        return np.zeros(
            shape[:2],
            dtype=np.uint8
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

        union_mask[
            mask > 127
        ] = 255

    return union_mask


# ============================================================
# EXPERIMENT 02 SEGMENTATION
# ============================================================

def segment_egg(image):

    """
    Experiment 02:

        BGR
         ↓
        Grayscale
         ↓
        CLAHE
         ↓
        Adaptive Gaussian Threshold
         ↓
        Morphological Opening
         ↓
        Morphological Closing
         ↓
        Connected Components
         ↓
        Binary foreground mask

    No Task 1 information is used here.
    """

    # --------------------------------------------------------
    # 1. Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 2. Local contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # 3. Adaptive threshold
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    # --------------------------------------------------------
    # 4. Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    # Remove small isolated noise
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # 5. Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )

    components = []

    for label in range(1, num_labels):

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

        components.append({
            "label": label,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    # Sort by area
    components.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    # Keep at most MAX_COMPONENTS
    components = components[
        :MAX_COMPONENTS
    ]

    final_mask = np.zeros_like(
        cleaned,
        dtype=np.uint8
    )

    for component in components:

        label = component["label"]

        final_mask[
            labels == label
        ] = 255

    return final_mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred_mask, gt_mask):

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

    # IoU
    if union == 0:
        iou = 1.0 if pred_area == 0 and gt_area == 0 else 0.0
    else:
        iou = intersection / union

    # Dice
    denominator = pred_area + gt_area

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
            intersection
            / pred_area
        )

    # Recall
    if gt_area == 0:
        recall = 0.0
    else:
        recall = (
            intersection
            / gt_area
        )

    return (
        iou,
        dice,
        precision,
        recall
    )


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image,
    pred_mask,
    gt_mask
):

    """
    Visualization:

        Yellow = TP / overlap
        Green  = False Positive
        Red    = False Negative
    """

    output = image.copy()

    pred = pred_mask > 0
    gt = gt_mask > 0

    true_positive = (
        pred &
        gt
    )

    false_positive = (
        pred &
        ~gt
    )

    false_negative = (
        ~pred &
        gt
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = output.copy()

    # OpenCV BGR colors
    overlay[true_positive] = (
        0,
        255,
        255
    )  # Yellow

    overlay[false_positive] = (
        0,
        255,
        0
    )  # Green

    overlay[false_negative] = (
        0,
        0,
        255
    )  # Red

    # Blend
    visualization = cv2.addWeighted(
        output,
        0.55,
        overlay,
        0.45,
        0
    )

    return visualization


# ============================================================
# SAVE BINARY MASK
# ============================================================

def save_mask(mask, image_path):

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
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("TASK 2 - EXPERIMENT 02")
    print("CLAHE + ADAPTIVE THRESHOLDING")
    print("=" * 70)

    print("\nParameters:")
    print(
        f"CLAHE clip limit      : "
        f"{CLAHE_CLIP_LIMIT}"
    )
    print(
        f"CLAHE tile grid       : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )
    print(
        f"Adaptive block size   : "
        f"{ADAPTIVE_BLOCK_SIZE}"
    )
    print(
        f"Adaptive C            : "
        f"{ADAPTIVE_C}"
    )
    print(
        f"Morphology kernel     : "
        f"{MORPH_KERNEL_SIZE}"
    )
    print(
        f"Minimum component area: "
        f"{MIN_AREA}"
    )
    print(
        f"Maximum components    : "
        f"{MAX_COMPONENTS}"
    )

    # --------------------------------------------------------
    # Load development split
    # --------------------------------------------------------

    image_paths = get_development_images()

    # --------------------------------------------------------
    # QUICK TEST
    # --------------------------------------------------------

    if RUN_QUICK_TEST:

        image_paths = image_paths[
            :QUICK_TEST_IMAGES
        ]

        print(
            f"\nQUICK TEST ENABLED"
        )

        print(
            f"Images to process: "
            f"{len(image_paths)}"
        )

    else:

        print(
            f"\nFULL DEVELOPMENT RUN"
        )

        print(
            f"Images to process: "
            f"{len(image_paths)}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )

        print(
            f"\n[{index}/{len(image_paths)}] "
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
        # Classical segmentation
        # ----------------------------------------------------

        pred_mask = segment_egg(
            image
        )

        # ----------------------------------------------------
        # Load frozen Task 1 mask
        # ----------------------------------------------------

        task1_mask = load_task1_union_mask(
            image_path,
            image.shape
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        iou, dice, precision, recall = (
            calculate_metrics(
                pred_mask,
                task1_mask
            )
        )

        # ----------------------------------------------------
        # Component statistics
        # ----------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                pred_mask,
                connectivity=8
            )
        )

        component_count = (
            num_labels - 1
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
        # Visualization
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
        # Print result
        # ----------------------------------------------------

        print(
            f"Components : {component_count}"
        )

        print(
            f"Pred area  : {predicted_area}"
        )

        print(
            f"Task1 area : {task1_area}"
        )

        print(
            f"IoU        : {iou:.4f}"
        )

        print(
            f"Dice       : {dice:.4f}"
        )

        print(
            f"Precision  : {precision:.4f}"
        )

        print(
            f"Recall     : {recall:.4f}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "image": filename,

            "image_path": image_path,

            "components": component_count,

            "predicted_area": predicted_area,

            "task1_area": task1_area,

            "iou": iou,

            "dice": dice,

            "precision": precision,

            "recall": recall,

            "mask_path": mask_path,

            "visualization_path": visualization_path
        })

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_df = pd.DataFrame(
        results
    )

    report_path = os.path.join(
        REPORT_OUTPUT_DIR,
        "experiment_02_report.csv"
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

    else:

        mean_iou = 0
        median_iou = 0
        mean_dice = 0
        mean_precision = 0
        mean_recall = 0

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 02 SUMMARY")
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

    print("\nReport:")
    print(report_path)

    print("\nVisualizations:")
    print(VIS_OUTPUT_DIR)

    print("\nMasks:")
    print(MASK_OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("EXPERIMENT 02 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()