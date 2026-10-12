import os
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from ultralytics import SAM


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data" / "egg.v1i.folder"))
)

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
)

ASSIGNMENT_FILE = (
    PROJECT_ROOT
    / "dataset_report"
    / "final_split_assignments.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "task1_gdino_sam"
    / "evaluation"
)

VIS_DIR = OUTPUT_ROOT / "visualizations"
MASK_DIR = OUTPUT_ROOT / "masks"
REPORT_DIR = OUTPUT_ROOT / "reports"

VIS_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN TASK 1 MODEL CONFIG
# ============================================================

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

SAM_CHECKPOINT = "sam2.1_t.pt"

TEXT_PROMPT = "egg."

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# UTILITY
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD GROUNDING DINO
# ============================================================

def load_grounding_dino():

    print_header("LOADING GROUNDING DINO")

    print(f"Model: {DINO_MODEL_ID}")
    print(f"Device: {DEVICE}")

    processor = AutoProcessor.from_pretrained(
        DINO_MODEL_ID
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL_ID
    ).to(DEVICE)

    model.eval()

    print("Grounding DINO loaded.")

    return processor, model


# ============================================================
# LOAD SAM
# ============================================================

def load_sam():

    print_header("LOADING SAM 2.1 TINY")

    print(f"Checkpoint: {SAM_CHECKPOINT}")

    sam = SAM(SAM_CHECKPOINT)

    print("SAM loaded.")

    return sam


# ============================================================
# GROUNDING DINO DETECTION
# ============================================================

@torch.no_grad()
def detect_eggs(image, processor, model):

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model(**inputs)

    try:

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (image.height, image.width)
            ],
        )

    except TypeError:

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (image.height, image.width)
            ],
        )

    result = results[0]

    boxes = result["boxes"].detach().cpu().numpy()

    scores = result["scores"].detach().cpu().numpy()

    if "text_labels" in result:
        labels = result["text_labels"]
    else:
        labels = ["egg"] * len(boxes)

    return boxes, scores, labels


# ============================================================
# SAM SEGMENTATION
# ============================================================

def segment_with_sam(sam, image, boxes):

    if len(boxes) == 0:
        return []

    boxes_list = boxes.tolist()

    results = sam.predict(
        source=image,
        bboxes=boxes_list,
        verbose=False,
    )

    result = results[0]

    if result.masks is None:
        return []

    masks = result.masks.data.cpu().numpy()

    return masks


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image,
    boxes,
    scores,
    masks,
    output_path,
):

    image_np = np.array(image).copy()

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR,
    )

    overlay = image_bgr.copy()

    # Draw masks
    for mask in masks:

        mask_bool = mask > 0.5

        if mask_bool.shape != image_bgr.shape[:2]:

            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        overlay[mask_bool] = (
            0.55 * overlay[mask_bool]
            + 0.45 * np.array([0, 255, 0])
        ).astype(np.uint8)

    image_bgr = cv2.addWeighted(
        overlay,
        0.7,
        image_bgr,
        0.3,
        0,
    )

    # Draw bounding boxes
    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        score = scores[i]

        label = f"egg {score:.2f}"

        cv2.putText(
            image_bgr,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        image_bgr,
    )


# ============================================================
# SAVE INDIVIDUAL MASKS
# ============================================================

def save_masks(masks, image_name):

    image_stem = Path(image_name).stem

    saved = 0

    for i, mask in enumerate(masks):

        mask_bool = mask > 0.5

        mask_uint8 = (
            mask_bool.astype(np.uint8) * 255
        )

        output_path = (
            MASK_DIR
            / f"{image_stem}_egg_{i + 1}.png"
        )

        cv2.imwrite(
            str(output_path),
            mask_uint8,
        )

        saved += 1

    return saved


# ============================================================
# FIND EXACT EVALUATION SPLIT
# ============================================================

def get_evaluation_images():

    print_header("LOADING LEAKAGE-AWARE EVALUATION SPLIT")

    if not ASSIGNMENT_FILE.exists():

        raise FileNotFoundError(
            f"Split assignment file not found:\n"
            f"{ASSIGNMENT_FILE}"
        )

    df = pd.read_csv(
        ASSIGNMENT_FILE
    )

    print("Split assignment columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Find split column
    # --------------------------------------------------------

    split_column = None

    possible_split_columns = [
        "final_split",
        "split",
        "dataset_split",
        "assigned_split",
    ]

    for column in possible_split_columns:

        if column in df.columns:
            split_column = column
            break

    if split_column is None:

        raise ValueError(
            "Could not identify split column."
        )

    # --------------------------------------------------------
    # Find image column
    # --------------------------------------------------------

    image_column = None

    possible_image_columns = [
        "image_path",
        "filepath",
        "file_path",
        "path",
        "image",
        "filename",
        "file_name",
    ]

    for column in possible_image_columns:

        if column in df.columns:
            image_column = column
            break

    if image_column is None:

        raise ValueError(
            "Could not identify image column."
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
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    if len(development_df) != 88:

        raise ValueError(
            f"EXPECTED 88 DEVELOPMENT IMAGES, "
            f"BUT FOUND {len(development_df)}."
        )

    if len(evaluation_df) != 22:

        raise ValueError(
            f"EXPECTED 22 EVALUATION IMAGES, "
            f"BUT FOUND {len(evaluation_df)}."
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
                DATASET_ROOT / raw_path
            )

            if candidate.exists():

                image_path = candidate

            else:

                filename = Path(
                    raw_path
                ).name

                matches = list(
                    DATASET_ROOT.rglob(
                        filename
                    )
                )

                if len(matches) == 1:

                    image_path = matches[0]

                elif len(matches) == 0:

                    raise FileNotFoundError(
                        f"Could not locate:\n"
                        f"{raw_path}"
                    )

                else:

                    raise RuntimeError(
                        f"Multiple files found for:\n"
                        f"{filename}"
                    )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Evaluation image does not exist:\n"
                f"{image_path}"
            )

        image_paths.append(
            image_path
        )

    # --------------------------------------------------------
    # Verify NO development overlap
    # --------------------------------------------------------

    development_filenames = {
        Path(str(p)).name
        for p in development_df[image_column]
    }

    evaluation_filenames = {
        p.name
        for p in image_paths
    }

    overlap = (
        development_filenames
        & evaluation_filenames
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED!\n"
            f"Overlapping images: {overlap}"
        )

    print()
    print("Leakage check: PASSED")
    print("Evaluation set: 22 images")
    print("Development/evaluation overlap: 0")

    print("=" * 70)

    return sorted(image_paths)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    processor,
    dino_model,
    sam,
):

    image_path = Path(image_path)

    print()
    print("-" * 70)
    print(f"IMAGE: {image_path.name}")
    print("-" * 70)

    start = time.perf_counter()

    image = Image.open(
        image_path
    ).convert("RGB")

    # Grounding DINO
    boxes, scores, labels = detect_eggs(
        image,
        processor,
        dino_model,
    )

    print(
        f"Grounding DINO detections: {len(boxes)}"
    )

    for i, (box, score) in enumerate(
        zip(boxes, scores)
    ):

        print(
            f"  Egg {i + 1}: "
            f"score={score:.3f}, "
            f"box={[round(float(x), 1) for x in box]}"
        )

    # SAM
    masks = segment_with_sam(
        sam,
        image,
        boxes,
    )

    print(
        f"SAM masks: {len(masks)}"
    )

    # Visualization
    output_visualization = (
        VIS_DIR / image_path.name
    )

    create_visualization(
        image,
        boxes,
        scores,
        masks,
        output_visualization,
    )

    # Masks
    saved_masks = save_masks(
        masks,
        image_path.name,
    )

    elapsed = (
        time.perf_counter() - start
    )

    if len(boxes) == 0:

        status = "NO_DETECTIONS"

    elif len(masks) == len(boxes):

        status = "GOOD"

    else:

        status = "REVIEW_MASK_COUNT"

    print(f"Status: {status}")
    print(f"Time: {elapsed:.2f}s")

    return {
        "image": image_path.name,
        "detections": len(boxes),
        "masks": saved_masks,
        "status": status,
        "time_sec": round(elapsed, 3),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "TASK 1 - FINAL EVALUATION"
    )

    print(
        "Grounding DINO Tiny -> SAM 2.1 Tiny"
    )

    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Grounding DINO threshold: "
        f"{BOX_THRESHOLD}"
    )

    print(
        f"Text threshold: "
        f"{TEXT_THRESHOLD}"
    )

    print()
    print(
        "IMPORTANT: This script uses the "
        "FROZEN Task 1 configuration."
    )

    print(
        "No evaluation results will be used "
        "to tune the model."
    )

    # --------------------------------------------------------
    # Load models ONCE
    # --------------------------------------------------------

    processor, dino_model = (
        load_grounding_dino()
    )

    sam = load_sam()

    # --------------------------------------------------------
    # Get EXACT 22 evaluation images
    # --------------------------------------------------------

    image_paths = (
        get_evaluation_images()
    )

    print(
        f"\nEvaluation images found: "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    reports = []

    total = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print(
            f"[{index}/{total}]"
        )

        try:

            report = process_image(
                image_path,
                processor,
                dino_model,
                sam,
            )

            reports.append(report)

        except Exception as e:

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )

            reports.append(
                {
                    "image": image_path.name,
                    "detections": 0,
                    "masks": 0,
                    "status":
                        "ERROR_" +
                        type(e).__name__,
                    "time_sec": 0,
                }
            )

        if DEVICE == "cuda":

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "evaluation_task1_report.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "detections",
                "masks",
                "status",
                "time_sec",
            ],
        )

        writer.writeheader()
        writer.writerows(reports)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    good = sum(
        r["status"] == "GOOD"
        for r in reports
    )

    no_detection = sum(
        r["status"] == "NO_DETECTIONS"
        for r in reports
    )

    review = sum(
        r["status"] == "REVIEW_MASK_COUNT"
        for r in reports
    )

    errors = sum(
        r["status"].startswith("ERROR")
        for r in reports
    )

    print_header(
        "TASK 1 EVALUATION COMPLETE"
    )

    print(
        f"Total images:       {len(reports)}"
    )

    print(
        f"GOOD:               {good}"
    )

    print(
        f"NO_DETECTIONS:      {no_detection}"
    )

    print(
        f"REVIEW_MASK_COUNT:  {review}"
    )

    print(
        f"ERRORS:             {errors}"
    )

    print()
    print(
        f"Visualizations: {VIS_DIR}"
    )

    print(
        f"Masks:          {MASK_DIR}"
    )

    print(
        f"Report:         {report_path}"
    )


if __name__ == "__main__":
    main()