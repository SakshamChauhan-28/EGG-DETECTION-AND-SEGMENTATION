import os
from pathlib import Path
import csv
import math

import cv2
import numpy as np


# ============================================================
# TASK 4 - VIDEO CANDIDATE ANALYSIS
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
)

VIDEO_PATH = PROJECT_ROOT / "Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm"

OUTPUT_ROOT = (
    PROJECT_ROOT / "task4_video" / "candidate_analysis"
)

FRAME_DIR = OUTPUT_ROOT / "frames"
REPORT_DIR = OUTPUT_ROOT / "reports"

FRAME_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VIDEO ANALYSIS WINDOW
# ============================================================

START_TIME_SEC = 5.0
END_TIME_SEC = 40.0

# Analyze approximately 4 frames per second.
SAMPLE_FPS = 4.0


# ============================================================
# FROZEN E1 PARAMETERS
# ============================================================

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7
MAX_COMPONENTS = 6


# ============================================================
# E1 SEGMENTATION
# ============================================================

def segment_e1(image):

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

    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= brightness_threshold
    )

    low_saturation = (
        saturation <= MAX_SATURATION
    )

    mask = (
        bright_mask &
        low_saturation
    ).astype(np.uint8) * 255

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


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

def get_components(mask):

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

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_AREA:
            continue

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        cx = float(
            centroids[label][0]
        )

        cy = float(
            centroids[label][1]
        )

        bbox_area = max(
            1,
            w * h
        )

        fill_ratio = (
            area / bbox_area
        )

        aspect_ratio = (
            max(w, h) /
            max(1, min(w, h))
        )

        # ----------------------------------------------------
        # Circularity
        # ----------------------------------------------------

        component_mask = (
            labels == label
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        circularity = 0.0

        if contours:

            contour = max(
                contours,
                key=cv2.contourArea
            )

            perimeter = cv2.arcLength(
                contour,
                True
            )

            contour_area = cv2.contourArea(
                contour
            )

            if perimeter > 0:

                circularity = (
                    4.0
                    * math.pi
                    * contour_area
                    / (perimeter * perimeter)
                )

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "aspect_ratio": aspect_ratio,
                "fill_ratio": fill_ratio,
                "circularity": circularity
            }
        )

    components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    return components[:MAX_COMPONENTS]


# ============================================================
# ANNOTATE FRAME
# ============================================================

def annotate_frame(
    frame,
    components,
    frame_number,
    timestamp
):

    output = frame.copy()

    cv2.putText(
        output,
        f"Frame: {frame_number}  "
        f"Time: {timestamp:.2f}s",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    for index, component in enumerate(
        components,
        start=1
    ):

        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]

        cx = int(
            round(component["cx"])
        )

        cy = int(
            round(component["cy"])
        )

        # Different colors only for visualization.
        # This does NOT affect the algorithm.
        if index == 1:
            color = (0, 255, 255)
        elif index == 2:
            color = (255, 0, 255)
        elif index == 3:
            color = (255, 255, 0)
        else:
            color = (0, 165, 255)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.circle(
            output,
            (cx, cy),
            6,
            color,
            -1
        )

        label = (
            f"C{index} "
            f"A={component['area']}"
        )

        cv2.putText(
            output,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 4 - VIDEO CANDIDATE ANALYSIS")
    print("=" * 70)

    print()
    print(f"Video: {VIDEO_PATH}")

    print()
    print("Frozen E1 parameters:")
    print(
        f"  Brightness percentile: "
        f"{BRIGHT_PERCENTILE}"
    )
    print(
        f"  Max saturation: "
        f"{MAX_SATURATION}"
    )
    print(
        f"  Minimum area: "
        f"{MIN_AREA}"
    )
    print(
        f"  Morphology kernel: "
        f"{MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}"
    )
    print(
        f"  Maximum components: "
        f"{MAX_COMPONENTS}"
    )

    print()
    print(
        f"Analysis interval: "
        f"{START_TIME_SEC:.1f}s -> "
        f"{END_TIME_SEC:.1f}s"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video.\n"
            "OpenCV could not decode the WebM/AV1 file."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print()
    print("Video properties:")
    print(
        f"  Resolution: "
        f"{width} x {height}"
    )
    print(
        f"  FPS: "
        f"{fps:.2f}"
    )
    print(
        f"  Frames: "
        f"{total_frames}"
    )
    print(
        f"  Duration: "
        f"{duration:.2f}s"
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    sample_step = max(
        1,
        int(round(fps / SAMPLE_FPS))
    )

    start_frame = int(
        START_TIME_SEC * fps
    )

    end_frame = min(
        total_frames - 1,
        int(END_TIME_SEC * fps)
    )

    print()
    print(
        f"Sampling every "
        f"{sample_step} frames "
        f"(~{fps / sample_step:.2f} FPS)"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        REPORT_DIR
        / "candidate_analysis.csv"
    )

    fieldnames = [
        "frame",
        "timestamp",
        "candidate_rank",
        "area",
        "x",
        "y",
        "w",
        "h",
        "cx",
        "cy",
        "aspect_ratio",
        "fill_ratio",
        "circularity"
    ]

    rows = []

    processed_frames = 0
    frame_index = start_frame

    # --------------------------------------------------------
    # Process sampled frames
    # --------------------------------------------------------

    while frame_index <= end_frame:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = cap.read()

        if not success:

            print(
                f"Warning: failed at "
                f"frame {frame_index}"
            )

            frame_index += sample_step
            continue

        timestamp = (
            frame_index / fps
        )

        mask = segment_e1(
            frame
        )

        components = get_components(
            mask
        )

        # ----------------------------------------------------
        # Save CSV rows
        # ----------------------------------------------------

        for rank, component in enumerate(
            components,
            start=1
        ):

            rows.append(
                {
                    "frame": frame_index,
                    "timestamp": round(
                        timestamp,
                        3
                    ),
                    "candidate_rank": rank,
                    "area": component["area"],
                    "x": component["x"],
                    "y": component["y"],
                    "w": component["w"],
                    "h": component["h"],
                    "cx": round(
                        component["cx"],
                        2
                    ),
                    "cy": round(
                        component["cy"],
                        2
                    ),
                    "aspect_ratio": round(
                        component["aspect_ratio"],
                        4
                    ),
                    "fill_ratio": round(
                        component["fill_ratio"],
                        4
                    ),
                    "circularity": round(
                        component["circularity"],
                        4
                    )
                }
            )

        # ----------------------------------------------------
        # Save representative frame
        #
        # Approximately one frame every 5 seconds.
        # ----------------------------------------------------

        if (
            processed_frames % 20 == 0
        ):

            annotated = annotate_frame(
                frame,
                components,
                frame_index,
                timestamp
            )

            filename = (
                f"frame_{frame_index:06d}_"
                f"{timestamp:07.2f}s.jpg"
            )

            cv2.imwrite(
                str(
                    FRAME_DIR / filename
                ),
                annotated
            )

        processed_frames += 1

        if processed_frames % 20 == 0:

            print(
                f"Processed "
                f"{processed_frames} samples | "
                f"video time "
                f"{timestamp:.2f}s | "
                f"candidates="
                f"{len(components)}"
            )

        frame_index += sample_step

    cap.release()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    unique_frames = len(
        {
            row["frame"]
            for row in rows
        }
    )

    candidate_counts = {}

    for row in rows:

        frame = row["frame"]

        candidate_counts[
            frame
        ] = candidate_counts.get(
            frame,
            0
        ) + 1

    if candidate_counts:

        mean_candidates = (
            np.mean(
                list(
                    candidate_counts.values()
                )
            )
        )

        max_candidates = max(
            candidate_counts.values()
        )

    else:

        mean_candidates = 0
        max_candidates = 0

    print()
    print("=" * 70)
    print("CANDIDATE ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Sampled frames with candidates: "
        f"{unique_frames}"
    )

    print(
        f"Total candidate components: "
        f"{len(rows)}"
    )

    print(
        f"Mean candidates/frame: "
        f"{mean_candidates:.2f}"
    )

    print(
        f"Maximum candidates/frame: "
        f"{max_candidates}"
    )

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Annotated frames: {FRAME_DIR}"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Inspect the annotated frames before "
        "we design the temporal tracker."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()