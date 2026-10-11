import os
from pathlib import Path
import csv
import math

import cv2
import numpy as np


# ============================================================
# TASK 4 - CLASSICAL VIDEO TRACKING PROTOTYPE
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
)

VIDEO_PATH = PROJECT_ROOT / (
    "Raw and Boiled Egg Spinning - Cool Science Experiment "
    "｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "task4_video"
    / "tracking_prototype"
)

FRAME_DIR = OUTPUT_ROOT / "debug_frames"
REPORT_DIR = OUTPUT_ROOT / "reports"

FRAME_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROTOTYPE INTERVAL
# ============================================================

START_TIME = 10.0
END_TIME = 35.0

# Save one debug frame every N frames.
DEBUG_EVERY = 48       # approximately every 2 seconds at 24 FPS


# ============================================================
# FROZEN E1 PARAMETERS
# ============================================================

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7
MAX_COMPONENTS = 6


# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Initialisation frame.
INIT_TIME = 10.0

# Local search region around predicted egg position.
ROI_SCALE = 2.2

# Minimum/maximum local ROI dimension.
MIN_ROI_SIZE = 220
MAX_ROI_SIZE = 650

# Maximum number of consecutive frames without a detection.
MAX_MISSED_FRAMES = 12

# Candidate must not jump farther than this from prediction.
MAX_CENTROID_DISTANCE = 300.0


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

    threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = (
        lightness >= threshold
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
# COMPONENT EXTRACTION
# ============================================================

def extract_components(mask):

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for label in range(1, count):

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
            max(w, h)
            / max(1, min(w, h))
        )

        # Circularity
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
                    / (perimeter ** 2)
                )

        components.append(
            {
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "fill": fill_ratio,
                "aspect": aspect_ratio,
                "circularity": circularity
            }
        )

    components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    return components[:MAX_COMPONENTS]


# ============================================================
# Hough initialization
# ============================================================

def initialize_tracks(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    height, width = gray.shape

    # The eggs are located in the lower-left/table
    # region around the initialization frame.
    roi_x1 = 0
    roi_x2 = int(width * 0.50)

    roi_y1 = int(height * 0.35)
    roi_y2 = int(height * 0.95)

    roi = gray[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ]

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=35,
        minRadius=70,
        maxRadius=190
    )

    if circles is None:

        raise RuntimeError(
            "Could not initialize the two egg tracks."
        )

    circles = np.round(
        circles[0]
    ).astype(int)

    candidates = []

    for x, y, r in circles:

        # Convert ROI coordinates to full-frame.
        x_full = x + roi_x1
        y_full = y + roi_y1

        candidates.append(
            {
                "cx": float(x_full),
                "cy": float(y_full),
                "r": float(r)
            }
        )

    # Keep circles in the expected egg region.
    candidates = [
        c for c in candidates
        if 50 < c["cx"] < width * 0.55
        and 350 < c["cy"] < height
    ]

    if len(candidates) < 2:

        raise RuntimeError(
            "Hough initialization found fewer "
            "than two plausible egg candidates."
        )

    # Prefer larger circles.
    candidates.sort(
        key=lambda c: c["r"],
        reverse=True
    )

    selected = candidates[:2]

    tracks = []

    for track_id, candidate in enumerate(
        selected,
        start=1
    ):

        r = candidate["r"]

        tracks.append(
            {
                "id": track_id,
                "cx": candidate["cx"],
                "cy": candidate["cy"],
                "prev_cx": candidate["cx"],
                "prev_cy": candidate["cy"],
                "w": 2 * r,
                "h": 2 * r,
                "area": math.pi * r * r,
                "missed": 0,
                "status": "INITIALIZED"
            }
        )

    # Sort left-to-right so IDs are stable.
    tracks.sort(
        key=lambda t: t["cx"]
    )

    # Reassign IDs after sorting.
    for idx, track in enumerate(
        tracks,
        start=1
    ):
        track["id"] = idx

    return tracks


# ============================================================
# Local ROI
# ============================================================

def get_roi(
    frame,
    track
):

    height, width = frame.shape[:2]

    size = max(
        MIN_ROI_SIZE,
        int(
            max(
                track["w"],
                track["h"]
            ) * ROI_SCALE
        )
    )

    size = min(
        MAX_ROI_SIZE,
        size
    )

    # Simple constant-velocity prediction.
    vx = (
        track["cx"]
        - track["prev_cx"]
    )

    vy = (
        track["cy"]
        - track["prev_cy"]
    )

    predicted_x = (
        track["cx"] + vx
    )

    predicted_y = (
        track["cy"] + vy
    )

    half = size // 2

    x1 = max(
        0,
        int(predicted_x - half)
    )

    y1 = max(
        0,
        int(predicted_y - half)
    )

    x2 = min(
        width,
        int(predicted_x + half)
    )

    y2 = min(
        height,
        int(predicted_y + half)
    )

    return (
        x1,
        y1,
        x2,
        y2,
        predicted_x,
        predicted_y
    )


# ============================================================
# Candidate scoring
# ============================================================

def score_candidate(
    candidate,
    track,
    predicted_x,
    predicted_y
):

    dx = (
        candidate["cx"]
        - predicted_x
    )

    dy = (
        candidate["cy"]
        - predicted_y
    )

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    if distance > MAX_CENTROID_DISTANCE:
        return None

    previous_area = max(
        1.0,
        track["area"]
    )

    area_ratio = (
        candidate["area"]
        / previous_area
    )

    # Avoid catastrophic size changes.
    if area_ratio < 0.20:
        return None

    if area_ratio > 5.0:
        return None

    # Prefer compact/egg-like shapes, but do not make
    # shape a hard requirement.
    shape_score = (
        0.45 * min(
            candidate["fill"],
            1.0
        )
        +
        0.35 * min(
            candidate["circularity"],
            1.0
        )
        +
        0.20 * (
            1.0
            / max(
                1.0,
                candidate["aspect"]
            )
        )
    )

    distance_score = max(
        0.0,
        1.0
        - distance / MAX_CENTROID_DISTANCE
    )

    area_score = math.exp(
        -abs(
            math.log(
                max(
                    1e-6,
                    area_ratio
                )
            )
        )
    )

    score = (
        0.50 * distance_score
        +
        0.30 * area_score
        +
        0.20 * shape_score
    )

    return score


# ============================================================
# Update track
# ============================================================

def update_track(
    frame,
    track
):

    (
        x1,
        y1,
        x2,
        y2,
        predicted_x,
        predicted_y
    ) = get_roi(
        frame,
        track
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:

        return False

    mask = segment_e1(
        roi
    )

    components = extract_components(
        mask
    )

    if not components:

        return False

    best = None
    best_score = -1.0

    for candidate in components:

        # Convert ROI coordinates back to full frame.
        candidate = candidate.copy()

        candidate["cx"] += x1
        candidate["cy"] += y1
        candidate["x"] += x1
        candidate["y"] += y1

        score = score_candidate(
            candidate,
            track,
            predicted_x,
            predicted_y
        )

        if (
            score is not None
            and score > best_score
        ):

            best_score = score
            best = candidate

    if best is None:

        return False

    # Update velocity history.
    track["prev_cx"] = track["cx"]
    track["prev_cy"] = track["cy"]

    track["cx"] = best["cx"]
    track["cy"] = best["cy"]

    track["w"] = best["w"]
    track["h"] = best["h"]

    track["area"] = best["area"]

    track["missed"] = 0
    track["status"] = "TRACKED"
    track["score"] = best_score

    return True


# ============================================================
# Draw track
# ============================================================

def draw_track(
    frame,
    track
):

    cx = int(
        round(track["cx"])
    )

    cy = int(
        round(track["cy"])
    )

    w = int(
        track["w"]
    )

    h = int(
        track["h"]
    )

    x1 = max(
        0,
        cx - w // 2
    )

    y1 = max(
        0,
        cy - h // 2
    )

    x2 = min(
        frame.shape[1] - 1,
        cx + w // 2
    )

    y2 = min(
        frame.shape[0] - 1,
        cy + h // 2
    )

    if track["status"] == "TRACKED":

        color = (
            0,
            255,
            0
        )

    else:

        color = (
            0,
            165,
            255
        )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        4
    )

    cv2.circle(
        frame,
        (cx, cy),
        8,
        color,
        -1
    )

    text = (
        f"Egg {track['id']} | "
        f"{track['status']}"
    )

    cv2.putText(
        frame,
        text,
        (
            x1,
            max(
                35,
                y1 - 12
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 4 - TRACKING PROTOTYPE")
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
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

    start_frame = int(
        START_TIME * fps
    )

    end_frame = min(
        total_frames - 1,
        int(END_TIME * fps)
    )

    init_frame_number = int(
        INIT_TIME * fps
    )

    print()
    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Prototype interval: "
        f"{START_TIME}s -> {END_TIME}s"
    )

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        init_frame_number
    )

    success, init_frame = cap.read()

    if not success:

        raise RuntimeError(
            "Could not read initialization frame."
        )

    tracks = initialize_tracks(
        init_frame
    )

    print()
    print("Initialized tracks:")

    for track in tracks:

        print(
            f"  Egg {track['id']}: "
            f"center=("
            f"{track['cx']:.1f}, "
            f"{track['cy']:.1f}) "
            f"size=("
            f"{track['w']:.1f}, "
            f"{track['h']:.1f})"
        )

    # --------------------------------------------------------
    # Reset to start
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    rows = []

    frame_number = start_frame
    processed = 0

    while frame_number <= end_frame:

        success, frame = cap.read()

        if not success:

            break

        timestamp = (
            frame_number / fps
        )

        # ----------------------------------------------------
        # Update both tracks
        # ----------------------------------------------------

        for track in tracks:

            detected = update_track(
                frame,
                track
            )

            if not detected:

                track["missed"] += 1

                if (
                    track["missed"]
                    <= MAX_MISSED_FRAMES
                ):

                    track["status"] = (
                        "TEMP_LOST"
                    )

                else:

                    track["status"] = (
                        "LOST"
                    )

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        annotated = frame.copy()

        for track in tracks:

            draw_track(
                annotated,
                track
            )

            rows.append(
                {
                    "frame": frame_number,
                    "timestamp": round(
                        timestamp,
                        3
                    ),
                    "track_id": track["id"],
                    "cx": round(
                        track["cx"],
                        2
                    ),
                    "cy": round(
                        track["cy"],
                        2
                    ),
                    "width": round(
                        track["w"],
                        2
                    ),
                    "height": round(
                        track["h"],
                        2
                    ),
                    "area": round(
                        track["area"],
                        2
                    ),
                    "status": track["status"],
                    "missed_frames": track["missed"]
                }
            )

        cv2.putText(
            annotated,
            (
                f"Task 4 Prototype | "
                f"{timestamp:.2f}s"
            ),
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save debug frame
        # ----------------------------------------------------

        if (
            processed % DEBUG_EVERY == 0
        ):

            filename = (
                f"frame_{frame_number:06d}_"
                f"{timestamp:07.2f}s.jpg"
            )

            cv2.imwrite(
                str(
                    FRAME_DIR / filename
                ),
                annotated
            )

        processed += 1

        if processed % 120 == 0:

            print(
                f"Processed "
                f"{processed} frames | "
                f"time={timestamp:.2f}s | "
                f"Egg1={tracks[0]['status']} | "
                f"Egg2={tracks[1]['status']}"
            )

        frame_number += 1

    cap.release()

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "tracking_prototype_report.csv"
    )

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "cx",
        "cy",
        "width",
        "height",
        "area",
        "status",
        "missed_frames"
    ]

    with open(
        report_path,
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

    print()
    print("=" * 70)
    print("TRACKING PROTOTYPE COMPLETE")
    print("=" * 70)

    for track_id in [1, 2]:

        track_rows = [
            row
            for row in rows
            if row["track_id"] == track_id
        ]

        tracked = sum(
            row["status"] == "TRACKED"
            for row in track_rows
        )

        temporary = sum(
            row["status"] == "TEMP_LOST"
            for row in track_rows
        )

        lost = sum(
            row["status"] == "LOST"
            for row in track_rows
        )

        total = len(
            track_rows
        )

        print()
        print(
            f"Egg {track_id}:"
        )

        print(
            f"  Frames: {total}"
        )

        print(
            f"  Tracked: {tracked}"
        )

        print(
            f"  Temporarily lost: "
            f"{temporary}"
        )

        print(
            f"  Lost: {lost}"
        )

        if total:

            print(
                f"  Tracking ratio: "
                f"{tracked / total:.2%}"
            )

    print()
    print(
        f"Debug frames: "
        f"{FRAME_DIR}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )

    print()
    print(
        "Inspect the debug frames before "
        "running the complete video."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()