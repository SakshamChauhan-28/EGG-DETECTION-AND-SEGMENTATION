import os
"""
TASK 4 - TRACKING PROTOTYPE V2

Classical segmentation-based multi-object tracking.

Pipeline:
    Video frame
        -> frozen E1 segmentation
        -> connected-component candidates
        -> geometric plausibility filtering
        -> temporal prediction
        -> joint assignment of candidates to Egg 1 / Egg 2
        -> persistent IDs

V2 changes from the first prototype:
1. Rejects implausibly large/small components relative to each track.
2. Rejects geometrically implausible components.
3. Uses predicted position from previous velocity.
4. Performs joint two-track assignment so one candidate cannot become both eggs.
5. Does not update a track from a bad frame; it predicts through short occlusions.
6. Reports TRACKED / PREDICTED / LOST separately.

IMPORTANT:
- No Task 1 masks are used.
- No neural model is used.
- E1 parameters are frozen.
"""

from pathlib import Path
import csv
import itertools
import math

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
)

VIDEO_PATH = Path(
    (Path(__file__).resolve().parent / r"Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm")
)

OUTPUT_ROOT = PROJECT_ROOT / "task4_video" / "tracking_v2"
DEBUG_DIR = OUTPUT_ROOT / "debug_frames"
REPORT_DIR = OUTPUT_ROOT / "reports"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN E1 PARAMETERS
# ============================================================

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7
MAX_COMPONENTS = 6


# ============================================================
# V2 TRACKING PARAMETERS
# ============================================================

# Prototype interval: deliberately targets the difficult section
# where V1 drifted.
# Start at the same 10 s point used to initialize V1 so that
# temporal state has time to evolve naturally before the difficult
# 18-34 s section.
START_TIME = 10.0
END_TIME = 34.0

# The difficult section is the main inspection region.
DEBUG_START_TIME = 18.0

# Save one debug frame every N processed frames.
DEBUG_EVERY_FRAMES = 48       # ~2 seconds at 24 FPS

# Initial states from the successful V1 initialization at 10 s.
# These are only initialization coordinates, not ground truth.
INITIAL_TRACKS = {
    1: {"cx": 173.0, "cy": 620.0, "w": 196.0, "h": 196.0},
    2: {"cx": 455.0, "cy": 652.0, "w": 210.0, "h": 210.0},
}

# Geometric gates.
MIN_AREA_RATIO = 0.25
MAX_AREA_RATIO = 2.50

MIN_ASPECT = 0.50
MAX_ASPECT = 1.80

MIN_FILL_RATIO = 0.35
MIN_CIRCULARITY = 0.30

# A candidate whose bbox dimensions grow much beyond the previous
# egg bbox is likely a merged egg + hand/background component.
MAX_SIZE_RATIO = 2.00

# Maximum center displacement accepted for one frame.
MAX_CENTER_DISTANCE = 120.0

# Short occlusion tolerance.
MAX_MISSED_FRAMES = 24

# Velocity update smoothing.
VELOCITY_ALPHA = 0.30


# ============================================================
# FROZEN E1 SEGMENTATION
# ============================================================

def e1_segment(image):
    """
    Frozen E1 algorithm from Task 2.

    Returns:
        binary mask
        candidate component dictionaries
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]

    brightness_threshold = np.percentile(
        lightness,
        BRIGHT_PERCENTILE
    )

    bright_mask = lightness >= brightness_threshold
    low_saturation = saturation <= MAX_SATURATION

    mask = (
        bright_mask & low_saturation
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
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

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    raw_components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        cx, cy = centroids[label]

        component_mask = (
            labels[y:y+h, x:x+w] == label
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = 0.0

        if contours:
            contour = max(
                contours,
                key=cv2.contourArea
            )
            perimeter = cv2.arcLength(
                contour,
                True
            )

        if perimeter > 0:
            circularity = (
                4.0 * math.pi * area
                / (perimeter * perimeter)
            )
        else:
            circularity = 0.0

        fill_ratio = area / float(max(1, w * h))
        aspect_ratio = w / float(max(1, h))

        raw_components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": float(cx),
                "cy": float(cy),
                "aspect": float(aspect_ratio),
                "fill": float(fill_ratio),
                "circularity": float(circularity),
            }
        )

    raw_components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    raw_components = raw_components[:MAX_COMPONENTS]

    final_mask = np.zeros_like(mask)

    for component in raw_components:
        final_mask[
            labels == component["label"]
        ] = 255

    return final_mask, raw_components


# ============================================================
# TRACK STATE
# ============================================================

class Track:
    def __init__(self, track_id, state):
        self.track_id = track_id

        self.cx = state["cx"]
        self.cy = state["cy"]
        self.w = state["w"]
        self.h = state["h"]
        self.area = state["w"] * state["h"]

        self.vx = 0.0
        self.vy = 0.0

        self.missed = 0
        self.total = 0
        self.tracked = 0
        self.predicted = 0
        self.lost = 0

        self.status = "TRACKED"

    def predict(self):
        return (
            self.cx + self.vx,
            self.cy + self.vy
        )

    def update(self, candidate):
        old_cx = self.cx
        old_cy = self.cy

        new_vx = candidate["cx"] - old_cx
        new_vy = candidate["cy"] - old_cy

        self.vx = (
            (1.0 - VELOCITY_ALPHA) * self.vx
            + VELOCITY_ALPHA * new_vx
        )

        self.vy = (
            (1.0 - VELOCITY_ALPHA) * self.vy
            + VELOCITY_ALPHA * new_vy
        )

        self.cx = candidate["cx"]
        self.cy = candidate["cy"]
        self.w = candidate["w"]
        self.h = candidate["h"]
        self.area = candidate["area"]

        self.missed = 0
        self.tracked += 1
        self.status = "TRACKED"

    def mark_missed(self):
        self.missed += 1

        # Continue prediction through a short occlusion.
        self.cx += self.vx
        self.cy += self.vy

        self.vx *= 0.85
        self.vy *= 0.85

        if self.missed <= MAX_MISSED_FRAMES:
            self.status = "PREDICTED"
            self.predicted += 1
        else:
            self.status = "LOST"
            self.lost += 1


# ============================================================
# GEOMETRIC FILTERING
# ============================================================

def candidate_is_plausible(candidate, track):
    """
    Reject candidates that are unlikely to represent the same egg.

    This is the main V2 protection against the V1 failure where
    egg + hand/background became one huge component.
    """

    if not (
        MIN_ASPECT
        <= candidate["aspect"]
        <= MAX_ASPECT
    ):
        return False

    if candidate["fill"] < MIN_FILL_RATIO:
        return False

    if candidate["circularity"] < MIN_CIRCULARITY:
        return False

    previous_area = max(
        1.0,
        track.area
    )

    area_ratio = (
        candidate["area"]
        / previous_area
    )

    if not (
        MIN_AREA_RATIO
        <= area_ratio
        <= MAX_AREA_RATIO
    ):
        return False

    previous_w = max(1.0, track.w)
    previous_h = max(1.0, track.h)

    width_ratio = (
        candidate["w"]
        / previous_w
    )

    height_ratio = (
        candidate["h"]
        / previous_h
    )

    if width_ratio > MAX_SIZE_RATIO:
        return False

    if height_ratio > MAX_SIZE_RATIO:
        return False

    if width_ratio < (1.0 / MAX_SIZE_RATIO):
        return False

    if height_ratio < (1.0 / MAX_SIZE_RATIO):
        return False

    return True


def candidate_cost(candidate, track):
    """
    Lower is better.

    Cost combines:
      - predicted-center distance
      - area consistency
      - shape consistency
    """

    predicted_x, predicted_y = track.predict()

    distance = math.hypot(
        candidate["cx"] - predicted_x,
        candidate["cy"] - predicted_y
    )

    if distance > MAX_CENTER_DISTANCE:
        return None

    distance_cost = (
        distance
        / MAX_CENTER_DISTANCE
    )

    previous_area = max(1.0, track.area)
    area_ratio = (
        candidate["area"]
        / previous_area
    )

    area_cost = abs(
        math.log(
            max(1e-6, area_ratio)
        )
    )
    area_cost = min(area_cost, 2.0) / 2.0

    previous_aspect = (
        track.w
        / max(1.0, track.h)
    )

    shape_cost = abs(
        candidate["aspect"]
        - previous_aspect
    )
    shape_cost = min(
        shape_cost,
        1.0
    )

    return (
        0.55 * distance_cost
        + 0.25 * area_cost
        + 0.20 * shape_cost
    )


# ============================================================
# JOINT ASSIGNMENT
# ============================================================

def assign_candidates(tracks, candidates):
    """
    Jointly assign candidates to the two tracks.

    A candidate can belong to at most one track.

    We explicitly enumerate the small assignment space:
      - each track can select one candidate
      - or select None
    """

    track_ids = sorted(tracks.keys())

    valid_costs = {}

    for track_id in track_ids:
        valid_costs[track_id] = {}

        for index, candidate in enumerate(candidates):
            if not candidate_is_plausible(
                candidate,
                tracks[track_id]
            ):
                continue

            cost = candidate_cost(
                candidate,
                tracks[track_id]
            )

            if cost is not None:
                valid_costs[track_id][index] = cost

    options_1 = [None] + list(
        valid_costs[track_ids[0]].keys()
    )

    options_2 = [None] + list(
        valid_costs[track_ids[1]].keys()
    )

    best_assignment = {
        track_ids[0]: None,
        track_ids[1]: None,
    }

    best_cost = float("inf")

    # A relatively large cost means "prefer no match"
    # rather than accepting a suspicious candidate.
    UNMATCHED_COST = 1.15

    for a in options_1:
        for b in options_2:

            if (
                a is not None
                and b is not None
                and a == b
            ):
                continue

            cost = 0.0

            if a is None:
                cost += UNMATCHED_COST
            else:
                cost += valid_costs[
                    track_ids[0]
                ][a]

            if b is None:
                cost += UNMATCHED_COST
            else:
                cost += valid_costs[
                    track_ids[1]
                ][b]

            if cost < best_cost:
                best_cost = cost

                best_assignment = {
                    track_ids[0]: a,
                    track_ids[1]: b,
                }

    # If the best option is still suspicious, reject it.
    if best_cost >= 2.0 * UNMATCHED_COST:
        return {
            track_ids[0]: None,
            track_ids[1]: None,
        }

    return best_assignment


# ============================================================
# DRAWING
# ============================================================

def draw_track(frame, track):
    cx = int(round(track.cx))
    cy = int(round(track.cy))

    x1 = int(round(
        track.cx - track.w / 2.0
    ))
    y1 = int(round(
        track.cy - track.h / 2.0
    ))
    x2 = int(round(
        track.cx + track.w / 2.0
    ))
    y2 = int(round(
        track.cy + track.h / 2.0
    ))

    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if track.status == "TRACKED":
        color = (0, 255, 0)
        status_text = "TRACKED"
    elif track.status == "PREDICTED":
        color = (0, 255, 255)
        status_text = "PREDICTED"
    else:
        color = (0, 0, 255)
        status_text = "LOST"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.circle(
        frame,
        (cx, cy),
        8,
        color,
        -1
    )

    label = (
        f"Egg {track.track_id} | "
        f"{status_text}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 4 - TRACKING PROTOTYPE V2")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

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

    frame_count = int(
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

    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f}s")
    print(
        f"Processing interval: "
        f"{START_TIME:.1f}s -> {END_TIME:.1f}s"
    )
    print(
        f"Debug inspection interval: "
        f"{DEBUG_START_TIME:.1f}s -> {END_TIME:.1f}s"
    )

    start_frame = int(
        START_TIME * fps
    )
    end_frame = min(
        frame_count - 1,
        int(END_TIME * fps)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    tracks = {
        track_id: Track(
            track_id,
            state
        )
        for track_id, state
        in INITIAL_TRACKS.items()
    }

    print()
    print("Initialized tracks:")

    for track_id, track in tracks.items():
        print(
            f"  Egg {track_id}: "
            f"center=({track.cx:.1f}, {track.cy:.1f}) "
            f"size=({track.w:.1f}, {track.h:.1f})"
        )

    rows = []
    processed = 0

    while True:

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame > end_frame:
            break

        ok, frame = cap.read()

        if not ok:
            break

        current_frame = (
            current_frame
        )

        timestamp = (
            current_frame / fps
            if fps > 0
            else 0.0
        )

        _, candidates = e1_segment(
            frame
        )

        assignment = assign_candidates(
            tracks,
            candidates
        )

        for track_id, track in tracks.items():

            candidate_index = assignment[
                track_id
            ]

            if candidate_index is None:
                track.mark_missed()
            else:
                track.update(
                    candidates[
                        candidate_index
                    ]
                )

            track.total += 1

        # Draw candidates faintly in blue for debugging.
        debug_frame = frame.copy()

        for candidate in candidates:
            x1 = candidate["x"]
            y1 = candidate["y"]
            x2 = x1 + candidate["w"]
            y2 = y1 + candidate["h"]

            cv2.rectangle(
                debug_frame,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                1
            )

        for track in tracks.values():
            draw_track(
                debug_frame,
                track
            )

        # Add time / candidate information.
        cv2.putText(
            debug_frame,
            f"Task 4 V2 | {timestamp:.2f}s | "
            f"Candidates: {len(candidates)}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if (
            timestamp >= DEBUG_START_TIME
            and processed % DEBUG_EVERY_FRAMES == 0
        ):
            output_path = (
                DEBUG_DIR
                / f"frame_{current_frame:06d}_{timestamp:06.2f}s.jpg"
            )

            cv2.imwrite(
                str(output_path),
                debug_frame
            )

        rows.append(
            {
                "frame": current_frame,
                "time_sec": round(
                    timestamp,
                    4
                ),
                "candidates": len(candidates),

                "egg1_status": tracks[1].status,
                "egg1_cx": round(
                    tracks[1].cx,
                    2
                ),
                "egg1_cy": round(
                    tracks[1].cy,
                    2
                ),
                "egg1_missed": tracks[1].missed,

                "egg2_status": tracks[2].status,
                "egg2_cx": round(
                    tracks[2].cx,
                    2
                ),
                "egg2_cy": round(
                    tracks[2].cy,
                    2
                ),
                "egg2_missed": tracks[2].missed,
            }
        )

        processed += 1

        if (
            processed % 120 == 0
        ):
            print(
                f"Processed {processed} frames | "
                f"time={timestamp:.2f}s | "
                f"Egg1={tracks[1].status} | "
                f"Egg2={tracks[2].status}"
            )

    cap.release()

    report_path = (
        REPORT_DIR
        / "tracking_v2_report.csv"
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("TRACKING V2 COMPLETE")
    print("=" * 70)

    for track_id, track in tracks.items():

        tracking_ratio = (
            track.tracked
            / max(1, track.total)
        )

        # Recompute tracked from report state count.
        tracked_count = sum(
            1
            for row in rows
            if row[
                f"egg{track_id}_status"
            ] == "TRACKED"
        )

        predicted_count = sum(
            1
            for row in rows
            if row[
                f"egg{track_id}_status"
            ] == "PREDICTED"
        )

        lost_count = sum(
            1
            for row in rows
            if row[
                f"egg{track_id}_status"
            ] == "LOST"
        )

        tracking_ratio = (
            tracked_count
            / max(1, len(rows))
        )

        print()
        print(f"Egg {track_id}:")
        print(f"  Frames: {len(rows)}")
        print(
            f"  Matched/Tracked: "
            f"{tracked_count}"
        )
        print(
            f"  Predicted: "
            f"{predicted_count}"
        )
        print(
            f"  Lost: "
            f"{lost_count}"
        )
        print(
            f"  True tracking ratio: "
            f"{tracking_ratio * 100:.2f}%"
        )

    print()
    print(
        f"Debug frames: {DEBUG_DIR}"
    )
    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
