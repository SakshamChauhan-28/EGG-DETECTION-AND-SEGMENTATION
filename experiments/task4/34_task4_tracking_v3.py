from pathlib import Path
import cv2
import numpy as np
import os
import math
from dataclasses import dataclass


# ============================================================
# TASK 4 - TRACKING V3
# Local refinement + global reacquisition
# Classical CV only
# ============================================================

VIDEO_PATH = (Path(__file__).resolve().parent.parent.parent / r"Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm")

OUTPUT_DIR = "task4_v3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "task4_tracking_v3_test.mp4"
)

# ------------------------------------------------------------
# TEST WINDOW
# ------------------------------------------------------------

START_TIME = 18.0
END_TIME = 34.0

# Process every frame
FRAME_SKIP = 1

# ------------------------------------------------------------
# E1 SEGMENTATION PARAMETERS
# ------------------------------------------------------------

BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100

MIN_AREA = 500

MORPH_KERNEL_SIZE = 7

# ------------------------------------------------------------
# TRACKING PARAMETERS
# ------------------------------------------------------------

# Local search around predicted position
LOCAL_SEARCH_RADIUS = 260

# Global reacquisition search
GLOBAL_MIN_AREA = 3000
GLOBAL_MAX_AREA = 150000

# Egg geometry
MIN_ASPECT = 0.45
MAX_ASPECT = 2.2

MIN_FILL = 0.35

# Maximum number of candidates retained
MAX_CANDIDATES = 12

# Track state
MAX_MISSES_BEFORE_LOST = 8

# Motion smoothing
VELOCITY_ALPHA = 0.65
POSITION_ALPHA = 0.70


# ============================================================
# INITIAL EGG SEEDS
# ============================================================

INITIAL_EGGS = [
    {
        "id": 1,
        "center": np.array([173.0, 620.0], dtype=np.float32),
        "size": np.array([196.0, 196.0], dtype=np.float32),
    },
    {
        "id": 2,
        "center": np.array([455.0, 652.0], dtype=np.float32),
        "size": np.array([210.0, 210.0], dtype=np.float32),
    }
]


# ============================================================
# TRACK CLASS
# ============================================================

@dataclass
class Track:

    track_id: int
    center: np.ndarray
    size: np.ndarray

    velocity: np.ndarray

    state: str = "TRACKING"

    misses: int = 0
    matched: int = 0
    predicted: int = 0
    lost: int = 0
    reacquired: int = 0

    last_score: float = 0.0

    def predict(self):

        self.center = self.center + self.velocity

    def update(self, candidate):

        new_center = candidate["center"]

        displacement = new_center - self.center

        new_velocity = displacement

        self.velocity = (
            VELOCITY_ALPHA * new_velocity
            + (1.0 - VELOCITY_ALPHA) * self.velocity
        )

        self.center = (
            POSITION_ALPHA * new_center
            + (1.0 - POSITION_ALPHA) * self.center
        )

        self.size = (
            0.8 * candidate["size"]
            + 0.2 * self.size
        )

        self.misses = 0
        self.state = "TRACKING"

        self.matched += 1
        self.last_score = candidate["score"]

    def mark_predicted(self):

        self.predicted += 1
        self.misses += 1

        if self.misses > MAX_MISSES_BEFORE_LOST:
            self.state = "LOST"

        if self.state == "LOST":
            self.lost += 1


# ============================================================
# E1 SEGMENTATION
# ============================================================

def e1_segmentation(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    brightness_threshold = np.percentile(
        v,
        BRIGHT_PERCENTILE
    )

    brightness_mask = (
        v >= brightness_threshold
    ).astype(np.uint8) * 255

    saturation_mask = (
        s <= MAX_SATURATION
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(
        brightness_mask,
        saturation_mask
    )

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

    return mask


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_candidates(mask, frame):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    candidates = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_AREA:
            continue

        aspect = w / max(h, 1)

        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        bbox_area = w * h

        fill = area / max(bbox_area, 1)

        if fill < MIN_FILL:
            continue

        cx, cy = centroids[i]

        # Mean brightness inside component
        component_pixels = frame[
            labels == i
        ]

        if len(component_pixels) == 0:
            mean_brightness = 0
        else:
            gray_values = cv2.cvtColor(
                component_pixels.reshape(
                    -1, 1, 3
                ),
                cv2.COLOR_BGR2GRAY
            )

            mean_brightness = float(
                np.mean(gray_values)
            )

        candidates.append(
            {
                "center": np.array(
                    [cx, cy],
                    dtype=np.float32
                ),

                "size": np.array(
                    [w, h],
                    dtype=np.float32
                ),

                "area": float(area),

                "aspect": float(aspect),

                "fill": float(fill),

                "brightness": mean_brightness,

                "bbox": (
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ),

                "label": i
            }
        )

    return candidates


# ============================================================
# LOCAL CANDIDATE FILTER
# ============================================================

def get_local_candidates(
    candidates,
    predicted_center
):

    result = []

    for c in candidates:

        distance = np.linalg.norm(
            c["center"] - predicted_center
        )

        if distance <= LOCAL_SEARCH_RADIUS:

            c2 = c.copy()
            c2["distance"] = float(distance)

            result.append(c2)

    return result


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    candidate,
    predicted_center,
    previous_size
):

    distance = np.linalg.norm(
        candidate["center"] - predicted_center
    )

    # --------------------------------------------------------
    # Spatial score
    # --------------------------------------------------------

    spatial_score = max(
        0.0,
        1.0 - distance / LOCAL_SEARCH_RADIUS
    )

    # --------------------------------------------------------
    # Size consistency
    # --------------------------------------------------------

    previous_area = (
        previous_size[0]
        * previous_size[1]
    )

    area_ratio = (
        candidate["area"]
        / max(previous_area, 1.0)
    )

    size_score = math.exp(
        -abs(math.log(
            max(area_ratio, 1e-6)
        ))
    )

    # --------------------------------------------------------
    # Shape score
    # --------------------------------------------------------

    aspect = candidate["aspect"]

    # Egg-like aspect preference
    shape_score = max(
        0.0,
        1.0 - abs(aspect - 1.15) / 1.0
    )

    # --------------------------------------------------------
    # Fill score
    # --------------------------------------------------------

    fill_score = min(
        1.0,
        candidate["fill"] / 0.85
    )

    # --------------------------------------------------------
    # Brightness score
    # --------------------------------------------------------

    brightness_score = min(
        1.0,
        candidate["brightness"] / 230.0
    )

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        0.40 * spatial_score
        + 0.20 * size_score
        + 0.15 * shape_score
        + 0.10 * fill_score
        + 0.15 * brightness_score
    )

    return score


# ============================================================
# LOCAL MATCH
# ============================================================

def find_local_match(
    track,
    candidates
):

    predicted = track.center.copy()

    local = get_local_candidates(
        candidates,
        predicted
    )

    if not local:
        return None

    best = None
    best_score = -1

    for candidate in local:

        score = score_candidate(
            candidate,
            predicted,
            track.size
        )

        candidate["score"] = score

        if score > best_score:

            best_score = score
            best = candidate

    # Conservative acceptance
    if best_score < 0.42:
        return None

    return best


# ============================================================
# GLOBAL REACQUISITION
# ============================================================

def global_reacquire(
    track,
    candidates
):

    if not candidates:
        return None

    scored = []

    previous_area = (
        track.size[0]
        * track.size[1]
    )

    for candidate in candidates:

        area = candidate["area"]

        if area < GLOBAL_MIN_AREA:
            continue

        if area > GLOBAL_MAX_AREA:
            continue

        # Size compatibility
        area_ratio = (
            area /
            max(previous_area, 1.0)
        )

        size_score = math.exp(
            -abs(math.log(
                max(area_ratio, 1e-6)
            ))
        )

        # Shape
        shape_score = max(
            0.0,
            1.0 -
            abs(candidate["aspect"] - 1.15)
            / 1.0
        )

        # Fill
        fill_score = min(
            1.0,
            candidate["fill"] / 0.85
        )

        # Brightness
        brightness_score = min(
            1.0,
            candidate["brightness"] / 230.0
        )

        score = (
            0.35 * size_score
            + 0.25 * shape_score
            + 0.20 * fill_score
            + 0.20 * brightness_score
        )

        candidate["score"] = score

        scored.append(candidate)

    if not scored:
        return None

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = scored[0]

    # Reacquisition threshold
    if best["score"] < 0.48:
        return None

    return best


# ============================================================
# DRAW
# ============================================================

def draw_track(
    frame,
    track
):

    cx, cy = track.center.astype(int)

    w, h = track.size.astype(int)

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)

    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    if track.state == "TRACKING":

        color = (0, 255, 255)

        label = (
            f"Egg {track.track_id} | TRACKING"
        )

    else:

        color = (0, 0, 255)

        label = (
            f"Egg {track.track_id} | LOST"
        )

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
        7,
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 4 - TRACKING V3")
    print("Local refinement + global reacquisition")
    print("=" * 70)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print()
        print("ERROR: Could not open video:")
        print(VIDEO_PATH)
        print()
        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = total_frames / fps

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"Duration: {duration:.2f}s"
    )

    print(
        f"Testing: {START_TIME:.1f}s -> "
        f"{END_TIME:.1f}s"
    )

    # --------------------------------------------------------
    # Seek to test start
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        START_TIME * 1000
    )

    # --------------------------------------------------------
    # Initialize tracks
    # --------------------------------------------------------

    tracks = []

    for egg in INITIAL_EGGS:

        track = Track(
            track_id=egg["id"],
            center=egg["center"].copy(),
            size=egg["size"].copy(),
            velocity=np.zeros(
                2,
                dtype=np.float32
            )
        )

        tracks.append(track)

        print(
            f"Egg {track.track_id}: "
            f"center={tuple(track.center.astype(int))} "
            f"size={tuple(track.size.astype(int))}"
        )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    frame_index = int(
        START_TIME * fps
    )

    end_frame = int(
        END_TIME * fps
    )

    processed = 0

    last_time = START_TIME

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    while frame_index <= end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        if processed % FRAME_SKIP != 0:

            frame_index += 1
            processed += 1
            continue

        current_time = (
            frame_index / fps
        )

        # ----------------------------------------------------
        # E1 segmentation
        # ----------------------------------------------------

        mask = e1_segmentation(frame)

        candidates = extract_candidates(
            mask,
            frame
        )

        # Keep reasonable number
        candidates.sort(
            key=lambda x: x["area"],
            reverse=True
        )

        candidates = candidates[
            :MAX_CANDIDATES
        ]

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        for track in tracks:

            if track.state != "LOST":

                track.predict()

        # ----------------------------------------------------
        # Match tracks
        # ----------------------------------------------------

        used_candidates = set()

        for track in tracks:

            # ------------------------------------------------
            # First try local tracking
            # ------------------------------------------------

            match = find_local_match(
                track,
                [
                    c for i, c in enumerate(candidates)
                    if i not in used_candidates
                ]
            )

            # ------------------------------------------------
            # If local tracking fails:
            # global reacquisition
            # ------------------------------------------------

            if match is None:

                match = global_reacquire(
                    track,
                    [
                        c for i, c in enumerate(candidates)
                        if i not in used_candidates
                    ]
                )

                if match is not None:

                    track.reacquired += 1

            # ------------------------------------------------
            # Update
            # ------------------------------------------------

            if match is not None:

                # find candidate index
                matched_index = None

                for i, c in enumerate(candidates):

                    if c is match:

                        matched_index = i
                        break

                if matched_index is not None:

                    used_candidates.add(
                        matched_index
                    )

                track.update(match)

            else:

                track.mark_predicted()

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        output = frame.copy()

        # Candidate visualization
        for candidate in candidates:

            x, y, w, h = candidate["bbox"]

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (255, 180, 0),
                1
            )

        for track in tracks:

            draw_track(
                output,
                track
            )

        # Header
        header = (
            f"Task 4 V3 | "
            f"{current_time:.2f}s | "
            f"Candidates: {len(candidates)}"
        )

        cv2.putText(
            output,
            header,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        writer.write(output)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if current_time - last_time >= 2.0:

            print(
                f"time={current_time:.2f}s | "
                f"candidates={len(candidates)} | "
                +
                " | ".join(
                    [
                        f"Egg {t.track_id}="
                        f"{t.state}"
                        for t in tracks
                    ]
                )
            )

            last_time = current_time

        frame_index += 1
        processed += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print("TRACKING V3 COMPLETE")
    print("=" * 70)

    for track in tracks:

        total = (
            track.matched
            + track.predicted
            + track.lost
        )

        ratio = (
            track.matched /
            max(total, 1)
        ) * 100

        print()
        print(
            f"Egg {track.track_id}:"
        )

        print(
            f"  Matched/Tracked: "
            f"{track.matched}"
        )

        print(
            f"  Predicted: "
            f"{track.predicted}"
        )

        print(
            f"  Lost: "
            f"{track.lost}"
        )

        print(
            f"  Reacquired: "
            f"{track.reacquired}"
        )

        print(
            f"  Tracking ratio: "
            f"{ratio:.2f}%"
        )

    print()
    print(
        f"Output: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()