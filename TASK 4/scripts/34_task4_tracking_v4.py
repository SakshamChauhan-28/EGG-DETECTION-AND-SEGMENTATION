from pathlib import Path
import cv2
import numpy as np
import os
import math
from dataclasses import dataclass

# ============================================================
# TASK 4 - TRACKING V4
# Detection-validated classical tracking
#
# GREEN  = real candidate match
# YELLOW = motion prediction (no valid candidate)
# RED    = lost
# ============================================================

VIDEO_PATH = (Path(__file__).resolve().parent / r"Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm")

OUTPUT_DIR = "task4_v4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "task4_tracking_v4_test.mp4")

# Test window
START_TIME = 18.0
END_TIME = 34.0
FRAME_SKIP = 1

# Frozen E1 segmentation
BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7

# Tracking
LOCAL_SEARCH_RADIUS = 220
MAX_CANDIDATES = 15
MAX_PREDICTED_FRAMES = 8

VELOCITY_ALPHA = 0.55
POSITION_ALPHA = 0.65
SIZE_ALPHA = 0.25

# Strict egg gates
MIN_ASPECT = 0.55
MAX_ASPECT = 1.90
MIN_FILL = 0.45
MIN_CIRCULARITY = 0.45
MIN_BRIGHTNESS = 150.0
MIN_WHITENESS = 0.45
MIN_AREA_RATIO = 0.35
MAX_AREA_RATIO = 2.50

LOCAL_MATCH_THRESHOLD = 0.58
GLOBAL_MATCH_THRESHOLD = 0.64

# Initial seeds at ~18 s
INITIAL_EGGS = [
    {"id": 1, "center": np.array([173., 620.], np.float32),
     "size": np.array([196., 196.], np.float32)},
    {"id": 2, "center": np.array([455., 652.], np.float32),
     "size": np.array([210., 210.], np.float32)},
]


@dataclass
class Track:
    track_id: int
    center: np.ndarray
    size: np.ndarray
    velocity: np.ndarray
    state: str = "MATCHED"
    predicted_frames: int = 0
    matched: int = 0
    predicted: int = 0
    lost: int = 0
    reacquired: int = 0
    last_score: float = 0.0

    def predict(self):
        self.center += self.velocity

    def update(self, c, was_lost=False):
        new_center = c["center"]
        displacement = new_center - self.center

        self.velocity = (
            VELOCITY_ALPHA * displacement
            + (1 - VELOCITY_ALPHA) * self.velocity
        )
        self.center = (
            POSITION_ALPHA * new_center
            + (1 - POSITION_ALPHA) * self.center
        )
        self.size = (
            SIZE_ALPHA * c["size"]
            + (1 - SIZE_ALPHA) * self.size
        )

        self.predicted_frames = 0
        self.state = "MATCHED"
        self.matched += 1
        self.last_score = c["score"]

        if was_lost:
            self.reacquired += 1

    def mark_predicted(self):
        self.predicted += 1
        self.predicted_frames += 1

        if self.predicted_frames > MAX_PREDICTED_FRAMES:
            if self.state != "LOST":
                self.lost += 1
            self.state = "LOST"
        else:
            self.state = "PREDICTED"


def e1_segmentation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    threshold = np.percentile(v, BRIGHT_PERCENTILE)
    bright = (v >= threshold).astype(np.uint8) * 255
    low_sat = (s <= MAX_SATURATION).astype(np.uint8) * 255

    mask = cv2.bitwise_and(bright, low_sat)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_candidates(mask, frame):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    candidates = []

    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_AREA:
            continue

        aspect = w / max(h, 1)
        fill = area / max(w * h, 1)

        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        if fill < MIN_FILL:
            continue

        component = labels == i
        cs = s[component]
        cv = v[component]

        if len(cs) == 0:
            continue

        brightness = float(np.mean(cv))
        saturation = float(np.mean(cs))
        whiteness = float(np.mean((cv >= 170) & (cs <= 100)))

        binary = component.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circularity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                contour_area = cv2.contourArea(contour)
                circularity = (
                    4.0 * math.pi * contour_area
                    / (perimeter * perimeter)
                )

        cx, cy = centroids[i]

        candidates.append({
            "center": np.array([cx, cy], np.float32),
            "size": np.array([w, h], np.float32),
            "area": float(area),
            "aspect": float(aspect),
            "fill": float(fill),
            "brightness": brightness,
            "saturation": saturation,
            "whiteness": whiteness,
            "circularity": circularity,
            "bbox": (int(x), int(y), int(w), int(h)),
            "label": i,
        })

    return candidates


def passes_gates(c, previous_size):
    previous_area = previous_size[0] * previous_size[1]
    area_ratio = c["area"] / max(previous_area, 1.0)

    return (
        MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
        and MIN_ASPECT <= c["aspect"] <= MAX_ASPECT
        and c["fill"] >= MIN_FILL
        and c["brightness"] >= MIN_BRIGHTNESS
        and c["whiteness"] >= MIN_WHITENESS
        and c["circularity"] >= MIN_CIRCULARITY
    )


def score_candidate(c, predicted, previous_size, global_mode=False):
    distance = np.linalg.norm(c["center"] - predicted)

    distance_limit = 900.0 if global_mode else LOCAL_SEARCH_RADIUS
    spatial = max(0.0, 1.0 - distance / distance_limit)

    previous_area = previous_size[0] * previous_size[1]
    ratio = c["area"] / max(previous_area, 1.0)

    size_score = math.exp(-abs(math.log(max(ratio, 1e-6))))
    shape_score = max(0.0, 1.0 - abs(c["aspect"] - 1.15) / 0.90)
    fill_score = min(1.0, c["fill"] / 0.85)
    brightness_score = min(1.0, c["brightness"] / 230.0)
    whiteness_score = min(1.0, c["whiteness"] / 0.80)
    circularity_score = min(1.0, c["circularity"] / 0.85)

    return (
        0.15 * spatial
        + 0.20 * size_score
        + 0.15 * shape_score
        + 0.10 * fill_score
        + 0.15 * brightness_score
        + 0.15 * whiteness_score
        + 0.10 * circularity_score
    )


def find_local_match(track, candidates):
    best = None
    best_score = -1.0

    for c in candidates:
        distance = np.linalg.norm(c["center"] - track.center)
        if distance > LOCAL_SEARCH_RADIUS:
            continue
        if not passes_gates(c, track.size):
            continue

        c = c.copy()
        c["score"] = score_candidate(
            c, track.center, track.size, global_mode=False
        )

        if c["score"] > best_score:
            best_score = c["score"]
            best = c

    if best is None or best_score < LOCAL_MATCH_THRESHOLD:
        return None
    return best


def global_reacquire(track, candidates):
    best = None
    best_score = -1.0

    for c in candidates:
        if not passes_gates(c, track.size):
            continue

        c = c.copy()
        c["score"] = score_candidate(
            c, track.center, track.size, global_mode=True
        )

        if c["score"] > best_score:
            best_score = c["score"]
            best = c

    if best is None or best_score < GLOBAL_MATCH_THRESHOLD:
        return None
    return best


def draw_track(frame, track):
    cx, cy = track.center.astype(int)
    w, h = track.size.astype(int)

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    if track.state == "MATCHED":
        color = (0, 255, 0)
        label = f"Egg {track.track_id} | TRACKED"
    elif track.state == "PREDICTED":
        color = (0, 255, 255)
        label = f"Egg {track.track_id} | PREDICTED"
    else:
        color = (0, 0, 255)
        label = f"Egg {track.track_id} | LOST"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, (cx, cy), 7, color, -1)
    cv2.putText(
        frame, label, (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )


def main():
    print("=" * 70)
    print("TASK 4 - TRACKING V4")
    print("Detection-validated tracking")
    print("=" * 70)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video:")
        print(VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")
    print(f"Testing: {START_TIME:.1f}s -> {END_TIME:.1f}s")
    print()
    print("GREEN  = real candidate match")
    print("YELLOW = prediction only")
    print("RED    = lost")
    print()

    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

    tracks = []
    for egg in INITIAL_EGGS:
        t = Track(
            track_id=egg["id"],
            center=egg["center"].copy(),
            size=egg["size"].copy(),
            velocity=np.zeros(2, np.float32),
        )
        tracks.append(t)
        print(
            f"Egg {t.track_id}: "
            f"center={tuple(t.center.astype(int))} "
            f"size={tuple(t.size.astype(int))}"
        )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_index = int(START_TIME * fps)
    end_frame = int(END_TIME * fps)
    processed = 0
    last_report = START_TIME

    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if processed % FRAME_SKIP != 0:
            frame_index += 1
            processed += 1
            continue

        current_time = frame_index / fps

        mask = e1_segmentation(frame)
        candidates = extract_candidates(mask, frame)
        candidates.sort(key=lambda c: c["area"], reverse=True)
        candidates = candidates[:MAX_CANDIDATES]

        for track in tracks:
            if track.state != "LOST":
                track.predict()

        used = set()

        for track in tracks:
            available = [
                c for i, c in enumerate(candidates)
                if i not in used
            ]

            previous_state = track.state

            match = find_local_match(track, available)

            if match is None:
                match = global_reacquire(track, available)

            if match is not None:
                index = next(
                    (
                        i for i, c in enumerate(candidates)
                        if i not in used
                        and np.array_equal(c["center"], match["center"])
                        and c["bbox"] == match["bbox"]
                    ),
                    None
                )

                if index is not None:
                    used.add(index)

                track.update(
                    match,
                    was_lost=(previous_state == "LOST")
                )
            else:
                track.mark_predicted()

        output = frame.copy()

        # Thin cyan boxes = all segmentation candidates.
        for c in candidates:
            x, y, w, h = c["bbox"]
            cv2.rectangle(
                output,
                (x, y), (x + w, y + h),
                (255, 180, 0), 1
            )

        for track in tracks:
            draw_track(output, track)

        header = (
            f"Task 4 V4 | {current_time:.2f}s | "
            f"Candidates: {len(candidates)}"
        )
        cv2.putText(
            output, header, (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2
        )

        writer.write(output)

        if current_time - last_report >= 2.0:
            print(
                f"time={current_time:.2f}s | "
                f"candidates={len(candidates)} | "
                + " | ".join(
                    f"Egg {t.track_id}={t.state}"
                    for t in tracks
                )
            )
            last_report = current_time

        frame_index += 1
        processed += 1

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print("TRACKING V4 COMPLETE")
    print("=" * 70)

    for t in tracks:
        total = t.matched + t.predicted + t.lost
        ratio = 100.0 * t.matched / max(total, 1)

        print()
        print(f"Egg {t.track_id}:")
        print(f"  Matched/Tracked: {t.matched}")
        print(f"  Predicted:       {t.predicted}")
        print(f"  Lost frames:     {t.lost}")
        print(f"  Reacquired:      {t.reacquired}")
        print(f"  Strict match ratio: {ratio:.2f}%")

    print()
    print(f"Output: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
