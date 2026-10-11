"""
TASK 4 - Egg Tracking
Grounding DINO Tiny + SORT

Purpose:
    Detect eggs with the already validated Grounding DINO Tiny detector,
    then maintain persistent IDs with a lightweight SORT tracker.

Why this approach:
    The classical Task-2 segmentation algorithm is frozen.
    Task 4 is treated as a tracking-by-detection problem because the
    assignment explicitly allows modifying the segmentation approach
    and/or implementing a tracking algorithm.

Pipeline:
    Video frame -> Grounding DINO ("egg.") -> detections -> SORT -> IDs

IMPORTANT:
    This script does NOT use the Task-2 E1 segmentation algorithm at runtime.
    It also does NOT use SAM, because Task 4 asks primarily for individual
    egg tracking and bounding boxes are sufficient.

FIRST RUN:
    Run the 18-34 second test window. Inspect the generated video.
    If it works, change TEST_ONLY = False for the complete video.
"""

import os
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
)

VIDEO_PATH = PROJECT_ROOT / (
    r"Raw and Boiled Egg Spinning - Cool Science Experiment ｜ "
    r"Educational Videos by Mocomi [Y4JyV94y2_A].webm"
)

OUTPUT_DIR = PROJECT_ROOT / "task4_final_tracking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "task4_final_gdino_sort_full.mp4"
METRICS_CSV = OUTPUT_DIR / "task4_final_tracking_metrics.csv"


# ============================================================
# RUN CONFIG
# ============================================================

# Start with True. This processes only the known 18-34 second
# two-egg window and is much faster for debugging.
TEST_ONLY = False

START_TIME = 18.0
END_TIME = 34.0

# Run Grounding DINO every N frames.
# SORT predicts the positions between detector frames.
DETECTION_INTERVAL = 3

# Grounding DINO configuration already validated in Task 1.
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "egg."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# Discard extremely tiny detections.
MIN_BOX_AREA = 2500

# SORT parameters.
MAX_AGE = 12
MIN_HITS = 2
IOU_THRESHOLD = 0.15

# Resize detector input to reduce computation.
# Boxes are mapped back to original frame coordinates.
DETECTOR_WIDTH = 960

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# BOX UTILITIES
# ============================================================

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
        ],
        dtype=np.float32,
    )


def xywh_to_xyxy(box):
    cx, cy, w, h = box
    return np.array(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        dtype=np.float32,
    )


def iou_matrix(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    t = np.asarray(tracks, dtype=np.float32)
    d = np.asarray(detections, dtype=np.float32)

    result = np.zeros((len(t), len(d)), dtype=np.float32)

    for i, a in enumerate(t):
        ax1, ay1, ax2, ay2 = a
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

        for j, b in enumerate(d):
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih

            ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = aa + ba - inter

            result[i, j] = inter / union if union > 0 else 0.0

    return result


# ============================================================
# SIMPLE KALMAN FILTER
# ============================================================

class KalmanBox:
    """
    Constant-velocity Kalman filter for [cx, cy, w, h].

    State:
        [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self, box):
        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0
        self.kf.transitionMatrix[1, 5] = 1.0
        self.kf.transitionMatrix[2, 6] = 1.0
        self.kf.transitionMatrix[3, 7] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 0.5

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        measurement = xyxy_to_xywh(box).reshape(4, 1)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = measurement

        self.last_prediction = box.copy()

    def predict(self):
        state = self.kf.predict().reshape(-1)
        box = xywh_to_xyxy(state[:4])
        self.last_prediction = box
        return box

    def update(self, box):
        measurement = xyxy_to_xywh(box).reshape(4, 1)
        self.kf.correct(measurement)
        self.last_prediction = box.copy()


# ============================================================
# SORT TRACK
# ============================================================

class Track:
    def __init__(self, box, score, track_id):
        self.kalman = KalmanBox(box)
        self.track_id = track_id

        self.score = float(score)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1

        self.state = "DETECTED"

    @property
    def box(self):
        return self.kalman.last_prediction

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = self.hit_streak if self.time_since_update == 1 else 0

        self.kalman.predict()

        if self.time_since_update > 0:
            self.state = "PREDICTED"

        return self.box

    def update(self, box, score):
        self.kalman.update(box)

        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.state = "TRACKING"


class SORT:
    def __init__(
        self,
        max_age=12,
        min_hits=2,
        iou_threshold=0.15,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of (box, confidence)
        returns current active tracks.
        """

        # Predict every existing track.
        predicted_boxes = []

        for track in self.tracks:
            predicted_boxes.append(track.predict())

        detection_boxes = [x[0] for x in detections]
        detection_scores = [x[1] for x in detections]

        matched = set()
        matched_tracks = set()

        if predicted_boxes and detection_boxes:
            ious = iou_matrix(
                predicted_boxes,
                detection_boxes,
            )

            # Hungarian assignment minimizes cost.
            rows, cols = linear_sum_assignment(1.0 - ious)

            for r, c in zip(rows, cols):
                if ious[r, c] >= self.iou_threshold:
                    self.tracks[r].update(
                        detection_boxes[c],
                        detection_scores[c],
                    )
                    matched_tracks.add(r)
                    matched.add(c)

        # Create new tracks for unmatched detections.
        for det_idx, (box, score) in enumerate(detections):
            if det_idx not in matched:
                self.tracks.append(
                    Track(
                        box=box,
                        score=score,
                        track_id=self.next_id,
                    )
                )
                self.next_id += 1

        # Remove tracks that have been absent too long.
        survivors = []

        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                survivors.append(track)

        self.tracks = survivors

        return self.tracks


# ============================================================
# GROUNDING DINO
# ============================================================

def load_grounding_dino():
    print("=" * 70)
    print("LOADING GROUNDING DINO TINY")
    print("=" * 70)

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


@torch.no_grad()
def detect_eggs(frame_bgr, processor, model):
    """
    Run Grounding DINO on a resized frame and map boxes back
    to the original frame dimensions.
    """

    original_h, original_w = frame_bgr.shape[:2]

    scale = min(
        1.0,
        DETECTOR_WIDTH / float(original_w),
    )

    if scale < 1.0:
        detector_w = int(round(original_w * scale))
        detector_h = int(round(original_h * scale))

        detector_frame = cv2.resize(
            frame_bgr,
            (detector_w, detector_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detector_frame = frame_bgr
        detector_h, detector_w = original_h, original_w

    image_rgb = cv2.cvtColor(
        detector_frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(image_rgb)

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (detector_h, detector_w)
            ],
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (detector_h, detector_w)
            ],
        )

    result = results[0]

    boxes = (
        result["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        result["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    # Map detector-resolution coordinates to original video coordinates.
    if scale < 1.0:
        boxes = boxes / scale

    detections = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(float)

        x1 = max(0.0, min(x1, original_w - 1))
        y1 = max(0.0, min(y1, original_h - 1))
        x2 = max(0.0, min(x2, original_w - 1))
        y2 = max(0.0, min(y2, original_h - 1))

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if area < MIN_BOX_AREA:
            continue

        detections.append(
            (
                np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32,
                ),
                float(score),
            )
        )

    # Highest confidence first.
    detections.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return detections


# ============================================================
# VISUALIZATION
# ============================================================

def draw_track(frame, track, fps):
    x1, y1, x2, y2 = [
        int(round(v))
        for v in track.box
    ]

    # Keep coordinates inside frame.
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if track.time_since_update == 0:
        color = (0, 220, 0)       # green = detector matched
        state = "TRACKING"
    else:
        color = (0, 220, 255)     # yellow = Kalman prediction
        state = "PREDICTED"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    cv2.circle(
        frame,
        (cx, cy),
        7,
        color,
        -1,
    )

    label = (
        f"Egg ID {track.track_id} | "
        f"{state} | "
        f"{track.score:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("TASK 4 - GROUNDING DINO + SORT EGG TRACKING")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Device: {DEVICE}")

    processor, dino_model = load_grounding_dino()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print()
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")

    if TEST_ONLY:
        start_time = START_TIME
        end_time = min(
            END_TIME,
            duration,
        )
    else:
        start_time = 0.0
        end_time = duration

    print(
        f"Processing: {start_time:.2f}s -> "
        f"{end_time:.2f}s"
    )
    print(
        f"Detector interval: every "
        f"{DETECTION_INTERVAL} frame(s)"
    )

    start_frame = int(
        round(start_time * fps)
    )

    end_frame = int(
        round(end_time * fps)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    # MP4 output.
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    tracker = SORT(
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD,
    )

    frame_index = start_frame
    processed = 0
    detection_runs = 0
    total_detections = 0

    # Tracking statistics.
    track_stats = {}

    overall_start = time.perf_counter()

    while frame_index <= end_frame:
        ok, frame = cap.read()

        if not ok:
            break

        current_time = frame_index / fps

        run_detection = (
            processed % DETECTION_INTERVAL == 0
        )

        if run_detection:
            detections = detect_eggs(
                frame,
                processor,
                dino_model,
            )

            detection_runs += 1
            total_detections += len(detections)

        else:
            detections = []

        active_tracks = tracker.update(
            detections
        )

        # Keep only currently relevant tracks.
        for track in active_tracks:
            if track.track_id not in track_stats:
                track_stats[track.track_id] = {
                    "matched_frames": 0,
                    "predicted_frames": 0,
                    "first_time": current_time,
                    "last_time": current_time,
                }

            stat = track_stats[track.track_id]

            if track.time_since_update == 0:
                stat["matched_frames"] += 1
            else:
                stat["predicted_frames"] += 1

            stat["last_time"] = current_time

            draw_track(
                frame,
                track,
                fps,
            )

        # Information overlay.
        cv2.putText(
            frame,
            f"Time: {current_time:.2f}s",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Grounding DINO + SORT",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_index += 1
        processed += 1

        if processed % max(1, int(fps * 2)) == 0:
            elapsed = time.perf_counter() - overall_start
            video_elapsed = (
                processed / fps
                if fps > 0
                else 0
            )

            print(
                f"time={current_time:.2f}s | "
                f"frames={processed} | "
                f"det_runs={detection_runs} | "
                f"last_detections={len(detections)} | "
                f"tracks={len(active_tracks)} | "
                f"runtime={elapsed:.1f}s"
            )

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - overall_start

    print()
    print("=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)

    print(f"Processed frames: {processed}")
    print(f"Detector runs: {detection_runs}")
    print(f"Total detections: {total_detections}")
    print(f"Runtime: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: "
            f"{processed / elapsed:.2f}"
        )

    print()
    print("Track summary:")

    with open(
        METRICS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer_csv = csv.writer(f)

        writer_csv.writerow(
            [
                "track_id",
                "matched_frames",
                "predicted_frames",
                "first_time_sec",
                "last_time_sec",
            ]
        )

        for track_id in sorted(track_stats):
            stat = track_stats[track_id]

            writer_csv.writerow(
                [
                    track_id,
                    stat["matched_frames"],
                    stat["predicted_frames"],
                    f"{stat['first_time']:.3f}",
                    f"{stat['last_time']:.3f}",
                ]
            )

            print(
                f"  ID {track_id}: "
                f"matched={stat['matched_frames']}, "
                f"predicted={stat['predicted_frames']}, "
                f"active={stat['first_time']:.2f}s-"
                f"{stat['last_time']:.2f}s"
            )

    print()
    print(f"Video saved to:\n{OUTPUT_VIDEO}")
    print(f"Metrics saved to:\n{METRICS_CSV}")


if __name__ == "__main__":
    main()
