import os
"""
TASK 4 - V3 / V1.5
Grounding DINO Tiny + identity-preserving SORT-style tracker.

V1 was visually better but unrestricted SORT created many new IDs.
V2 fixed the ID count but became too conservative and lost Egg 1.

V3 keeps V1's permissive association/prediction behavior while
recycling only the two original identities.

Key design:
    - Exactly two persistent egg identities: ID 1 and ID 2.
    - No ID 3+ is ever created.
    - Grounding DINO runs periodically.
    - Kalman prediction runs every frame.
    - Candidate association is deliberately more permissive than V2.
    - If a track is temporarily lost, a later detection can reacquire
      that SAME ID.
    - No SAM or Task-2 E1 segmentation is used at runtime.
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


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

OUTPUT_DIR = PROJECT_ROOT / "task4_v3_identity_tracker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "task4_tracking_v3_test.mp4"
METRICS_CSV = OUTPUT_DIR / "task4_tracking_v3_metrics.csv"


# ============================================================
# RUN CONFIG
# ============================================================

TEST_ONLY = True

START_TIME = 18.0
END_TIME = 34.0

# V1 used 3. Keep this because V1 was visually better.
DETECTION_INTERVAL = 3

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "egg."

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detector resize.
DETECTOR_WIDTH = 960

MIN_BOX_AREA = 2500

# ------------------------------------------------------------
# Association
# ------------------------------------------------------------

# More permissive than V2.
MAX_CENTER_DISTANCE = 350.0

# Track may survive this many frames without a detector match.
MAX_MISSES = 24

# Do not require IoU. During fast movement/partial occlusion,
# the detector box can have little overlap with the Kalman box.
MIN_MATCH_SCORE = 0.20

# ------------------------------------------------------------
# Identity protection
# ------------------------------------------------------------

# If a track has been lost for a long time, allow reacquisition
# over a larger area, but still prefer the predicted position.
REACQUISITION_DISTANCE = 500.0

# ============================================================
# INITIAL EGG BOXES
# ============================================================

INITIAL_TRACKS = [
    {
        "id": 1,
        "box": np.array(
            [75.0, 522.0, 271.0, 718.0],
            dtype=np.float32,
        ),
    },
    {
        "id": 2,
        "box": np.array(
            [350.0, 547.0, 560.0, 757.0],
            dtype=np.float32,
        ),
    },
]


# ============================================================
# BOX FUNCTIONS
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


def box_center(box):
    return np.array(
        [
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        ],
        dtype=np.float32,
    )


def box_size(box):
    return np.array(
        [
            max(1.0, box[2] - box[0]),
            max(1.0, box[3] - box[1]),
        ],
        dtype=np.float32,
    )


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(
        0.0, box[3] - box[1]
    )


def compute_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    union = (
        box_area(a)
        + box_area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return float(inter / union)


# ============================================================
# KALMAN BOX
# ============================================================

class KalmanBox:
    def __init__(self, box):
        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.transitionMatrix = np.eye(
            8,
            dtype=np.float32,
        )

        self.kf.transitionMatrix[0, 4] = 1.0
        self.kf.transitionMatrix[1, 5] = 1.0
        self.kf.transitionMatrix[2, 6] = 1.0
        self.kf.transitionMatrix[3, 7] = 1.0

        self.kf.measurementMatrix = np.zeros(
            (4, 8),
            dtype=np.float32,
        )

        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = (
            np.eye(8, dtype=np.float32) * 1e-2
        )
        self.kf.processNoiseCov[4:, 4:] *= 0.5

        self.kf.measurementNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.1
        )

        self.kf.errorCovPost = (
            np.eye(8, dtype=np.float32)
        )

        self.kf.statePost = np.zeros(
            (8, 1),
            dtype=np.float32,
        )

        self.kf.statePost[:4] = (
            xyxy_to_xywh(box).reshape(4, 1)
        )

        self.last_box = box.copy()

    def predict(self):
        state = self.kf.predict().reshape(-1)
        self.last_box = xywh_to_xyxy(state[:4])
        return self.last_box.copy()

    def update(self, box):
        self.kf.correct(
            xyxy_to_xywh(box).reshape(4, 1)
        )
        self.last_box = box.copy()


# ============================================================
# FIXED ID TRACK
# ============================================================

class Track:
    def __init__(self, track_id, box):
        self.track_id = track_id

        self.kalman = KalmanBox(box)
        self.current_box = box.copy()
        self.last_detection_box = box.copy()

        self.expected_size = box_size(box)

        self.misses = 0
        self.matched = 0
        self.predicted = 0
        self.lost = 0
        self.reacquired = 0

        self.state = "TRACKING"

        self.last_score = 0.0

    def predict(self):
        self.current_box = self.kalman.predict()

        self.misses += 1

        if self.misses > MAX_MISSES:
            self.state = "LOST"
            self.lost += 1
        else:
            self.state = "PREDICTED"
            self.predicted += 1

        return self.current_box.copy()

    def update(self, box, detector_score, was_reacquisition=False):
        self.kalman.update(box)

        self.current_box = box.copy()
        self.last_detection_box = box.copy()

        observed_size = box_size(box)

        self.expected_size = (
            0.20 * observed_size
            + 0.80 * self.expected_size
        )

        self.misses = 0
        self.matched += 1
        self.last_score = float(detector_score)

        if was_reacquisition:
            self.reacquired += 1
            self.state = "REACQUIRED"
        else:
            self.state = "TRACKING"


# ============================================================
# GROUNDING DINO
# ============================================================

def load_detector():
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
        detector_h = original_h
        detector_w = original_w

    rgb = cv2.cvtColor(
        detector_frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(rgb)

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    )

    inputs = {
        k: (
            v.to(DEVICE)
            if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(detector_h, detector_w)],
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(detector_h, detector_w)],
        )

    result = results[0]

    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()

    if scale < 1.0:
        boxes = boxes / scale

    detections = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(float)

        clean_box = np.array(
            [
                max(0.0, min(x1, original_w - 1)),
                max(0.0, min(y1, original_h - 1)),
                max(0.0, min(x2, original_w - 1)),
                max(0.0, min(y2, original_h - 1)),
            ],
            dtype=np.float32,
        )

        if box_area(clean_box) < MIN_BOX_AREA:
            continue

        detections.append(
            {
                "box": clean_box,
                "score": float(score),
            }
        )

    detections.sort(
        key=lambda d: d["score"],
        reverse=True,
    )

    return detections


# ============================================================
# ASSOCIATION
# ============================================================

def pair_score(track, detection):
    predicted = track.current_box
    candidate = detection["box"]

    pc = box_center(predicted)
    cc = box_center(candidate)

    distance = float(
        np.linalg.norm(cc - pc)
    )

    # Normal tracking association.
    max_distance = MAX_CENTER_DISTANCE

    # If track has been lost, permit wider reacquisition.
    if track.misses > MAX_MISSES:
        max_distance = REACQUISITION_DISTANCE

    distance_score = max(
        0.0,
        1.0 - distance / max_distance,
    )

    iou = compute_iou(
        predicted,
        candidate,
    )

    # IoU is deliberately a soft cue.
    iou_score = min(
        1.0,
        iou / 0.40,
    )

    expected_w, expected_h = (
        track.expected_size
    )

    candidate_w, candidate_h = (
        box_size(candidate)
    )

    wr = candidate_w / max(1.0, expected_w)
    hr = candidate_h / max(1.0, expected_h)

    # Soft size score rather than hard rejection.
    size_error = (
        abs(np.log(max(wr, 1e-6)))
        + abs(np.log(max(hr, 1e-6)))
    )

    size_score = float(
        np.exp(-0.75 * size_error)
    )

    detector_score = float(
        detection["score"]
    )

    # This is intentionally close to V1's permissive behavior.
    score = (
        0.45 * distance_score
        + 0.25 * iou_score
        + 0.15 * size_score
        + 0.15 * detector_score
    )

    if distance > max_distance:
        return -1.0

    return float(score)


def associate(tracks, detections):
    if not detections:
        return []

    scores = np.full(
        (len(tracks), len(detections)),
        -1.0,
        dtype=np.float32,
    )

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            scores[i, j] = pair_score(
                track,
                detection,
            )

    # Hungarian assignment.
    cost = 1.0 - np.maximum(scores, 0.0)

    rows, cols = linear_sum_assignment(cost)

    matches = []

    for r, c in zip(rows, cols):
        if scores[r, c] >= MIN_MATCH_SCORE:
            matches.append(
                (
                    r,
                    c,
                    float(scores[r, c]),
                )
            )

    return matches


# ============================================================
# VISUALIZATION
# ============================================================

def draw_track(frame, track):
    x1, y1, x2, y2 = [
        int(round(v))
        for v in track.current_box
    ]

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if track.state == "TRACKING":
        color = (0, 220, 0)
        state = "TRACKING"

    elif track.state == "REACQUIRED":
        color = (255, 180, 0)
        state = "REACQUIRED"

    elif track.state == "PREDICTED":
        color = (0, 220, 255)
        state = "PREDICTED"

    else:
        color = (0, 0, 255)
        state = "LOST"

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
        f"{state}"
    )

    if track.state in ("TRACKING", "REACQUIRED"):
        label += f" | DINO {track.last_score:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
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
    print("TASK 4 V3 - IDENTITY-PRESERVING TRACKING")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    processor, model = load_detector()

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (
        total_frames / fps
        if fps > 0 else 0.0
    )

    print()
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")

    if TEST_ONLY:
        start_time = START_TIME
        end_time = min(END_TIME, duration)
    else:
        start_time = 0.0
        end_time = duration

    print(
        f"Processing: {start_time:.2f}s -> "
        f"{end_time:.2f}s"
    )

    print(
        f"DINO every {DETECTION_INTERVAL} frames"
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

    tracks = [
        Track(
            track_id=item["id"],
            box=item["box"],
        )
        for item in INITIAL_TRACKS
    ]

    frame_index = start_frame
    processed = 0
    detector_runs = 0
    total_detections = 0
    accepted_matches = 0

    runtime_start = time.perf_counter()

    while frame_index <= end_frame:
        ok, frame = cap.read()

        if not ok:
            break

        current_time = frame_index / fps

        # Predict all fixed identities every frame.
        for track in tracks:
            track.predict()

        detections = []

        if processed % DETECTION_INTERVAL == 0:
            detections = detect_eggs(
                frame,
                processor,
                model,
            )

            detector_runs += 1
            total_detections += len(detections)

            matches = associate(
                tracks,
                detections,
            )

            matched_track_indices = set()

            for track_idx, det_idx, score in matches:
                track = tracks[track_idx]

                was_reacquisition = (
                    track.misses > 0
                )

                track.update(
                    detections[det_idx]["box"],
                    detections[det_idx]["score"],
                    was_reacquisition=was_reacquisition,
                )

                matched_track_indices.add(
                    track_idx
                )

                accepted_matches += 1

        # Draw only the two persistent identities.
        for track in tracks:
            draw_track(
                frame,
                track,
            )

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
            "Grounding DINO + Identity Tracker V3",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_index += 1
        processed += 1

        if processed % max(
            1,
            int(fps * 2),
        ) == 0:
            elapsed = (
                time.perf_counter()
                - runtime_start
            )

            print(
                f"time={current_time:.2f}s | "
                f"frames={processed} | "
                f"det_runs={detector_runs} | "
                f"detections={len(detections)} | "
                f"ID1={tracks[0].state} | "
                f"ID2={tracks[1].state} | "
                f"runtime={elapsed:.1f}s"
            )

    cap.release()
    writer.release()

    elapsed = (
        time.perf_counter()
        - runtime_start
    )

    print()
    print("=" * 70)
    print("V3 COMPLETE")
    print("=" * 70)

    print(f"Processed frames: {processed}")
    print(f"Detector runs: {detector_runs}")
    print(f"Total detections: {total_detections}")
    print(f"Accepted matches: {accepted_matches}")
    print(f"Runtime: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: "
            f"{processed / elapsed:.2f}"
        )

    print()
    print("TRACK SUMMARY:")

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
                "lost_frames",
                "reacquired",
                "total_processed_frames",
            ]
        )

        for track in tracks:
            writer_csv.writerow(
                [
                    track.track_id,
                    track.matched,
                    track.predicted,
                    track.lost,
                    track.reacquired,
                    processed,
                ]
            )

            print(
                f"  Egg ID {track.track_id}: "
                f"matched={track.matched}, "
                f"predicted={track.predicted}, "
                f"lost={track.lost}, "
                f"reacquired={track.reacquired}"
            )

    print()
    print(f"Video: {OUTPUT_VIDEO}")
    print(f"Metrics: {METRICS_CSV}")


if __name__ == "__main__":
    main()
