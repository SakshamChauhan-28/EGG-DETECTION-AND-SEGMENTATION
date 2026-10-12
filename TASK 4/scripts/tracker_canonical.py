"""
Task 4: Temporal Multi-Object Video Tracking
Architecture: Grounding DINO Tiny (Zero-Shot Detection) + SORT (Kalman Filter + Hungarian IoU Matching)

Usage:
  python src/task4/tracker.py --video path/to/video.webm --output-video output_tracking.mp4 --metrics-csv metrics.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Allow running directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.utils import xyxy_to_xywh, xywh_to_xyxy, compute_iou_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4: Grounding DINO + SORT Video Egg Tracking")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.webm or .mp4)")
    parser.add_argument("--output-video", type=str, default="tracking_output.mp4", help="Path to save annotated output video")
    parser.add_argument("--metrics-csv", type=str, default="tracking_metrics.csv", help="Path to save tracker diagnostics CSV")
    parser.add_argument("--stride", type=int, default=3, help="Detection interval (run detector every N frames, default: 3)")
    parser.add_argument("--detector-width", type=int, default=960, help="Width for detector inference downscaling (default: 960)")
    parser.add_argument("--dino-model", type=str, default="IDEA-Research/grounding-dino-tiny", help="Grounding DINO model ID")
    parser.add_argument("--prompt", type=str, default="egg.", help="Text prompt for grounding (default: 'egg.')")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.20, help="Text grounding threshold")
    parser.add_argument("--max-age", type=int, default=12, help="Maximum frames to preserve lost track (default: 12)")
    parser.add_argument("--min-hits", type=int, default=2, help="Minimum detection hits before track confirmation (default: 2)")
    parser.add_argument("--iou-threshold", type=float, default=0.15, help="IoU assignment threshold for matching (default: 0.15)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    return parser.parse_args()


class KalmanBox:
    """Constant-velocity Kalman filter for tracking 2D bounding boxes in [cx, cy, w, h]."""
    def __init__(self, box: np.ndarray):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 0.5
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        measurement = xyxy_to_xywh(box).reshape(4, 1)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = measurement
        self.last_prediction = box.copy()

    def predict(self) -> np.ndarray:
        state = self.kf.predict().reshape(-1)
        box = xywh_to_xyxy(state[:4])
        self.last_prediction = box
        return box

    def update(self, box: np.ndarray):
        measurement = xyxy_to_xywh(box).reshape(4, 1)
        self.kf.correct(measurement)
        self.last_prediction = box.copy()


class Track:
    def __init__(self, box: np.ndarray, score: float, track_id: int):
        self.kalman = KalmanBox(box)
        self.track_id = track_id
        self.score = float(score)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.state = "DETECTED"

    @property
    def box(self) -> np.ndarray:
        return self.kalman.last_prediction

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = self.hit_streak if self.time_since_update == 1 else 0
        self.kalman.predict()
        if self.time_since_update > 0:
            self.state = "PREDICTED"
        return self.box

    def update(self, box: np.ndarray, score: float):
        self.kalman.update(box)
        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.state = "TRACKING"


class SORT:
    def __init__(self, max_age: int = 12, min_hits: int = 2, iou_threshold: float = 0.15):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Tuple[np.ndarray, float]]) -> List[Track]:
        predicted_boxes = [t.predict() for t in self.tracks]
        detection_boxes = [d[0] for d in detections]
        detection_scores = [d[1] for d in detections]

        matched_detections = set()
        if predicted_boxes and detection_boxes:
            ious = compute_iou_matrix(np.array(predicted_boxes), np.array(detection_boxes))
            rows, cols = linear_sum_assignment(1.0 - ious)
            for r, c in zip(rows, cols):
                if ious[r, c] >= self.iou_threshold:
                    self.tracks[r].update(detection_boxes[c], detection_scores[c])
                    matched_detections.add(c)

        for det_idx, (box, score) in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks.append(Track(box, score, self.next_id))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks


def run_tracker(video_path: Path, output_video_path: Path, metrics_csv_path: Path,
                dino_model_id: str, prompt: str, box_threshold: float, text_threshold: float,
                stride: int, detector_width: int, max_age: int, min_hits: int, iou_threshold: float, device: str):
    
    print(f"Loading Grounding DINO ({dino_model_id}) on {device}...")
    processor = AutoProcessor.from_pretrained(dino_model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)
    dino_model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {w}x{h} @ {fps:.2f} FPS | {total_frames} frames ({total_frames/fps:.1f}s)")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    tracker = SORT(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    track_stats: Dict[int, Dict[str, Any]] = {}

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_sec = frame_idx / fps
        run_detection = (frame_idx % stride == 0)

        detections = []
        if run_detection:
            # Rescale for detector efficiency
            scale = min(1.0, detector_width / float(w))
            dw, dh = (int(w * scale), int(h * scale)) if scale < 1.0 else (w, h)
            d_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame

            pil_img = Image.fromarray(cv2.cvtColor(d_frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)

            try:
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(dh, dw)]
                )[0]
            except TypeError:
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(dh, dw)]
                )[0]

            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy()

            if scale < 1.0:
                boxes = boxes / scale

            for b, s in zip(boxes, scores):
                if (b[2] - b[0]) * (b[3] - b[1]) >= 2000: # Discard tiny noise
                    detections.append((b, float(s)))

        active_tracks = tracker.update(detections)

        # Telemetry recording and visualization
        vis = frame.copy()
        for t in active_tracks:
            if t.track_id not in track_stats:
                track_stats[t.track_id] = {
                    "matched_frames": 0, "predicted_frames": 0, "first_sec": current_sec, "last_sec": current_sec
                }
            stat = track_stats[t.track_id]
            stat["last_sec"] = current_sec
            if t.time_since_update == 0:
                stat["matched_frames"] += 1
                color = (0, 220, 0) # Green for active detection match
                state_str = "TRACKING"
            else:
                stat["predicted_frames"] += 1
                color = (0, 220, 255) # Yellow for Kalman prediction
                state_str = "PREDICTED"

            x1, y1, x2, y2 = [int(round(v)) for v in t.box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(vis, f"Egg {t.track_id} | {state_str}", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(vis, f"Task 4: Grounding DINO + SORT | {current_sec:.2f}s", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        writer.write(vis)

        if frame_idx % int(fps * 5) == 0:
            print(f"Processed frame {frame_idx}/{total_frames} ({current_sec:.1f}s) | Active tracks: {len(active_tracks)}")
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"Tracking complete in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS). Video saved: {output_video_path}")

    # Save metrics CSV
    import pandas as pd
    rows = []
    for tid, st in sorted(track_stats.items()):
        rows.append({
            "track_id": tid,
            "matched_frames": st["matched_frames"],
            "predicted_frames": st["predicted_frames"],
            "first_time_sec": f"{st['first_sec']:.3f}",
            "last_time_sec": f"{st['last_sec']:.3f}",
        })
    df = pd.DataFrame(rows)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(metrics_csv_path), index=False)
    print(f"Tracking metrics saved to: {metrics_csv_path}")


def main():
    args = parse_args()
    run_tracker(
        video_path=Path(args.video),
        output_video_path=Path(args.output_video),
        metrics_csv_path=Path(args.metrics_csv),
        dino_model_id=args.dino_model,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        stride=args.stride,
        detector_width=args.detector_width,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
