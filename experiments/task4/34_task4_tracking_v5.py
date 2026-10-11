import cv2
import numpy as np
import os
import math
from dataclasses import dataclass, field
from itertools import permutations

VIDEO_PATH = None
OUTPUT_DIR = "task4_v5_output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "task4_tracking_v5_test.mp4")
START_TIME = 18.0
END_TIME = 34.0
FRAME_SKIP = 1

# Frozen E1 segmentation
BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7

# Candidate filtering
MIN_ASPECT = 0.45
MAX_ASPECT = 2.2
MIN_FILL = 0.35
MAX_CANDIDATES = 15
GLOBAL_MIN_AREA = 2500
GLOBAL_MAX_AREA = 90000

# V5 tracking
LOCAL_SEARCH_RADIUS = 330.0
GLOBAL_REACQUIRE_RADIUS = 700.0
HIGH_CONFIDENCE = 0.55
LOW_CONFIDENCE = 0.32
MAX_PREDICTION_FRAMES = 18
MAX_SPEED = 35.0
VELOCITY_ALPHA = 0.45
POSITION_ALPHA = 0.70
SIZE_ALPHA = 0.65
MAX_SIZE_RATIO = 2.2
MAX_JUMP_DISTANCE = 330.0
BRIGHTNESS_REFERENCE = 220.0

INITIAL_EGGS = [
    {"id": 1, "center": np.array([173.0, 620.0], dtype=np.float32), "size": np.array([196.0, 196.0], dtype=np.float32)},
    {"id": 2, "center": np.array([455.0, 652.0], dtype=np.float32), "size": np.array([210.0, 210.0], dtype=np.float32)},
]

@dataclass
class Track:
    track_id: int
    center: np.ndarray
    size: np.ndarray
    velocity: np.ndarray
    state: str = "MATCHED"
    misses: int = 0
    matched: int = 0
    predicted: int = 0
    lost: int = 0
    reacquired: int = 0
    last_score: float = 1.0
    predicted_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def predict_position(self):
        self.predicted_center = self.center + self.velocity
        return self.predicted_center.copy()

    def update(self, candidate, score, frame_shape):
        new_center = candidate["center"].astype(np.float32)
        displacement = new_center - self.center
        distance = float(np.linalg.norm(displacement))
        if distance > MAX_JUMP_DISTANCE:
            displacement *= MAX_JUMP_DISTANCE / max(distance, 1e-6)
        self.velocity = VELOCITY_ALPHA * displacement + (1.0 - VELOCITY_ALPHA) * self.velocity
        speed = float(np.linalg.norm(self.velocity))
        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / speed
        self.center = POSITION_ALPHA * new_center + (1.0 - POSITION_ALPHA) * self.center
        self.size = SIZE_ALPHA * candidate["size"] + (1.0 - SIZE_ALPHA) * self.size
        h, w = frame_shape[:2]
        self.center[0] = float(np.clip(self.center[0], 0, w - 1))
        self.center[1] = float(np.clip(self.center[1], 0, h - 1))
        self.predicted_center = self.center.copy()
        was_reacquired = self.state in ("PREDICTED", "LOST")
        self.state = "MATCHED"
        self.misses = 0
        self.matched += 1
        self.last_score = float(score)
        if was_reacquired:
            self.reacquired += 1

    def mark_prediction(self):
        self.predicted_center = self.predict_position()
        self.predicted += 1
        self.misses += 1
        if self.misses > MAX_PREDICTION_FRAMES:
            self.state = "LOST"
            self.lost += 1
        else:
            self.state = "PREDICTED"

def find_video():
    if VIDEO_PATH is not None:
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(VIDEO_PATH)
        return VIDEO_PATH
    exts = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
    candidates = [p for p in os.listdir(".") if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts and "task4_tracking_v5" not in p.lower()]
    if not candidates:
        raise FileNotFoundError("No video found in current directory. Set VIDEO_PATH at top of script.")
    candidates.sort(key=lambda x: ("raw and boiled" not in x.lower(), len(x)))
    return candidates[0]

def e1_segmentation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    threshold = np.percentile(v, BRIGHT_PERCENTILE)
    bright = (v >= threshold).astype(np.uint8) * 255
    low_sat = (s <= MAX_SATURATION).astype(np.uint8) * 255
    mask = cv2.bitwise_and(bright, low_sat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def extract_candidates(mask, frame):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < GLOBAL_MIN_AREA or area > GLOBAL_MAX_AREA or w < 35 or h < 35:
            continue
        aspect = w / max(h, 1)
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue
        bbox_area = max(w * h, 1)
        fill = area / bbox_area
        if fill < MIN_FILL:
            continue
        cmask = labels[y:y+h, x:x+w] == i
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sat = hsv[y:y+h, x:x+w, 1]
        mean_brightness = float(gray[cmask].mean()) if cmask.any() else 0.0
        mean_sat = float(sat[cmask].mean()) if cmask.any() else 255.0
        shape_score = 1.0 - min(abs(aspect - 1.0), 1.0)
        fill_score = float(np.clip(fill / 0.75, 0.0, 1.0))
        brightness_score = float(np.clip(mean_brightness / BRIGHTNESS_REFERENCE, 0.0, 1.0))
        saturation_score = float(np.clip(1.0 - mean_sat / 180.0, 0.0, 1.0))
        intrinsic = 0.30 * shape_score + 0.30 * fill_score + 0.25 * brightness_score + 0.15 * saturation_score
        candidates.append({
            "bbox": (x, y, w, h), "center": np.array([x+w/2.0, y+h/2.0], dtype=np.float32),
            "size": np.array([float(w), float(h)], dtype=np.float32), "area": area,
            "aspect": aspect, "fill": fill, "brightness": mean_brightness,
            "intrinsic_score": intrinsic,
        })
    candidates.sort(key=lambda c: (c["intrinsic_score"], c["area"]), reverse=True)
    return candidates[:MAX_CANDIDATES]

def candidate_score(track, candidate, predicted_center):
    distance = float(np.linalg.norm(candidate["center"] - predicted_center))
    distance_score = float(np.exp(-((distance / LOCAL_SEARCH_RADIUS) ** 2)))
    old_w, old_h = track.size
    new_w, new_h = candidate["size"]
    ratio_error = abs(math.log(max(new_w / max(old_w, 1.0), 1e-6))) + abs(math.log(max(new_h / max(old_h, 1.0), 1e-6)))
    size_score = float(np.exp(-ratio_error))
    size_ratio = max(new_w/max(old_w,1.0), old_w/max(new_w,1.0), new_h/max(old_h,1.0), old_h/max(new_h,1.0))
    if size_ratio > MAX_SIZE_RATIO:
        size_score *= 0.25
    score = 0.45 * distance_score + 0.20 * size_score + 0.35 * candidate["intrinsic_score"]
    if distance > LOCAL_SEARCH_RADIUS:
        score *= 0.55
    if distance > MAX_JUMP_DISTANCE:
        score *= 0.45
    return float(np.clip(score, 0.0, 1.0))

def assign_tracks(tracks, candidates):
    if not candidates:
        return {}
    predicted = [t.predict_position() for t in tracks]
    scores = np.zeros((len(tracks), len(candidates)), dtype=np.float32)
    for ti, track in enumerate(tracks):
        for ci, cand in enumerate(candidates):
            scores[ti, ci] = candidate_score(track, cand, predicted[ti])
    best = {}
    best_total = -1.0
    if len(candidates) >= len(tracks):
        for perm in permutations(range(len(candidates)), len(tracks)):
            total = sum(float(scores[ti, ci]) for ti, ci in enumerate(perm))
            if total > best_total:
                best_total = total
                best = {ti: ci for ti, ci in enumerate(perm)}
    elif len(candidates) == 1:
        best_t = int(np.argmax(scores[:, 0]))
        best = {best_t: 0}
    return {ti: (ci, float(scores[ti, ci])) for ti, ci in best.items()}

def try_global_reacquisition(track, candidates):
    if not candidates:
        return None
    predicted = track.predict_position()
    best, best_score = None, -1.0
    for c in candidates:
        d = float(np.linalg.norm(c["center"] - predicted))
        if d > GLOBAL_REACQUIRE_RADIUS:
            continue
        distance_score = float(np.exp(-((d / 500.0) ** 2)))
        score = 0.45 * distance_score + 0.55 * c["intrinsic_score"]
        if score > best_score:
            best_score, best = score, c
    if best is not None and best_score >= 0.60:
        return best, best_score
    return None

def draw_track(frame, track):
    if track.state == "MATCHED":
        color, label, center = (0,255,0), f"Egg {track.track_id} | TRACKED", track.center
    elif track.state == "PREDICTED":
        color, label, center = (0,255,255), f"Egg {track.track_id} | PREDICTED", track.predicted_center
    else:
        color, label, center = (0,0,255), f"Egg {track.track_id} | LOST", track.predicted_center
    cx, cy = center.astype(int)
    w, h = track.size.astype(int)
    x1, y1 = int(cx-w/2), int(cy-h/2)
    x2, y2 = int(cx+w/2), int(cy+h/2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
    cv2.circle(frame, (cx,cy), 7, color, -1)
    cv2.putText(frame, label, (max(5,x1), max(30,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    cv2.putText(frame, f"conf={track.last_score:.2f}", (max(5,x1), min(frame.shape[0]-10,y2+25)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = find_video()
    print("="*70)
    print("TASK 4 - TRACKING V5")
    print("Confidence-based hybrid tracker")
    print("="*70)
    print(f"Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f}s")
    print(f"Testing: {START_TIME:.1f}s -> {END_TIME:.1f}s")
    print("GREEN  = real candidate match / TRUE TRACK")
    print("YELLOW = prediction only / temporary occlusion")
    print("RED    = lost")
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME*1000)
    tracks = []
    for egg in INITIAL_EGGS:
        t = Track(egg["id"], egg["center"].copy(), egg["size"].copy(), np.zeros(2,dtype=np.float32))
        t.predicted_center = t.center.copy()
        tracks.append(t)
        print(f"Egg {t.track_id}: center={tuple(t.center.astype(int))} size={tuple(t.size.astype(int))}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width,height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError(f"Could not create output: {OUTPUT_VIDEO}")
    frame_index, end_frame = int(START_TIME*fps), int(END_TIME*fps)
    processed, last_report = 0, START_TIME
    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        if processed % FRAME_SKIP != 0:
            frame_index += 1; processed += 1; continue
        current_time = frame_index/fps
        mask = e1_segmentation(frame)
        candidates = extract_candidates(mask, frame)
        for t in tracks:
            if t.state != "LOST": t.predict_position()
        assignments = assign_tracks(tracks, candidates)
        used = set()
        for ti, track in enumerate(tracks):
            accepted = False
            a = assignments.get(ti)
            if a is not None:
                ci, score = a
                if ci not in used and score >= HIGH_CONFIDENCE:
                    track.update(candidates[ci], score, frame.shape)
                    used.add(ci); accepted = True
            if not accepted:
                available = [c for i,c in enumerate(candidates) if i not in used]
                reacquired = try_global_reacquisition(track, available)
                if reacquired is not None:
                    c, score = reacquired
                    try:
                        ci = candidates.index(c)
                    except ValueError:
                        ci = -1
                    track.update(c, score, frame.shape)
                    track.reacquired += 1
                    if ci >= 0: used.add(ci)
                    accepted = True
            if not accepted:
                track.mark_prediction()
        output = frame.copy()
        for c in candidates:
            x,y,w,h = c["bbox"]
            cv2.rectangle(output, (x,y), (x+w,y+h), (255,180,0), 1)
        for t in tracks: draw_track(output,t)
        cv2.putText(output, f"Task 4 V5 | {current_time:.2f}s | Candidates: {len(candidates)}", (25,40), cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        cv2.putText(output, "GREEN=TRACKED  YELLOW=PREDICTED  RED=LOST", (25,70), cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),2)
        writer.write(output)
        if current_time-last_report >= 2.0:
            print(f"time={current_time:.2f}s | candidates={len(candidates)} | " + " | ".join(f"Egg {t.track_id}={t.state}" for t in tracks))
            last_report=current_time
        frame_index += 1; processed += 1
    cap.release(); writer.release()
    print("\n"+"="*70)
    print("TRACKING V5 COMPLETE")
    print("="*70)
    for t in tracks:
        total=t.matched+t.predicted+t.lost
        print(f"\nEgg {t.track_id}:")
        print(f"  Matched / TRUE TRACK: {t.matched}")
        print(f"  Predicted / OCCLUSION: {t.predicted}")
        print(f"  Lost: {t.lost}")
        print(f"  Reacquired: {t.reacquired}")
        print(f"  True tracking ratio: {100*t.matched/max(total,1):.2f}%")
        print(f"  Visible/predicted coverage: {100*(t.matched+t.predicted)/max(total,1):.2f}%")
    print(f"\nOutput: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
