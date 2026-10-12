from pathlib import Path

import cv2
import numpy as np
import os
import math
from itertools import permutations

# ============================================================
# TASK 4 - TRACKING V6
# Optical-flow + local segmentation + appearance validation
#
# Design principle:
#   NEVER switch to a random egg-like object just because it
#   looks plausible. Motion and appearance must agree.
# ============================================================

VIDEO_PATH =(Path(__file__).resolve().parent / r"Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm")
# If automatic discovery does not find the video, paste the full path here.

OUTPUT_DIR = "task4_v6_output"
START_TIME = 18.0
END_TIME = 34.0

# ------------------------------------------------------------
# Initial manually verified egg positions at 18 s
# ------------------------------------------------------------
INITIAL_TRACKS = [
    {"id": 1, "center": (173, 620), "size": (196, 196)},
    {"id": 2, "center": (455, 652), "size": (210, 210)},
]

# ------------------------------------------------------------
# E1 segmentation (same frozen Task-2 algorithm)
# ------------------------------------------------------------
BRIGHT_PERCENTILE = 65
MAX_SATURATION = 100
MIN_AREA = 500
MORPH_KERNEL_SIZE = 7

# ------------------------------------------------------------
# Local search
# ------------------------------------------------------------
LOCAL_SEARCH_RADIUS = 300

# Candidate filtering is intentionally conservative but not
# as aggressive as E8.
MIN_ASPECT = 0.45
MAX_ASPECT = 2.20
MIN_FILL = 0.30
MAX_CANDIDATES = 12

# ------------------------------------------------------------
# Optical flow
# ------------------------------------------------------------
MAX_FLOW_POINTS = 80
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3

# ------------------------------------------------------------
# Appearance / template matching
# ------------------------------------------------------------
TEMPLATE_SIZE = 128
TEMPLATE_UPDATE_ALPHA = 0.10
MIN_TEMPLATE_SCORE = 0.38

# ------------------------------------------------------------
# Acceptance thresholds
# A candidate must have enough evidence from several signals.
# ------------------------------------------------------------
ACCEPT_SCORE = 0.58
WEAK_SCORE = 0.43
MAX_JUMP_DISTANCE = 170
MAX_SIZE_RATIO = 1.65

# Motion / appearance weights
W_MOTION = 0.35
W_TEMPLATE = 0.35
W_SHAPE = 0.15
W_BRIGHTNESS = 0.15

# ------------------------------------------------------------
# Tracking state
# ------------------------------------------------------------
MAX_PREDICTION_FRAMES = 24
VELOCITY_ALPHA = 0.45
POSITION_ALPHA = 0.70
SIZE_ALPHA = 0.65

# Reinitialise optical-flow points periodically.
FLOW_REFRESH_INTERVAL = 8

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def find_video():
    if VIDEO_PATH is not None and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH

    extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    candidates = []

    for root, _, files in os.walk("."):
        # Do not search inside generated output folders.
        if OUTPUT_DIR in root:
            continue
        for f in files:
            if f.lower().endswith(extensions):
                candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError(
            "Video not found. Set VIDEO_PATH at the top of this script."
        )

    # Prefer a video whose name contains Mocomi/Y4JyV94y2.
    preferred = [
        p for p in candidates
        if "Y4JyV94y2" in os.path.basename(p)
        or "Mocomi" in os.path.basename(p)
    ]

    return preferred[0] if preferred else candidates[0]


def clamp_box(cx, cy, w, h, frame_w, frame_h):
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))

    return x1, y1, x2, y2


def crop_box(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def safe_ratio(a, b):
    if b <= 1e-6:
        return 0.0
    return a / b


def iou_boxes(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return safe_ratio(inter, union)


def resize_gray(img, size=(TEMPLATE_SIZE, TEMPLATE_SIZE)):
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def normalized_histogram(img):
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def histogram_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    value = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def template_similarity(template_gray, crop):
    if template_gray is None or crop is None or crop.size == 0:
        return 0.0

    current = resize_gray(crop)
    if current is None:
        return 0.0

    # Normalized correlation between the previous egg appearance
    # and the candidate crop.
    a = template_gray.astype(np.float32)
    b = current.astype(np.float32)

    a = a - a.mean()
    b = b - b.mean()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 0.0

    corr = float(np.sum(a * b) / denom)
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def detect_e1_candidates(frame):
    """Frozen E1 brightness + saturation segmentation."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness_threshold = np.percentile(gray, BRIGHT_PERCENTILE)

    bright = gray >= brightness_threshold
    low_sat = hsv[:, :, 1] <= MAX_SATURATION

    mask = (bright & low_sat).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_local_candidates(mask, frame, predicted_center, previous_size):
    h, w = mask.shape[:2]
    px, py = predicted_center

    rx1 = max(0, int(px - LOCAL_SEARCH_RADIUS))
    ry1 = max(0, int(py - LOCAL_SEARCH_RADIUS))
    rx2 = min(w, int(px + LOCAL_SEARCH_RADIUS))
    ry2 = min(h, int(py + LOCAL_SEARCH_RADIUS))

    local = mask[ry1:ry2, rx1:rx2]

    if local.size == 0:
        return []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        local, connectivity=8
    )

    candidates = []

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]

        if area < MIN_AREA:
            continue

        aspect = safe_ratio(bw, bh)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        fill = safe_ratio(area, max(1, bw * bh))
        if fill < MIN_FILL:
            continue

        cx_local, cy_local = centroids[i]
        cx = rx1 + float(cx_local)
        cy = ry1 + float(cy_local)

        # Candidate bounding box.
        box = (
            rx1 + int(x),
            ry1 + int(y),
            rx1 + int(x + bw),
            ry1 + int(y + bh),
        )

        crop = crop_box(frame, box)

        if crop.size == 0:
            continue

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        brightness = float(np.mean(gray_crop)) / 255.0
        saturation = float(np.mean(hsv_crop[:, :, 1])) / 255.0

        candidates.append({
            "center": (cx, cy),
            "size": (float(bw), float(bh)),
            "area": float(area),
            "aspect": float(aspect),
            "fill": float(fill),
            "brightness": brightness,
            "saturation": saturation,
            "box": box,
            "crop": crop,
        })

    # Keep candidates closest to the predicted position first.
    candidates.sort(
        key=lambda c: math.hypot(
            c["center"][0] - px,
            c["center"][1] - py
        )
    )

    return candidates[:MAX_CANDIDATES]


def candidate_shape_score(candidate, previous_size):
    pw, ph = previous_size
    cw, ch = candidate["size"]

    size_ratio_w = min(
        safe_ratio(cw, max(pw, 1)),
        safe_ratio(pw, max(cw, 1))
    )
    size_ratio_h = min(
        safe_ratio(ch, max(ph, 1)),
        safe_ratio(ph, max(ch, 1))
    )

    size_score = 0.5 * (size_ratio_w + size_ratio_h)

    # Egg-like candidates tend to have reasonable fill and
    # non-extreme aspect ratios. Keep this weak so it cannot
    # override temporal/appearance evidence.
    aspect_score = 1.0 - min(
        abs(math.log(max(candidate["aspect"], 1e-6))),
        1.0
    )

    fill_score = min(candidate["fill"] / 0.75, 1.0)

    return float(
        0.45 * size_score +
        0.25 * aspect_score +
        0.30 * fill_score
    )


def brightness_score(candidate, reference_brightness):
    current = candidate["brightness"]
    diff = abs(current - reference_brightness)
    return float(np.clip(1.0 - diff / 0.45, 0.0, 1.0))


def motion_score(candidate_center, predicted_center):
    d = math.hypot(
        candidate_center[0] - predicted_center[0],
        candidate_center[1] - predicted_center[1]
    )

    return float(
        np.exp(-0.5 * (d / max(LOCAL_SEARCH_RADIUS * 0.55, 1.0)) ** 2)
    )


def choose_candidate(
    candidates,
    predicted_center,
    previous_box,
    previous_size,
    previous_template,
    previous_hist,
    reference_brightness
):
    if not candidates:
        return None

    scored = []

    for c in candidates:
        d = math.hypot(
            c["center"][0] - predicted_center[0],
            c["center"][1] - predicted_center[1]
        )

        if d > MAX_JUMP_DISTANCE:
            continue

        cw, ch = c["size"]
        pw, ph = previous_size

        ratio_w = max(
            safe_ratio(cw, max(pw, 1)),
            safe_ratio(pw, max(cw, 1))
        )
        ratio_h = max(
            safe_ratio(ch, max(ph, 1)),
            safe_ratio(ph, max(ch, 1))
        )

        if ratio_w > MAX_SIZE_RATIO or ratio_h > MAX_SIZE_RATIO:
            continue

        candidate_template = resize_gray(c["crop"])
        t_score = template_similarity(previous_template, c["crop"])

        h_score = histogram_similarity(
            previous_hist,
            normalized_histogram(c["crop"])
        )

        # Combine two appearance cues conservatively.
        appearance_score = 0.70 * t_score + 0.30 * h_score

        m_score = motion_score(c["center"], predicted_center)
        s_score = candidate_shape_score(c, previous_size)
        b_score = brightness_score(c, reference_brightness)

        total = (
            W_MOTION * m_score +
            W_TEMPLATE * appearance_score +
            W_SHAPE * s_score +
            W_BRIGHTNESS * b_score
        )

        scored.append({
            "candidate": c,
            "score": float(total),
            "motion_score": m_score,
            "template_score": t_score,
            "hist_score": h_score,
            "appearance_score": appearance_score,
            "shape_score": s_score,
            "brightness_score": b_score,
            "distance": d,
        })

    if not scored:
        return None

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    # Strong acceptance:
    # - high total score, OR
    # - reasonable total score AND strong temporal+appearance evidence.
    strong_evidence = (
        best["motion_score"] >= 0.45 and
        best["appearance_score"] >= 0.52 and
        best["template_score"] >= MIN_TEMPLATE_SCORE
    )

    if best["score"] >= ACCEPT_SCORE and strong_evidence:
        return best

    return None


class EggTrack:
    def __init__(self, track_id, center, size, frame):
        self.id = track_id

        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)

        self.velocity = np.zeros(2, dtype=np.float32)

        self.state = "TRACKING"
        self.prediction_frames = 0

        self.matched = 0
        self.predicted = 0
        self.lost = 0
        self.reacquired = 0

        self.last_score = 1.0
        self.last_template_score = 1.0
        self.last_motion_score = 1.0

        self.reference_brightness = self._get_brightness(frame)

        box = self.get_box(frame.shape[1], frame.shape[0])
        crop = crop_box(frame, box)

        self.template = resize_gray(crop)
        self.histogram = normalized_histogram(crop)

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.flow_points = None
        self.flow_frame_counter = 0

        self.last_box = box

    def _get_brightness(self, frame):
        box = self.get_box(frame.shape[1], frame.shape[0])
        crop = crop_box(frame, box)
        if crop.size == 0:
            return 0.75
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) / 255.0

    def get_box(self, frame_w, frame_h):
        return clamp_box(
            float(self.center[0]),
            float(self.center[1]),
            float(self.size[0]),
            float(self.size[1]),
            frame_w,
            frame_h
        )

    def initialize_flow_points(self, gray):
        box = self.last_box
        x1, y1, x2, y2 = box

        # Slightly shrink the ROI to avoid background edges.
        margin_x = max(4, int((x2 - x1) * 0.12))
        margin_y = max(4, int((y2 - y1) * 0.12))

        mask = np.zeros_like(gray)
        cv2.rectangle(
            mask,
            (x1 + margin_x, y1 + margin_y),
            (x2 - margin_x, y2 - margin_y),
            255,
            -1
        )

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_FLOW_POINTS,
            qualityLevel=FLOW_QUALITY,
            minDistance=FLOW_MIN_DISTANCE,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False
        )

        self.flow_points = points
        self.prev_gray = gray.copy()
        self.flow_frame_counter = 0

    def optical_flow_prediction(self, gray):
        if self.flow_points is None or len(self.flow_points) < 4:
            self.initialize_flow_points(self.prev_gray)

        if self.flow_points is None or len(self.flow_points) < 4:
            return self.center.copy(), 0

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.flow_points,
            None,
            winSize=FLOW_WIN_SIZE,
            maxLevel=FLOW_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if next_points is None or status is None:
            self.initialize_flow_points(gray)
            return self.center.copy(), 0

        good_old = self.flow_points[status.ravel() == 1]
        good_new = next_points[status.ravel() == 1]

        if len(good_old) < 4:
            self.initialize_flow_points(gray)
            predicted = self.center + self.velocity
            return predicted, len(good_old)

        # Robust translation/scale/rotation estimate.
        affine, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99
        )

        if affine is not None:
            old_center = self.center.reshape(1, 1, 2)
            transformed = cv2.transform(old_center, affine)[0, 0]
            predicted = transformed.astype(np.float32)

            inlier_count = int(inliers.sum()) if inliers is not None else 0
        else:
            displacement = np.median(
                good_new - good_old,
                axis=0
            ).astype(np.float32)

            predicted = self.center + displacement
            inlier_count = len(good_old)

        # Blend optical flow with the learned velocity.
        predicted = (
            0.75 * predicted +
            0.25 * (self.center + self.velocity)
        )

        self.flow_points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        self.flow_frame_counter += 1

        if (
            self.flow_frame_counter >= FLOW_REFRESH_INTERVAL
            or len(self.flow_points) < 10
        ):
            self.initialize_flow_points(gray)

        return predicted, inlier_count

    def update_from_candidate(self, candidate_result, frame):
        c = candidate_result["candidate"]
        new_center = np.array(c["center"], dtype=np.float32)
        new_size = np.array(c["size"], dtype=np.float32)

        displacement = new_center - self.center

        self.velocity = (
            VELOCITY_ALPHA * displacement +
            (1.0 - VELOCITY_ALPHA) * self.velocity
        )

        self.center = (
            POSITION_ALPHA * new_center +
            (1.0 - POSITION_ALPHA) * self.center
        )

        self.size = (
            SIZE_ALPHA * new_size +
            (1.0 - SIZE_ALPHA) * self.size
        )

        was_prediction = self.state != "TRACKING"

        self.state = "TRACKING"
        self.prediction_frames = 0

        self.matched += 1

        if was_prediction:
            self.reacquired += 1

        self.last_score = candidate_result["score"]
        self.last_template_score = candidate_result["template_score"]
        self.last_motion_score = candidate_result["motion_score"]

        self.last_box = self.get_box(frame.shape[1], frame.shape[0])

        # Slow template adaptation. This prevents one bad frame from
        # immediately replacing the identity template.
        crop = crop_box(frame, self.last_box)

        new_template = resize_gray(crop)
        if new_template is not None and self.template is not None:
            self.template = (
                (1.0 - TEMPLATE_UPDATE_ALPHA) * self.template +
                TEMPLATE_UPDATE_ALPHA * new_template
            ).astype(np.uint8)

        new_hist = normalized_histogram(crop)
        if new_hist is not None and self.histogram is not None:
            self.histogram = (
                (1.0 - TEMPLATE_UPDATE_ALPHA) * self.histogram +
                TEMPLATE_UPDATE_ALPHA * new_hist
            )

        current_brightness = self._get_brightness(frame)
        self.reference_brightness = (
            0.90 * self.reference_brightness +
            0.10 * current_brightness
        )

        self.initialize_flow_points(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )

    def update_prediction(self, predicted_center, frame):
        predicted_center = np.asarray(predicted_center, dtype=np.float32)

        displacement = predicted_center - self.center
        self.velocity = (
            0.20 * displacement +
            0.80 * self.velocity
        )

        self.center = predicted_center

        self.prediction_frames += 1

        if self.prediction_frames <= MAX_PREDICTION_FRAMES:
            self.state = "PREDICTED"
            self.predicted += 1
        else:
            self.state = "LOST"
            self.lost += 1

        self.last_score = 0.0
        self.last_template_score = 0.0
        self.last_motion_score = 0.0

        self.last_box = self.get_box(
            frame.shape[1],
            frame.shape[0]
        )

    def total_frames(self):
        return self.matched + self.predicted + self.lost

    def true_tracking_ratio(self):
        total = self.total_frames()
        return self.matched / total if total else 0.0

    def visible_prediction_coverage(self):
        total = self.total_frames()
        return (
            (self.matched + self.predicted) / total
            if total else 0.0
        )


def assign_candidates(track_results, used_indices):
    """
    Since there are only two tracks, evaluate all one-to-one assignments.
    This prevents both tracks from choosing the same candidate.
    """
    available = [
        i for i in range(len(track_results[0]["scored"]))
        if i not in used_indices
    ]

    return available


def draw_track(frame, track):
    if track.state == "TRACKING":
        color = (0, 255, 0)       # GREEN
    elif track.state == "PREDICTED":
        color = (0, 255, 255)     # YELLOW
    else:
        color = (0, 0, 255)       # RED

    x1, y1, x2, y2 = track.last_box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = (
        f"Egg {track.id} | {track.state} | "
        f"S={track.last_score:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

    # Center point.
    cx, cy = map(int, track.center)
    cv2.circle(frame, (cx, cy), 5, color, -1)


def main():
    video_path = find_video()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_video = os.path.join(
        OUTPUT_DIR,
        "task4_tracking_v6_test.mp4"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print("=" * 70)
    print("TASK 4 - TRACKING V6")
    print("Optical-flow + local segmentation + appearance validation")
    print("=" * 70)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")
    print(f"Testing: {START_TIME:.1f}s -> {END_TIME:.1f}s")
    print("GREEN  = validated true track")
    print("YELLOW = prediction only / temporary occlusion")
    print("RED    = lost")
    print()
    print("V6 principle:")
    print("  Optical flow predicts motion.")
    print("  Local E1 segmentation proposes candidates.")
    print("  Appearance + motion validate identity.")
    print("  No reliable candidate -> prediction, NOT object switching.")
    print()

    for t in INITIAL_TRACKS:
        print(
            f"Egg {t['id']}: center={t['center']} "
            f"size={t['size']}"
        )

    start_frame = int(round(START_TIME * fps))
    end_frame = int(round(END_TIME * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read starting frame.")

    # Initialize tracks.
    tracks = [
        EggTrack(
            t["id"],
            t["center"],
            t["size"],
            first_frame
        )
        for t in INITIAL_TRACKS
    ]

    # We use E1 masks only to generate LOCAL candidates.
    # Task-1 masks are never used here.
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Write initial frame.
    vis = first_frame.copy()

    for track in tracks:
        draw_track(vis, track)

    cv2.putText(
        vis,
        "V6 | Optical Flow + Appearance Validation",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(vis)

    last_report_second = -1

    # Main frame loop.
    for frame_idx in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # E1 segmentation.
        e1_mask = detect_e1_candidates(frame)

        predictions = []

        for track in tracks:
            predicted_center, flow_inliers = track.optical_flow_prediction(
                gray
            )

            # Keep prediction inside frame.
            predicted_center[0] = np.clip(
                predicted_center[0], 0, width - 1
            )
            predicted_center[1] = np.clip(
                predicted_center[1], 0, height - 1
            )

            predictions.append(
                (predicted_center, flow_inliers)
            )

        # Generate candidate lists independently around each prediction.
        candidate_lists = []

        for track, (predicted_center, flow_inliers) in zip(
            tracks, predictions
        ):
            candidates = extract_local_candidates(
                e1_mask,
                frame,
                predicted_center,
                track.size
            )

            scored = []

            for candidate in candidates:
                # Reuse choose_candidate scoring logic by passing a
                # one-element list, then collect the result.
                result = choose_candidate(
                    [candidate],
                    predicted_center,
                    track.last_box,
                    track.size,
                    track.template,
                    track.histogram,
                    track.reference_brightness
                )

                if result is not None:
                    scored.append(result)

            scored.sort(
                key=lambda r: r["score"],
                reverse=True
            )

            candidate_lists.append({
                "predicted": predicted_center,
                "flow_inliers": flow_inliers,
                "scored": scored
            })

        # --------------------------------------------------------
        # One-to-one assignment.
        # Evaluate combinations so Egg 1 and Egg 2 cannot share
        # the same candidate.
        # --------------------------------------------------------
        best_assignment = None
        best_assignment_score = -1.0

        if len(tracks) == 2:
            list0 = candidate_lists[0]["scored"]
            list1 = candidate_lists[1]["scored"]

            options0 = [None] + list0
            options1 = [None] + list1

            for r0 in options0:
                for r1 in options1:
                    if r0 is None and r1 is None:
                        total = 0.0
                        if total > best_assignment_score:
                            best_assignment = (None, None)
                            best_assignment_score = total
                        continue

                    if r0 is not None and r1 is not None:
                        # Do not allow both tracks to use the same candidate.
                        if (
                            r0["candidate"]["box"] ==
                            r1["candidate"]["box"]
                        ):
                            continue

                    total = (
                        (r0["score"] if r0 is not None else 0.0) +
                        (r1["score"] if r1 is not None else 0.0)
                    )

                    if total > best_assignment_score:
                        best_assignment = (r0, r1)
                        best_assignment_score = total
        else:
            best_assignment = tuple(
                cl["scored"][0] if cl["scored"] else None
                for cl in candidate_lists
            )

        # --------------------------------------------------------
        # Update each track.
        # --------------------------------------------------------
        for idx, track in enumerate(tracks):
            result = (
                best_assignment[idx]
                if best_assignment is not None
                else None
            )

            if result is not None:
                # Extra identity protection:
                # if appearance is weak, reject even when total score
                # is acceptable.
                if (
                    result["template_score"] >= MIN_TEMPLATE_SCORE
                    and result["motion_score"] >= 0.35
                    and result["score"] >= ACCEPT_SCORE
                ):
                    track.update_from_candidate(result, frame)
                else:
                    track.update_prediction(
                        predictions[idx][0],
                        frame
                    )
            else:
                track.update_prediction(
                    predictions[idx][0],
                    frame
                )

        # --------------------------------------------------------
        # Visualization
        # --------------------------------------------------------
        vis = frame.copy()

        # Draw local candidate boxes in CYAN.
        # These are proposals, not accepted tracks.
        for cl in candidate_lists:
            for result in cl["scored"]:
                c = result["candidate"]

                x1, y1, x2, y2 = c["box"]

                cv2.rectangle(
                    vis,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    1
                )

        for track in tracks:
            draw_track(vis, track)

        current_time = frame_idx / fps

        cv2.putText(
            vis,
            f"V6 | t={current_time:.2f}s",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(vis)

        # Terminal report every 2 seconds.
        current_second = int(current_time)

        if (
            current_second % 2 == 0
            and current_second != last_report_second
        ):
            last_report_second = current_second

            print(
                f"time={current_time:.2f}s | "
                f"candidates="
                f"{sum(len(cl['scored']) for cl in candidate_lists)} | "
                f"Egg 1={tracks[0].state} | "
                f"Egg 2={tracks[1].state}"
            )

    writer.release()
    cap.release()

    print()
    print("=" * 70)
    print("TRACKING V6 COMPLETE")
    print("=" * 70)

    for track in tracks:
        print(f"Egg {track.id}:")
        print(f"  Matched / TRUE TRACK: {track.matched}")
        print(f"  Predicted / OCCLUSION: {track.predicted}")
        print(f"  Lost: {track.lost}")
        print(f"  Reacquired: {track.reacquired}")
        print(
            f"  True tracking ratio: "
            f"{track.true_tracking_ratio() * 100:.2f}%"
        )
        print(
            f"  Visible/predicted coverage: "
            f"{track.visible_prediction_coverage() * 100:.2f}%"
        )
        print()

    print(f"Output: {output_video}")
    print()
    print("IMPORTANT:")
    print("These percentages measure accepted algorithmic matches.")
    print("They do NOT prove identity correctness.")
    print("Inspect the output video before interpreting the numbers.")


if __name__ == "__main__":
    main()
