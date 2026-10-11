
import cv2
import numpy as np
import os
import math

# ============================================================
# TASK 4 - TRACKING V7
# Template/shape tracking + optical-flow prediction
#
# Major change from V5/V6:
#   V7 DOES NOT depend on E1 connected components.
#
# E1 produced large merged regions around eggs, causing V6 to
# reject real eggs. V7 instead searches directly for the egg's
# visual appearance inside a local region around the predicted
# position.
#
# Signals:
#   1. Optical-flow motion prediction
#   2. Multi-scale grayscale template matching
#   3. Edge/shape template matching
#   4. HSV histogram similarity
#   5. Position / size consistency
#
# If confidence is low:
#   -> prediction only
#   -> never switch to an arbitrary bright object.
# ============================================================

VIDEO_PATH = None
OUTPUT_DIR = "task4_v7_output"

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
# Local search
# ------------------------------------------------------------
SEARCH_RADIUS = 280

# Multi-scale template search.
SCALES = [0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25]

# ------------------------------------------------------------
# Template processing
# ------------------------------------------------------------
TEMPLATE_SIZE = 128
EDGE_LOW = 50
EDGE_HIGH = 150

# ------------------------------------------------------------
# Acceptance
# ------------------------------------------------------------
ACCEPT_SCORE = 0.56
STRONG_SCORE = 0.66

# A candidate too far from the optical-flow prediction is rejected.
MAX_CENTER_DISTANCE = 190

# Prevent extreme scale jumps.
MAX_SIZE_RATIO = 1.55

# ------------------------------------------------------------
# Score weights
# ------------------------------------------------------------
W_GRAY = 0.40
W_EDGE = 0.30
W_HIST = 0.20
W_MOTION = 0.10

# ------------------------------------------------------------
# Template update
# Slow adaptation prevents the tracker from learning the hand.
# ------------------------------------------------------------
TEMPLATE_UPDATE_ALPHA = 0.06
HIST_UPDATE_ALPHA = 0.05

# Only update the appearance template when confidence is strong.
MIN_TEMPLATE_UPDATE_SCORE = 0.68

# ------------------------------------------------------------
# Optical flow
# ------------------------------------------------------------
MAX_FLOW_POINTS = 80
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3
FLOW_REFRESH_INTERVAL = 6

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
MAX_PREDICTION_FRAMES = 30
VELOCITY_ALPHA = 0.45

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
DRAW_SEARCH_REGION = True


# ============================================================
# Utility
# ============================================================

def find_video():
    if VIDEO_PATH is not None and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH

    extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    candidates = []

    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if not d.startswith("task4_v7_output")
        ]

        for f in files:
            if f.lower().endswith(extensions):
                candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError(
            "Video not found. Set VIDEO_PATH at the top of the script."
        )

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

    x1 = max(0, min(frame_w - 2, x1))
    y1 = max(0, min(frame_h - 2, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))

    return x1, y1, x2, y2


def crop_box(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def safe_ratio(a, b):
    return a / b if abs(b) > 1e-8 else 0.0


def resize_gray(image, size=(TEMPLATE_SIZE, TEMPLATE_SIZE)):
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return cv2.resize(
        gray,
        size,
        interpolation=cv2.INTER_AREA
    )


def normalize_gray(image):
    if image is None or image.size == 0:
        return None

    image = image.astype(np.float32)

    mean = image.mean()
    std = image.std()

    if std < 1e-6:
        return np.zeros_like(image)

    return (image - mean) / std


def edge_image(image):
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return cv2.Canny(gray, EDGE_LOW, EDGE_HIGH)


def resize_edge(image, size=(TEMPLATE_SIZE, TEMPLATE_SIZE)):
    edges = edge_image(image)

    if edges is None:
        return None

    return cv2.resize(
        edges,
        size,
        interpolation=cv2.INTER_NEAREST
    )


def histogram(image):
    if image is None or image.size == 0:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256]
    )

    cv2.normalize(hist, hist)

    return hist


def histogram_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0

    correlation = cv2.compareHist(
        h1,
        h2,
        cv2.HISTCMP_CORREL
    )

    return float(
        np.clip(
            (correlation + 1.0) / 2.0,
            0.0,
            1.0
        )
    )


def normalized_template_score(template, candidate):
    if template is None or candidate is None:
        return 0.0

    a = normalize_gray(template)
    b = normalize_gray(candidate)

    if a is None or b is None:
        return 0.0

    score = float(np.mean(a * b))

    # Convert approximately [-1, 1] -> [0, 1].
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def edge_template_score(template_edges, candidate_edges):
    if template_edges is None or candidate_edges is None:
        return 0.0

    a = template_edges.astype(np.float32) / 255.0
    b = candidate_edges.astype(np.float32) / 255.0

    # Dice-like overlap of edge pixels.
    intersection = np.sum(a * b)
    denom = np.sum(a) + np.sum(b)

    if denom < 1e-6:
        return 0.0

    dice = 2.0 * intersection / denom

    return float(np.clip(dice, 0.0, 1.0))


def motion_score(candidate_center, predicted_center):
    distance = math.hypot(
        candidate_center[0] - predicted_center[0],
        candidate_center[1] - predicted_center[1]
    )

    return float(
        np.exp(
            -0.5 *
            (distance / 100.0) ** 2
        )
    )


def size_score(candidate_size, previous_size):
    cw, ch = candidate_size
    pw, ph = previous_size

    rw = min(
        safe_ratio(cw, max(pw, 1)),
        safe_ratio(pw, max(cw, 1))
    )

    rh = min(
        safe_ratio(ch, max(ph, 1)),
        safe_ratio(ph, max(ch, 1))
    )

    return float(
        0.5 * (rw + rh)
    )


def get_search_box(center, radius, frame_w, frame_h):
    cx, cy = center

    return (
        max(0, int(cx - radius)),
        max(0, int(cy - radius)),
        min(frame_w, int(cx + radius)),
        min(frame_h, int(cy + radius))
    )


# ============================================================
# Optical flow
# ============================================================

class FlowPredictor:

    def __init__(self):
        self.points = None
        self.previous_gray = None
        self.frame_counter = 0

    def initialize(self, gray, box):
        x1, y1, x2, y2 = box

        mask = np.zeros_like(gray)

        # Use the inner region to avoid tracking background edges.
        mx = max(5, int((x2 - x1) * 0.12))
        my = max(5, int((y2 - y1) * 0.12))

        x1i = min(x2 - 2, x1 + mx)
        y1i = min(y2 - 2, y1 + my)
        x2i = max(x1i + 2, x2 - mx)
        y2i = max(y1i + 2, y2 - my)

        cv2.rectangle(
            mask,
            (x1i, y1i),
            (x2i, y2i),
            255,
            -1
        )

        self.points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_FLOW_POINTS,
            qualityLevel=FLOW_QUALITY,
            minDistance=FLOW_MIN_DISTANCE,
            blockSize=7,
            mask=mask
        )

        self.previous_gray = gray.copy()
        self.frame_counter = 0

    def predict(self, gray, box, center, velocity):

        if (
            self.previous_gray is None
            or self.points is None
            or len(self.points) < 5
        ):
            self.initialize(
                self.previous_gray
                if self.previous_gray is not None
                else gray,
                box
            )

        if self.points is None or len(self.points) < 5:
            self.previous_gray = gray.copy()

            return (
                np.asarray(center, dtype=np.float32)
                + np.asarray(velocity, dtype=np.float32),
                0
            )

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.points,
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
            self.initialize(gray, box)

            return (
                np.asarray(center, dtype=np.float32)
                + np.asarray(velocity, dtype=np.float32),
                0
            )

        valid = status.ravel() == 1

        old_points = self.points[valid]
        new_points = next_points[valid]

        if len(old_points) < 5:
            self.initialize(gray, box)

            return (
                np.asarray(center, dtype=np.float32)
                + np.asarray(velocity, dtype=np.float32),
                len(old_points)
            )

        affine, inliers = cv2.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99
        )

        if affine is not None:
            old_center = np.asarray(
                center,
                dtype=np.float32
            ).reshape(1, 1, 2)

            transformed = cv2.transform(
                old_center,
                affine
            )[0, 0]

            predicted = transformed.astype(
                np.float32
            )

            inlier_count = (
                int(inliers.sum())
                if inliers is not None
                else len(old_points)
            )

        else:
            displacement = np.median(
                new_points - old_points,
                axis=0
            )

            predicted = (
                np.asarray(center, dtype=np.float32)
                + displacement.astype(np.float32)
            )

            inlier_count = len(old_points)

        # Blend optical flow and velocity prediction.
        predicted = (
            0.80 * predicted +
            0.20 * (
                np.asarray(center, dtype=np.float32)
                + np.asarray(velocity, dtype=np.float32)
            )
        )

        self.points = new_points.reshape(-1, 1, 2)
        self.previous_gray = gray.copy()
        self.frame_counter += 1

        if (
            self.frame_counter >= FLOW_REFRESH_INTERVAL
            or len(self.points) < 12
        ):
            self.initialize(gray, box)

        return predicted, inlier_count


# ============================================================
# Egg tracker
# ============================================================

class EggTracker:

    def __init__(
        self,
        track_id,
        center,
        size,
        frame
    ):
        self.id = track_id

        self.center = np.asarray(
            center,
            dtype=np.float32
        )

        self.size = np.asarray(
            size,
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.state = "TRACKING"

        self.prediction_frames = 0

        self.matched = 0
        self.predicted = 0
        self.lost = 0
        self.reacquired = 0

        self.last_score = 1.0
        self.gray_score = 1.0
        self.edge_score = 1.0
        self.hist_score = 1.0
        self.motion_score = 1.0

        self.last_box = self.get_box(
            frame.shape[1],
            frame.shape[0]
        )

        crop = crop_box(
            frame,
            self.last_box
        )

        self.template_gray = resize_gray(
            crop
        )

        self.template_edge = resize_edge(
            crop
        )

        self.template_hist = histogram(
            crop
        )

        self.flow = FlowPredictor()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.flow.initialize(
            gray,
            self.last_box
        )

    def get_box(
        self,
        frame_w,
        frame_h
    ):
        return clamp_box(
            self.center[0],
            self.center[1],
            self.size[0],
            self.size[1],
            frame_w,
            frame_h
        )

    def predict(
        self,
        gray,
        frame_w,
        frame_h
    ):
        predicted, inliers = self.flow.predict(
            gray,
            self.last_box,
            self.center,
            self.velocity
        )

        predicted[0] = np.clip(
            predicted[0],
            0,
            frame_w - 1
        )

        predicted[1] = np.clip(
            predicted[1],
            0,
            frame_h - 1
        )

        return predicted, inliers

    def search(
        self,
        frame,
        predicted_center
    ):
        frame_h, frame_w = frame.shape[:2]

        sx1, sy1, sx2, sy2 = get_search_box(
            predicted_center,
            SEARCH_RADIUS,
            frame_w,
            frame_h
        )

        search = frame[
            sy1:sy2,
            sx1:sx2
        ]

        if search.size == 0:
            return None, (sx1, sy1, sx2, sy2)

        search_gray = cv2.cvtColor(
            search,
            cv2.COLOR_BGR2GRAY
        )

        search_gray = cv2.GaussianBlur(
            search_gray,
            (5, 5),
            0
        )

        # --------------------------------------------------------
        # Multi-scale template matching.
        # --------------------------------------------------------
        best = None

        base_w = float(self.size[0])
        base_h = float(self.size[1])

        for scale in SCALES:

            tw = max(
                40,
                int(round(base_w * scale))
            )

            th = max(
                40,
                int(round(base_h * scale))
            )

            if (
                tw >= search_gray.shape[1]
                or th >= search_gray.shape[0]
            ):
                continue

            # Current template is resized to this scale.
            source_template = self.template_gray

            if source_template is None:
                continue

            template = cv2.resize(
                source_template,
                (tw, th),
                interpolation=cv2.INTER_AREA
            )

            # Normalized correlation.
            result = cv2.matchTemplate(
                search_gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(
                result
            )

            # Candidate center in full-frame coordinates.
            cx = (
                sx1 +
                max_loc[0] +
                tw / 2.0
            )

            cy = (
                sy1 +
                max_loc[1] +
                th / 2.0
            )

            candidate_box = (
                int(round(cx - tw / 2)),
                int(round(cy - th / 2)),
                int(round(cx + tw / 2)),
                int(round(cy + th / 2))
            )

            candidate_box = clamp_box(
                cx,
                cy,
                tw,
                th,
                frame_w,
                frame_h
            )

            candidate_crop = crop_box(
                frame,
                candidate_box
            )

            if candidate_crop.size == 0:
                continue

            candidate_gray = resize_gray(
                candidate_crop
            )

            candidate_edge = resize_edge(
                candidate_crop
            )

            candidate_hist = histogram(
                candidate_crop
            )

            # Grayscale appearance.
            gray_score = normalized_template_score(
                self.template_gray,
                candidate_gray
            )

            # Edge/shape appearance.
            edge_score_value = edge_template_score(
                self.template_edge,
                candidate_edge
            )

            # Color/texture distribution.
            hist_score_value = histogram_similarity(
                self.template_hist,
                candidate_hist
            )

            # Motion agreement.
            m_score = motion_score(
                (cx, cy),
                predicted_center
            )

            # MatchTemplate is useful, but it should not dominate.
            tm_score = float(
                np.clip(
                    (max_val + 1.0) / 2.0,
                    0.0,
                    1.0
                )
            )

            total = (
                0.30 * tm_score +
                W_GRAY * gray_score +
                W_EDGE * edge_score_value +
                W_HIST * hist_score_value +
                W_MOTION * m_score
            )

            # The weights above intentionally sum to >1 because
            # template matching is an auxiliary cue. Normalize.
            total /= (
                0.30 +
                W_GRAY +
                W_EDGE +
                W_HIST +
                W_MOTION
            )

            center_distance = math.hypot(
                cx - predicted_center[0],
                cy - predicted_center[1]
            )

            if center_distance > MAX_CENTER_DISTANCE:
                continue

            # Reject only extreme size changes.
            candidate_size = (
                float(tw),
                float(th)
            )

            size_consistency = size_score(
                candidate_size,
                self.size
            )

            if (
                safe_ratio(
                    max(tw, th),
                    max(self.size[0], self.size[1])
                ) > MAX_SIZE_RATIO
                or
                safe_ratio(
                    max(self.size[0], self.size[1]),
                    max(tw, th)
                ) > MAX_SIZE_RATIO
            ):
                continue

            # Small size consistency contribution.
            total = (
                0.90 * total +
                0.10 * size_consistency
            )

            result_item = {
                "score": float(total),
                "tm_score": tm_score,
                "gray_score": gray_score,
                "edge_score": edge_score_value,
                "hist_score": hist_score_value,
                "motion_score": m_score,
                "size_score": size_consistency,
                "center": (cx, cy),
                "size": candidate_size,
                "box": candidate_box,
                "crop": candidate_crop,
                "distance": center_distance,
            }

            if (
                best is None
                or result_item["score"] > best["score"]
            ):
                best = result_item

        return best, (
            sx1,
            sy1,
            sx2,
            sy2
        )

    def update(
        self,
        result,
        frame
    ):
        new_center = np.asarray(
            result["center"],
            dtype=np.float32
        )

        displacement = (
            new_center -
            self.center
        )

        self.velocity = (
            VELOCITY_ALPHA * displacement +
            (1.0 - VELOCITY_ALPHA) * self.velocity
        )

        self.center = (
            0.70 * new_center +
            0.30 * self.center
        )

        old_state = self.state

        self.state = "TRACKING"
        self.prediction_frames = 0

        self.matched += 1

        if old_state != "TRACKING":
            self.reacquired += 1

        self.last_score = result["score"]
        self.gray_score = result["gray_score"]
        self.edge_score = result["edge_score"]
        self.hist_score = result["hist_score"]
        self.motion_score = result["motion_score"]

        old_size = self.size.copy()

        candidate_size = np.asarray(
            result["size"],
            dtype=np.float32
        )

        self.size = (
            0.65 * candidate_size +
            0.35 * old_size
        )

        self.last_box = self.get_box(
            frame.shape[1],
            frame.shape[0]
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # Only adapt appearance when the match is VERY strong.
        # This prevents the hand from becoming the new template.
        # --------------------------------------------------------
        if (
            result["score"] >=
            MIN_TEMPLATE_UPDATE_SCORE
        ):
            crop = crop_box(
                frame,
                self.last_box
            )

            new_gray = resize_gray(
                crop
            )

            new_edge = resize_edge(
                crop
            )

            new_hist = histogram(
                crop
            )

            if new_gray is not None:
                self.template_gray = (
                    (1.0 - TEMPLATE_UPDATE_ALPHA)
                    * self.template_gray.astype(np.float32)
                    +
                    TEMPLATE_UPDATE_ALPHA
                    * new_gray.astype(np.float32)
                ).astype(np.uint8)

            if new_edge is not None:
                self.template_edge = new_edge

            if new_hist is not None:
                self.template_hist = (
                    (1.0 - HIST_UPDATE_ALPHA)
                    * self.template_hist
                    +
                    HIST_UPDATE_ALPHA
                    * new_hist
                )

        # Reinitialize flow around the validated egg.
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.flow.initialize(
            gray,
            self.last_box
        )

    def predict_only(
        self,
        predicted_center,
        frame
    ):
        predicted_center = np.asarray(
            predicted_center,
            dtype=np.float32
        )

        displacement = (
            predicted_center -
            self.center
        )

        self.velocity = (
            0.20 * displacement +
            0.80 * self.velocity
        )

        self.center = predicted_center

        self.prediction_frames += 1

        if (
            self.prediction_frames <=
            MAX_PREDICTION_FRAMES
        ):
            self.state = "PREDICTED"
            self.predicted += 1
        else:
            self.state = "LOST"
            self.lost += 1

        self.last_score = 0.0
        self.gray_score = 0.0
        self.edge_score = 0.0
        self.hist_score = 0.0
        self.motion_score = 0.0

        self.last_box = self.get_box(
            frame.shape[1],
            frame.shape[0]
        )

    def total_frames(self):
        return (
            self.matched +
            self.predicted +
            self.lost
        )

    def tracking_ratio(self):
        total = self.total_frames()

        return (
            self.matched / total
            if total > 0
            else 0.0
        )

    def coverage(self):
        total = self.total_frames()

        return (
            (self.matched + self.predicted) / total
            if total > 0
            else 0.0
        )


# ============================================================
# Visualization
# ============================================================

def draw_tracker(
    frame,
    tracker
):
    if tracker.state == "TRACKING":
        color = (0, 255, 0)

    elif tracker.state == "PREDICTED":
        color = (0, 255, 255)

    else:
        color = (0, 0, 255)

    x1, y1, x2, y2 = tracker.last_box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cx, cy = map(
        int,
        tracker.center
    )

    cv2.circle(
        frame,
        (cx, cy),
        5,
        color,
        -1
    )

    label = (
        f"Egg {tracker.id} | "
        f"{tracker.state} | "
        f"S={tracker.last_score:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(25, y1 - 10)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


def draw_search_box(
    frame,
    box
):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 180, 0),
        1
    )


# ============================================================
# Main
# ============================================================

def main():

    video_path = find_video()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "task4_tracking_v7_test.mp4"
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print("=" * 70)
    print("TASK 4 - TRACKING V7")
    print("Template/shape tracking + optical-flow prediction")
    print("=" * 70)

    print(
        f"Video: {os.path.basename(video_path)}"
    )

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

    print(
        "GREEN  = validated true track"
    )

    print(
        "YELLOW = prediction only / temporary occlusion"
    )

    print(
        "RED    = lost"
    )

    print()

    print(
        "V7 principle:"
    )

    print(
        "  No E1 connected components."
    )

    print(
        "  Optical flow predicts where the egg should be."
    )

    print(
        "  Multi-scale template + edge matching finds the egg."
    )

    print(
        "  Histogram checks appearance."
    )

    print(
        "  Weak match -> prediction, NOT object switching."
    )

    print()

    for t in INITIAL_TRACKS:
        print(
            f"Egg {t['id']}: "
            f"center={t['center']} "
            f"size={t['size']}"
        )

    start_frame = int(
        round(
            START_TIME * fps
        )
    )

    end_frame = int(
        round(
            END_TIME * fps
        )
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ok, first_frame = cap.read()

    if not ok:
        cap.release()

        raise RuntimeError(
            "Could not read starting frame."
        )

    trackers = [
        EggTracker(
            t["id"],
            t["center"],
            t["size"],
            first_frame
        )
        for t in INITIAL_TRACKS
    ]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    # Initial visualization.
    first_vis = first_frame.copy()

    for tracker in trackers:
        draw_tracker(
            first_vis,
            tracker
        )

    cv2.putText(
        first_vis,
        "V7 | Template + Optical Flow",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(
        first_vis
    )

    last_report_second = -1

    # ========================================================
    # Frame loop
    # ========================================================

    for frame_idx in range(
        start_frame + 1,
        end_frame + 1
    ):

        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        predictions = []
        searches = []

        # ----------------------------------------------------
        # 1. Motion prediction + local template search
        # ----------------------------------------------------

        for tracker in trackers:

            predicted_center, flow_inliers = (
                tracker.predict(
                    gray,
                    width,
                    height
                )
            )

            result, search_box = tracker.search(
                frame,
                predicted_center
            )

            predictions.append(
                (
                    predicted_center,
                    flow_inliers
                )
            )

            searches.append(
                (
                    result,
                    search_box
                )
            )

        # ----------------------------------------------------
        # 2. Resolve two-track identity conflicts.
        #
        # If both trackers select essentially the same region,
        # the stronger score wins. The weaker tracker predicts.
        # ----------------------------------------------------

        chosen = [None, None]

        if len(trackers) == 2:

            r0 = searches[0][0]
            r1 = searches[1][0]

            if r0 is None:
                chosen[0] = None

            if r1 is None:
                chosen[1] = None

            if (
                r0 is not None
                and r1 is not None
            ):
                b0 = r0["box"]
                b1 = r1["box"]

                # IoU between boxes.
                ix1 = max(
                    b0[0],
                    b1[0]
                )

                iy1 = max(
                    b0[1],
                    b1[1]
                )

                ix2 = min(
                    b0[2],
                    b1[2]
                )

                iy2 = min(
                    b0[3],
                    b1[3]
                )

                inter = (
                    max(0, ix2 - ix1) *
                    max(0, iy2 - iy1)
                )

                area0 = (
                    (b0[2] - b0[0]) *
                    (b0[3] - b0[1])
                )

                area1 = (
                    (b1[2] - b1[0]) *
                    (b1[3] - b1[1])
                )

                union = (
                    area0 +
                    area1 -
                    inter
                )

                overlap = (
                    inter / union
                    if union > 0
                    else 0
                )

                if overlap > 0.45:
                    if (
                        r0["score"] >=
                        r1["score"]
                    ):
                        chosen[0] = r0
                        chosen[1] = None
                    else:
                        chosen[0] = None
                        chosen[1] = r1

                else:
                    chosen[0] = r0
                    chosen[1] = r1

        else:

            for i in range(
                len(trackers)
            ):
                chosen[i] = searches[i][0]

        # ----------------------------------------------------
        # 3. Confidence validation and update.
        # ----------------------------------------------------

        for i, tracker in enumerate(
            trackers
        ):

            result = chosen[i]

            if result is not None:

                # Strong appearance agreement.
                appearance_ok = (
                    result["gray_score"] >= 0.45
                    and
                    result["edge_score"] >= 0.20
                    and
                    result["hist_score"] >= 0.35
                )

                # Motion agreement.
                motion_ok = (
                    result["motion_score"] >= 0.25
                )

                # Overall acceptance.
                accept = (
                    result["score"] >=
                    ACCEPT_SCORE
                    and
                    appearance_ok
                    and
                    motion_ok
                )

                if accept:
                    tracker.update(
                        result,
                        frame
                    )

                else:
                    tracker.predict_only(
                        predictions[i][0],
                        frame
                    )

            else:
                tracker.predict_only(
                    predictions[i][0],
                    frame
                )

        # ----------------------------------------------------
        # 4. Visualization.
        # ----------------------------------------------------

        vis = frame.copy()

        if DRAW_SEARCH_REGION:
            for (
                result,
                search_box
            ) in searches:
                draw_search_box(
                    vis,
                    search_box
                )

        for tracker in trackers:
            draw_tracker(
                vis,
                tracker
            )

        current_time = (
            frame_idx / fps
        )

        cv2.putText(
            vis,
            f"V7 | t={current_time:.2f}s",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(
            vis
        )

        # ----------------------------------------------------
        # Terminal progress.
        # ----------------------------------------------------

        current_second = int(
            current_time
        )

        if (
            current_second % 2 == 0
            and
            current_second !=
            last_report_second
        ):

            last_report_second = (
                current_second
            )

            print(
                f"time={current_time:.2f}s | "
                f"Egg 1={trackers[0].state} "
                f"S={trackers[0].last_score:.2f} | "
                f"Egg 2={trackers[1].state} "
                f"S={trackers[1].last_score:.2f}"
            )

    writer.release()
    cap.release()

    print()
    print("=" * 70)
    print("TRACKING V7 COMPLETE")
    print("=" * 70)

    for tracker in trackers:

        print(
            f"Egg {tracker.id}:"
        )

        print(
            f"  Matched / TRUE TRACK: "
            f"{tracker.matched}"
        )

        print(
            f"  Predicted / OCCLUSION: "
            f"{tracker.predicted}"
        )

        print(
            f"  Lost: "
            f"{tracker.lost}"
        )

        print(
            f"  Reacquired: "
            f"{tracker.reacquired}"
        )

        print(
            f"  Algorithmic tracking ratio: "
            f"{tracker.tracking_ratio() * 100:.2f}%"
        )

        print(
            f"  Visible/predicted coverage: "
            f"{tracker.coverage() * 100:.2f}%"
        )

        print()

    print(
        f"Output: {output_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The percentages measure accepted algorithmic matches."
    )

    print(
        "They do NOT prove identity correctness."
    )

    print(
        "Inspect the video visually before using the numbers."
    )


if __name__ == "__main__":
    main()
