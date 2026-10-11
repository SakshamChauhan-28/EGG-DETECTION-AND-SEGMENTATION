from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import SAM


# ============================================================
# CONFIG
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path(
    "task1_segmentation/candidate_points_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "candidate_points_v3.csv"


# ============================================================
# PARAMETERS
# ============================================================

MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.70

MIN_MEAN_BRIGHTNESS = 80

MIN_DISTANCE = 35

MAX_POINTS_PER_REGION = 5


# ============================================================
# LOAD SAM
# ============================================================

print("=" * 70)
print("TASK 1 - CANDIDATE GENERATOR V3")
print("SAM AUTOMATIC MASKS -> INTERIOR POINTS")
print("=" * 70)

print("\nLoading SAM 2.1 Tiny...")

model = SAM("sam2.1_t.pt")

print("SAM loaded.")


# ============================================================
# IMAGES
# ============================================================

image_paths = sorted([
    p for p in DEVELOPMENT_DIR.iterdir()
    if p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png"
    }
])

print(
    f"\nDevelopment images: {len(image_paths)}"
)


# ============================================================
# CSV
# ============================================================

rows = []


# ============================================================
# PROCESS
# ============================================================

for image_index, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"\n[{image_index}/{len(image_paths)}] "
        f"{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("  Could not read image.")
        continue

    height, width = image.shape[:2]

    image_area = height * width

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # SAM AUTOMATIC SEGMENTATION
    # --------------------------------------------------------

    try:

        results = model(
            str(image_path),
            verbose=False
        )

    except Exception as e:

        print(
            f"  SAM ERROR: {e}"
        )

        continue


    result = results[0]

    if result.masks is None:

        print(
            "  No automatic masks."
        )

        continue


    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )

    print(
        f"  SAM masks: {len(masks)}"
    )


    candidate_visual = image.copy()

    image_rows = []

    candidate_count = 0


    # ========================================================
    # ANALYZE EACH SAM MASK
    # ========================================================

    for mask_index, raw_mask in enumerate(
        masks
    ):

        mask = (
            raw_mask > 0.5
        ).astype(np.uint8)


        area = int(
            np.sum(mask)
        )

        area_ratio = (
            area /
            float(image_area)
        )


        # ----------------------------------------------------
        # Area filtering
        # ----------------------------------------------------

        if area_ratio < MIN_AREA_RATIO:
            continue

        if area_ratio > MAX_AREA_RATIO:
            continue


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())

        y1 = int(ys.min())
        y2 = int(ys.max())

        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1

        aspect_ratio = (
            box_width /
            float(box_height)
        )


        # ----------------------------------------------------
        # Mask geometry
        # ----------------------------------------------------

        bbox_area = (
            box_width *
            box_height
        )

        extent = (
            area /
            float(bbox_area)
        )


        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = cv2.contourArea(
            contour
        )

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area > 0:

            solidity = (
                contour_area /
                hull_area
            )

        else:

            solidity = 0


        # ----------------------------------------------------
        # Image appearance
        # ----------------------------------------------------

        mask_pixels = mask > 0

        mean_brightness = float(
            gray[mask_pixels].mean()
        )

        mean_saturation = float(
            hsv[:, :, 1][mask_pixels].mean()
        )


        # ----------------------------------------------------
        # Egg candidate score
        # ----------------------------------------------------

        score = 0.0


        # Eggs in this dataset are generally bright.

        if mean_brightness >= 80:
            score += 0.30

        if mean_brightness >= 120:
            score += 0.15


        # Egg-like compactness

        if extent >= 0.45:
            score += 0.20

        if solidity >= 0.90:
            score += 0.20


        # Reasonable egg aspect ratio

        if 0.35 <= aspect_ratio <= 2.8:
            score += 0.15


        # ----------------------------------------------------
        # Candidate threshold
        # ----------------------------------------------------

        if score < 0.45:
            continue


        # ====================================================
        # DISTANCE TRANSFORM
        # ====================================================

        dist = cv2.distanceTransform(
            mask,
            cv2.DIST_L2,
            5
        )


        # ----------------------------------------------------
        # Find local maxima using dilation
        # ----------------------------------------------------

        kernel = np.ones(
            (31, 31),
            np.uint8
        )

        dilated = cv2.dilate(
            dist,
            kernel
        )

        maxima = (
            (dist == dilated) &
            (dist >= MIN_DISTANCE)
        )


        ys_peak, xs_peak = np.where(
            maxima
        )


        peak_values = []

        for px, py in zip(
            xs_peak,
            ys_peak
        ):

            peak_values.append(
                (
                    float(dist[py, px]),
                    int(px),
                    int(py)
                )
            )


        peak_values.sort(
            reverse=True
        )


        # ----------------------------------------------------
        # Non-maximum suppression
        # ----------------------------------------------------

        selected_points = []

        for depth, px, py in peak_values:

            too_close = False

            for _, sx, sy in selected_points:

                distance = np.sqrt(
                    (px - sx) ** 2 +
                    (py - sy) ** 2
                )

                if distance < MIN_DISTANCE:

                    too_close = True
                    break


            if too_close:
                continue


            selected_points.append(
                (
                    depth,
                    px,
                    py
                )
            )


            if len(selected_points) >= MAX_POINTS_PER_REGION:
                break


        if not selected_points:
            continue


        candidate_count += len(
            selected_points
        )


        # ====================================================
        # SAVE CSV ROWS
        # ====================================================

        for point_id, (
            depth,
            px,
            py
        ) in enumerate(
            selected_points,
            start=1
        ):

            rows.append({

                "image": image_path.name,

                "region_id":
                    mask_index,

                "point_id":
                    point_id,

                "x":
                    px,

                "y":
                    py,

                "point_distance":
                    round(depth, 2),

                "area":
                    area,

                "area_ratio":
                    round(area_ratio, 4),

                "aspect_ratio":
                    round(aspect_ratio, 3),

                "extent":
                    round(extent, 3),

                "solidity":
                    round(solidity, 3),

                "mean_brightness":
                    round(mean_brightness, 2),

                "mean_saturation":
                    round(mean_saturation, 2),

                "candidate_score":
                    round(score, 3),
            })


        # ====================================================
        # VISUALIZATION
        # ====================================================

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            candidate_visual,
            contours,
            -1,
            (0, 255, 0),
            2
        )


        for point_number, (
            depth,
            px,
            py
        ) in enumerate(
            selected_points,
            start=1
        ):

            cv2.circle(
                candidate_visual,
                (px, py),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                candidate_visual,
                str(point_number),
                (px + 8, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    output_visual = (
        VIS_DIR /
        f"{image_path.stem}_candidates.jpg"
    )

    cv2.imwrite(
        str(output_visual),
        candidate_visual
    )


    print(
        f"  Candidate points: {candidate_count}"
    )


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "image",
    "region_id",
    "point_id",
    "x",
    "y",
    "point_distance",
    "area",
    "area_ratio",
    "aspect_ratio",
    "extent",
    "solidity",
    "mean_brightness",
    "mean_saturation",
    "candidate_score",
]


with open(
    CSV_PATH,
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


# ============================================================
# SUMMARY
# ============================================================

unique_images = len(
    set(
        row["image"]
        for row in rows
    )
)

total_points = len(rows)

print("\n" + "=" * 70)
print("V3 CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nImages processed: {len(image_paths)}"
)

print(
    f"Images with candidate points: "
    f"{unique_images}"
)

print(
    f"Total candidate points: "
    f"{total_points}"
)

print(
    f"\nCSV:"
)

print(CSV_PATH)

print(
    "\nVisualization:"
)

print(VIS_DIR)

print(
    "\nIMPORTANT:"
)

print(
    "These are SAM-derived candidate prompts, "
    "not final egg masks."
)

print(
    "The 22-image evaluation set was not used."
)