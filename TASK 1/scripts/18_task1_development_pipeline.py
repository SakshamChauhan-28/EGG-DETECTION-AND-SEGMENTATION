from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import SAM


# ============================================================
# CONFIGURATION
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

CANDIDATE_CSV = Path(
    "task1_segmentation/candidate_points_v2/candidate_points_v2.csv"
)

OUTPUT_DIR = Path(
    "task1_segmentation/final_development"
)

MASK_DIR = OUTPUT_DIR / "masks"
VIS_DIR = OUTPUT_DIR / "visualizations"

MASK_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CANDIDATE POINTS
# ============================================================

candidate_points = {}

with open(
    CANDIDATE_CSV,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        image_name = row["image"]

        candidate_points.setdefault(
            image_name,
            []
        )

        candidate_points[image_name].append({
            "x": int(row["x"]),
            "y": int(row["y"]),
            "score": float(row["region_score"]),
            "distance": float(row["point_distance"]),
        })


# ============================================================
# LOAD SAM
# ============================================================

print("=" * 70)
print("TASK 1 - FULL DEVELOPMENT SEGMENTATION")
print("=" * 70)

print("\nLoading SAM 2.1 Tiny...")

model = SAM("sam2.1_t.pt")

print("SAM loaded.")


# ============================================================
# IMAGE LIST
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
# CSV REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "development_segmentation_report.csv"
)

report_rows = []


# ============================================================
# PROCESS ALL IMAGES
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

        print("  ERROR: Could not read image.")

        report_rows.append({
            "image": image_path.name,
            "candidate_points": 0,
            "successful_masks": 0,
            "status": "FAIL_READ",
        })

        continue


    height, width = image.shape[:2]

    points = candidate_points.get(
        image_path.name,
        []
    )

    print(
        f"  Candidate points: {len(points)}"
    )


    # --------------------------------------------------------
    # No candidate points
    # --------------------------------------------------------

    if len(points) == 0:

        print(
            "  STATUS: REVIEW - no candidate points"
        )

        report_rows.append({
            "image": image_path.name,
            "candidate_points": 0,
            "successful_masks": 0,
            "status": "REVIEW_NO_CANDIDATES",
        })

        continue


    image_mask_dir = (
        MASK_DIR /
        image_path.stem
    )

    image_mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    successful_masks = 0

    mask_areas = []

    boundary_touch_count = 0

    point_inside_count = 0


    # --------------------------------------------------------
    # SAM for each point
    # --------------------------------------------------------

    for instance_id, point in enumerate(
        points,
        start=1
    ):

        x = point["x"]
        y = point["y"]

        try:

            results = model(
                str(image_path),
                points=[[x, y]],
                labels=[1],
                device="cuda:0",
                imgsz=640,
                verbose=False,
            )

            result = results[0]

            if result.masks is None:

                continue

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            if len(masks) == 0:

                continue

            mask = (
                masks[0] > 0.5
            ).astype(np.uint8)

            area = int(
                np.sum(mask)
            )

            if area == 0:

                continue


            # ------------------------------------------------
            # Check prompt point
            # ------------------------------------------------

            if (
                0 <= x < width and
                0 <= y < height and
                mask[y, x] > 0
            ):
                point_inside_count += 1


            # ------------------------------------------------
            # Check image boundary
            # ------------------------------------------------

            touches_boundary = (
                np.any(mask[0, :] > 0) or
                np.any(mask[-1, :] > 0) or
                np.any(mask[:, 0] > 0) or
                np.any(mask[:, -1] > 0)
            )

            if touches_boundary:

                boundary_touch_count += 1


            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            mask_path = (
                image_mask_dir /
                f"egg_{instance_id:02d}.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask * 255
            )


            # ------------------------------------------------
            # Add instance ID
            # ------------------------------------------------

            combined_mask[
                mask > 0
            ] = instance_id

            successful_masks += 1

            mask_areas.append(area)


        except Exception as e:

            print(
                f"    SAM error for "
                f"instance {instance_id}: {e}"
            )


    # ========================================================
    # QUALITY CHECK
    # ========================================================

    if successful_masks == 0:

        status = "FAIL_NO_MASKS"

    else:

        point_success_rate = (
            point_inside_count /
            len(points)
        )

        boundary_rate = (
            boundary_touch_count /
            successful_masks
        )

        # Normal egg masks should not normally occupy
        # almost the entire image.

        largest_area_ratio = (
            max(mask_areas) /
            float(width * height)
        )

        if point_success_rate < 0.5:

            status = "REVIEW_POINT_OUTSIDE"

        elif boundary_rate > 0.5:

            status = "REVIEW_BOUNDARY"

        elif largest_area_ratio > 0.80:

            status = "REVIEW_HUGE_MASK"

        elif successful_masks < len(points):

            status = "REVIEW_MISSING_MASK"

        else:

            status = "GOOD"


    print(
        f"  Masks: {successful_masks}"
    )

    print(
        f"  Status: {status}"
    )


    # ========================================================
    # SAVE REPORT ROW
    # ========================================================

    report_rows.append({
        "image": image_path.name,
        "candidate_points": len(points),
        "successful_masks": successful_masks,
        "points_inside_masks": point_inside_count,
        "boundary_touch_masks": boundary_touch_count,
        "status": status,
    })


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "image",
    "candidate_points",
    "successful_masks",
    "points_inside_masks",
    "boundary_touch_masks",
    "status",
]

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        report_rows
    )


# ============================================================
# SUMMARY
# ============================================================

from collections import Counter

status_counts = Counter(
    row["status"]
    for row in report_rows
)

print("\n" + "=" * 70)
print("DEVELOPMENT SEGMENTATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal images: {len(image_paths)}"
)

for status, count in status_counts.items():

    print(
        f"{status}: {count}"
    )

print(
    f"\nReport:"
)

print(report_path)

print(
    "\nIMPORTANT:"
)

print(
    "The 22-image evaluation set was not used."
)