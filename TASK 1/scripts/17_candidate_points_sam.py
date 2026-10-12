from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import SAM


# ============================================================
# Configuration
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

CANDIDATE_DIR = Path(
    "task1_segmentation/candidate_points_v2"
)

OUTPUT_DIR = Path(
    "task1_segmentation/automatic_point_sam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TEST_FILENAMES = [
    "not_damaged_100_jpg.rf.5d780a6024004fa2f68741fa56ddf4c9.jpg",
    "not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a.jpg",
    "not_damaged_128_jpg.rf.95a7ef96dcc9e14fa7137708df6f2e60.jpg",
    "not_damaged_148_jpg.rf.8a3e1c0e1dae236abe0e9a75a4882bda.jpg",
]


# ============================================================
# Load candidate points
# ============================================================

CSV_PATH = (
    CANDIDATE_DIR /
    "candidate_points_v2.csv"
)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Candidate CSV not found: {CSV_PATH}"
    )


candidate_points = {}

with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        image_name = row["image"]

        if image_name not in candidate_points:
            candidate_points[image_name] = []

        candidate_points[image_name].append({
            "region_rank": int(row["region_rank"]),
            "point_rank": int(row["point_rank"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "score": float(row["region_score"]),
            "distance": float(row["point_distance"]),
        })


# ============================================================
# Load SAM
# ============================================================

print("=" * 70)
print("TASK 1 - AUTOMATIC POINT -> SAM 2.1 TINY")
print("=" * 70)

print("\nLoading SAM 2.1 Tiny...")

model = SAM("sam2.1_t.pt")

print("SAM 2.1 Tiny loaded.")


# ============================================================
# Process images
# ============================================================

for image_index, filename in enumerate(
    TEST_FILENAMES,
    start=1
):

    print("\n" + "-" * 70)
    print(
        f"[{image_index}/{len(TEST_FILENAMES)}] "
        f"{filename}"
    )
    print("-" * 70)

    image_path = (
        DEVELOPMENT_DIR /
        filename
    )

    if not image_path.exists():

        print("Image not found.")
        continue

    points = candidate_points.get(
        filename,
        []
    )

    print(
        f"Candidate points: {len(points)}"
    )

    if len(points) == 0:

        print("No candidate points.")
        continue


    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("Could not read image.")
        continue


    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    image_output_dir = (
        OUTPUT_DIR /
        image_path.stem
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Combined instance mask
    combined_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Process each candidate point independently
    # --------------------------------------------------------

    for point_index, point in enumerate(
        points,
        start=1
    ):

        x = point["x"]
        y = point["y"]

        print(
            f"\n  Egg candidate {point_index}: "
            f"point=({x},{y})"
        )

        try:

            # ------------------------------------------------
            # SAM positive point prompt
            # ------------------------------------------------

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

                print(
                    "    SAM returned no mask."
                )

                continue


            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            print(
                f"    SAM masks returned: "
                f"{len(masks)}"
            )


            # ------------------------------------------------
            # SAM can return multiple mask hypotheses.
            #
            # Select the first/highest-ranked mask.
            # ------------------------------------------------

            mask = masks[0]

            mask_uint8 = (
                (mask > 0.5)
                .astype(np.uint8)
                * 255
            )


            # ------------------------------------------------
            # Save individual mask
            # ------------------------------------------------

            mask_path = (
                image_output_dir /
                f"egg_{point_index:02d}.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask_uint8
            )

            print(
                f"    Saved: {mask_path}"
            )


            # ------------------------------------------------
            # Add to combined instance mask
            #
            # Each egg gets a unique integer ID.
            # ------------------------------------------------

            combined_mask[
                mask_uint8 > 0
            ] = point_index


            # ------------------------------------------------
            # Create per-egg visualization
            # ------------------------------------------------

            visualization = image.copy()

            overlay = image.copy()

            overlay[
                mask_uint8 > 0
            ] = (
                255,
                0,
                0
            )

            visualization = cv2.addWeighted(
                visualization,
                0.70,
                overlay,
                0.30,
                0
            )

            cv2.circle(
                visualization,
                (x, y),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                visualization,
                f"egg {point_index}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            visualization_path = (
                image_output_dir /
                f"egg_{point_index:02d}_result.jpg"
            )

            cv2.imwrite(
                str(visualization_path),
                visualization
            )


        except RuntimeError as e:

            print(
                f"    Runtime error: {e}"
            )

            continue

        except Exception as e:

            print(
                f"    Error: {e}"
            )

            continue


    # --------------------------------------------------------
    # Save combined instance mask
    # --------------------------------------------------------

    combined_path = (
        image_output_dir /
        "combined_instance_mask.png"
    )

    # Convert instance IDs into visible grayscale values
    visible_mask = (
        combined_mask.astype(np.float32)
        / max(1, len(points))
        * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(combined_path),
        visible_mask
    )

    print(
        f"\n  Combined mask saved: "
        f"{combined_path}"
    )


print("\n" + "=" * 70)
print("AUTOMATIC POINT -> SAM TEST COMPLETE")
print("=" * 70)

print("\nResults:")
print(OUTPUT_DIR)

print(
    "\nThese images are still development-set experiments."
)

print(
    "The 22-image evaluation set remains untouched."
)