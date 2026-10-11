from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path(
    "task1_segmentation/candidate_points_v2"
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
# Find separated interior points
# ============================================================

def find_interior_points(
    contour_mask,
    min_distance=55,
    min_peak_ratio=0.30,
):
    """
    Find multiple interior points inside one connected region.

    This is particularly useful when multiple eggs touch
    and therefore form one connected bright contour.
    """

    distance = cv2.distanceTransform(
        contour_mask,
        cv2.DIST_L2,
        5
    )

    max_distance = distance.max()

    if max_distance <= 0:
        return []

    # Only consider sufficiently deep interior pixels.
    threshold = max(
        5.0,
        max_distance * min_peak_ratio
    )

    candidate_region = (
        distance >= threshold
    ).astype(np.uint8) * 255

    # Find local maxima.
    kernel_size = (
        min_distance * 2 + 1
    )

    dilated = cv2.dilate(
        distance,
        np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8
        )
    )

    local_maxima = (
        (distance >= dilated)
        & (distance >= threshold)
    )

    ys, xs = np.where(local_maxima)

    peaks = []

    for x, y in zip(xs, ys):

        peaks.append({
            "x": int(x),
            "y": int(y),
            "distance": float(distance[y, x])
        })

    # Sort strongest first.
    peaks.sort(
        key=lambda p: p["distance"],
        reverse=True
    )

    # Non-maximum suppression.
    selected = []

    for peak in peaks:

        too_close = False

        for chosen in selected:

            dx = (
                peak["x"]
                - chosen["x"]
            )

            dy = (
                peak["y"]
                - chosen["y"]
            )

            distance_between = (
                dx * dx + dy * dy
            ) ** 0.5

            if distance_between < min_distance:

                too_close = True
                break

        if not too_close:

            selected.append(peak)

    return selected


# ============================================================
# Generate candidate points
# ============================================================

def generate_candidates(image):

    h, w = image.shape[:2]

    image_area = h * w

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------------------------
    # Brightness threshold
    # --------------------------------------------------------

    otsu_threshold, _ = cv2.threshold(
        value,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    percentile_threshold = np.percentile(
        value,
        70
    )

    threshold = max(
        otsu_threshold,
        percentile_threshold
    )

    binary = (
        value >= threshold
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Find connected regions
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        area_ratio = (
            area / image_area
        )

        # Reject tiny noise.
        if area_ratio < 0.005:
            continue

        # Reject very large regions.
        if area_ratio > 0.30:
            continue

        x, y, bw, bh = cv2.boundingRect(
            contour
        )

        # ----------------------------------------------------
        # Reject regions touching image boundary.
        # ----------------------------------------------------

        touches_border = (
            x <= 1
            or y <= 1
            or x + bw >= w - 1
            or y + bh >= h - 1
        )

        if touches_border:
            continue

        if bw <= 0 or bh <= 0:
            continue

        aspect_ratio = bw / bh

        if (
            aspect_ratio < 0.25
            or aspect_ratio > 4.0
        ):
            continue

        # ----------------------------------------------------
        # Create contour mask.
        # ----------------------------------------------------

        contour_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        # ----------------------------------------------------
        # Shape statistics.
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * np.pi
            * area
            / (perimeter * perimeter)
        )

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area <= 0:
            continue

        solidity = (
            area / hull_area
        )

        # ----------------------------------------------------
        # Mean brightness / saturation.
        # ----------------------------------------------------

        mean_value = cv2.mean(
            value,
            mask=contour_mask
        )[0]

        mean_saturation = cv2.mean(
            saturation,
            mask=contour_mask
        )[0]

        # ----------------------------------------------------
        # Find multiple interior points.
        # ----------------------------------------------------

        points = find_interior_points(
            contour_mask,
            min_distance=55,
            min_peak_ratio=0.30
        )

        if len(points) == 0:
            continue

        # ----------------------------------------------------
        # Candidate score.
        # ----------------------------------------------------

        brightness_score = (
            mean_value / 255.0
        )

        low_saturation_score = (
            1.0
            - mean_saturation / 255.0
        )

        circularity_score = np.clip(
            circularity,
            0.0,
            1.0
        )

        solidity_score = np.clip(
            solidity,
            0.0,
            1.0
        )

        score = (
            0.30 * brightness_score
            + 0.20 * low_saturation_score
            + 0.25 * circularity_score
            + 0.25 * solidity_score
        )

        candidates.append({
            "contour": contour,
            "mask": contour_mask,
            "bbox": (x, y, bw, bh),
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "mean_value": mean_value,
            "mean_saturation": mean_saturation,
            "score": score,
            "points": points,
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return candidates, binary


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("TASK 1 - CANDIDATE POINT GENERATOR V2")
print("=" * 70)

csv_path = (
    OUTPUT_DIR /
    "candidate_points_v2.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "image,"
        "region_rank,"
        "point_rank,"
        "x,"
        "y,"
        "region_score,"
        "point_distance,"
        "area_ratio,"
        "circularity,"
        "solidity\n"
    )


for index, filename in enumerate(
    TEST_FILENAMES,
    start=1
):

    print("\n" + "-" * 70)
    print(
        f"[{index}/{len(TEST_FILENAMES)}] "
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

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("Could not read image.")
        continue

    candidates, binary = (
        generate_candidates(image)
    )

    print(
        f"Candidate regions: "
        f"{len(candidates)}"
    )

    visualization = image.copy()

    total_points = 0

    for region_rank, candidate in enumerate(
        candidates[:10],
        start=1
    ):

        contour = candidate["contour"]

        x, y, bw, bh = (
            candidate["bbox"]
        )

        # Draw candidate contour.
        cv2.drawContours(
            visualization,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Draw bounding box.
        cv2.rectangle(
            visualization,
            (x, y),
            (x + bw, y + bh),
            (255, 0, 0),
            2
        )

        cv2.putText(
            visualization,
            f"region {region_rank}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Draw every interior point.
        # ----------------------------------------------------

        for point_rank, point in enumerate(
            candidate["points"],
            start=1
        ):

            px = point["x"]
            py = point["y"]

            cv2.circle(
                visualization,
                (px, py),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                visualization,
                str(point_rank),
                (px + 8, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            with open(
                csv_path,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f'"{filename}",'
                    f"{region_rank},"
                    f"{point_rank},"
                    f"{px},"
                    f"{py},"
                    f"{candidate['score']:.5f},"
                    f"{point['distance']:.2f},"
                    f"{candidate['area_ratio']:.5f},"
                    f"{candidate['circularity']:.5f},"
                    f"{candidate['solidity']:.5f}\n"
                )

            total_points += 1

    # --------------------------------------------------------
    # Save visualization.
    # --------------------------------------------------------

    visualization_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_candidates_v2.jpg"
    )

    cv2.imwrite(
        str(visualization_path),
        visualization
    )

    # Threshold visualization.
    threshold_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_threshold_v2.png"
    )

    cv2.imwrite(
        str(threshold_path),
        binary
    )

    print(
        f"Interior points generated: "
        f"{total_points}"
    )

    for region_rank, candidate in enumerate(
        candidates[:10],
        start=1
    ):

        print(
            f"  Region {region_rank}: "
            f"points={len(candidate['points'])}, "
            f"score={candidate['score']:.3f}"
        )

        for point_rank, point in enumerate(
            candidate["points"],
            start=1
        ):

            print(
                f"      Point {point_rank}: "
                f"({point['x']}, {point['y']}) "
                f"depth={point['distance']:.1f}"
            )


print("\n" + "=" * 70)
print("V2 CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(f"\nResults:")
print(OUTPUT_DIR)

print(f"\nCSV:")
print(csv_path)