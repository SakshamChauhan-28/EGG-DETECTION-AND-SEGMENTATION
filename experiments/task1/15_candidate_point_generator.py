from pathlib import Path
import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path(
    "task1_segmentation/candidate_points"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Test the same representative images first
TEST_FILENAMES = [
    "not_damaged_100_jpg.rf.5d780a6024004fa2f68741fa56ddf4c9.jpg",
    "not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a.jpg",
    "not_damaged_128_jpg.rf.95a7ef96dcc9e14fa7137708df6f2e60.jpg",
    "not_damaged_148_jpg.rf.8a3e1c0e1dae236abe0e9a75a4882bda.jpg",
]


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates(image):
    """
    Generate candidate egg regions using brightness.

    These candidates are ONLY used to generate prompts.
    They are NOT the final segmentation masks.
    """

    h, w = image.shape[:2]
    image_area = h * w

    # --------------------------------------------------------
    # Convert to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------------------------
    # Brightness threshold
    # --------------------------------------------------------

    # Otsu threshold
    otsu_threshold, _ = cv2.threshold(
        value,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Percentile threshold
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
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        area_ratio = area / image_area

        # Ignore tiny noise
        if area_ratio < 0.005:
            continue

        # Ignore huge regions such as table/background
        if area_ratio > 0.35:
            continue

        x, y, bw, bh = cv2.boundingRect(
            contour
        )

        if bw <= 0 or bh <= 0:
            continue

        aspect_ratio = bw / bh

        # Eggs are generally compact/elliptical.
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue

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

        # Solidity
        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        # Mean brightness inside contour
        contour_mask = np.zeros(
            value.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        mean_value = cv2.mean(
            value,
            mask=contour_mask
        )[0]

        mean_saturation = cv2.mean(
            saturation,
            mask=contour_mask
        )[0]

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------

        # Prefer:
        # - reasonable size
        # - compact shape
        # - high solidity
        # - bright region
        # - relatively low saturation

        size_score = 1.0 - abs(
            area_ratio - 0.08
        ) / 0.08

        size_score = np.clip(
            size_score,
            0.0,
            1.0
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

        brightness_score = (
            mean_value / 255.0
        )

        low_saturation_score = (
            1.0 - mean_saturation / 255.0
        )

        score = (
            0.20 * size_score
            + 0.20 * circularity_score
            + 0.20 * solidity_score
            + 0.25 * brightness_score
            + 0.15 * low_saturation_score
        )

        # ----------------------------------------------------
        # Find a safe interior point
        # ----------------------------------------------------

        # Distance transform gives the point furthest
        # from the candidate boundary.
        candidate_mask = np.zeros(
            value.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            candidate_mask,
            [contour],
            -1,
            255,
            -1
        )

        distance = cv2.distanceTransform(
            candidate_mask,
            cv2.DIST_L2,
            5
        )

        _, max_distance, _, max_location = (
            cv2.minMaxLoc(distance)
        )

        point_x, point_y = max_location

        candidates.append({
            "contour": contour,
            "area": area,
            "area_ratio": area_ratio,
            "bbox": (x, y, bw, bh),
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "mean_value": mean_value,
            "mean_saturation": mean_saturation,
            "score": score,
            "point": (point_x, point_y),
            "point_distance": max_distance,
        })

    # Highest score first
    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return candidates, binary


# ============================================================
# Process images
# ============================================================

print("=" * 70)
print("TASK 1 - AUTOMATIC EGG CANDIDATE POINT GENERATION")
print("=" * 70)

csv_path = OUTPUT_DIR / "candidate_points.csv"

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "image,"
        "candidate_rank,"
        "x,"
        "y,"
        "score,"
        "area_ratio,"
        "aspect_ratio,"
        "circularity,"
        "solidity,"
        "mean_value,"
        "mean_saturation\n"
    )


for index, filename in enumerate(
    TEST_FILENAMES,
    start=1
):

    image_path = DEVELOPMENT_DIR / filename

    print("\n" + "-" * 70)
    print(
        f"[{index}/{len(TEST_FILENAMES)}] "
        f"{filename}"
    )
    print("-" * 70)

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
        f"Candidate regions found: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Save threshold image
    # --------------------------------------------------------

    threshold_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_threshold.png"
    )

    cv2.imwrite(
        str(threshold_path),
        binary
    )

    # --------------------------------------------------------
    # Draw candidates
    # --------------------------------------------------------

    visualization = image.copy()

    # Show top 5 candidates
    for rank, candidate in enumerate(
        candidates[:5],
        start=1
    ):

        contour = candidate["contour"]

        x, y, bw, bh = (
            candidate["bbox"]
        )

        px, py = candidate["point"]

        # Draw contour
        cv2.drawContours(
            visualization,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Draw bounding box
        cv2.rectangle(
            visualization,
            (x, y),
            (x + bw, y + bh),
            (255, 0, 0),
            2
        )

        # Draw candidate point
        cv2.circle(
            visualization,
            (px, py),
            7,
            (0, 0, 255),
            -1
        )

        label = (
            f"#{rank} "
            f"{candidate['score']:.2f}"
        )

        cv2.putText(
            visualization,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with open(
            csv_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                f'"{filename}",'
                f"{rank},"
                f"{px},"
                f"{py},"
                f"{candidate['score']:.5f},"
                f"{candidate['area_ratio']:.5f},"
                f"{candidate['aspect_ratio']:.5f},"
                f"{candidate['circularity']:.5f},"
                f"{candidate['solidity']:.5f},"
                f"{candidate['mean_value']:.2f},"
                f"{candidate['mean_saturation']:.2f}\n"
            )

    visualization_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_candidates.jpg"
    )

    cv2.imwrite(
        str(visualization_path),
        visualization
    )

    print(
        f"Saved visualization: "
        f"{visualization_path}"
    )

    # Print candidates
    for rank, candidate in enumerate(
        candidates[:5],
        start=1
    ):

        px, py = candidate["point"]

        print(
            f"  #{rank}: "
            f"point=({px},{py}), "
            f"score={candidate['score']:.3f}, "
            f"area={candidate['area_ratio']:.3f}, "
            f"circularity={candidate['circularity']:.3f}, "
            f"solidity={candidate['solidity']:.3f}"
        )


print("\n" + "=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(f"\nResults:")
print(OUTPUT_DIR)

print(f"\nCSV:")
print(csv_path)

print(
    "\nIMPORTANT: These candidate regions are "
    "only prompts for SAM."
)

print(
    "They are NOT being treated as final egg masks."
)