from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOWorld, SAM


# ============================================================
# Configuration
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path("task1_segmentation/stress_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.05


# ============================================================
# Select representative development images
# ============================================================

image_paths = sorted([
    p for p in DEVELOPMENT_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

if len(image_paths) == 0:
    raise RuntimeError("No development images found.")

# Select 12 images distributed across the development set
if len(image_paths) > 12:
    indices = np.linspace(
        0,
        len(image_paths) - 1,
        12,
        dtype=int
    )

    image_paths = [image_paths[i] for i in indices]


# ============================================================
# Load models
# ============================================================

print("=" * 70)
print("TASK 1 - YOLO-WORLD + SAM 2.1 TINY STRESS TEST")
print("=" * 70)

print(f"\nImages selected: {len(image_paths)}")

print("\nLoading YOLO-World...")

detector = YOLOWorld("yolov8s-worldv2.pt")
detector.set_classes(["egg"])

print("YOLO-World loaded.")

print("\nLoading SAM 2.1 Tiny...")

sam = SAM("sam2.1_t.pt")

print("SAM 2.1 Tiny loaded.")


# ============================================================
# Results log
# ============================================================

log_path = OUTPUT_DIR / "stress_test_results.csv"

with open(log_path, "w", encoding="utf-8") as f:
    f.write(
        "image,"
        "detections,"
        "masks,"
        "confidence_min,"
        "confidence_max,"
        "status\n"
    )


# ============================================================
# Process images
# ============================================================

for index, image_path in enumerate(image_paths, start=1):

    print("\n" + "-" * 70)
    print(f"[{index}/{len(image_paths)}] {image_path.name}")
    print("-" * 70)

    try:

        # ------------------------------------------------------
        # YOLO-World
        # ------------------------------------------------------

        detection_results = detector.predict(
            source=str(image_path),
            device="cuda:0",
            imgsz=640,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        detection = detection_results[0]

        if (
            detection.boxes is None
            or len(detection.boxes) == 0
        ):

            print("  YOLO: NO EGGS DETECTED")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f'"{image_path.name}",'
                    f"0,0,,,NO_DETECTION\n"
                )

            continue


        boxes = detection.boxes.xyxy.cpu().numpy()
        confidences = detection.boxes.conf.cpu().numpy()

        print(f"  YOLO detections: {len(boxes)}")

        for i, confidence in enumerate(confidences):
            print(
                f"    Egg {i + 1}: "
                f"confidence={confidence:.3f}"
            )


        # ------------------------------------------------------
        # Save detection visualization
        # ------------------------------------------------------

        detection_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_detection.jpg"
        )

        detection.save(
            filename=str(detection_path)
        )


        # ------------------------------------------------------
        # SAM
        # ------------------------------------------------------

        sam_results = sam(
            source=str(image_path),
            bboxes=boxes.tolist(),
            device="cuda:0",
            imgsz=640,
            verbose=False,
        )

        result = sam_results[0]


        if result.masks is None:

            print("  SAM: NO MASKS")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f'"{image_path.name}",'
                    f"{len(boxes)},0,"
                    f"{confidences.min():.4f},"
                    f"{confidences.max():.4f},"
                    f"NO_MASK\n"
                )

            continue


        masks = result.masks.data.cpu().numpy()

        print(f"  SAM masks: {len(masks)}")


        # ------------------------------------------------------
        # Save segmentation visualization
        # ------------------------------------------------------

        segmentation_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_segmentation.jpg"
        )

        result.save(
            filename=str(segmentation_path)
        )


        # ------------------------------------------------------
        # Save individual masks
        # ------------------------------------------------------

        mask_dir = (
            OUTPUT_DIR /
            f"{image_path.stem}_masks"
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for mask_index, mask in enumerate(masks):

            mask_uint8 = (
                (mask > 0.5).astype(np.uint8) * 255
            )

            mask_path = (
                mask_dir /
                f"egg_{mask_index + 1:02d}.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask_uint8
            )


        # ------------------------------------------------------
        # Combined mask
        # ------------------------------------------------------

        combined_mask = (
            np.any(masks > 0.5, axis=0)
            .astype(np.uint8)
            * 255
        )

        combined_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_combined_mask.png"
        )

        cv2.imwrite(
            str(combined_path),
            combined_mask
        )


        # ------------------------------------------------------
        # Log
        # ------------------------------------------------------

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f'"{image_path.name}",'
                f"{len(boxes)},"
                f"{len(masks)},"
                f"{confidences.min():.4f},"
                f"{confidences.max():.4f},"
                f"SUCCESS\n"
            )

        print("  Status: SUCCESS")


    except RuntimeError as e:

        print(f"  RUNTIME ERROR: {e}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f'"{image_path.name}",'
                f",,,,"
                f"RUNTIME_ERROR\n"
            )

    except Exception as e:

        print(f"  ERROR: {e}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f'"{image_path.name}",'
                f",,,,"
                f"ERROR\n"
            )


# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 70)
print("STRESS TEST COMPLETE")
print("=" * 70)

print("\nResults:")
print(OUTPUT_DIR)

print("\nCSV log:")
print(log_path)

print(
    "\nIMPORTANT: These are development-set experiments only."
)
print(
    "The 22-image evaluation set has not been used."
)