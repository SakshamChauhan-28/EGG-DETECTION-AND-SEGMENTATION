from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLOWorld, SAM


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path("task1_segmentation/yolo_sam_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Select test image
# ============================================================

image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

if not image_paths:
    raise RuntimeError("No development images found.")

IMAGE_PATH = image_paths[0]


# ============================================================
# Load models
# ============================================================

print("=" * 70)
print("YOLO-World + SAM 2.1 Tiny Pipeline")
print("=" * 70)

print(f"\nImage: {IMAGE_PATH.name}")

print("\n[1/2] Loading YOLO-World...")

detector = YOLOWorld("yolov8s-worldv2.pt")
detector.set_classes(["egg"])

print("YOLO-World loaded.")


print("\n[2/2] Loading SAM 2.1 Tiny...")

sam = SAM("sam2.1_t.pt")

print("SAM 2.1 Tiny loaded.")


# ============================================================
# Step 1: Detect egg
# ============================================================

print("\n" + "-" * 70)
print("STEP 1 - EGG DETECTION")
print("-" * 70)

detection_results = detector.predict(
    source=str(IMAGE_PATH),
    device="cuda:0",
    imgsz=640,
    conf=0.05,
    verbose=False,
)

detection = detection_results[0]

if detection.boxes is None or len(detection.boxes) == 0:
    raise RuntimeError("YOLO-World did not detect an egg.")

print(f"Egg detections: {len(detection.boxes)}")


# ============================================================
# Extract bounding boxes
# ============================================================

boxes = detection.boxes.xyxy.cpu().numpy()
confidences = detection.boxes.conf.cpu().numpy()


for i, (box, confidence) in enumerate(
    zip(boxes, confidences),
    start=1
):

    x1, y1, x2, y2 = box

    print(
        f"Egg {i}: "
        f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
        f"confidence={confidence:.3f}"
    )


# ============================================================
# Step 2: SAM segmentation
# ============================================================

print("\n" + "-" * 70)
print("STEP 2 - SAM SEGMENTATION")
print("-" * 70)

sam_boxes = boxes.tolist()

sam_results = sam(
    source=str(IMAGE_PATH),
    bboxes=sam_boxes,
    device="cuda:0",
    imgsz=640,
    verbose=False,
)

result = sam_results[0]


# ============================================================
# Save SAM visualization
# ============================================================

sam_visualization = OUTPUT_DIR / (
    f"{IMAGE_PATH.stem}_sam_result.jpg"
)

result.save(filename=str(sam_visualization))


# ============================================================
# Save masks
# ============================================================

if result.masks is None:

    print("SAM did not generate any masks.")

else:

    masks = result.masks.data.cpu().numpy()

    print(f"\nSAM masks generated: {len(masks)}")

    mask_dir = OUTPUT_DIR / f"{IMAGE_PATH.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i, mask in enumerate(masks):

        mask_uint8 = (
            (mask > 0.5).astype(np.uint8) * 255
        )

        mask_path = mask_dir / f"egg_{i + 1:02d}.png"

        cv2.imwrite(
            str(mask_path),
            mask_uint8
        )

        print(f"Saved: {mask_path}")


# ============================================================
# Save combined binary mask
# ============================================================

if result.masks is not None:

    combined_mask = np.any(
        masks > 0.5,
        axis=0
    ).astype(np.uint8) * 255

    combined_path = (
        OUTPUT_DIR /
        f"{IMAGE_PATH.stem}_combined_mask.png"
    )

    cv2.imwrite(
        str(combined_path),
        combined_mask
    )

    print(f"\nCombined mask saved: {combined_path}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("YOLO-WORLD + SAM PIPELINE COMPLETE")
print("=" * 70)

print(f"\nDetection image:")
print(OUTPUT_DIR)

print("\nPipeline:")
print("Image → YOLO-World → Egg bounding box → SAM 2.1 Tiny → Egg mask")