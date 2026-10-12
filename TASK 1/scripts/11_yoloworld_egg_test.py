from pathlib import Path
import cv2
from ultralytics import YOLOWorld


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path("task1_segmentation/yoloworld_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Select one image
# ============================================================

image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

if not image_paths:
    raise RuntimeError("No development images found.")

IMAGE_PATH = image_paths[0]


# ============================================================
# Load YOLO-World
# ============================================================

print("=" * 70)
print("YOLO-World Egg Detection Test")
print("=" * 70)

print(f"\nImage: {IMAGE_PATH.name}")

print("\nLoading YOLO-World...")

model = YOLOWorld("yolov8s-worldv2.pt")

print("Model loaded.")


# ============================================================
# Set custom vocabulary
# ============================================================

model.set_classes(["egg"])

print("Custom class: egg")


# ============================================================
# Run detection
# ============================================================

print("\nRunning egg detection...")

results = model.predict(
    source=str(IMAGE_PATH),
    device="cuda:0",
    imgsz=640,
    conf=0.05,
    verbose=False,
)

result = results[0]


# ============================================================
# Print detections
# ============================================================

print("\nDetections:")

if result.boxes is None or len(result.boxes) == 0:

    print("NO EGGS DETECTED.")

else:

    boxes = result.boxes

    print(f"Number of detections: {len(boxes)}")

    for i, box in enumerate(boxes):

        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu())

        x1, y1, x2, y2 = xyxy.astype(int)

        print(
            f"  Egg {i + 1}: "
            f"box=({x1}, {y1}, {x2}, {y2}), "
            f"confidence={confidence:.3f}"
        )


# ============================================================
# Save visualization
# ============================================================

output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_detection.jpg"

result.save(filename=str(output_path))

print(f"\nVisualization saved to:")
print(output_path)


print("\n" + "=" * 70)
print("YOLO-WORLD TEST COMPLETE")
print("=" * 70)