from pathlib import Path
import cv2
from ultralytics import SAM

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = Path("split_dataset/development")

# Pick the first development image
image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

IMAGE_PATH = image_paths[0]

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 70)
print("SAM 2.1 Tiny - Point Prompt Test")
print("=" * 70)

print(f"\nImage: {IMAGE_PATH.name}")

model = SAM("sam2.1_t.pt")

print("\nModel loaded.")

# --------------------------------------------------
# Read image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError("Could not read image.")

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")

# --------------------------------------------------
# Temporary center point
# --------------------------------------------------

point_x = width // 2
point_y = height // 2

print(f"Prompt point: ({point_x}, {point_y})")

# --------------------------------------------------
# Prompted segmentation
# --------------------------------------------------

results = model(
    str(IMAGE_PATH),
    points=[[point_x, point_y]],
    labels=[1],
    device="cuda:0",
    imgsz=640,
    verbose=False,
)

result = results[0]

# --------------------------------------------------
# Save result
# --------------------------------------------------

output_dir = Path("task1_segmentation/point_test")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{IMAGE_PATH.stem}_point_result.jpg"

result.save(filename=str(output_path))

print("\nSegmentation complete.")

if result.masks is not None:
    print(f"Masks generated: {len(result.masks)}")
else:
    print("No mask generated.")

print(f"Result saved to: {output_path}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)