from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import SAM


# ============================================================
# Configuration
# ============================================================

DEVELOPMENT_DIR = Path("split_dataset/development")

OUTPUT_DIR = Path("task1_segmentation/sam_tiny_auto_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Use a few difficult development images
TEST_FILENAMES = [
    "not_damaged_100_jpg.rf.5d780a6024004fa2f68741fa56ddf4c9.jpg",
    "not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a.jpg",
    "not_damaged_128_jpg.rf.95a7ef96dcc9e14fa7137708df6f2e60.jpg",
    "not_damaged_148_jpg.rf.8a3e1c0e1dae236abe0e9a75a4882bda.jpg",
]


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("SAM 2.1 TINY - AUTOMATIC SEGMENTATION TEST")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading SAM 2.1 Tiny...")

model = SAM("sam2.1_t.pt")

print("Model loaded.")


# ============================================================
# Process images
# ============================================================

for index, filename in enumerate(TEST_FILENAMES, start=1):

    image_path = DEVELOPMENT_DIR / filename

    print("\n" + "-" * 70)
    print(f"[{index}/{len(TEST_FILENAMES)}] {filename}")
    print("-" * 70)

    if not image_path.exists():
        print("Image not found.")
        continue

    try:

        # Clear cached GPU memory between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Running automatic segmentation...")

        results = model(
            str(image_path),
            device="cuda:0",
            imgsz=640,
            verbose=False,
        )

        result = results[0]

        if result.masks is None:

            print("No masks generated.")

            continue

        masks = result.masks.data.cpu().numpy()

        print(f"Masks generated: {len(masks)}")

        # ------------------------------------------------------
        # Save visualization
        # ------------------------------------------------------

        visualization_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_result.jpg"
        )

        result.save(
            filename=str(visualization_path)
        )

        print(
            f"Visualization: {visualization_path}"
        )

        # ------------------------------------------------------
        # Save masks
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
                f"mask_{mask_index:03d}.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask_uint8
            )

        print("Masks saved.")

    except RuntimeError as e:

        print(f"RUNTIME ERROR: {e}")

        if "out of memory" in str(e).lower():
            print("CUDA OUT OF MEMORY.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:

        print(f"ERROR: {e}")


print("\n" + "=" * 70)
print("AUTOMATIC SAM TEST COMPLETE")
print("=" * 70)