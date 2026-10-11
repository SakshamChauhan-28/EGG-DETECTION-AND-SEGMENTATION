"""
Task 2: Classical Image-Processing Segmentation (Algorithm E1)
Final Selected Classical Model: Dual-Space Brightness & Low Saturation Baseline

Parameters:
  BRIGHT_PERCENTILE = 65 (LAB Lightness channel)
  MAX_SATURATION = 100   (HSV Saturation channel)
  MIN_AREA = 500         (Minimum connected component pixel area)
  MORPH_KERNEL_SIZE = 7  (Elliptical structuring element for opening/closing)
  MAX_COMPONENTS = 6     (Maximum retained high-confidence components)

Usage:
  python src/task2/baseline_e1.py --image path/to/image.jpg --output-dir outputs/task2/
  python src/task2/baseline_e1.py --dataset-dir split_dataset/development --output-dir task2_dev_output/
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Classical Image-Processing Segmentation (Algorithm E1)")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for segmentation")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing images to process")
    parser.add_argument("--output-dir", type=str, default="outputs/task2", help="Directory for saved masks & overlays")
    parser.add_argument("--bright-percentile", type=float, default=65.0, help="LAB Lightness percentile threshold (default: 65)")
    parser.add_argument("--max-saturation", type=int, default=100, help="Maximum HSV saturation threshold (default: 100)")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum connected component pixel area (default: 500)")
    parser.add_argument("--morph-kernel", type=int, default=7, help="Morphology kernel size (default: 7)")
    parser.add_argument("--max-components", type=int, default=6, help="Maximum components retained (default: 6)")
    return parser.parse_args()


class ClassicalSegmenterE1:
    def __init__(self, bright_percentile: float = 65.0,
                 max_saturation: int = 100,
                 min_area: int = 500,
                 morph_kernel_size: int = 7,
                 max_components: int = 6):
        self.bright_percentile = bright_percentile
        self.max_saturation = max_saturation
        self.min_area = min_area
        self.morph_kernel_size = morph_kernel_size
        self.max_components = max_components
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    def segment(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes Algorithm E1 on an input BGR image.
        Returns:
            binary_mask: uint8 array (0 or 255)
            components: list of retained connected component metadata
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        saturation = hsv[:, :, 1]
        lightness = lab[:, :, 0]

        # 1. Intensity and Saturation Thresholding
        thresh_lightness = np.percentile(lightness, self.bright_percentile)
        bright_mask = lightness >= thresh_lightness
        low_saturation = saturation <= self.max_saturation

        raw_mask = (bright_mask & low_saturation).astype(np.uint8) * 255

        # 2. Morphological Opening (noise removal) and Closing (hole filling)
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)

        # 3. Connected Component Extraction & Filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        candidates = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]

            candidates.append({
                "label": i,
                "area": area,
                "bbox": (x, y, w, h),
                "centroid": (float(cx), float(cy))
            })

        # Sort by area descending and cap to max_components
        candidates.sort(key=lambda c: c["area"], reverse=True)
        retained = candidates[:self.max_components]

        # 4. Construct Final Filtered Binary Mask
        final_mask = np.zeros_like(closed)
        for c in retained:
            final_mask[labels == c["label"]] = 255

        return final_mask, retained


def process_image(img_path: Path, segmenter: ClassicalSegmenterE1, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
        
    mask, comps = segmenter.segment(img)
    
    stem = img_path.stem
    mask_dir = out_dir / "masks"
    vis_dir = out_dir / "visualizations"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mask
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
    
    # Create side-by-side visualization overlay
    vis = img.copy()
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 127] = (0, 220, 220) # Yellow overlay
    vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.45, 0)
    
    for c in comps:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(vis, (int(c["centroid"][0]), int(c["centroid"][1])), 4, (0, 0, 255), -1)
        
    combo = np.hstack([img, vis])
    cv2.imwrite(str(vis_dir / f"{stem}_e1.jpg"), combo)


def main():
    args = parse_args()
    segmenter = ClassicalSegmenterE1(
        bright_percentile=args.bright_percentile,
        max_saturation=args.max_saturation,
        min_area=args.min_area,
        morph_kernel_size=args.morph_kernel,
        max_components=args.max_components
    )
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        p = Path(args.image)
        print(f"Executing Algorithm E1 on {p}...")
        process_image(p, segmenter, out_dir)
        print(f"Results saved in: {out_dir}")
    elif args.dataset_dir:
        ds_path = Path(args.dataset_dir)
        imgs = sorted(list(ds_path.glob("*.jpg")) + list(ds_path.glob("*.png")))
        print(f"Executing Algorithm E1 across {len(imgs)} images in {ds_path}...")
        for p in imgs:
            process_image(p, segmenter, out_dir)
        print(f"Processing complete. Masks and visualizations saved in: {out_dir}")
    else:
        print("Please provide --image or --dataset-dir. Use --help for options.")


if __name__ == "__main__":
    main()
