"""
Task 3: Quantitative and Qualitative Segmentation Evaluation
Benchmarks classical segmentation predictions against Task 1 pseudo-ground-truth masks.

Metrics computed:
  - Intersection over Union (IoU)
  - Dice Similarity Coefficient (F1)
  - Precision
  - Recall

Usage:
  python src/task3/evaluate.py --pred-dir task2_output/masks --gt-dir task1_output/masks --output-csv evaluation_report.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd

# Allow running directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.utils import compute_mask_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3: Segmentation Evaluation against Pseudo-Ground-Truth")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing prediction masks (.png or .jpg)")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory containing pseudo-GT masks")
    parser.add_argument("--output-csv", type=str, default="evaluation_report.csv", help="Path to save output CSV metrics report")
    return parser.parse_args()


def load_union_gt_mask(gt_dir: Path, stem: str, shape: tuple) -> np.ndarray:
    """
    Finds and unions all instance masks for an image in Task 1 output.
    E.g., matches stem_egg_1.png, stem_egg_2.png or stem.png.
    """
    # 1. Check exact stem match
    direct = gt_dir / f"{stem}.png"
    if direct.is_file():
        m = cv2.imread(str(direct), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return m

    # 2. Check multi-instance matching
    instance_masks = sorted(gt_dir.glob(f"{stem}_egg_*.png"))
    if not instance_masks:
        instance_masks = sorted(gt_dir.glob(f"{stem}*.png"))
        
    union = np.zeros(shape, dtype=np.uint8)
    for p in instance_masks:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            union[m > 127] = 255
    return union


def evaluate_directories(pred_dir: Path, gt_dir: Path, output_csv: Path = None) -> pd.DataFrame:
    pred_files = sorted(list(pred_dir.glob("*.png")) + list(pred_dir.glob("*.jpg")))
    if not pred_files:
        raise ValueError(f"No masks found in prediction directory: {pred_dir}")
        
    records = []
    print(f"Evaluating {len(pred_files)} prediction masks against ground-truth in {gt_dir}...")
    
    for pf in pred_files:
        stem = pf.stem
        # Remove suffixes like '_e1' if present
        clean_stem = stem.replace("_e1", "")
        
        pred = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
            
        gt = load_union_gt_mask(gt_dir, clean_stem, pred.shape)
        metrics = compute_mask_metrics(pred, gt)
        metrics["image"] = pf.name
        metrics["clean_stem"] = clean_stem
        records.append(metrics)
        
    df = pd.DataFrame(records)
    
    print("=" * 65)
    print("QUANTITATIVE EVALUATION SCORECARD")
    print("=" * 65)
    print(f"Evaluated Samples (N): {len(df)}")
    print(f"Mean IoU:              {df['iou'].mean():.4f}")
    print(f"Median IoU:            {df['iou'].median():.4f}")
    print(f"Mean Dice (F1):        {df['dice'].mean():.4f}")
    print(f"Mean Precision:        {df['precision'].mean():.4f}")
    print(f"Mean Recall:           {df['recall'].mean():.4f}")
    print("=" * 65)
    
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_csv), index=False)
        print(f"Detailed per-image report saved to: {output_csv}")
        
    return df


def main():
    args = parse_args()
    evaluate_directories(Path(args.pred_dir), Path(args.gt_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()
