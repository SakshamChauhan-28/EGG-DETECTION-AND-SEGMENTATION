"""
Common utility functions for Nosh Robotics Computer Vision Pipeline.
Provides standardized metric calculations, image I/O, and box conversions.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Computes Intersection over Union (IoU), Dice, Precision, and Recall
    between binary prediction and ground-truth masks.
    """
    p = pred_mask > 127
    g = gt_mask > 127
    
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()
    
    iou = float(intersection / union) if union > 0 else (1.0 if p_sum == 0 and g_sum == 0 else 0.0)
    dice = float(2 * intersection / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 1.0
    precision = float(intersection / p_sum) if p_sum > 0 else 0.0
    recall = float(intersection / g_sum) if g_sum > 0 else 0.0
    
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "predicted_area": int(p_sum),
        "target_area": int(g_sum)
    }


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = box
    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
        ],
        dtype=np.float32,
    )


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    return np.array(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        dtype=np.float32,
    )


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    matrix = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box_a in enumerate(a):
        ax1, ay1, ax2, ay2 = box_a
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j, box_b in enumerate(b):
            bx1, by1, bx2, by2 = box_b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            matrix[i, j] = inter / union if union > 0 else 0.0

    return matrix
