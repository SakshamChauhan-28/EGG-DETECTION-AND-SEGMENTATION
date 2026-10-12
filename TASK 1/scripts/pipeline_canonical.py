"""
Task 1: Pretrained Foundation Segmentation Pipeline
Architecture: Grounding DINO Tiny (Semantic Detection) + SAM 2.1 Tiny (Instance Segmentation)

Usage:
  python src/task1/pipeline.py --image path/to/image.jpg --output-dir outputs/
  python src/task1/pipeline.py --dataset-dir split_dataset/development --output-dir task1_output/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Import Ultralytics SAM
try:
    from ultralytics import SAM
except ImportError:
    SAM = None


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: Grounding DINO + SAM Egg Segmentation Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for inference")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing images to process")
    parser.add_argument("--output-dir", type=str, default="outputs/task1", help="Output directory for masks & visual overlays")
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="Path to sam2.1_t.pt checkpoint")
    parser.add_argument("--dino-model", type=str, default="IDEA-Research/grounding-dino-tiny", help="Grounding DINO model ID")
    parser.add_argument("--prompt", type=str, default="egg.", help="Text prompt for grounding")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Detection bounding-box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.20, help="Text grounding threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    return parser.parse_args()


def find_checkpoint(user_path: str = None) -> Path:
    """Locate SAM 2.1 Tiny checkpoint in user path, weights/, or repository root."""
    if user_path and Path(user_path).is_file():
        return Path(user_path)
    
    candidates = [
        Path("weights/sam2.1_t.pt"),
        Path("sam2.1_t.pt"),
        Path("../sam2.1_t.pt"),
        Path("../../sam2.1_t.pt")
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
            
    # Default fallback to expected local name
    return Path("sam2.1_t.pt")


class GroundingDinoSAM:
    def __init__(self, dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
                 sam_checkpoint: str = None,
                 prompt: str = "egg.",
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.20,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        print(f"Loading Grounding DINO Tiny ({dino_model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
        self.dino_model.eval()
        
        checkpoint_path = find_checkpoint(sam_checkpoint)
        print(f"Loading SAM 2.1 Tiny from: {checkpoint_path}...")
        if SAM is None:
            raise ImportError("Ultralytics SAM is not installed. Please install via: pip install ultralytics")
        if not checkpoint_path.is_file():
            print(f"[Warning] SAM checkpoint '{checkpoint_path}' not found locally. It may be downloaded automatically or must be placed in repository root.")
        self.sam = SAM(str(checkpoint_path))

    @torch.no_grad()
    def detect_boxes(self, pil_image: Image.Image) -> List[Tuple[np.ndarray, float]]:
        w, h = pil_image.size
        inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)
        
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[(h, w)]
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[(h, w)]
            )
            
        res = results[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        
        detections = []
        for box, score in zip(boxes, scores):
            detections.append((box, float(score)))
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    def segment_image(self, image_path: Path) -> Dict[str, Any]:
        """Runs Grounding DINO detection and SAM prompting on a single image."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        detections = self.detect_boxes(pil_img)
        masks = []
        
        if detections:
            boxes = [d[0] for d in detections]
            # SAM prediction with box prompts
            sam_results = self.sam(str(image_path), bboxes=boxes, verbose=False)
            if sam_results and sam_results[0].masks is not None:
                raw_masks = sam_results[0].masks.data.cpu().numpy()
                for m in raw_masks:
                    # Resize to original resolution if needed
                    m_bin = (m > 0.5).astype(np.uint8) * 255
                    if m_bin.shape[:2] != img_bgr.shape[:2]:
                        m_bin = cv2.resize(m_bin, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    masks.append(m_bin)
                    
        return {
            "image_path": str(image_path),
            "detections": detections,
            "masks": masks,
            "image_shape": img_bgr.shape
        }


def save_outputs(result: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "visualizations"
    mask_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    stem = Path(result["image_path"]).stem
    img = cv2.imread(result["image_path"])
    vis = img.copy()
    
    # Save individual masks and composite
    union = np.zeros(img.shape[:2], dtype=np.uint8)
    for idx, (mask, det) in enumerate(zip(result["masks"], result["detections"]), 1):
        m_path = mask_dir / f"{stem}_egg_{idx}.png"
        cv2.imwrite(str(m_path), mask)
        union = np.maximum(union, mask)
        
        # Color overlay on visualization
        color = np.array([30, 200, 30], dtype=np.uint8)
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 127] = color
        vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.45, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in det[0]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"Egg {idx}: {det[1]:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
    cv2.imwrite(str(vis_dir / f"{stem}_seg.jpg"), vis)


def main():
    args = parse_args()
    pipeline = GroundingDinoSAM(
        dino_model_id=args.dino_model,
        sam_checkpoint=args.sam_checkpoint,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        img_path = Path(args.image)
        print(f"Processing image: {img_path}...")
        res = pipeline.segment_image(img_path)
        save_outputs(res, out_dir)
        print(f"Saved results to: {out_dir}")
    elif args.dataset_dir:
        ds_path = Path(args.dataset_dir)
        imgs = sorted(list(ds_path.glob("*.jpg")) + list(ds_path.glob("*.png")))
        print(f"Processing {len(imgs)} images in {ds_path}...")
        for p in imgs:
            res = pipeline.segment_image(p)
            save_outputs(res, out_dir)
        print(f"Batch processing complete. Outputs in: {out_dir}")
    else:
        print("Please provide either --image or --dataset-dir. Use --help for usage.")


if __name__ == "__main__":
    main()
