#!/usr/bin/env python3
"""
Step 1: Auto-generate bounding boxes for skin lesions and save in YOLO format.
Uses Segment Anything Model (SAM) when available for pseudo-labels; otherwise
uses a simple OpenCV-based fallback (weaker, for testing the pipeline).
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

# Filter masks by area: keep regions between min_ratio and max_ratio of image area (lesion-like size)
MIN_AREA_RATIO = 0.005   # 0.5% of image
MAX_AREA_RATIO = 0.85    # avoid full-image segments
MIN_SIDE_PX = 15         # minimum width or height in pixels

# SAM checkpoint: download from https://github.com/facebookresearch/segment-anything#model-checkpoints
# e.g. sam_vit_b_01ec64.pth (default), sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")  # vit_b, vit_l, vit_h


def mask_to_yolo_boxes(masks_ret, height: int, width: int):
    """Convert SAM mask generator output to YOLO format boxes (normalized xywh)."""
    boxes = []
    for m in masks_ret:
        b = m.get("bbox")
        if b is None or len(b) != 4:
            continue
        # SAM can return either xyxy (x1,y1,x2,y2) or xywh (x,y,w,h)
        v0, v1, v2, v3 = b
        if v2 <= width and v3 <= height and v0 + v2 <= width + 10 and v1 + v3 <= height + 10:
            x1, y1 = v0, v1
            x2, y2 = v0 + v2, v1 + v3
        else:
            x1, y1, x2, y2 = v0, v1, v2, v3
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        w = x2 - x1
        h = y2 - y1
        if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
            continue
        area_ratio = (w * h) / (width * height)
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        xc_n = x_center / width
        yc_n = y_center / height
        wn = w / width
        hn = h / height
        if wn <= 0 or hn <= 0 or xc_n < 0 or yc_n < 0 or xc_n > 1 or yc_n > 1:
            continue
        boxes.append((xc_n, yc_n, wn, hn))
    return boxes


def fallback_boxes_opencv(image_bgr, height: int, width: int):
    """
    Simple fallback: skin/lesion-like regions via color and luminance.
    Use only when SAM is not available; quality is limited.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Emphasize darker regions (often lesions) and blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold or simple threshold on darker areas
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
            continue
        area_ratio = (w * h) / (width * height)
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        xc_n = x_center / width
        yc_n = y_center / height
        wn = w / width
        hn = h / height
        boxes.append((xc_n, yc_n, wn, hn))
    # NMS-like: merge overlapping boxes (simple merge by keeping larger)
    return _merge_overlapping_boxes(boxes)


def _merge_overlapping_boxes(boxes, iou_thresh=0.5):
    """Simple merge: remove smaller box if IoU with a larger one is high."""
    if len(boxes) <= 1:
        return boxes
    # Sort by area descending
    areas = [b[2] * b[3] for b in boxes]
    order = np.argsort(areas)[::-1]
    keep = []
    for i in order:
        xc, yc, w, h = boxes[i]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        discard = False
        for j in keep:
            xc2, yc2, w2, h2 = boxes[j]
            x12 = xc2 - w2 / 2
            y12 = yc2 - h2 / 2
            x22 = xc2 + w2 / 2
            y22 = yc2 + h2 / 2
            xi1 = max(x1, x12)
            yi1 = max(y1, y12)
            xi2 = min(x2, x22)
            yi2 = min(y2, y22)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            a1 = w * h
            a2 = w2 * h2
            iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
            if iou >= iou_thresh:
                discard = True
                break
        if not discard:
            keep.append(i)
    return [boxes[i] for i in sorted(keep)]


def get_sam_generator():
    """Build SAM automatic mask generator if segment_anything is available."""
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        return None
    ckpt = Path(SAM_CHECKPOINT)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / "checkpoints" / ckpt
    if not ckpt.exists():
        print(f"SAM checkpoint not found at {ckpt}. Using OpenCV fallback.")
        return None
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(ckpt))
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=0,  # faster
        min_mask_region_area=100,
    )
    return mask_generator


def process_image(image_path: Path, mask_generator, out_label_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    if mask_generator is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img_rgb)
        boxes = mask_to_yolo_boxes(masks, h, w)
    else:
        boxes = fallback_boxes_opencv(img, h, w)
    # If no boxes from SAM/fallback, write one conservative box (center 80% region) so training has a label
    if len(boxes) == 0:
        boxes = [(0.5, 0.5, 0.4, 0.4)]
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        for (xc, yc, wn, hn) in boxes:
            f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
    return len(boxes)


def main():
    mask_generator = get_sam_generator()
    if mask_generator is None:
        print("Using OpenCV fallback for pseudo-labels. For better results, install SAM:")
        print("  pip install segment-anything")
        print("  Download checkpoint: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print("  Place e.g. sam_vit_b_01ec64.pth in project_01/checkpoints/")
    for subset in ("train", "val"):
        img_dir = IMAGES_DIR / subset
        label_dir = LABELS_DIR / subset
        label_dir.mkdir(parents=True, exist_ok=True)
        if not img_dir.exists():
            continue
        for im_path in sorted(img_dir.iterdir()):
            if not im_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
                continue
            label_path = label_dir / (im_path.stem + ".txt")
            n = process_image(im_path, mask_generator, label_path)
            print(f"  {subset}/{im_path.name} -> {n} box(es)")
    print("Labels saved to dataset/labels/train and dataset/labels/val (YOLO format).")


if __name__ == "__main__":
    main()
