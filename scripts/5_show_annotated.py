#!/usr/bin/env python3
"""
Draw current YOLO labels on dataset images and save to dataset/annotated/.
Use this to visually verify auto-generated or manual annotations.
"""
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
ANNOTATED_DIR = DATASET_ROOT / "annotated"

CLASS_NAMES = {0: "lesion"}
# Patch: semi-transparent fill over lesion area (BGR)
PATCH_COLOR = (0, 200, 100)   # green tint
PATCH_ALPHA = 0.35             # opacity of patch overlay
# Border and label
BOX_COLOR = (0, 255, 0)       # BGR green
TEXT_COLOR = (255, 255, 255)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6


def draw_yolo_labels_on_image(image_path: Path, label_path: Path, class_names: dict) -> np.ndarray:
    """Load image and draw YOLO boxes as patches (filled overlay) + border and 'lesion' label."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    # Parse all boxes (class_id, x1, y1, x2, y2 in pixels)
    boxes = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((xc - nw / 2) * w))
            y1 = max(0, int((yc - nh / 2) * h))
            x2 = min(w, int((xc + nw / 2) * w))
            y2 = min(h, int((yc + nh / 2) * h))
            boxes.append((cls_id, x1, y1, x2, y2))
    # 1) Patches: semi-transparent fill over each lesion area
    overlay = img.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PATCH_COLOR, -1)
    cv2.addWeighted(overlay, PATCH_ALPHA, img, 1 - PATCH_ALPHA, 0, img)
    # 2) Borders and "lesion" labels on top
    for cls_id, x1, y1, x2, y2 in boxes:
        label = class_names.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, TEXT_COLOR, 1)
    return img


def main():
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "val"):
        img_dir = IMAGES_DIR / subset
        label_dir = LABELS_DIR / subset
        out_dir = ANNOTATED_DIR / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        if not img_dir.exists():
            continue
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        for im_path in sorted(img_dir.iterdir()):
            if im_path.suffix.lower() not in exts:
                continue
            label_path = label_dir / (im_path.stem + ".txt")
            out_img = draw_yolo_labels_on_image(im_path, label_path, CLASS_NAMES)
            if out_img is not None:
                out_path = out_dir / im_path.name
                cv2.imwrite(str(out_path), out_img)
                print(f"  {subset}/{im_path.name} -> annotated/{subset}/")
    print(f"Annotated images saved to {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()
