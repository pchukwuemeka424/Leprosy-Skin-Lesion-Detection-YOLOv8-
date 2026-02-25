#!/usr/bin/env python3
"""
Step 2: Train YOLOv8 on leprosy lesion data with transfer learning and augmentation.
Best practices for small medical datasets: augmentation, pretrained backbone, early stopping.
"""
import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

def main():
    # Use absolute dataset path so training works from any cwd
    data_yaml = DATA_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    # Load pretrained YOLOv8 model (transfer learning)
    model = YOLO("yolov8n.pt")  # n=nano, s=small, m=medium; nano good for small datasets

    model.train(
        data=str(data_yaml.resolve()),
        epochs=int(os.environ.get("EPOCHS", 150)),
        imgsz=640,
        batch=8,
        patience=30,
        save=True,
        project=str(RUNS_DIR),
        name="leprosy",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation (helpful for small medical datasets)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Reduce overfitting on small data
        dropout=0.0,
        # Device
        device=None,
        workers=4,
        verbose=True,
    )
    print("Training complete. Best weights saved in runs/detect/leprosy/weights/best.pt")


if __name__ == "__main__":
    main()
