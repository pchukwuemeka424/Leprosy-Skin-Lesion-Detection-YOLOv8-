#!/usr/bin/env python3
"""
Step 4: Run inference and save images with predicted bounding boxes drawn.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "leprosy" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "leprosy" / "predict"


def main(weights_path: Path = None, source: Path = None, save_dir: Path = None):
    weights_path = weights_path or DEFAULT_WEIGHTS
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Train first: python scripts/2_train.py")
        return
    source = source or IMAGES_DIR  # can be train, val, or a folder of images
    save_dir = save_dir or OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source),
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        line_width=2,
    )
    print(f"Predictions saved to {save_dir}")
    return results


if __name__ == "__main__":
    import sys
    w = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    main(weights_path=w, source=src)
