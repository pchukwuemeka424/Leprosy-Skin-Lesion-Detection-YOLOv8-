#!/usr/bin/env python3
"""
Step 3: Evaluate trained YOLOv8 model — mAP, precision, recall.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"
# Default: use latest run; override with CLI arg
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "leprosy" / "weights" / "best.pt"


def main(weights_path: Path = None):
    weights_path = weights_path or DEFAULT_WEIGHTS
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Train first: python scripts/2_train.py")
        return
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(DATA_YAML.resolve()), split="val", verbose=True)
    # Ultralytics DetectionMetrics: .box has map50, map, mp (precision), mr (recall)
    box = metrics.box
    mAP50 = getattr(box, "map50", 0.0)
    mAP = getattr(box, "map", 0.0)
    prec = getattr(box, "mp", 0.0)
    rec = getattr(box, "mr", 0.0)
    print("\n--- Summary ---")
    print(f"mAP50:      {mAP50:.4f}")
    print(f"mAP50-95:   {mAP:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    return metrics


if __name__ == "__main__":
    import sys
    w = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(w)
