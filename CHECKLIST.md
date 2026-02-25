# Pipeline checklist: Leprosy lesion detection

| Requirement | Status | Where |
|-------------|--------|--------|
| **1. Annotate lesions (bounding boxes)** | Done | `scripts/1_auto_annotate.py` generates boxes (SAM or OpenCV fallback). View in `dataset/annotated/` via `scripts/5_show_annotated.py`. |
| **2. Convert to YOLO format** | Done | Labels in `dataset/labels/` are one `.txt` per image: `class_id x_center y_center width height` (normalized 0–1). Config: `dataset/data.yaml`. |
| **3. Train using pretrained weights** | Done | `scripts/2_train.py` loads `yolov8n.pt` and trains on your data (transfer learning). |
| **4. Evaluate: mAP** | Done | `scripts/3_evaluate.py` reports **mAP50** and **mAP50-95**. |
| **5. Evaluate: Precision** | Done | Same script reports **Precision** (box.mp). |
| **6. Evaluate: Recall** | Done | Same script reports **Recall** (box.mr). |

## Quick verify

```bash
pip install ultralytics opencv-python  # if not already

# 0. Prepare + annotate (already done)
python3 scripts/0_prepare_dataset.py
python3 scripts/1_auto_annotate.py

# 1. Train (produces runs/detect/leprosy/weights/best.pt)
python3 scripts/2_train.py
# Quick test: EPOCHS=5 python3 scripts/2_train.py

# 2. Evaluate (mAP, Precision, Recall)
python3 scripts/3_evaluate.py
```

After training, evaluation prints:
```
--- Summary ---
mAP50:      0.xxxx
mAP50-95:   0.xxxx
Precision:  0.xxxx
Recall:     0.xxxx
```
