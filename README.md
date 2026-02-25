# Leprosy Skin Lesion Detection (YOLOv8)

Early detection of leprosy skin lesions using YOLOv8 with auto-generated pseudo-labels, transfer learning, and data augmentation.

---

## Process overview

End-to-end flow from raw images to screening:

```
Raw skin images → [0] Prepare dataset → [1] Pseudo-labels (SAM/OpenCV) → [2] Train YOLOv8
                                                                              ↓
Dashboard / Inference ← [4] Inference (optional) ← [3] Evaluate (mAP, P, R) ←┘
```

| Step | Script | What it does |
|------|--------|--------------|
| **0** | `0_prepare_dataset.py` | Split source images into `dataset/images/train` and `dataset/images/val` (80/20). |
| **1** | `1_auto_annotate.py` | Generate YOLO-format labels (pseudo-labels) via SAM or OpenCV fallback. |
| **2** | `2_train.py` | Train YOLOv8n on `dataset/data.yaml`; save best/last weights. |
| **3** | `3_evaluate.py` | Compute validation mAP, precision, recall. |
| **4** | `4_inference.py` | Run model on images and save predictions with bounding boxes. |
| — | `app.py` | Streamlit dashboard: upload image → Leprosy / Not Leprosy + confidence. |

---

## Architecture

- **Model**: [YOLOv8](https://docs.ultralytics.com/) (Ultralytics), **nano** variant (`yolov8n.pt`) for object detection.
- **Task**: Single-class detection — one class `lesion` (leprosy skin lesion).
- **Input**: RGB images; training/inference at **640×640** (with letterboxing).
- **Output**: Bounding boxes (xywh) + confidence per detection; dashboard uses **≥25% confidence** as “Leprosy” threshold.

**Training setup:**

| Component | Choice |
|-----------|--------|
| Pretrained backbone | YOLOv8n (transfer learning) |
| Optimizer | AdamW |
| Learning rate | `lr0=1e-3`, `lrf=0.01`, warmup 3 epochs |
| Regularization | Weight decay `0.0005`, momentum `0.937` |
| Epochs | 150 (env `EPOCHS`); early stopping **patience=30** |
| Augmentation | Mosaic, mixup, HSV, fliplr, scale, translate, shear, degrees |

**Dataset config** (`dataset/data.yaml`): paths to `images/train`, `images/val`; `nc: 1`, class name `lesion`.

---

## Setup

Use **Python 3.10, 3.11, or 3.12** (PyTorch/Ultralytics do not support Python 3.14 yet).

```bash
cd /Users/acehub/Desktop/project_01
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Or: pip install ultralytics opencv-python
```

## Dataset layout

- **Images**: `dataset/images/train/`, `dataset/images/val/`
- **Labels (YOLO)**: `dataset/labels/train/`, `dataset/labels/val/`
- **Config**: `dataset/data.yaml`

---

## Pipeline steps (run in order)

Follow these steps after setup. Step numbers match the process overview above.

### 0. Prepare dataset

Copy images into train/val (80/20 split):

```bash
python scripts/0_prepare_dataset.py
# Or from another folder: python scripts/0_prepare_dataset.py /path/to/images
```

### 1. Auto-generate bounding boxes (pseudo-labels)

Generates YOLO-format labels from a pre-trained model or a simple fallback:

- **With SAM (recommended)**  
  - Install: `pip install segment-anything` (and PyTorch if needed).  
  - Download a [SAM checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) (e.g. `sam_vit_b_01ec64.pth`).  
  - Place it in `checkpoints/` and run:

  ```bash
  mkdir -p checkpoints
  # put sam_vit_b_01ec64.pth in checkpoints/
  python scripts/1_auto_annotate.py
  ```

- **Without SAM**  
  Uses an OpenCV-based fallback (weaker). Run:

  ```bash
  python scripts/1_auto_annotate.py
  ```

Labels are written to `dataset/labels/train/` and `dataset/labels/val/` (one `.txt` per image, class 0 = lesion in normalized xywh).

### 2. Train YOLOv8

Transfer learning from a pretrained YOLOv8n model, with augmentation and early stopping:

```bash
python scripts/2_train.py
```

Outputs: `runs/detect/leprosy/weights/best.pt`, `last.pt`.

### 3. Evaluate

Compute mAP, precision, and recall on the validation set:

```bash
python scripts/3_evaluate.py
# Or: python scripts/3_evaluate.py runs/detect/leprosy/weights/best.pt
```

### 4. Inference (draw boxes on images)

Save predicted images with bounding boxes:

```bash
python scripts/4_inference.py
# Or: python scripts/4_inference.py path/to/best.pt path/to/images
```

Predictions are saved under `runs/detect/leprosy/predict/`.

## Small-dataset practices

- **Transfer learning**: Pretrained YOLOv8n.
- **Augmentation**: Mosaic, mixup, HSV, fliplr, scale, translate, shear (see `scripts/2_train.py`).
- **Early stopping**: `patience=30`.
- **Regularization**: AdamW, weight decay, warmup.

## Streamlit dashboard

Modern healthcare-themed screening dashboard: upload a skin image and get **Leprosy** / **Not Leprosy** with confidence.

**Features:**
- Teal/clinical UI with white background, metric cards, and status badges
- Prediction and confidence shown as healthcare-style KPIs (percentage + progress bar)
- Sidebar with screening assistant info and detection threshold (≥25%)
- Input image and results side-by-side; detection overlay at reduced size when lesions are found

```bash
streamlit run app.py
```

Requires trained weights at `runs/detect/leprosy/weights/best.pt` (run `scripts/2_train.py` first).

## File overview

| Script | Purpose |
|--------|--------|
| `app.py` | Streamlit healthcare dashboard: upload image → Leprosy / Not Leprosy + confidence, metric cards, detection overlay. |
| `0_prepare_dataset.py` | Split images into train/val under `dataset/images/`. |
| `1_auto_annotate.py` | Pseudo-labels (SAM or OpenCV) → YOLO labels. |
| `2_train.py` | Train YOLOv8 on `dataset/data.yaml`. |
| `3_evaluate.py` | Validation metrics (mAP, precision, recall). |
| `4_inference.py` | Run model and save images with drawn boxes. |
| `5_show_annotated.py` | Draw current YOLO labels on images → `dataset/annotated/`. |
