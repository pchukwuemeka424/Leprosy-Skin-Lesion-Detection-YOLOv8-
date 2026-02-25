# Leprosy Skin Lesion Detection (YOLOv8)

Early detection of leprosy skin lesions using YOLOv8 with auto-generated pseudo-labels, transfer learning, and data augmentation.

**Product summary:** Raw skin images are split into train/val, pseudo-labeled (SAM or OpenCV), then used to fine-tune YOLOv8n. The trained model powers a Streamlit healthcare dashboard where users upload an image and receive **Leprosy** / **Not Leprosy** with confidence and optional detection overlay. Pipeline scripts: prepare → annotate → train → evaluate → inference; optional batch inference and annotated previews.

---

## System architecture

The system has four main layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                                         │
│  • Streamlit app (app.py): upload image → prediction + confidence + overlay │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER                                                                │
│  • YOLOv8n (Ultralytics): object detection, single class "lesion"            │
│  • Weights: runs/detect/leprosy/weights/best.pt (after training)              │
│  • Inference: conf threshold 0.15; dashboard uses ≥0.25 for "Leprosy" label   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE LAYER (scripts/)                                                  │
│  • 0_prepare_dataset.py  → train/val split (80/20)                          │
│  • 1_auto_annotate.py    → pseudo-labels (SAM or OpenCV) → YOLO .txt         │
│  • 2_train.py            → train YOLOv8n on data.yaml                      │
│  • 3_evaluate.py         → mAP, precision, recall on val set               │
│  • 4_inference.py        → batch predict, save images with boxes           │
│  • 5_show_annotated.py   → draw labels on images → dataset/annotated/       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                                 │
│  • Source: leprosy-images/ (or custom path)                                 │
│  • dataset/images/{train,val}/  → images                                     │
│  • dataset/labels/{train,val}/  → YOLO format (class_id xc yc w h normalized)│
│  • dataset/data.yaml          → paths + nc:1, names: {0: lesion}            │
│  • dataset/annotated/         → visual verification (from script 5)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Dependencies:** Python 3.10–3.12, Ultralytics (YOLOv8), OpenCV, Streamlit; optional: Segment Anything (SAM) for better pseudo-labels.

---

## System flow

**End-to-end (training pipeline):**

```
Raw skin images (e.g. leprosy-images/)
        │
        ▼  [0] 0_prepare_dataset.py  (80/20 split by deterministic hash)
dataset/images/train/   dataset/images/val/
        │
        ▼  [1] 1_auto_annotate.py    (SAM or OpenCV → bounding boxes)
dataset/labels/train/   dataset/labels/val/   (.txt per image: class_id xc yc w h)
        │
        ▼  [2] 2_train.py            (YOLOv8n + data.yaml → train with augmentation)
runs/detect/leprosy/weights/best.pt   last.pt
        │
        ├──► [3] 3_evaluate.py       (val set → mAP50, mAP50-95, precision, recall)
        │
        └──► [4] 4_inference.py  or  app.py   (predict on images / upload)
```

**Inference flow (dashboard):**

```
User uploads image → temp file → YOLO.predict(conf=0.15)
        → boxes + confidence
        → max confidence ≥ 0.25 ? "Leprosy" : "Not Leprosy"
        → display: prediction badge, confidence %, progress bar, detection overlay (if boxes)
```

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

## Data and training process

Detailed flow from raw images to a trained model and screening.

### Step 0: Data preparation

| Item | Detail |
|------|--------|
| **Input** | Source folder of skin images (default: `leprosy-images/`). |
| **Formats** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`. |
| **Split** | 80% train / 20% val, deterministic (MD5 hash of filename + seed 42). |
| **Output** | `dataset/images/train/`, `dataset/images/val/` (copies; originals unchanged). |
| **This repo** | 39 train, 10 val (49 total images). |

### Step 1: Pseudo-labeling (annotations)

| Item | Detail |
|------|--------|
| **Input** | Images in `dataset/images/train/` and `dataset/images/val/`. |
| **Methods** | **SAM** (recommended): Segment Anything Model → masks → bounding boxes. **OpenCV fallback**: grayscale + Otsu threshold + contours → boxes. |
| **Filters** | Box area 0.5%–85% of image; min side 15 px; SAM: `points_per_side=12`, `pred_iou_thresh=0.7`, `stability_score_thresh=0.85`. |
| **Output** | One `.txt` per image in `dataset/labels/{train,val}/`. Format: `0 x_center y_center width height` (normalized 0–1). Class `0` = lesion. |
| **Fallback** | If no boxes found, one conservative box `(0.5, 0.5, 0.4, 0.4)` is written so training has a label. |

### Step 2: Training

| Item | Detail |
|------|--------|
| **Config** | `dataset/data.yaml`: `path`, `train`, `val`, `nc: 1`, `names: {0: lesion}`. |
| **Model** | YOLOv8n pretrained (`yolov8n.pt`), transfer learning. |
| **Input size** | 640×640 (letterboxing). Batch size 8. |
| **Optimizer** | AdamW, `lr0=1e-3`, `lrf=0.01`, weight decay `0.0005`, warmup 3 epochs. |
| **Augmentation** | Mosaic 1.0, mixup 0.1, HSV, fliplr 0.5, degrees 15, translate 0.1, scale 0.5, shear 5. |
| **Stopping** | Early stopping `patience=30`; max epochs 150 (override: env `EPOCHS`). |
| **Training set size (this repo)** | 39 images (train); 10 images (val). |
| **Output** | `runs/detect/leprosy/weights/best.pt`, `last.pt`; curves and logs in `runs/detect/leprosy/`. |

### Step 3: Evaluation

| Item | Detail |
|------|--------|
| **Input** | `best.pt` (or path as CLI arg), `dataset/data.yaml` (val split). |
| **Metrics** | mAP50, mAP50-95, Precision (box.mp), Recall (box.mr). |
| **Script** | `scripts/3_evaluate.py` [optional weights path]. |

### Step 4: Inference and dashboard

| Item | Detail |
|------|--------|
| **Batch** | `4_inference.py`: predict on folder (default: `dataset/images/`), save images with boxes to `runs/detect/leprosy/predict/`. |
| **Dashboard** | `app.py`: upload single image → YOLO predict (conf=0.15) → max conf ≥ 0.25 → "Leprosy" else "Not Leprosy"; show confidence and detection overlay. |

---

## Project structure

```
project_01/
├── app.py                    # Streamlit dashboard (upload → Leprosy / Not Leprosy)
├── requirements.txt          # ultralytics, opencv-python, streamlit, etc.
├── run_train_eval.sh         # Train + evaluate in one go
├── yolov8n.pt                # Pretrained YOLOv8 nano (downloaded by Ultralytics if missing)
├── dataset/
│   ├── data.yaml             # Dataset config (paths, nc, class names)
│   ├── images/{train,val}/   # Images
│   ├── labels/{train,val}/   # YOLO labels (.txt per image)
│   └── annotated/{train,val}/ # Visual verification (from 5_show_annotated.py)
├── leprosy-images/           # Default source for step 0 (optional)
├── checkpoints/              # SAM checkpoint (e.g. sam_vit_b_01ec64.pth) if using SAM
├── runs/detect/leprosy/
│   ├── weights/best.pt       # Best weights (used by app and 4_inference)
│   ├── weights/last.pt
│   └── predict/              # Output of 4_inference.py
└── scripts/
    ├── 0_prepare_dataset.py  # Train/val split
    ├── 1_auto_annotate.py    # Pseudo-labels (SAM or OpenCV)
    ├── 2_train.py           # Train YOLOv8
    ├── 3_evaluate.py        # mAP, P, R
    ├── 4_inference.py       # Batch predict, save images
    └── 5_show_annotated.py   # Draw labels → dataset/annotated/
```

---

## Key configuration

| Item | Value | Where |
|------|--------|--------|
| **Number of images (this repo)** | **39 train, 10 val (49 total)** | `dataset/images/train/`, `dataset/images/val/` |
| Train/val split | 80% / 20% | `0_prepare_dataset.py` (`TRAIN_RATIO`, `RANDOM_SEED`) |
| Pseudo-label area | 0.5%–85% of image | `1_auto_annotate.py` (`MIN_AREA_RATIO`, `MAX_AREA_RATIO`) |
| Min box side | 15 px | `1_auto_annotate.py` (`MIN_SIDE_PX`) |
| Detection threshold (dashboard) | ≥ 25% → "Leprosy" | `app.py` (`CONF_THRESHOLD`) |
| Inference confidence (model) | 0.15 | `app.py` (`model.predict(conf=0.15)`); lower = more detections |
| Epochs / early stopping | 150 / patience 30 | `2_train.py` (env `EPOCHS` to override) |
| Image size | 640×640 | `2_train.py` (`imgsz=640`) |

---

## Setup

Use **Python 3.10, 3.11, or 3.12** (PyTorch/Ultralytics do not support Python 3.14 yet).

```bash
cd /Users/acehub/Desktop/project_01
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Or: pip install ultralytics opencv-python streamlit
# Optional: pip install segment-anything for better pseudo-labels (Step 1)
```

## Dataset layout

- **Images**: `dataset/images/train/`, `dataset/images/val/`
- **Labels (YOLO)**: `dataset/labels/train/`, `dataset/labels/val/`
- **Config**: `dataset/data.yaml`
- **Number of images used to train (this repo):** **39** training images, **10** validation images (**49** total).

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

**Quick train + evaluate:** `./run_train_eval.sh` (trains then runs evaluation). See `CHECKLIST.md` for a pipeline checklist and quick verify commands.

---

## Deployment

**Vercel is not suitable** for this app: it’s a Streamlit + PyTorch/YOLO app that needs a long-running Python process and large dependencies (model weights, Ultralytics). Vercel’s serverless limits (e.g. 50 MB function size) and request/response model don’t support this.

**Recommended: Streamlit Community Cloud** (free, one-click from GitHub)

1. Push this repo to GitHub (you already have `origin` set).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **“New app”** → choose this repo, branch `main`, main file **`app.py`**.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and run `streamlit run app.py`.
5. Ensure **trained weights** are in the repo at `runs/detect/leprosy/weights/best.pt` (commit and push them if you train locally), or the app will show “Model not found”.

**Other options**

- **[Hugging Face Spaces](https://huggingface.co/spaces)** (Streamlit): create a Space, select Streamlit SDK, upload `app.py` and `requirements.txt`, and add `best.pt` (e.g. via Git LFS or “Files and versions”).
- **Railway / Render**: deploy as a web service with `streamlit run app.py --server.port=$PORT`; add a `Procfile` or start command and include the weights in the repo or as build artifacts.

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
