#!/usr/bin/env python3
"""
Step 0: Prepare dataset structure.
Copies images from a source folder into dataset/images/train and dataset/images/val
with an 80/20 train/val split (stratified by filename for reproducibility).
"""
import os
import shutil
import hashlib
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "leprosy-images"  # or set to Path("dataset/images") if already flat
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def get_image_files(dir_path: Path):
    return sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS and not f.name.startswith(".")
    )


def stable_split(files, train_ratio: float, seed: int):
    """Deterministic train/val split based on filename hash."""
    n = len(files)
    if n == 0:
        return [], []
    indices = list(range(n))
    # Use seed + filename for reproducible ordering
    def key(i):
        h = hashlib.md5(files[i].name.encode()).hexdigest()
        return (int(h[:8], 16) + seed) % (2**32)
    indices.sort(key=key)
    n_train = max(1, int(n * train_ratio))
    train_files = [files[i] for i in indices[:n_train]]
    val_files = [files[i] for i in indices[n_train:]]
    return train_files, val_files


def main(source_dir: Path = None):
    source_dir = source_dir or DEFAULT_SOURCE
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Create it and add images, or run with: 0_prepare_dataset.py <path_to_images>")
        return

    files = get_image_files(source_dir)
    if not files:
        print(f"No images found in {source_dir}")
        return

    train_files, val_files = stable_split(files, TRAIN_RATIO, RANDOM_SEED)
    print(f"Total images: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    for subset, file_list in [("train", train_files), ("val", val_files)]:
        img_dir = IMAGES_DIR / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        for f in file_list:
            dest = img_dir / f.name
            if dest.resolve() != f.resolve():
                shutil.copy2(f, dest)
                print(f"  Copy {f.name} -> images/{subset}/")

    print("Done. Images are in dataset/images/train and dataset/images/val")
    print("Run step 1 to auto-generate labels: python scripts/1_auto_annotate.py")


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(src)
