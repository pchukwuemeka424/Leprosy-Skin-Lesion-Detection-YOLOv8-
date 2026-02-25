#!/bin/bash
# Train then evaluate (mAP, Precision, Recall).
# Requires: Python 3.10–3.12 (PyTorch/Ultralytics do not support 3.14 yet).
#   e.g. pyenv install 3.12 && pyenv local 3.12, or: python3.12 -m venv .venv && source .venv/bin/activate
set -e
cd "$(dirname "$0")"

echo "Installing dependencies..."
pip install ultralytics opencv-python

echo "Training..."
python3 scripts/2_train.py

echo "Evaluating (mAP, Precision, Recall)..."
python3 scripts/3_evaluate.py
