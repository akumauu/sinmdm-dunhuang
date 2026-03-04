#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SINMDM_VENV:-/tmp/sinmdm-product-venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

if ! "${PYTHON_BIN}" -c "import torch, torchvision, gradio, cv2, mediapipe, blobfile" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install numpy scipy matplotlib plotly "gradio>=4.44.0,<6.0.0" opencv-python mediapipe pytest blobfile
  "${PYTHON_BIN}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" app.py
