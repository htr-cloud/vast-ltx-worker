#!/usr/bin/env bash

set -euo pipefail

echo "============================================================"
echo " ComfyUI Worker"
echo "============================================================"

echo
python --version

echo
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA Runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )
PY

echo
echo "Starting ComfyUI..."

cd /opt/ComfyUI

exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    "$@"
