#!/usr/bin/env bash

set -euo pipefail

echo "============================================================"
echo " LTX Worker"
echo "============================================================"
echo

cd /opt/LTX-2

echo "Python:"
python3 --version

echo

echo "LTX revision:"
git rev-parse --short HEAD

echo

echo "PyTorch / CUDA:"

uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA Runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(device.total_memory / 1024**3, 2),
        "GB"
    )
PY

echo
echo "Model directory:  ${MODEL_DIR}"
echo "Input directory:  ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo

exec "$@"#!/usr/bin/env bash

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
