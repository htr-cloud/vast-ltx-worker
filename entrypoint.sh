#!/usr/bin/env bash

set -euo pipefail

LTX_HOME="${LTX_HOME:-/opt/LTX-2}"
LTX_PYTHON="${LTX_PYTHON:-${LTX_HOME}/.venv/bin/python}"

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
INPUT_DIR="${INPUT_DIR:-/workspace/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"

echo "============================================================"
echo " LTX-2 Runtime"
echo "============================================================"
echo

cd "${LTX_HOME}"

echo "Python:"
"${LTX_PYTHON}" --version

echo

echo "LTX revision:"
if command -v git >/dev/null 2>&1 && [ -d "${LTX_HOME}/.git" ]; then
    git rev-parse --short HEAD || true
else
    echo "git metadata unavailable"
fi

echo

echo "PyTorch / CUDA / LTX kernels:"

"${LTX_PYTHON}" - <<'PY'
import importlib.util
import torch

print("PyTorch:", torch.__version__)
print("CUDA Runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

print(
    "ltx_kernels:",
    importlib.util.find_spec("ltx_kernels") is not None,
)

print(
    "natten:",
    importlib.util.find_spec("natten") is not None,
)

try:
    import nvfp4_cpp
    print("nvfp4_cpp: True")
except Exception as exc:
    print("nvfp4_cpp: False")
    print("nvfp4 error:", exc)

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(props.total_memory / 1024**3, 2),
        "GB",
    )

    print(
        "Compute Capability:",
        ".".join(
            str(x)
            for x in torch.cuda.get_device_capability(0)
        ),
    )
PY

echo

echo "Directories:"
echo "  Models: ${MODEL_DIR}"
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"

mkdir -p \
    "${MODEL_DIR}" \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    "${OUTPUT_DIR}/af_jobs" \
    /workspace/af

echo
echo "Runtime ready."
echo "============================================================"
echo

exec "$@"
