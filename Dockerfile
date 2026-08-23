# ============================================================
# Build stage
# ============================================================

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LTX_REF=main

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LTX_HOME=/opt/LTX-2 \
    TORCH_CUDA_ARCH_LIST=12.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        git-lfs \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv installieren
RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    uv

RUN uv --version

# ------------------------------------------------------------
# LTX-2
# ------------------------------------------------------------

RUN git clone \
    --filter=blob:none \
    https://github.com/Lightricks/LTX-2.git \
    ${LTX_HOME} && \
    cd ${LTX_HOME} && \
    git checkout ${LTX_REF}

WORKDIR ${LTX_HOME}

# ------------------------------------------------------------
# LTX Runtime + NATTEN + RTX-5090 / Blackwell Kernels
#
# Auf realer RTX 5090 erfolgreich getestet mit:
# Torch 2.13.0+cu132
# CUDA 13.2
# Compute Capability 12.0
# ltx-kernels 1.2.0
# ------------------------------------------------------------

RUN uv sync \
    --extra natten \
    --group kernels

# ------------------------------------------------------------
# Build-Verifikation
#
# Noch kein GPU-Test möglich, aber alle kompilierten Extensions
# müssen importierbar sein.
# ------------------------------------------------------------

RUN /opt/LTX-2/.venv/bin/python - <<'PY'
import importlib
import torch
import ltx_kernels

from ltx_pipelines.utils.quantization_factory import QuantizationKind

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("ltx_kernels:", ltx_kernels.__file__)

for name in (
    "all2all_cpp",
    "ops_cpp",
    "blockwise_cpp",
    "nvfp4_cpp",
):
    mod = importlib.import_module(name)
    print(f"{name}: OK -> {mod.__file__}")

q = [x.value for x in QuantizationKind]
print("Quantization:", q)

assert "fp8-cast" in q
assert "fp8-scaled-mm" in q
assert "nvfp4-cast" in q
assert "nvfp4-prequant" in q

print("LTX build check: OK")
PY


# ============================================================
# Runtime stage
# ============================================================

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    LTX_HOME=/opt/LTX-2 \
    MODEL_DIR=/workspace/models \
    INPUT_DIR=/workspace/input \
    OUTPUT_DIR=/workspace/output \
    HF_HOME=/workspace/cache/huggingface \
    PATH="/opt/LTX-2/.venv/bin:/usr/local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        ffmpeg \
        ca-certificates \
        curl \
        rclone \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Fertige LTX-Installation inklusive:
# - .venv
# - torch
# - natten
# - ltx-kernels
# - nvfp4_cpp
# ------------------------------------------------------------

COPY --from=builder /opt/LTX-2 /opt/LTX-2

# Der aktuelle ltx_worker verwendet uv auch zur Runtime-Prüfung.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# uvx ggf. ebenfalls mitnehmen, falls vorhanden/benötigt
# COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------

RUN mkdir -p \
    ${MODEL_DIR} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    ${HF_HOME} \
    /workspace/af

RUN rm -rf ${LTX_HOME}/models && \
    ln -s ${MODEL_DIR} ${LTX_HOME}/models

COPY ltx_worker.py /workspace/af/ltx_worker.py
COPY entrypoint.sh /entrypoint.sh

RUN chmod 700 /workspace/af/ltx_worker.py && \
    chmod +x /entrypoint.sh

# ------------------------------------------------------------
# Runtime-Prüfung ohne GPU
# ------------------------------------------------------------

RUN python3 --version && \
    /opt/LTX-2/.venv/bin/python --version && \
    uv --version && \
    ffmpeg -version | head -n 1 && \
    rclone version | head -n 1

RUN /opt/LTX-2/.venv/bin/python - <<'PY'
import importlib
import torch
import ltx_kernels

from ltx_pipelines.utils.quantization_factory import QuantizationKind

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("ltx_kernels:", ltx_kernels.__file__)

for name in (
    "all2all_cpp",
    "ops_cpp",
    "blockwise_cpp",
    "nvfp4_cpp",
):
    mod = importlib.import_module(name)
    print(f"{name}: OK -> {mod.__file__}")

q = [x.value for x in QuantizationKind]
print("Quantization:", q)

assert "nvfp4-cast" in q
assert "nvfp4-prequant" in q

print("Runtime image check: OK")
PY

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
