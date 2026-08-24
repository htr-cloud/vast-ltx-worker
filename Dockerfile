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


# ------------------------------------------------------------
# uv nur fuer Build / Dependency-Management
# ------------------------------------------------------------

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
# LTX Runtime
#
# Getestet auf RTX 5090:
# - Torch 2.13.0+cu132
# - CUDA 13.2
# - Compute Capability 12.0
# - NATTEN
# - ltx-kernels
# - nvfp4_cpp
#
# TORCH_CUDA_ARCH_LIST=12.0:
# gezielter Build fuer RTX 5090 / Blackwell Consumer
# ------------------------------------------------------------

RUN uv sync \
    --extra natten \
    --group kernels


# ------------------------------------------------------------
# Build-Verifikation
#
# Hier gibt es noch keine GPU.
# Deshalb pruefen wir Imports und gebaute Extensions.
# ------------------------------------------------------------

RUN /opt/LTX-2/.venv/bin/python - <<'PY'
import importlib
import torch
import ltx_kernels

from ltx_pipelines.utils.quantization_factory import QuantizationKind

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("ltx_kernels:", ltx_kernels.__file__)

required_extensions = (
    "all2all_cpp",
    "ops_cpp",
    "blockwise_cpp",
    "nvfp4_cpp",
)

for name in required_extensions:
    module = importlib.import_module(name)
    print(f"{name}: OK -> {module.__file__}")

quantization = [x.value for x in QuantizationKind]

print("Quantization:", quantization)

assert "fp8-cast" in quantization
assert "fp8-scaled-mm" in quantization
assert "nvfp4-cast" in quantization
assert "nvfp4-prequant" in quantization

print("LTX BUILD CHECK: OK")
PY# ============================================================
# Runtime stage
# ============================================================

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    LTX_HOME=/opt/LTX-2 \
    MODEL_DIR=/workspace/models \
    INPUT_DIR=/workspace/input \
    OUTPUT_DIR=/workspace/output \
    HF_HOME=/workspace/cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        gcc \
        g++ \
        git \
        git-lfs \
        ffmpeg \
        rclone \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# uv im Runtime-Image
# WICHTIG: explizit System-Python verwenden
# ------------------------------------------------------------

RUN /usr/bin/python3 -m pip install \
        --break-system-packages \
        --no-cache-dir \
        uv

# ------------------------------------------------------------
# Fertige LTX-Installation inklusive .venv
# ------------------------------------------------------------

COPY --from=builder /opt/LTX-2 /opt/LTX-2

# ------------------------------------------------------------
# pip auch innerhalb der LTX-venv installieren
# ------------------------------------------------------------

RUN /usr/local/bin/uv pip install \
        --python /opt/LTX-2/.venv/bin/python \
        pip

# Erst JETZT die LTX-venv an den Anfang des PATH setzen
ENV PATH="/opt/LTX-2/.venv/bin:${PATH}"

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------

RUN mkdir -p \
    ${MODEL_DIR} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    ${OUTPUT_DIR}/af_jobs \
    ${HF_HOME} \
    /workspace/af

RUN rm -rf ${LTX_HOME}/models && \
    ln -s ${MODEL_DIR} ${LTX_HOME}/models

COPY entrypoint.sh /entrypoint.sh

RUN chmod 755 /entrypoint.sh




# ------------------------------------------------------------
# Runtime Image Check
#
# Noch ohne GPU, aber alle Runtime-Module und kompilierten
# Extensions muessen vorhanden sein.
# ------------------------------------------------------------

RUN python3 --version && \
    python3 -m pip --version && \
    uv --version && \
    gcc --version | head -n 1 && \
    g++ --version | head -n 1 && \
    /opt/LTX-2/.venv/bin/python --version && \
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
    module = importlib.import_module(name)
    print(f"{name}: OK -> {module.__file__}")

quantization = [x.value for x in QuantizationKind]

print("Quantization:", quantization)

assert "nvfp4-cast" in quantization
assert "nvfp4-prequant" in quantization

print("LTX RUNTIME IMAGE CHECK: OK")
PY


# ============================================================
# Container
# ============================================================

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
