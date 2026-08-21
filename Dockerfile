# ============================================================
# Build stage
# ============================================================

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LTX_REF=main

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LTX_HOME=/opt/LTX-2

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

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    uv

RUN git clone \
    --filter=blob:none \
    https://github.com/Lightricks/LTX-2.git \
    ${LTX_HOME} && \
    cd ${LTX_HOME} && \
    git checkout ${LTX_REF}

WORKDIR ${LTX_HOME}

RUN uv sync --extra natten


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
    PATH="/opt/LTX-2/.venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        ffmpeg \
        ca-certificates \
        curl \
        rclone \
    && rm -rf /var/lib/apt/lists/*

# Fertige LTX-Installation inklusive .venv
COPY --from=builder /opt/LTX-2 /opt/LTX-2

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

RUN chmod +x /entrypoint.sh

# ------------------------------------------------------------
# Runtime-Prüfung
# ------------------------------------------------------------

RUN python3 --version && \
    /opt/LTX-2/.venv/bin/python --version && \
    ffmpeg -version | head -n 1 && \
    rclone version | head -n 1

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
