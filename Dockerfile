FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LTX_REF=main

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LTX_HOME=/opt/LTX-2 \
    MODEL_DIR=/workspace/models \
    INPUT_DIR=/workspace/input \
    OUTPUT_DIR=/workspace/output \
    HF_HOME=/workspace/cache/huggingface

# ------------------------------------------------------------
# System
# ------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        git-lfs \
        ffmpeg \
        curl \
        wget \
        aria2 \
        ca-certificates \
        jq \
        procps \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# uv
# ------------------------------------------------------------

RUN python3 -m pip install --break-system-packages uv

# ------------------------------------------------------------
# LTX-2
# ------------------------------------------------------------

RUN git clone https://github.com/Lightricks/LTX-2.git ${LTX_HOME} && \
    cd ${LTX_HOME} && \
    git checkout ${LTX_REF}

WORKDIR ${LTX_HOME}

# Exakt die vom LTX-Projekt gelockten Dependencies verwenden.
# natten = schnellerer Diffusion-VAE-Decoder auf Linux/CUDA.

RUN uv sync --frozen

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------

RUN mkdir -p \
    ${MODEL_DIR} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    ${HF_HOME}

# Das offizielle LTX-Beispiel erwartet standardmäßig ./models.
# Deshalb verlinken wir es auf unseren persistenten Workspace.

RUN rm -rf ${LTX_HOME}/models && \
    ln -s ${MODEL_DIR} ${LTX_HOME}/models

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Vast-Container bleibt zunächst aktiv.
CMD ["sleep", "infinity"]
