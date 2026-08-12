FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REF=master

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_PATH=/opt/ComfyUI \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        git-lfs \
        ffmpeg \
        curl \
        wget \
        aria2 \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        procps \
        unzip \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    python -m pip install --upgrade pip setuptools wheel

RUN pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

RUN git clone https://github.com/Comfy-Org/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && \
    git checkout ${COMFYUI_REF}

WORKDIR ${COMFYUI_PATH}

RUN pip install -r requirements.txt

RUN mkdir -p \
    input \
    output \
    temp \
    models/checkpoints \
    models/diffusion_models \
    models/text_encoders \
    models/clip \
    models/vae \
    models/loras \
    models/controlnet

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8188

ENTRYPOINT ["/entrypoint.sh"]
