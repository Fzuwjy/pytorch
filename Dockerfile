FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
    https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n medcagd \
        -c conda-forge \
        python=3.13.3 \
        pip=25.1 \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/envs/medcagd/bin:/opt/conda/bin:${PATH}"

RUN python -m pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install --no-cache-dir \
    numpy==2.1.2 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    Pillow==11.0.0 \
    tqdm==4.67.1 \
    opencv-python-headless==4.11.0.86 \
    albucore==0.0.24 \
    albumentations==2.0.6 \
    timm==1.0.15 \
    segmentation-models-pytorch==0.5.0 \
    cupy-cuda12x==13.4.1 \
    thop==0.1.1.post2209072238 \
    fvcore==0.1.5.post20221221 \
    patchify==0.2.3

RUN python -m pip check

RUN python - <<'PY'
import sys
import torch
import torchvision
import timm
import albumentations
import cupy
import segmentation_models_pytorch

from cupyx.scipy.ndimage import distance_transform_edt, binary_erosion

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("Timm:", timm.__version__)
print("Albumentations:", albumentations.__version__)
print("CuPy:", cupy.__version__)

assert sys.version_info[:3] == (3, 13, 3)
assert torch.__version__.startswith("2.7.0")
assert torchvision.__version__.startswith("0.22.0")
assert torch.version.cuda == "12.8"
assert timm.__version__ == "1.0.15"
assert albumentations.__version__ == "2.0.6"
assert cupy.__version__ == "13.4.1"

print("MedCAGD environment verification: OK")
PY

WORKDIR /workspace

CMD ["/bin/bash"]
