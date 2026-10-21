
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    unzip \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 environment
RUN conda create -y -n change3d python=3.11 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/change3d/bin:$PATH

RUN python -m pip install \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# Official PyTorch stack: 2.2.0 + CUDA 11.8
RUN python -m pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Change3D official dependencies
RUN python -m pip install --no-cache-dir \
    einops==0.8.1 \
    fvcore==0.1.5.post20221221 \
    h5py==3.13.0 \
    imageio==2.37.0 \
    matplotlib==3.10.1 \
    numpy==1.26.1 \
    opencv-python==4.11.0.86 \
    Pillow==11.1.0 \
    pytorchvideo==0.1.5 \
    scipy==1.15.2 \
    scikit-image==0.25.2 \
    tqdm==4.67.1

# Check dependency compatibility
RUN python -m pip check

# Build-time environment verification
RUN python - <<'PY'
import sys
import torch
import torchvision
import torchaudio
import numpy
import cv2
import h5py
import einops
import pytorchvideo

from pytorchvideo.layers.convolutions import Conv2plus1d

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)
print("CUDA:", torch.version.cuda)
print("NumPy:", numpy.__version__)

assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split("+")[0] == "2.2.0"
assert torchvision.__version__.split("+")[0] == "0.17.0"
assert torch.version.cuda == "11.8"
PY

RUN java -version

WORKDIR /workspace

CMD ["/bin/bash"]
