
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies and Java for METEOR evaluation
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
    openjdk-11-jre-headless \
    locales \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Dedicated Chg2Cap environment
RUN conda create -y -n chg2cap python=3.9 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/chg2cap/bin:$PATH

RUN python -m pip install pip==23.3.2 setuptools==68.2.2 wheel==0.42.0

# Official PyTorch 1.11 CUDA 11.3 stack
RUN python -m pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Compatible numerical and data dependencies
RUN python -m pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    h5py==3.8.0 \
    lmdb==1.4.1 \
    Pillow==9.5.0 \
    opencv-python==4.5.5.64 \
    matplotlib==3.5.3 \
    scikit-image==0.19.3 \
    scikit-learn==1.1.3 \
    imageio==2.31.5 \
    einops==0.4.1 \
    tqdm==4.66.1

# Logging and evaluation dependencies
RUN python -m pip install --no-cache-dir \
    tensorboard==2.9.1 \
    protobuf==3.20.3

# Validate the environment during image build
RUN python - <<'PY'
import sys
import torch
import torchvision
import numpy
import h5py
import cv2
import scipy
import skimage
import sklearn
import imageio
import einops

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("h5py:", h5py.__version__)

assert sys.version_info[:2] == (3, 9)
assert torch.__version__.startswith("1.11.0")
assert torchvision.__version__.startswith("0.12.0")
assert torch.version.cuda == "11.3"
PY

RUN java -version

WORKDIR /workspace

CMD ["/bin/bash"]
