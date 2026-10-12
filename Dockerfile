
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

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
    default-jre-headless \
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

# Keep the official PyTorch environment
RUN python -m pip install --upgrade pip setuptools wheel

# SCNet training and data dependencies
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    imageio==2.37.0 \
    einops==0.8.1 \
    timm==0.9.16 \
    huggingface-hub==0.25.2 \
    tqdm==4.67.1 \
    Pillow==11.1.0 \
    opencv-python==4.10.0.84 \
    matplotlib==3.9.2

# Model complexity and parameter analysis
RUN python -m pip install --no-cache-dir \
    fvcore==0.1.5.post20221221 \
    ptflops==0.7.5 \
    thop==0.1.1.post2209072238 \
    torchinfo==1.8.0

# Build-time environment verification
RUN python - <<'PY'
import sys
import torch
import torchvision
import numpy
import timm
import einops

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from fvcore.nn import FlopCountAnalysis

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("timm:", timm.__version__)
print("torch.compile:", hasattr(torch, "compile"))

assert sys.version_info >= (3, 11)
assert torch.__version__.startswith("2.7.1")
assert torchvision.__version__.startswith("0.22.1")
assert torch.version.cuda == "11.8"
assert hasattr(torch, "compile")
PY

RUN java -version

WORKDIR /workspace

CMD ["/bin/bash"]
