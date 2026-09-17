FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 系统依赖
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
    && rm -rf /var/lib/apt/lists/*

# 升级基础 Python 工具
RUN python -m pip install --upgrade pip setuptools wheel

# MM-DINO / DINOv3 基础依赖
RUN pip install --no-cache-dir \
    ftfy \
    omegaconf \
    regex \
    scikit-learn \
    submitit \
    termcolor \
    torchmetrics \
    numpy \
    pillow \
    scikit-image \
    tqdm \
    transformers \
    matplotlib \
    opencv-python

# 构建阶段直接检查关键版本
RUN python - <<'PY'
import sys
import torch
import torchvision

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)

assert sys.version_info >= (3, 11), sys.version
assert torch.__version__.startswith("2.7.1"), torch.__version__
PY

WORKDIR /workspace

CMD ["/bin/bash"]
