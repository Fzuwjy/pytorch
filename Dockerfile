FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 系统工具 + 后续研究可能需要的编译环境
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

# EndoDAC 官方使用 Python 3.9
RUN conda create -y -n endodac python=3.9 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/endodac/bin:$PATH

# 严格对齐 EndoDAC 官方 PyTorch 栈
RUN conda install -y -n endodac \
    -c pytorch \
    -c nvidia \
    pytorch=2.0.0 \
    torchvision=0.15.0 \
    pytorch-cuda=11.7 \
    && conda clean -afy

RUN python -m pip install --upgrade pip setuptools wheel

# 避免老 PyTorch / CUDA 扩展与 NumPy 2.x 的兼容问题
RUN pip install --no-cache-dir numpy==1.26.4

# EndoDAC 官方依赖
RUN pip install --no-cache-dir \
    omegaconf==2.3.0 \
    torchmetrics==0.10.3 \
    fvcore \
    iopath==0.1.10 \
    xformers==0.0.18 \
    submitit \
    tensorboardX \
    tqdm \
    opencv-python \
    open3d==0.18.0 \
    scikit-image==0.22.0

# 官方 requirements 中存在 cuML，但原作者没有锁版本。
# 固定到仍支持 Python 3.9 的 2024 版本，避免 2026 最新版破坏环境。
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    cuml-cu11==24.4.0

# 构建时直接检查关键环境
RUN python - <<'PY'
import sys
import torch
import torchvision

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available during build:", torch.cuda.is_available())

assert sys.version_info[:2] == (3, 9), sys.version
assert torch.__version__.startswith("2.0.0"), torch.__version__
assert torchvision.__version__.startswith("0.15.0"), torchvision.__version__
assert torch.version.cuda.startswith("11.7"), torch.version.cuda
PY

RUN echo 'export PATH=/opt/conda/envs/endodac/bin:$PATH' \
    > /etc/profile.d/endodac-env.sh \
    && echo 'export PATH=/opt/conda/envs/endodac/bin:$PATH' \
    >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]
