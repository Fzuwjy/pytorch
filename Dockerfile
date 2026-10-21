
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=8.6

# System and CUDA build dependencies
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

# Keep the PyTorch environment provided by the official image
RUN python -m pip install \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# Stable dependencies for VMamba / ChangeMamba
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    triton==2.2.0 \
    timm==0.4.12 \
    einops==0.8.1 \
    packaging==24.2 \
    pytest==8.3.4 \
    chardet==5.2.0 \
    yacs==0.1.8 \
    termcolor==2.4.0 \
    submitit==1.5.2 \
    tensorboardX==2.6.2.2 \
    protobuf==4.25.5 \
    fvcore==0.1.5.post20221221 \
    seaborn==0.13.2 \
    matplotlib==3.8.4 \
    pandas==2.2.3 \
    scipy==1.11.4 \
    Pillow==10.4.0 \
    PyYAML==6.0.2 \
    opencv-python-headless==4.10.0.84 \
    tqdm==4.67.1

# Check the CUDA compiler
RUN nvcc --version

# Pin ChangeMamba to the inspected upstream revision
RUN git clone \
    https://github.com/ChenHongruixuan/ChangeMamba.git \
    /opt/ChangeMamba \
    && cd /opt/ChangeMamba \
    && git checkout 9ce9cec13f9ea14bc0ad91f071577ec9b3a97983

# Build only for RTX 3090 (sm_86).
# Limit NVCC threading for GitHub Actions memory.
# This changes build flags only, not model or kernel algorithms.
RUN python - <<'PY'
from pathlib import Path

path = Path(
    "/opt/ChangeMamba/kernels/selective_scan/setup.py"
)

source = path.read_text()

start = source.index('    cc_flag.append("-gencode")')
end = source.index("    # HACK", start)

source = (
    source[:start]
    + '    cc_flag.extend(["-gencode", '
      '"arch=compute_86,code=sm_86"])\n\n'
    + source[end:]
)

assert '["--threads", "4"]' in source
source = source.replace(
    '["--threads", "4"]',
    '["--threads", "1"]'
)

path.write_text(source)
print("Selective scan configured for sm_86")
PY

# Build all three official selective_scan CUDA extensions
RUN cd /opt/ChangeMamba/kernels/selective_scan \
    && MAX_JOBS=1 python -m pip install \
       --no-build-isolation --no-deps .

# Verify dependencies and compiled CUDA extensions
# Install the Python ninja package required by selective_scan
RUN python -m pip install --no-cache-dir ninja==1.11.1.1 \
    && python -m pip check

RUN python - <<'PY'
import sys
import torch
import torchvision
import triton
import timm

import selective_scan_cuda_core
import selective_scan_cuda_ndstate
import selective_scan_cuda_oflex

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
print("Timm:", timm.__version__)
print("All selective_scan CUDA extensions imported")

assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.2.0"
assert torchvision.__version__.split("+")[0] == "0.17.0"
assert torch.version.cuda == "11.8"
PY

# Check ChangeMamba model imports
RUN PYTHONPATH=/opt/ChangeMamba python - <<'PY'
from changedetection.models.vmamba import VSSM
print("ChangeMamba VSSM import successful")
PY

WORKDIR /workspace

CMD ["/bin/bash"]
