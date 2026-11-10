
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=1
ENV MAMBA_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "pip<25" \
    "setuptools<70" \
    wheel \
    ninja \
    packaging \
    psutil

RUN python -m pip install --no-cache-dir \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pillow==10.4.0 \
    matplotlib==3.9.2 \
    scikit-learn==1.5.2 \
    einops==0.8.0 \
    timm==0.9.16 \
    tensorboardX==2.6.2.2 \
    tqdm==4.66.5 \
    PyYAML==6.0.2 \
    yacs==0.1.8

RUN python -m pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.1.1

RUN python -m pip install --no-cache-dir --no-build-isolation \
    mamba-ssm==1.1.2

RUN python -c "import torch; assert torch.__version__.startswith('2.3.1'); assert torch.version.cuda == '11.8'; print('PyTorch:', torch.__version__)"

RUN python -c "import causal_conv1d; import mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('Mamba CUDA extensions imported successfully')"

WORKDIR /workspace

CMD ["/bin/bash"]
