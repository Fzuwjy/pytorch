
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=1
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=TRUE

RUN python -m pip install --no-cache-dir \
    "pip<25" \
    "setuptools<70" \
    wheel \
    ninja \
    packaging \
    psutil \
    einops==0.8.0

RUN command -v nvcc && \
    command -v gcc && \
    command -v g++ && \
    command -v git && \
    command -v cmake

RUN python -c "import torch; assert torch.__version__.startswith('2.3.1'); assert torch.version.cuda == '11.8'; print(torch.__version__, torch.version.cuda)"

RUN mkdir -p /wheels

RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
    --wheel-dir=/wheels \
    "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.1"

RUN python -m pip install --no-cache-dir --no-deps \
    /wheels/causal_conv1d-*.whl

RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
    --wheel-dir=/wheels \
    "git+https://github.com/state-spaces/mamba.git@v1.1.2"


FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime AS runtime

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir \
    torchvision==0.18.1 \
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
    yacs==0.1.8 \
    packaging \
    psutil \
    transformers==4.38.2

RUN python -m pip install --no-cache-dir --no-index --no-deps \
    /wheels/causal_conv1d-*.whl \
    /wheels/mamba_ssm-*.whl \
    && rm -rf /wheels

RUN python -c "import torch, torchvision, torchaudio; assert torch.__version__.startswith('2.3.1'); assert torchvision.__version__.startswith('0.18.1'); assert torchaudio.__version__.startswith('2.3.1'); assert torch.version.cuda == '11.8'; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

RUN python -c "import causal_conv1d; from causal_conv1d import causal_conv1d_fn; import mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput; print('All KMUNet dependencies imported successfully')"

WORKDIR /workspace

CMD ["/bin/bash"]
