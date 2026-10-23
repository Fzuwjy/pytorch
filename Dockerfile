
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# FLUX Fill + OneReward inference dependencies
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==11.3.0 \
    diffusers==0.35.1 \
    transformers==4.51.3 \
    accelerate==1.10.1 \
    huggingface-hub==0.35.3 \
    tokenizers==0.21.4 \
    safetensors==0.5.3 \
    sentencepiece==0.2.1 \
    protobuf==3.20.3 \
    ftfy==6.3.1 \
    opencv-python-headless==4.10.0.84 \
    psutil==6.1.1

# Verify package versions and FLUX APIs during build
RUN python - <<'PY'
import sys
import inspect
import torch
import torchvision
import diffusers
import transformers
import accelerate

from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_fill import (
    calculate_shift,
    retrieve_timesteps,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)

assert sys.version_info[:2] == (3, 11)
assert torch.__version__.startswith("2.7.1")
assert torchvision.__version__.startswith("0.22.1")
assert torch.version.cuda == "11.8"
assert diffusers.__version__ == "0.35.1"
assert transformers.__version__ == "4.51.3"
assert "strength" in inspect.signature(
    FluxFillPipeline.check_inputs
).parameters

print("FLUX environment import check: OK")
PY

WORKDIR /workspace

CMD ["/bin/bash"]
