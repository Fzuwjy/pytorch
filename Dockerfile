FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
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

RUN conda create -y -n work python=3.12.3 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/work/bin:$PATH

RUN python -m pip install --upgrade pip

RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

RUN echo 'export PATH=/opt/conda/envs/work/bin:$PATH' > /etc/profile.d/work-env.sh \
    && echo 'export PATH=/opt/conda/envs/work/bin:$PATH' >> /root/.bashrc
    
WORKDIR /workspace

CMD ["/bin/bash"]
