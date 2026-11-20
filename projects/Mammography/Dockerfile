FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
        ffmpeg \
        git \
        ninja-build \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        python3.8 \
        python3.8-dev \
        python3-pip \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 20.04 supplies Python 3.8.10; install PyTorch's CUDA 11.3 wheels
# directly so the build does not need Conda to solve a Python upgrade.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# Install MIM and project dependencies
RUN python -m pip install --no-cache-dir \
        timm openmim pandas pydicom scikit-learn Pillow==9.5.0

# Install MMPretrain
RUN git clone https://github.com/dorazhang93/Mammography-nils.git
WORKDIR /Mammography-nils
ENV PYTHONPATH=/Mammography-nils
RUN mim install --no-cache-dir -e .

COPY . .
