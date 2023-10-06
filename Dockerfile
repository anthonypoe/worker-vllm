ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"

# Base NVidia CUDA Ubuntu image
FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION AS base

ENV HOME /root
WORKDIR $HOME
ENV PYTHON_VERSION=3.10
ENV PATH="/usr/local/cuda/bin:${PATH}"


ARG APTPKGS="wget software-properties-common"
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git git-lfs && \
    python3 -m pip install --upgrade pip && \
    apt-get install -y --no-install-recommends $APTPKGS && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for Python env management
ENV PATH="${HOME}/miniconda3/bin:${PATH}"
ENV BASEPATH="${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ${HOME}/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Make base conda environment
ENV CONDA=pytorch
RUN conda create -n "${CONDA}" python="${PYTHON_VERSION}"
ENV PATH="${HOME}/miniconda3/envs/${CONDA}/bin:${BASEPATH}"

# Install pytorch
ARG PYTORCH="2.0.1"
ARG CUDA="118"
RUN pip3 install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu$CUDA

RUN pip3 install vllm huggingface-hub

# Set up git to support LFS, and to store credentials; useful for Huggingface Hub
RUN git config --global credential.helper store && \
    git lfs install

# Add src files (Worker Template)
ADD src .  

# Quick temporary updates
RUN pip3 install git+https://github.com/runpod/runpod-python@a1#egg=runpod --compile

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_REVISION="main"
ENV MODEL_REVISION=$MODEL_REVISION
ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
ARG TOKENIZER=
ENV TOKENIZER=$TOKENIZER
ARG STREAMING=
ENV STREAMING=$STREAMING

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Download the models
RUN mkdir -p /model

# Set environment variables
ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    MODEL_BASE_PATH=$MODEL_BASE_PATH \
    HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Run the Python script to download the model
RUN python3 -u download_model.py

# Start the handler
CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER python3 -u handler.py 
