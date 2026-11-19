# syntax=docker/dockerfile:1.7
# Every step of a Litmus run (corpus, teacher, student) on one GPU machine. The build context is the `tree/` of a
# run directory shipped to the machine. From the shipped project folder:
#   sudo docker build -t litmus -f launch/Dockerfile ../../..
# CUDA 13.0 (cu130, the default) needs driver 580 or newer. On a 570 driver (CUDA 12.8 at most) build with
#   --build-arg CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 --build-arg TORCH_INDEX=cu126
# At run time launch/in_container.sh mounts the shipped directory at /workspace, read-only.

ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG TORCH_INDEX=cu130
ARG PROJECT=projects/tooling/quill

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1

# build-essential serves Triton, which compiles the e-means kernels and torch.compile's graphs at run time.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
        python3.12 python3.12-dev python3.12-venv && \
    python3.12 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip

COPY ${PROJECT}/launch/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade "pip>=26.2.2" setuptools && \
    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/${TORCH_INDEX} && \
    pip install -r /tmp/requirements.txt && \
    python -c "import faiss, faiss.contrib.e_means, optuna, torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'faiss', faiss.__version__)"

WORKDIR /workspace
CMD ["bash"]
