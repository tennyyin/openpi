# SageMaker container for openpi (JAX) fine-tuning on the cv-wfm H200 queue.
# Base: cosmos-predict2-base in the robotics-old account (124224456861) ECR — an
# Ubuntu/py312 SageMaker DLC derivative with git + a working changehostname setup
# (same base the open-world image uses; pull auth via the in-account `docker login`
# in launch_sm.py). openpi pins python 3.11, which uv provisions in its own venv.
FROM 124224456861.dkr.ecr.us-west-2.amazonaws.com/cosmos-predict2-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
WORKDIR ${SAGEMAKER_SUBMIT_DIRECTORY}

# Install deps FIRST from lockfiles so the large .venv layer is cached on its own and
# only re-pushed when pyproject.toml / uv.lock change (not on every code edit). openpi is
# a uv workspace; the `openpi-client` member (packages/*) must be present for resolution.
COPY pyproject.toml uv.lock .python-version ${SAGEMAKER_SUBMIT_DIRECTORY}/
COPY packages ${SAGEMAKER_SUBMIT_DIRECTORY}/packages
RUN uv sync --frozen --no-dev --no-install-project

# Then the code (small layer) + the baked assets (norm stats). The project install is
# cheap because deps are already present from the cached layer above.
COPY . ${SAGEMAKER_SUBMIT_DIRECTORY}
RUN uv sync --frozen --no-dev

ENV PYTHONPATH=${SAGEMAKER_SUBMIT_DIRECTORY}/src:${SAGEMAKER_SUBMIT_DIRECTORY}:${PYTHONPATH}
# Offline: base checkpoints are mounted from a channel and passed via CLI; no gs:// egress.
ENV HF_HUB_OFFLINE=1
ENV HF_HOME=/tmp/hf_home
ENV TOKENIZERS_PARALLELISM=false
ENV GIT_PYTHON_REFRESH=quiet

# Reset WORKDIR to / so the DLC entrypoint (start_with_right_hostname.sh) finds
# /changehostname.c via its relative gcc reference.
WORKDIR /
