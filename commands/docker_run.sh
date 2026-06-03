#!/usr/bin/env bash
set -euo pipefail

source .env.secrets
source .env.paths

HF_HOME="${NVME}/hf-home"
ROOT_CACHE="${NVME}/container-root-cache"
image_tag="cosmos-predict2:nightly"

mkdir -p "$HF_HOME" "$ROOT_CACHE" outputs

docker run -it --runtime=nvidia --ipc=host --rm \
  -v .:/workspace \
  -v /workspace/.venv \
  -v "$ROOT_CACHE:/root/.cache" \
  -v "$HF_HOME:$HF_HOME" \
  -v /opt/dlami/nvme:/opt/dlami/nvme \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME="$HF_HOME" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" \
  "$image_tag"
