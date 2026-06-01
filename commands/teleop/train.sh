#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ -z "${NVME_DIR:-}" ]]; then
  for candidate in /mnt/nvme /mnt/local_nvme /local_nvme /mnt/instance-store /scratch; do
    if [[ -d "${candidate}" && -w "${candidate}" ]]; then
      NVME_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -n "${NVME_DIR:-}" ]]; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${NVME_DIR}/cosmos-predict2-venv}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${NVME_DIR}/uv-cache}"
else
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
fi

mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv}")" "${UV_CACHE_DIR}"

if [[ -z "${COSMOS_CHECKPOINT_DIR:-}" ]]; then
  if [[ -n "${NVME_DIR:-}" ]]; then
    COSMOS_CHECKPOINT_DIR="${NVME_DIR}/cosmos-predict2-checkpoints"
  else
    COSMOS_CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
  fi
fi

if [[ -z "${MIMIC_VIDEO_ROOT:-}" ]]; then
  if [[ -d "/mnt/mimic-video-ebs/mimic-video" ]]; then
    MIMIC_VIDEO_ROOT="/mnt/mimic-video-ebs/mimic-video"
  else
    MIMIC_VIDEO_ROOT="${HOME}/code/mimic-video"
  fi
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${FROZEN_CHECKPOINT_DIR:-${MIMIC_VIDEO_ROOT}/model/checkpoints}}"
T5_DIR="${T5_DIR:-${CHECKPOINT_DIR}/text_encoder/t5-11b}"
COSMOS_MODEL_DIR="${COSMOS_CHECKPOINT_DIR}/nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_TOKENIZER="${COSMOS_MODEL_DIR}/tokenizer/tokenizer.pth"
COSMOS_MODEL="${COSMOS_MODEL_DIR}/model-480p-10fps.pt"

DATASET_PATH="${DATASET_PATH:-${MIMIC_VIDEO_ROOT}/data/teleop_raw}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${DATASET_PATH}/t5_xxl_instruction_cache}"
CAMERA_NAME="${CAMERA_NAME:-camera_top}"
EPISODE_GLOB="${EPISODE_GLOB:-*/episode_*}"
VIDEO_SIZE="${VIDEO_SIZE:-256,256}"
NUM_FRAMES="${NUM_FRAMES:-93}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"

NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
UV_EXTRA="${UV_EXTRA:-cu128}"
if [[ -z "${UV_SYNC_ARGS:-}" ]]; then
  if [[ "${UV_EXTRA}" == "cu126" ]]; then
    UV_SYNC_ARGS="--frozen --extra ${UV_EXTRA}"
  else
    UV_SYNC_ARGS="--extra ${UV_EXTRA}"
  fi
fi
if [[ -z "${UV_RUN_ARGS:-}" ]]; then
  if [[ "${UV_EXTRA}" == "cu126" ]]; then
    UV_RUN_ARGS="--frozen --extra ${UV_EXTRA}"
  else
    UV_RUN_ARGS="--extra ${UV_EXTRA}"
  fi
fi
read -r -a uv_sync_args <<< "${UV_SYNC_ARGS}"
read -r -a uv_run_args <<< "${UV_RUN_ARGS}"

BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

MAX_ITER="${MAX_ITER:-7000}"
VALIDATION_ITER="${VALIDATION_ITER:-200}"
MAX_VAL_ITER="${MAX_VAL_ITER:-10}"
SAVE_ITER="${SAVE_ITER:-500}"
LR="${LR:-1.778e-4}"

SKIP_EMBEDDINGS="${SKIP_EMBEDDINGS:-0}"
DOWNLOAD_COSMOS_CHECKPOINTS="${DOWNLOAD_COSMOS_CHECKPOINTS:-1}"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Dataset path does not exist: ${DATASET_PATH}" >&2
  echo "Set DATASET_PATH=/path/to/teleop_raw or MIMIC_VIDEO_ROOT=/path/to/mimic-video." >&2
  exit 1
fi

if [[ ! -d "${T5_DIR}" ]]; then
  echo "T5 checkpoint directory does not exist: ${T5_DIR}" >&2
  echo "Set T5_DIR=/path/to/text_encoder/t5-11b or FROZEN_CHECKPOINT_DIR=/path/to/checkpoints." >&2
  exit 1
fi

mkdir -p "${COSMOS_CHECKPOINT_DIR}"

export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${COSMOS_CHECKPOINT_DIR}}"

log "Repo root: ${REPO_ROOT}"
log "Mimic-video root: ${MIMIC_VIDEO_ROOT}"
log "Dataset path: ${DATASET_PATH}"
log "Episode glob: ${EPISODE_GLOB}"
log "Embedding cache: ${EMBEDDING_CACHE_DIR}"
log "T5 checkpoint: ${T5_DIR}"
log "Frozen checkpoints: ${CHECKPOINT_DIR}"
log "Cosmos checkpoints: ${COSMOS_CHECKPOINT_DIR}"
log "COSMOS_PREDICT2_ARGS: ${COSMOS_PREDICT2_ARGS}"
log "UV cache: ${UV_CACHE_DIR}"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  log "UV venv: ${UV_PROJECT_ENVIRONMENT}"
else
  log "UV venv: ${REPO_ROOT}/.venv"
fi
log "CUDA extra: ${UV_EXTRA}"
log "Training: NPROC=${NPROC}, context_parallel=${CONTEXT_PARALLEL_SIZE}, batch=${BATCH_SIZE}, val_batch=${VAL_BATCH_SIZE}"

log "Syncing uv environment: uv sync ${UV_SYNC_ARGS}"
uv sync "${uv_sync_args[@]}"

if [[ ! -f "${COSMOS_TOKENIZER}" || ! -f "${COSMOS_MODEL}" ]]; then
  if [[ "${DOWNLOAD_COSMOS_CHECKPOINTS}" == "1" ]]; then
    log "Downloading Cosmos 2B Video2World 480p/10fps checkpoints into ${COSMOS_CHECKPOINT_DIR}"
    uv run "${uv_run_args[@]}" python scripts/download_checkpoints.py \
      --checkpoint_dir "${COSMOS_CHECKPOINT_DIR}" \
      --model_sizes 2B \
      --model_types video2world \
      --resolution 480 \
      --fps 10
  else
    echo "Missing Cosmos checkpoint files:" >&2
    echo "  ${COSMOS_TOKENIZER}" >&2
    echo "  ${COSMOS_MODEL}" >&2
    echo "Set CHECKPOINT_DIR/FROZEN_CHECKPOINT_DIR to the Cosmos checkpoint root, or rerun with DOWNLOAD_COSMOS_CHECKPOINTS=1." >&2
    exit 1
  fi
fi

if [[ "${SKIP_EMBEDDINGS}" != "1" ]]; then
  log "Preparing unique instruction embeddings"
  uv run "${uv_run_args[@]}" python -m scripts.get_t5_embeddings_from_teleop_raw \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${EMBEDDING_CACHE_DIR}" \
    --episode_glob "${EPISODE_GLOB}" \
    --cache_dir "${T5_DIR}"
else
  log "Skipping embedding preparation because SKIP_EMBEDDINGS=1"
fi

log "Launching training"
IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}" uv run "${uv_run_args[@]}" torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_teleop_raw \
  dataloader_train.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_val.dataset.dataset_dir="${DATASET_PATH}" \
  dataloader_train.dataset.embedding_cache_dir="${EMBEDDING_CACHE_DIR}" \
  dataloader_val.dataset.embedding_cache_dir="${EMBEDDING_CACHE_DIR}" \
  dataloader_train.dataset.camera_name="${CAMERA_NAME}" \
  dataloader_val.dataset.camera_name="${CAMERA_NAME}" \
  dataloader_train.dataset.video_size="[${VIDEO_SIZE}]" \
  dataloader_val.dataset.video_size="[${VIDEO_SIZE}]" \
  dataloader_train.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_val.dataset.num_frames="${NUM_FRAMES}" \
  dataloader_train.dataset.val_fraction="${VAL_FRACTION}" \
  dataloader_val.dataset.val_fraction="${VAL_FRACTION}" \
  dataloader_train.dataset.split_seed="${SPLIT_SEED}" \
  dataloader_val.dataset.split_seed="${SPLIT_SEED}" \
  dataloader_train.dataset.episode_glob="${EPISODE_GLOB}" \
  dataloader_val.dataset.episode_glob="${EPISODE_GLOB}" \
  dataloader_train.batch_size="${BATCH_SIZE}" \
  dataloader_val.batch_size="${VAL_BATCH_SIZE}" \
  dataloader_train.num_workers="${NUM_WORKERS}" \
  dataloader_val.num_workers="${VAL_NUM_WORKERS}" \
  model_parallel.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
  trainer.max_iter="${MAX_ITER}" \
  trainer.validation_iter="${VALIDATION_ITER}" \
  trainer.max_val_iter="${MAX_VAL_ITER}" \
  checkpoint.save_iter="${SAVE_ITER}" \
  optimizer.lr="${LR}"
