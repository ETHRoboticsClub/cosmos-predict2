#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.env.paths"

DATASET_PATH="${DATASET_PATH:-${OUT_ROOT:-/nvme/datasets/teleop/preprocessed}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nvme/checkpoints}"
T5_CACHE_DIR="${T5_CACHE_DIR:-${CHECKPOINT_DIR}/google-t5/t5-11b}"

cd "${REPO_ROOT}"

echo "Precomputing unique T5 embeddings"
echo "  dataset path: ${DATASET_PATH}"
echo "  T5 cache dir: ${T5_CACHE_DIR}"

if [[ ! -d "${T5_CACHE_DIR}" ]]; then
  echo "Missing T5 checkpoint directory: ${T5_CACHE_DIR}" >&2
  echo "Download it with: python scripts/download_checkpoints.py --checkpoint_dir ${CHECKPOINT_DIR} --model_types video2world --model_sizes 2B --resolution 480 --fps 10" >&2
  exit 1
fi

python -m scripts.get_t5_embeddings --dataset_path "${DATASET_PATH}" --cache_dir "${T5_CACHE_DIR}" "$@"
