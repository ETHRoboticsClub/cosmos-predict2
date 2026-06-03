#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.env.paths"

DATASET_PATH="${DATASET_PATH:-${OUT_ROOT:-${NVME}/datasets/teleop/preprocessed}}"

cd "${REPO_ROOT}"

echo "Precomputing unique T5 embeddings"
echo "  dataset path: ${DATASET_PATH}"

python -m scripts.get_t5_embeddings --dataset_path "${DATASET_PATH}" "$@"
