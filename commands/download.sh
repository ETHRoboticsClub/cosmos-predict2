#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

S3_URI="${S3_URI:-s3://ethrc-ml-data-916780037007/robot-learning/teleop/}"
OUTPUT_DIR="${OUTPUT_DIR:-/nvme/datasets/teleop/raw}"
S3_REGION="${S3_REGION:-us-east-1}"

cd "${REPO_ROOT}"
source ".env.secrets"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is not installed in this container." >&2
  echo "Install it with: apt-get update && apt-get install -y awscli" >&2
  echo "Or rebuild the Docker image after pulling this script/Dockerfile change." >&2
  exit 2
fi

aws --region "${S3_REGION}" s3 sync "${S3_URI}" "${OUTPUT_DIR}" "$@"


# HF_OUTPUT_DIR="${HF_OUTPUT_DIR:-${OUTPUT_DIR}}"
# HF_DATASETS=(
#   "ETHRC/yams-carton-box-closing-fri-tom-mat-varing-fan-position"
#   "ETHRC/yams-carton-box-closing-wed-tom-elias"
#   "ETHRC/yams-carton-box-closing-tue-tom-mat-2"
#   "ETHRC/towelspring26_3-trimmed"
#   "ETHRC/towelspring26_2"
#   "ETHRC/yams-carton-box-closing-noe-exploring-30-04-2026"
#   "ETHRC/yams-closed-carton-box-to-migros-basket-go2"
#   "ETHRC/yams-carton-box-closing-sat-michael-mat-varing-fan-position-25-04-2025"
#   "ETHRC/yams-carton-box-closing-combined"
#   "ETHRC/yams-carton-box-closing-mon-tom-mat"
#   "ETHRC/yams-carton-box-closing-19-04-2026"
#   "ETHRC/towelspring26_3"
#   "ETHRC/towelspring26_realsense"
# )

# mkdir -p "${HF_OUTPUT_DIR}"

# for dataset in "${HF_DATASETS[@]}"; do
#   dataset_name="${dataset#*/}"
#   hf download "${dataset}" \
#     --repo-type dataset \
#     --local-dir "${HF_OUTPUT_DIR}/${dataset_name}" \
#     "$@"
# done
