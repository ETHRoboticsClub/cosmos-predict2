#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Upload all video files from a local folder to the cosmos-2-eval S3 prefix.
#
# Usage:
#   ./scripts/upload_eval_videos_to_s3.sh <source_dir> [s3_prefix]
#
# Examples:
#   ./scripts/upload_eval_videos_to_s3.sh output
#   ./scripts/upload_eval_videos_to_s3.sh output s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval/run-001/

set -euo pipefail

DEFAULT_S3_PREFIX="s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval"

usage() {
    cat <<EOF
Usage: $(basename "$0") <source_dir> [s3_prefix]

Upload every .mp4/.MP4 in <source_dir> to S3 (basename preserved).

Arguments:
  source_dir   Local folder containing generated videos
  s3_prefix    S3 destination prefix (default: ${DEFAULT_S3_PREFIX}/)

Environment:
  DRY_RUN=1    Print commands without uploading

Example:
  $(basename "$0") output
  $(basename "$0") output s3://ethrc-ml-data-916780037007/robot-learning/cosmos-2-eval/yams-ep0/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SOURCE_DIR="${1:-}"
S3_PREFIX="${2:-$DEFAULT_S3_PREFIX}"

if [[ -z "$SOURCE_DIR" ]]; then
    echo "Error: source_dir is required." >&2
    usage >&2
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: not a directory: $SOURCE_DIR" >&2
    exit 1
fi

# Normalize S3 prefix to end with /
if [[ "$S3_PREFIX" != */ ]]; then
    S3_PREFIX="${S3_PREFIX}/"
fi

shopt -s nullglob
videos=("$SOURCE_DIR"/*.mp4 "$SOURCE_DIR"/*.MP4)
shopt -u nullglob

if [[ ${#videos[@]} -eq 0 ]]; then
    echo "No .mp4 files found in $SOURCE_DIR"
    exit 1
fi

echo "Uploading ${#videos[@]} video(s) from $SOURCE_DIR to ${S3_PREFIX}"

for video in "${videos[@]}"; do
  dest="${S3_PREFIX}$(basename "$video")"
  if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "aws s3 cp $(printf '%q' "$video") $(printf '%q' "$dest")"
  else
    echo "-> $dest"
    aws s3 cp "$video" "$dest"
  fi
done

echo "Done."
