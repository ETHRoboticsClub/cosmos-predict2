#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="cosmos-predict2:nightly"
IMAGE_ARCHIVE="$REPO_ROOT/docker-images/cosmos-predict2-nightly.tar.gz"
OVERWRITE=1

usage() {
  echo "Usage: $0 [--overwrite]" >&2
}

while (($#)); do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

cd "$REPO_ROOT"
"$SCRIPT_DIR/configure_docker_nvme.sh"

if [[ "$OVERWRITE" == "1" ]]; then
  rm -f "$IMAGE_ARCHIVE"
  docker image rm -f "$IMAGE_NAME" >/dev/null 2>&1 || true
else
  docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 && exit 0

  if [ -f "$IMAGE_ARCHIVE" ]; then
    gunzip -c "$IMAGE_ARCHIVE" | docker load
    exit 0
  fi
fi

docker build --progress=plain -t "$IMAGE_NAME" -f Dockerfile .
mkdir -p "$(dirname "$IMAGE_ARCHIVE")"
echo "Saving Docker image archive to $IMAGE_ARCHIVE"
(
  docker save "$IMAGE_NAME" | gzip -1 > "$IMAGE_ARCHIVE"
) &
archive_pid=$!

(
  while kill -0 "$archive_pid" 2>/dev/null; do
    sleep "${SAVE_PROGRESS_INTERVAL:-30}"
    kill -0 "$archive_pid" 2>/dev/null || break
    if [ -f "$IMAGE_ARCHIVE" ]; then
      archive_size="$(du -h "$IMAGE_ARCHIVE" | cut -f1)"
    else
      archive_size="0"
    fi
    echo "Still saving image archive... ${archive_size} written"
  done
) &
progress_pid=$!

if ! wait "$archive_pid"; then
  kill "$progress_pid" >/dev/null 2>&1 || true
  wait "$progress_pid" 2>/dev/null || true
  echo "Failed to save Docker image archive" >&2
  exit 1
fi

kill "$progress_pid" >/dev/null 2>&1 || true
wait "$progress_pid" 2>/dev/null || true
ls -lh "$IMAGE_ARCHIVE"
