#!/usr/bin/env bash
set -euo pipefail

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/opt/dlami/nvme/docker}"
CONTAINERD_DATA_ROOT="${CONTAINERD_DATA_ROOT:-/opt/dlami/nvme/containerd}"
CONTAINERD_BACKUP_ROOT="${CONTAINERD_BACKUP_ROOT:-/opt/dlami/nvme/containerd.bak-cosmos-predict2}"

sudo systemctl stop docker containerd
sudo mkdir -p "$DOCKER_DATA_ROOT" "$CONTAINERD_DATA_ROOT"
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak-cosmos-predict2 2>/dev/null || true

if [ ! -L /var/lib/containerd ]; then
  sudo mv /var/lib/containerd "$CONTAINERD_BACKUP_ROOT"
  sudo ln -s "$CONTAINERD_DATA_ROOT" /var/lib/containerd
fi

if [ -e /var/lib/containerd.bak-cosmos-predict2 ]; then
  sudo mv /var/lib/containerd.bak-cosmos-predict2 "$CONTAINERD_BACKUP_ROOT"
fi

sudo tee /etc/docker/daemon.json >/dev/null <<CONFIG
{
    "data-root": "$DOCKER_DATA_ROOT",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
CONFIG

sudo systemctl start containerd docker
docker info --format '{{.DockerRootDir}}'
ls -ld /var/lib/containerd
