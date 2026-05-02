#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git gnupg jq lsb-release \
  python3 python3-pip python3-venv unzip docker.io

if [[ ! -e /dev/kfd ]]; then
  echo "[bootstrap-phase26-mi300x] ERROR: /dev/kfd is missing; this host is not exposing ROCm compute." >&2
  exit 1
fi
if [[ ! -d /dev/dri ]]; then
  echo "[bootstrap-phase26-mi300x] ERROR: /dev/dri is missing; this host is not exposing GPU render devices." >&2
  exit 1
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "[bootstrap-phase26-mi300x] ERROR: rocm-smi is not on PATH; use the gpu-amd-base image or install ROCm tooling before deploy." >&2
  exit 1
fi
if ! command -v amd-smi >/dev/null 2>&1; then
  echo "[bootstrap-phase26-mi300x] ERROR: amd-smi is not on PATH; use the gpu-amd-base image or install ROCm tooling before deploy." >&2
  exit 1
fi
rocm-smi
amd-smi static

systemctl enable --now docker

install -d /opt/clypt-phase26
install -d /opt/clypt-phase26/.cache/huggingface
install -d /opt/clypt-phase26/.cache/torch/kernels
install -d /opt/clypt-phase26/venvs
install -d /etc/clypt-phase26
install -d /etc/clypt
install -d /var/lib/clypt/phase26
install -d /var/log/clypt/phase26

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase26/repo"
echo "  2. create /etc/clypt-phase26/phase26.env for the MI300X Phase26 host"
echo "  3. bash scripts/do_phase26/deploy_phase26_mi300x_services.sh"
