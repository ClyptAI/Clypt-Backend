#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

if [[ ! -e /dev/kfd ]]; then
  echo "[bootstrap-phase1-mi300x] ERROR: /dev/kfd is missing; use the gpu-amd-base image." >&2
  exit 1
fi
if [[ ! -d /dev/dri ]] || ! compgen -G "/dev/dri/renderD*" >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: /dev/dri/renderD* is missing; AMD VAAPI decode is required." >&2
  exit 1
fi
if ! command -v rocm-smi >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: rocm-smi is missing; use the gpu-amd-base image." >&2
  exit 1
fi
if ! command -v amd-smi >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: amd-smi is missing; use the gpu-amd-base image." >&2
  exit 1
fi

rocm-smi >/dev/null
amd-smi static >/dev/null

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl docker.io ffmpeg git gnupg lsb-release \
  libva2 mesa-va-drivers vainfo python3 python3-pip python3-venv unzip

systemctl enable --now docker

install -d /opt/clypt-phase1
install -d /opt/clypt-phase1/hf-cache
install -d /opt/clypt-phase1/.cache/torch/kernels
install -d /opt/clypt-phase1/.cache/huggingface
install -d /opt/clypt-phase1/venvs
install -d /etc/clypt-phase1
install -d /var/lib/clypt/phase1
install -d /var/log/clypt/phase1

echo "Bootstrap complete for Phase1 MI300X. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase1/repo"
echo "  2. create /etc/clypt-phase1/phase1.env for AMD/ROCm Phase1"
echo "  3. bash scripts/do_phase1/deploy_phase1_mi300x_services.sh"
