#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

wait_for_apt_locks() {
  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi
  local waited_s=0
  local max_wait_s="${APT_LOCK_WAIT_S:-600}"
  local locks=(/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock)
  while fuser "${locks[@]}" >/dev/null 2>&1; do
    if (( waited_s >= max_wait_s )); then
      echo "[bootstrap-phase26-mi300x] ERROR: timed out waiting for apt/dpkg locks." >&2
      return 1
    fi
    echo "[bootstrap-phase26-mi300x] waiting for apt/dpkg locks..."
    sleep 5
    waited_s=$((waited_s + 5))
  done
}

wait_for_apt_locks
apt-get update
wait_for_apt_locks
packages=(
  build-essential ca-certificates curl ffmpeg git gnupg jq lsb-release
  python3 python3-pip python3-venv unzip
)
if ! command -v docker >/dev/null 2>&1; then
  packages+=(docker.io)
fi
wait_for_apt_locks
DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"

if [[ ! -e /dev/kfd ]]; then
  echo "[bootstrap-phase26-mi300x] ERROR: /dev/kfd is missing; this host is not exposing ROCm compute." >&2
  exit 1
fi
if [[ ! -d /dev/dri ]]; then
  echo "[bootstrap-phase26-mi300x] ERROR: /dev/dri is missing; this host is not exposing GPU render devices." >&2
  exit 1
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "[bootstrap-phase26-mi300x] ERROR: rocm-smi is not on PATH; use the DigitalOcean ROCm 7.2 MI300X application image or install ROCm tooling before deploy." >&2
  exit 1
fi
if ! command -v amd-smi >/dev/null 2>&1; then
  echo "[bootstrap-phase26-mi300x] ERROR: amd-smi is not on PATH; use the DigitalOcean ROCm 7.2 MI300X application image or install ROCm tooling before deploy." >&2
  exit 1
fi
rocm-smi
amd-smi static

systemctl enable --now docker

install -d /opt/clypt-phase26
install -d /opt/clypt-phase26/.cache/huggingface
install -d /opt/clypt-phase26/.cache/torch/kernels
install -d /opt/clypt-phase26/test-bank-cache/audio
install -d /opt/clypt-phase26/test-bank-cache/videos
install -d /opt/clypt-phase26/venvs
install -d /etc/clypt-phase26
install -d /etc/clypt
install -d /var/lib/clypt/phase1
install -d /var/lib/clypt/phase1/work
install -d /var/lib/clypt/phase26
install -d /var/log/clypt/phase1/logs
install -d /var/log/clypt/phase26

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase26/repo"
echo "  2. create /etc/clypt-phase26/phase26.env for the colocated Phase1 + Phase26 MI300X host"
echo "  3. bash scripts/do_phase26/deploy_phase26_mi300x_services.sh"
