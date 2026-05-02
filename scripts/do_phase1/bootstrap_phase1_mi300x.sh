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
      echo "[bootstrap-phase1-mi300x] ERROR: timed out waiting for apt/dpkg locks." >&2
      return 1
    fi
    echo "[bootstrap-phase1-mi300x] waiting for apt/dpkg locks..."
    sleep 5
    waited_s=$((waited_s + 5))
  done
}

if [[ ! -e /dev/kfd ]]; then
  echo "[bootstrap-phase1-mi300x] ERROR: /dev/kfd is missing; use the DigitalOcean ROCm 7.2 MI300X application image." >&2
  exit 1
fi
if [[ ! -d /dev/dri ]] || ! compgen -G "/dev/dri/renderD*" >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: /dev/dri/renderD* is missing; AMD VAAPI decode is required." >&2
  exit 1
fi
if ! command -v rocm-smi >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: rocm-smi is missing; use the DigitalOcean ROCm 7.2 MI300X application image." >&2
  exit 1
fi
if ! command -v amd-smi >/dev/null; then
  echo "[bootstrap-phase1-mi300x] ERROR: amd-smi is missing; use the DigitalOcean ROCm 7.2 MI300X application image." >&2
  exit 1
fi

rocm-smi >/dev/null
amd-smi static >/dev/null

wait_for_apt_locks
apt-get update
wait_for_apt_locks
packages=(
  build-essential ca-certificates curl ffmpeg git gnupg lsb-release
  libva2 mesa-va-drivers vainfo python3 python3-pip python3-venv unzip
)
if ! command -v docker >/dev/null 2>&1; then
  packages+=(docker.io)
fi
wait_for_apt_locks
DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"

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
