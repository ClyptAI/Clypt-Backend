#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

detect_tensorrt_version() {
  local cuda_suffix
  local trt_version

  if dpkg-query -W -f='${Status}\n' cuda-toolkit-12-9 2>/dev/null | grep -q "ok installed"; then
    cuda_suffix="cuda12.9"
  elif command -v nvcc >/dev/null 2>&1; then
    cuda_suffix="cuda$(nvcc --version | awk -F'[ ,]+' '/release/ {print $6; exit}')"
  else
    cuda_suffix="cuda12.9"
  fi

  trt_version="$(apt-cache madison tensorrt | awk -v suffix="$cuda_suffix" '$0 ~ suffix {print $3; exit}')"
  if [[ -z "$trt_version" ]]; then
    echo "Unable to find a TensorRT package matching ${cuda_suffix}" >&2
    return 1
  fi

  printf '%s\n' "$trt_version"
}

TENSORRT_VERSION="$(detect_tensorrt_version)"

apt-get install -y \
  build-essential \
  cmake \
  ffmpeg \
  git \
  jq \
  libgl1 \
  libglib2.0-0 \
  libsndfile1 \
  python3 \
  python3-pip \
  python3-venv \
  rsync

apt-get install -y --allow-downgrades "tensorrt=${TENSORRT_VERSION}"

mkdir -p /opt/clypt-phase1
mkdir -p /etc/clypt-phase1
mkdir -p /var/lib/clypt/do_phase1_service

chmod 755 /opt/clypt-phase1
chmod 755 /etc/clypt-phase1
chmod 755 /var/lib/clypt/do_phase1_service

echo "Bootstrap complete."
echo "Next:"
echo "1. Copy backend/do_phase1_service/.env.example to /etc/clypt-phase1/do-phase1.env and fill it in."
echo "2. Copy GCP service-account JSON to /etc/clypt-phase1/gcp-sa.json."
echo "3. Run scripts/do_phase1/deploy_phase1_service.sh from the repo checkout on the droplet."
