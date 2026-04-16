#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  gnupg \
  lsb-release \
  ninja-build \
  python3 \
  unzip \
  python3-venv \
  python3-pip

install -d /opt/clypt-phase1
install -d /opt/clypt-phase1/hf-cache
install -d /opt/clypt-phase1/venvs
install -d /opt/clypt-phase1/videos
install -d /opt/clypt-phase1/.cache/torch/kernels
install -d /opt/clypt-phase1/.cache/huggingface
install -d /etc/clypt-phase1
install -d /var/lib/clypt/v3_1_phase1_service
install -d /var/log/clypt/v3_1_phase1

if [[ ! -x /root/.bun/bin/bun ]]; then
  curl -fsSL https://bun.sh/install | bash
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This host may not be a GPU base image." >&2
fi

echo "Bootstrap complete. Next step: sync repo to /opt/clypt-phase1/repo, create /etc/clypt-phase1/v3_1_phase1.env, run deploy_vllm_service.sh (Phase 1 env + VibeVoice), then run deploy_sglang_qwen_service.sh (separate SGLang env for Qwen)."
