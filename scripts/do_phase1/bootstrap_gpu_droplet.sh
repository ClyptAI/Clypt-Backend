#!/usr/bin/env bash
set -euo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  cuda-toolkit-12-4 \
  ffmpeg \
  git \
  gnupg \
  lsb-release \
  python3 \
  unzip \
  python3-venv \
  python3-pip

install -d /opt/clypt-phase1
install -d /var/lib/clypt/v3_1_phase1_service
install -d /var/log/clypt/v3_1_phase1

if [[ ! -x /root/.bun/bin/bun ]]; then
  curl -fsSL https://bun.sh/install | bash
fi

echo "Bootstrap complete. Next step: sync the repo to /opt/clypt-phase1/repo and run deploy_vllm_service.sh"
