#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y \
  ffmpeg \
  git \
  python3 \
  python3-venv \
  python3-pip

install -d /opt/clypt-phase1
install -d /var/lib/clypt/v3_1_phase1_service
install -d /var/log/clypt/v3_1_phase1

echo "Bootstrap complete. Next step: sync the repo to /opt/clypt-phase1/repo and run deploy_phase1_service.sh"
