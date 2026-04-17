#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git gnupg lsb-release \
  python3 python3-pip python3-venv unzip

install -d /opt/clypt-phase1
install -d /opt/clypt-phase1/hf-cache
install -d /opt/clypt-phase1/.cache/torch/kernels
install -d /opt/clypt-phase1/.cache/huggingface
install -d /opt/clypt-phase1/venvs
install -d /etc/clypt-phase1
install -d /var/lib/clypt/phase1
install -d /var/log/clypt/phase1

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase1/repo"
echo "  2. create /etc/clypt-phase1/phase1.env (see docs/runtime/known-good-phase1-h200.env)"
echo "  3. bash scripts/do_phase1/deploy_phase1_services.sh"
