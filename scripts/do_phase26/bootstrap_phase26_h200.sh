#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git gnupg lsb-release \
  ninja-build python3 python3-pip python3-venv unzip

install -d /opt/clypt-phase26
install -d /opt/clypt-phase26/hf-cache
install -d /opt/clypt-phase26/.cache/torch/kernels
install -d /opt/clypt-phase26/.cache/huggingface
install -d /opt/clypt-phase26/venvs
install -d /etc/clypt-phase26
install -d /var/lib/clypt/phase26
install -d /var/log/clypt/phase26

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase26/repo"
echo "  2. create /etc/clypt-phase26/phase26.env (see docs/runtime/known-good-phase26-h200.env)"
echo "  3. bash scripts/do_phase26/deploy_phase26_services.sh"
