#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/preamble.sh
source "$SCRIPT_DIR/../lib/preamble.sh"

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/phase1.env}"
PHASE1_VENV_DIR="${PHASE1_VENV_DIR:-/opt/clypt-phase1/venvs/phase1}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-phase1-vllm-vibevoice:latest}"

require_root
require_dir "$REPO_DIR"
require_file "$ENV_FILE"
cd "$REPO_DIR"
load_env_file "$ENV_FILE"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl docker.io ffmpeg git gnupg lsb-release python3 python3-pip python3-venv unzip

if ! command -v nvidia-ctk &>/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

python3 -m venv "$PHASE1_VENV_DIR"
. "$PHASE1_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1" "Cython<4"
python -m pip install --no-build-isolation "youtokentome>=1.0.5"
python -m pip install -r requirements-do-phase1-h200.txt

install -d -m 0755 /etc/clypt
install -D -m 0644 scripts/do_phase1/clypt-phase1-runtime.env /etc/clypt/clypt-phase1-runtime.env

docker build -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/

install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-vibevoice.service /etc/systemd/system/clypt-phase1-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-vllm-vibevoice.service /etc/systemd/system/clypt-phase1-vllm-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-visual.service /etc/systemd/system/clypt-phase1-visual.service

systemctl daemon-reload
systemctl enable \
  clypt-phase1-vllm-vibevoice.service \
  clypt-phase1-vibevoice.service \
  clypt-phase1-visual.service \
  clypt-phase1-api.service \
  clypt-phase1-worker.service

systemctl restart clypt-phase1-vllm-vibevoice.service
systemctl restart clypt-phase1-vibevoice.service
systemctl restart clypt-phase1-visual.service
systemctl restart clypt-phase1-api.service
systemctl restart clypt-phase1-worker.service

echo "[deploy-phase1] done."
