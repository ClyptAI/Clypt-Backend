#!/usr/bin/env bash
# Bootstrap the H200 Phase 1 VISUAL + Phase 2-4 host.
#
# This droplet runs the visual chain (RF-DETR + ByteTrack), the Phase 1
# orchestrator, the SGLang Qwen service, and the Phase 2-4 local worker.
# The audio chain (VibeVoice, NFA, emotion2vec+, YAMNet) and node-media prep
# live on the RTX 6000 Ada host — see scripts/do_phase1_audio/.
#
# Run as root on a fresh NVIDIA AI/ML base image droplet.
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

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase1/repo"
echo "  2. create /etc/clypt-phase1/v3_1_phase1.env (see docs/runtime/known-good.env)"
echo "  3. bash scripts/do_phase1_visual/deploy_visual_service.sh"
echo "  4. bash scripts/do_phase1_visual/deploy_sglang_qwen_service.sh"
