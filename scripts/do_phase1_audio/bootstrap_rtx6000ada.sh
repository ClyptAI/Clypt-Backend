#!/usr/bin/env bash
# Bootstrap script for the RTX 6000 Ada Phase 1 AUDIO host.
#
# This droplet runs:
#   * VibeVoice vLLM ASR (GPU, native dtype — 48 GB is plenty)
#   * NeMo Forced Aligner + emotion2vec+ + YAMNet (GPU + CPU)
#   * ffmpeg NVENC/NVDEC node-clip extraction for Phase 2 node-media prep
#   * The FastAPI host wired at backend.runtime.phase1_audio_service.app:create_app()
#
# It does NOT run RF-DETR, SGLang, or the Phase 2–4 worker — those live on the H200.
#
# Run as root on a fresh NVIDIA AI/ML base image droplet. Idempotent.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[bootstrap-rtx6000ada] ERROR: run as root." >&2
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
  python3-pip \
  python3-venv \
  unzip

install -d /opt/clypt-audio-host
install -d /opt/clypt-audio-host/venvs
install -d /opt/clypt-audio-host/hf-cache
install -d /opt/clypt-audio-host/.cache/torch/kernels
install -d /opt/clypt-audio-host/.cache/huggingface
install -d /opt/clypt-audio-host/scratch
install -d /etc/clypt-audio-host
install -d /var/log/clypt/audio-host

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap-rtx6000ada] WARNING: nvidia-smi not found — this host may not be an NVIDIA AI/ML image." >&2
fi

# Smoke-test NVENC. This is the whole reason we chose RTX 6000 Ada over H200.
if command -v ffmpeg >/dev/null 2>&1; then
  if ffmpeg -hide_banner -loglevel error -init_hw_device cuda=cu -c:v h264_nvenc -f null - </dev/null 2>/dev/null; then
    echo "[bootstrap-rtx6000ada] NVENC (h264_nvenc) smoke test OK."
  else
    echo "[bootstrap-rtx6000ada] WARNING: h264_nvenc init failed — node-media prep will fail." >&2
  fi
fi

echo "[bootstrap-rtx6000ada] bootstrap complete."
echo "[bootstrap-rtx6000ada] Next:"
echo "  1. rsync the repo to /opt/clypt-audio-host/repo"
echo "  2. create /etc/clypt-audio-host/audio_host.env (see docs/runtime/known-good-audio-host.env)"
echo "  3. bash scripts/do_phase1_audio/deploy_vllm_service.sh"
echo "  4. bash scripts/do_phase1_audio/deploy_audio_service.sh"
