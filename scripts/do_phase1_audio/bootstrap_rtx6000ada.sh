#!/usr/bin/env bash
# Bootstrap script for the RTX 6000 Ada Phase 1 VibeVoice ASR host.
#
# This droplet runs:
#   * VibeVoice vLLM ASR (sole tenant on the card, native dtype — 48 GB is plenty)
#   * ffmpeg NVENC/NVDEC node-clip extraction for Phase 2 node-media prep
#   * The FastAPI host wired at backend.runtime.phase1_audio_service.app:create_app()
#
# It does NOT run NFA / emotion2vec+ / YAMNet (those moved back to the H200 —
# see docs/ERROR_LOG.md 2026-04-17) nor RF-DETR / SGLang / Phase 2–4 worker.
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
# NVENC requires frame dimensions >= 80x80, so use a synthetic 256x256 color
# source as the input. If this fails, h264_nvenc cannot init on this host and
# node-media prep will fail.
if command -v ffmpeg >/dev/null 2>&1; then
  if ffmpeg -hide_banner -loglevel error \
      -f lavfi -i color=black:s=256x256:d=1 \
      -c:v h264_nvenc -f null - </dev/null 2>/dev/null; then
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
