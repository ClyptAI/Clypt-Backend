#!/usr/bin/env bash
# Deploy VibeVoice vLLM on the RTX 6000 Ada audio host.
#
# VibeVoice on 48 GB VRAM does NOT need the L4-era bf16 encoder patch; we run
# native dtype. We use the baked docker/vibevoice-vllm/ image — all heavy
# setup (apt packages, `pip install -e vibevoice[vllm]`) happens at image
# build time, NOT on every container restart. The systemd unit calls
# `vllm serve` directly; it does NOT go through the upstream
# start_server.py wrapper (which reruns installs on every boot and cost us
# ~5 min per restart — see docs/ERROR_LOG.md 2026-04-17).
#
# GPU memory budget on the 48 GiB RTX 6000 Ada. vLLM is now sole tenant for
# model weights (NFA / emotion2vec+ / YAMNet moved back to the H200 on
# 2026-04-17), but ffmpeg node-media-prep still runs here and needs ~11 GiB
# of VRAM headroom for NVDEC CUDA contexts on concurrent h264_cuvid jobs.
# That's why the systemd unit pins --gpu-memory-utilization 0.77 instead of
# vLLM's 0.90 default — at 0.90 cuCtxCreate OOMs and every ffmpeg clip falls
# back to CPU decode. See clypt-vllm-vibevoice.service and
# docs/ERROR_LOG.md 2026-04-17.
#
# Usage (on the RTX host, as root):
#   REPO_DIR=/opt/clypt-audio-host/repo bash scripts/do_phase1_audio/deploy_vllm_service.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-audio-host/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-audio-host/audio_host.env}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-vllm-vibevoice:latest}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:${VLLM_HOST_PORT}/health}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-1200}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/clypt-audio-host/hf-cache}"
VIBEVOICE_REPO_URL="${VIBEVOICE_REPO_URL:-https://github.com/microsoft/VibeVoice.git}"
VIBEVOICE_REPO_REF="${VIBEVOICE_REPO_REF:-main}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-vllm-rtx] ERROR: run as root." >&2
  exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-vllm-rtx] ERROR: repo dir not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-vllm-rtx] ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

# Docker + NVIDIA runtime (NVIDIA AI/ML base image usually already has these).
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
fi
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

# Build the vLLM image. All vibevoice + system deps are baked in at build
# time (see docker/vibevoice-vllm/Dockerfile), so container restarts are
# fast and deterministic. We forward the VIBEVOICE_REF build arg so you can
# pin to a known-good vibevoice SHA/tag without editing the Dockerfile.
docker build \
  --build-arg "VIBEVOICE_REPO_URL=${VIBEVOICE_REPO_URL}" \
  --build-arg "VIBEVOICE_REF=${VIBEVOICE_REPO_REF}" \
  -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/

install -d -m 0755 "$HF_CACHE_DIR"

install -D -m 0644 \
  scripts/do_phase1_audio/systemd/clypt-vllm-vibevoice.service \
  /etc/systemd/system/clypt-vllm-vibevoice.service

systemctl daemon-reload
systemctl enable clypt-vllm-vibevoice.service
systemctl restart clypt-vllm-vibevoice.service

deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT_S ))
while true; do
  if curl -fsS "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
    echo "[deploy-vllm-rtx] vLLM healthy at $VLLM_HEALTH_URL"
    break
  fi
  if (( $(date +%s) >= deadline )); then
    echo "[deploy-vllm-rtx] ERROR: vLLM did not become healthy within ${VLLM_READY_TIMEOUT_S}s" >&2
    systemctl --no-pager --full status clypt-vllm-vibevoice.service >&2 || true
    docker logs --tail 200 clypt-vllm-vibevoice >&2 || true
    exit 1
  fi
  sleep 5
done

models_json="$(curl -fsS "http://127.0.0.1:${VLLM_HOST_PORT}/v1/models" || true)"
if ! grep -Eq '"id"[[:space:]]*:[[:space:]]*"vibevoice"' <<<"$models_json"; then
  echo "[deploy-vllm-rtx] ERROR: served model id is not 'vibevoice'." >&2
  printf '%s\n' "$models_json" >&2
  exit 1
fi

echo "[deploy-vllm-rtx] done."
