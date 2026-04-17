#!/usr/bin/env bash
# Deploy VibeVoice vLLM on the RTX 6000 Ada audio host.
#
# VibeVoice on 48 GB VRAM does NOT need the L4-era bf16 encoder patch; we run
# native dtype. We reuse the existing docker/vibevoice-vllm/ image definition
# and mount the VibeVoice repo for the entrypoint script.
#
# GPU memory: VibeVoice weights take ~18.2 GiB in bfloat16. We set
# --gpu-memory-utilization=0.85 (~40.8 GiB budget on a 48 GiB RTX 6000 Ada)
# to leave ~22 GiB for KV cache + activations. Values below ~0.55 trip
# `No available memory for the cache blocks` on this model.
#
# Usage (on the RTX host, as root):
#   REPO_DIR=/opt/clypt-audio-host/repo bash scripts/do_phase1_audio/deploy_vllm_service.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-audio-host/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-audio-host/audio_host.env}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-vllm-vibevoice:latest}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:${VLLM_HOST_PORT}/health}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-2400}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/clypt-audio-host/hf-cache}"
VIBEVOICE_REPO_DIR="${VIBEVOICE_REPO_DIR:-/opt/clypt-audio-host/vibevoice-repo}"
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

# Sync VibeVoice repo for the container mount.
if [[ -d "$VIBEVOICE_REPO_DIR/.git" ]]; then
  git -C "$VIBEVOICE_REPO_DIR" fetch --depth 1 origin "$VIBEVOICE_REPO_REF"
  git -C "$VIBEVOICE_REPO_DIR" checkout --force FETCH_HEAD
else
  git clone --depth 1 --branch "$VIBEVOICE_REPO_REF" "$VIBEVOICE_REPO_URL" "$VIBEVOICE_REPO_DIR"
fi

# Build the vLLM image (shared with H200 image def — entrypoint script is the same).
docker build -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/

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
