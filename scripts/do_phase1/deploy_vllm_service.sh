#!/usr/bin/env bash
# Full vLLM-path deployment script for the GPU droplet.
#
# This is the single script to run on a fresh droplet (or after an rsync)
# when deploying with VIBEVOICE_BACKEND=vllm.
#
# What it does:
#   1. Install Docker if not present (needed for the vLLM container)
#   2. pip-install requirements in the main worker venv
#   3. Validate torchaudio (used by NFA + emotion2vec+ in the main venv)
#   4. Install/refresh the API and worker systemd units
#   5. Build the vLLM Docker image
#   6. Install the vLLM systemd unit and start the container
#   7. Wait for the health endpoint to respond
#   8. Restart the Phase 1 worker so it picks up new code
#
# What it does NOT do (intentional):
#   - install the native VibeVoice venv
#   - build flash-attn from source
#   - touch cuda-toolkit (GPU base image CUDA driver is sufficient)
#
# Usage (run on the droplet as root):
#   REPO_DIR=/opt/clypt-phase1/repo bash scripts/do_phase1/deploy_vllm_service.sh
#
# Environment overrides:
#   REPO_DIR             — repo root (default: /opt/clypt-phase1/repo)
#   ENV_FILE             — systemd env file (default: /etc/clypt-phase1/v3_1_phase1.env)
#   REQUIREMENTS_FILE    — pip requirements for main venv (default: requirements-do-phase1.txt)
#   VLLM_IMAGE_TAG       — Docker image tag (default: clypt-vllm-vibevoice:latest)
#   VLLM_HOST_PORT       — loopback port for vLLM (default: 8000)
#   VLLM_HEALTH_URL      — full health URL to poll (default: http://127.0.0.1:8000/health)
#   VLLM_READY_TIMEOUT_S — seconds to wait for healthy (default: 600)
#   HF_CACHE_DIR         — HuggingFace model cache dir (default: /opt/clypt-phase1/hf-cache)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1.txt}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-vllm-vibevoice:latest}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:${VLLM_HOST_PORT}/health}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-600}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/clypt-phase1/hf-cache}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-vllm] ERROR: repo dir not found: $REPO_DIR" >&2
  echo "[deploy-vllm] rsync the repo first, then re-run this script." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-vllm] ERROR: env file not found: $ENV_FILE" >&2
  echo "[deploy-vllm] Create it from .env.example and set VIBEVOICE_BACKEND=vllm." >&2
  exit 1
fi

cd "$REPO_DIR"

# --- 1. Install Docker if not present -------------------------------------
if ! command -v docker &>/dev/null; then
  echo "[deploy-vllm] Docker not found — installing ..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
  echo "[deploy-vllm] Docker installed: $(docker --version)"
else
  echo "[deploy-vllm] Docker already present: $(docker --version)"
fi

# --- 2. Main worker venv — pip install (no native venv, no flash-attn) ----
echo "[deploy-vllm] installing main worker requirements ($REQUIREMENTS_FILE) ..."
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"

# --- 3. Validate torchaudio (used by NFA and emotion2vec+ in main venv) ---
python - <<'PY'
from backend.providers.vibevoice import validate_torchaudio_runtime
meta = validate_torchaudio_runtime()
print(f"[deploy-vllm] main venv torchaudio OK: {meta['torchaudio_version']}")
PY

# --- 4. Install / refresh API and worker systemd units --------------------
install -d -m 0755 /var/lib/clypt/v3_1_phase1_service
install -d -m 0755 /var/log/clypt/v3_1_phase1
install -d -m 0755 /opt/clypt-phase1/.cache/torch/kernels

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-v31-phase1-api.service \
  /etc/systemd/system/clypt-v31-phase1-api.service

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-v31-phase1-worker.service \
  /etc/systemd/system/clypt-v31-phase1-worker.service

systemctl daemon-reload
systemctl enable clypt-v31-phase1-api.service clypt-v31-phase1-worker.service

# --- 5. Build vLLM Docker image -------------------------------------------
echo "[deploy-vllm] building Docker image: $VLLM_IMAGE_TAG ..."
docker build -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/
echo "[deploy-vllm] image built: $VLLM_IMAGE_TAG"

# --- 6. HF cache dir + vLLM systemd unit ----------------------------------
install -d -m 0755 "$HF_CACHE_DIR"

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-vllm-vibevoice.service \
  /etc/systemd/system/clypt-vllm-vibevoice.service

systemctl daemon-reload
systemctl enable clypt-vllm-vibevoice.service

echo "[deploy-vllm] starting clypt-vllm-vibevoice.service ..."
systemctl restart clypt-vllm-vibevoice.service

# --- 7. Wait for vLLM health ----------------------------------------------
echo "[deploy-vllm] waiting for health OK at $VLLM_HEALTH_URL (timeout=${VLLM_READY_TIMEOUT_S}s) ..."
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT_S ))
while true; do
  if curl -fsS "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
    echo "[deploy-vllm] vLLM service is healthy ✓"
    break
  fi
  now=$(date +%s)
  if (( now >= deadline )); then
    echo "[deploy-vllm] ERROR: health check did not pass within ${VLLM_READY_TIMEOUT_S}s" >&2
    systemctl --no-pager --full status clypt-vllm-vibevoice.service >&2 || true
    docker logs --tail 100 clypt-vllm-vibevoice >&2 || true
    exit 1
  fi
  sleep 5
done

# --- 8. Restart Phase 1 API + worker with new code ------------------------
echo "[deploy-vllm] restarting Phase 1 API and worker ..."
systemctl restart clypt-v31-phase1-api.service
systemctl restart clypt-v31-phase1-worker.service

systemctl --no-pager --full status clypt-vllm-vibevoice.service || true
systemctl --no-pager --full status clypt-v31-phase1-worker.service || true
echo ""
echo "[deploy-vllm] deployment complete."
echo "[deploy-vllm] Ensure /etc/clypt-phase1/v3_1_phase1.env contains:"
echo "               VIBEVOICE_BACKEND=vllm"
echo "               VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:${VLLM_HOST_PORT}"
