#!/usr/bin/env bash
# Deploy script for the H200 Phase 1 VISUAL + Phase 2-4 host.
#
# What this host runs:
#   * Phase 1 visual chain: RF-DETR + ByteTrack (TensorRT FP16 fast path)
#   * Phase 1 API + worker (orchestrator; calls the RTX 6000 Ada audio host over HTTP)
#   * Phase 2-4 local SQLite queue worker (calls the RTX host for node-media prep)
#   * SGLang Qwen service is installed/started separately via deploy_sglang_qwen_service.sh
#
# What this host does NOT run:
#   * VibeVoice vLLM (on RTX 6000 Ada)
#   * NFA / emotion2vec+ / YAMNet (on RTX 6000 Ada)
#   * ffmpeg NVENC node-clip extraction (on RTX 6000 Ada — H200 NVENC is broken)
#
# Run on the H200 droplet as root after rsyncing the repo + creating the env file.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1-visual.txt}"
PHASE1_VENV_DIR="${PHASE1_VENV_DIR:-/opt/clypt-phase1/venvs/phase1}"
PIP_FALLBACK_LEGACY_RESOLVER="${PIP_FALLBACK_LEGACY_RESOLVER:-1}"
PHASE1_CACHE_HOME="${PHASE1_CACHE_HOME:-/opt/clypt-phase1/.cache}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-visual] ERROR: run as root." >&2
  exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-visual] ERROR: repo dir not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-visual] ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

# --- 1. Host prereqs -------------------------------------------------------
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git gnupg lsb-release \
  python3 python3-pip python3-venv unzip

# --- 2. Validate env file --------------------------------------------------
if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a" >/dev/null 2>&1; then
  echo "[deploy-visual] ERROR: env file is not shell-sourceable: $ENV_FILE" >&2
  exit 1
fi

# Required env vars for the H200 worker.
for required in \
  GOOGLE_CLOUD_PROJECT \
  GCS_BUCKET \
  CLYPT_PHASE1_AUDIO_HOST_URL \
  CLYPT_PHASE1_AUDIO_HOST_TOKEN \
  CLYPT_PHASE24_NODE_MEDIA_PREP_URL \
  CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN \
  CLYPT_PHASE24_QUEUE_BACKEND \
  GENAI_GENERATION_BACKEND \
  CLYPT_LOCAL_LLM_BASE_URL; do
  if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$required:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-visual] ERROR: required env var missing/empty in $ENV_FILE: $required" >&2
    exit 1
  fi
done

# Hard-fail if legacy audio-provider envs are still on the H200 — they have no
# effect here anymore and their presence signals a half-migrated host.
for banned in VIBEVOICE_BACKEND VIBEVOICE_VLLM_BASE_URL VIBEVOICE_VLLM_MODEL; do
  if /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$banned:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-visual] ERROR: $banned must not be set on the H200 env file." >&2
    echo "[deploy-visual] Those belong on the RTX 6000 Ada audio host (see docs/runtime/known-good-audio-host.env)." >&2
    exit 1
  fi
done

set -a
source "$ENV_FILE"
set +a

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" && ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  echo "[deploy-visual] ERROR: GOOGLE_APPLICATION_CREDENTIALS points to a missing file: ${GOOGLE_APPLICATION_CREDENTIALS}" >&2
  exit 1
fi

# --- 3. Phase 1 + Phase 2-4 venv ------------------------------------------
install -d -m 0755 "$(dirname "$PHASE1_VENV_DIR")"
python3 -m venv "$PHASE1_VENV_DIR"
. "$PHASE1_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1"
if ! python -m pip install -r "$REQUIREMENTS_FILE"; then
  if [[ "$PIP_FALLBACK_LEGACY_RESOLVER" != "1" ]]; then
    echo "[deploy-visual] ERROR: pip install failed and legacy-resolver fallback is disabled." >&2
    exit 1
  fi
  PIP_USE_DEPRECATED=legacy-resolver python -m pip install -r "$REQUIREMENTS_FILE"
fi

if [[ "${CLYPT_PHASE1_VISUAL_BACKEND:-}" == tensorrt* ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y libnvinfer-bin
  python -m pip install tensorrt-cu13
  if ! command -v trtexec >/dev/null 2>&1; then
    echo "[deploy-visual] ERROR: trtexec not found after installing libnvinfer-bin." >&2
    exit 1
  fi
  python - <<'PY'
import tensorrt as trt
print(f"[deploy-visual] TensorRT runtime OK: {trt.__version__}")
PY
fi

# --- 4. Ping the RTX audio host before starting services -------------------
echo "[deploy-visual] probing audio host at $CLYPT_PHASE1_AUDIO_HOST_URL ..."
if ! curl -fsS \
  -H "Authorization: Bearer ${CLYPT_PHASE1_AUDIO_HOST_TOKEN}" \
  "${CLYPT_PHASE1_AUDIO_HOST_URL%/}/health" >/dev/null; then
  echo "[deploy-visual] WARN: audio host /health did not respond. Workers will still start but will fail on first job." >&2
  echo "[deploy-visual] Deploy scripts/do_phase1_audio/deploy_audio_service.sh on the RTX box if you haven't yet." >&2
fi

# --- 5. systemd units ------------------------------------------------------
install -d -m 0755 /var/lib/clypt/v3_1_phase1_service
install -d -m 0755 /var/log/clypt/v3_1_phase1
install -d -m 0755 "$PHASE1_CACHE_HOME/torch/kernels"

install -D -m 0644 \
  scripts/do_phase1_visual/systemd/clypt-v31-phase1-api.service \
  /etc/systemd/system/clypt-v31-phase1-api.service

install -D -m 0644 \
  scripts/do_phase1_visual/systemd/clypt-v31-phase1-worker.service \
  /etc/systemd/system/clypt-v31-phase1-worker.service

install -D -m 0644 \
  scripts/do_phase1_visual/systemd/clypt-v31-phase24-local-worker.service \
  /etc/systemd/system/clypt-v31-phase24-local-worker.service

systemctl daemon-reload
systemctl enable \
  clypt-v31-phase1-api.service \
  clypt-v31-phase1-worker.service \
  clypt-v31-phase24-local-worker.service

systemctl restart clypt-v31-phase1-api.service
systemctl restart clypt-v31-phase1-worker.service
systemctl restart clypt-v31-phase24-local-worker.service

systemctl is-active --quiet clypt-v31-phase1-api.service
systemctl is-active --quiet clypt-v31-phase1-worker.service
systemctl is-active --quiet clypt-v31-phase24-local-worker.service

echo "[deploy-visual] done."
echo "[deploy-visual] Next: bash scripts/do_phase1_visual/deploy_sglang_qwen_service.sh"
