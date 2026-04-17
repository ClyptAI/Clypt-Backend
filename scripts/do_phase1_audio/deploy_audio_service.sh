#!/usr/bin/env bash
# Deploy the VibeVoice ASR + node-media-prep FastAPI service on the RTX 6000 Ada droplet.
#
# This installs:
#   * the vibevoice-asr venv (requirements-do-phase1-audio.txt)
#   * the systemd unit that runs the FastAPI app
#
# Prereq: deploy_vllm_service.sh has already started clypt-vllm-vibevoice.service.
#
# NFA / emotion2vec+ / YAMNet no longer run here — they moved back to the H200
# because NFA's global alignment needs ~17 GiB of VRAM that it couldn't get
# while co-located with vLLM on the 48 GiB RTX card. See docs/ERROR_LOG.md
# 2026-04-17. No NFA/emotion/YAMNet prewarm is performed by this script.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-audio-host/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-audio-host/audio_host.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1-audio.txt}"
VENV_DIR="${VENV_DIR:-/opt/clypt-audio-host/venvs/audio}"
PHASE1_CACHE_HOME="${PHASE1_CACHE_HOME:-/opt/clypt-audio-host/.cache}"
PIP_FALLBACK_LEGACY_RESOLVER="${PIP_FALLBACK_LEGACY_RESOLVER:-1}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-vibevoice-asr] ERROR: run as root." >&2
  exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-vibevoice-asr] ERROR: repo dir not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-vibevoice-asr] ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

# Host prereqs.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git python3 python3-pip python3-venv

# Validate env file and required vars for the vibevoice-asr host.
if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a" >/dev/null 2>&1; then
  echo "[deploy-vibevoice-asr] ERROR: env file is not shell-sourceable: $ENV_FILE" >&2
  exit 1
fi
for required in \
  CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_BIND \
  CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_PORT \
  CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN \
  VIBEVOICE_VLLM_BASE_URL \
  VIBEVOICE_VLLM_MODEL \
  GOOGLE_CLOUD_PROJECT \
  GCS_BUCKET; do
  if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$required:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-vibevoice-asr] ERROR: required env var missing/empty in $ENV_FILE: $required" >&2
    exit 1
  fi
done

# Hard-fail if NFA/emotion/YAMNet model envs are set here — those stages run
# on the H200 now.
for banned in \
  CLYPT_PHASE1_YAMNET_DEVICE \
  CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT \
  FUNASR_MODEL_SOURCE; do
  if /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$banned:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-vibevoice-asr] ERROR: $banned is set in $ENV_FILE but NFA/emotion/YAMNet run on the H200 now; remove it." >&2
    exit 1
  fi
done

set -a
source "$ENV_FILE"
set +a

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" && ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  echo "[deploy-vibevoice-asr] ERROR: GOOGLE_APPLICATION_CREDENTIALS points to a missing file: ${GOOGLE_APPLICATION_CREDENTIALS}" >&2
  exit 1
fi

# VibeVoice-ASR venv (FastAPI + GCS client + ffmpeg NVENC wrapper).
install -d -m 0755 "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
if ! python -m pip install -r "$REQUIREMENTS_FILE"; then
  if [[ "$PIP_FALLBACK_LEGACY_RESOLVER" != "1" ]]; then
    echo "[deploy-vibevoice-asr] ERROR: pip install failed and legacy-resolver fallback is disabled." >&2
    exit 1
  fi
  PIP_USE_DEPRECATED=legacy-resolver python -m pip install -r "$REQUIREMENTS_FILE"
fi

install -d -m 0755 "$PHASE1_CACHE_HOME"

install -d -m 0755 /var/log/clypt/audio-host

install -D -m 0644 \
  scripts/do_phase1_audio/systemd/clypt-audio-host.service \
  /etc/systemd/system/clypt-audio-host.service

systemctl daemon-reload
systemctl enable clypt-audio-host.service
systemctl restart clypt-audio-host.service

sleep 3
systemctl --no-pager --full status clypt-audio-host.service | head -n 25

echo "[deploy-vibevoice-asr] deployment complete."
echo "[deploy-vibevoice-asr] Smoke-test health:"
echo "    curl -fsS http://127.0.0.1:${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_PORT}/health"
