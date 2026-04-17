#!/usr/bin/env bash
# Deploy the Phase 1 audio host FastAPI service on the RTX 6000 Ada droplet.
#
# This installs:
#   * the audio-host venv (requirements-do-phase1-audio.txt)
#   * pre-warmed NFA + emotion2vec+ caches
#   * the systemd unit that runs the FastAPI app
#
# Prereq: deploy_vllm_service.sh has already started clypt-vllm-vibevoice.service.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-audio-host/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-audio-host/audio_host.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1-audio.txt}"
VENV_DIR="${VENV_DIR:-/opt/clypt-audio-host/venvs/audio}"
PHASE1_CACHE_HOME="${PHASE1_CACHE_HOME:-/opt/clypt-audio-host/.cache}"
PIP_FALLBACK_LEGACY_RESOLVER="${PIP_FALLBACK_LEGACY_RESOLVER:-1}"
PREWARM_AUDIO_MODELS="${PREWARM_AUDIO_MODELS:-1}"
PREWARM_RETRIES="${PREWARM_RETRIES:-3}"
PREWARM_RETRY_BACKOFF_S="${PREWARM_RETRY_BACKOFF_S:-20}"
PREWARM_TIMEOUT_S="${PREWARM_TIMEOUT_S:-1800}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-audio-host] ERROR: run as root." >&2
  exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-audio-host] ERROR: repo dir not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-audio-host] ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

# Host prereqs.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git python3 python3-pip python3-venv

# Validate env file and required vars for the audio host.
if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a" >/dev/null 2>&1; then
  echo "[deploy-audio-host] ERROR: env file is not shell-sourceable: $ENV_FILE" >&2
  exit 1
fi
for required in \
  CLYPT_PHASE1_AUDIO_HOST_BIND \
  CLYPT_PHASE1_AUDIO_HOST_PORT \
  CLYPT_PHASE1_AUDIO_HOST_TOKEN \
  VIBEVOICE_VLLM_BASE_URL \
  VIBEVOICE_VLLM_MODEL \
  GOOGLE_CLOUD_PROJECT \
  GCS_BUCKET; do
  if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$required:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-audio-host] ERROR: required env var missing/empty in $ENV_FILE: $required" >&2
    exit 1
  fi
done

set -a
source "$ENV_FILE"
set +a

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" && ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  echo "[deploy-audio-host] ERROR: GOOGLE_APPLICATION_CREDENTIALS points to a missing file: ${GOOGLE_APPLICATION_CREDENTIALS}" >&2
  exit 1
fi

# Audio-host venv.
install -d -m 0755 "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1" "Cython<3.1"
python -m pip install --no-build-isolation youtokentome
if ! python -m pip install -r "$REQUIREMENTS_FILE"; then
  if [[ "$PIP_FALLBACK_LEGACY_RESOLVER" != "1" ]]; then
    echo "[deploy-audio-host] ERROR: pip install failed and legacy-resolver fallback is disabled." >&2
    exit 1
  fi
  PIP_USE_DEPRECATED=legacy-resolver python -m pip install -r "$REQUIREMENTS_FILE"
fi
python -m pip install "setuptools==69.5.1"

# Validate torchaudio (NFA + emotion2vec+ depend on it).
python - <<'PY'
from backend.providers.audio_runtime import validate_torchaudio_runtime
meta = validate_torchaudio_runtime()
print(f"[deploy-audio-host] torchaudio OK: {meta['torchaudio_version']}")
PY

# Prewarm NFA + emotion2vec+ caches so first request doesn't pay the download.
install -d -m 0755 "$PHASE1_CACHE_HOME"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch/kernels"
install -d -m 0755 "$PHASE1_CACHE_HOME/huggingface"

if [[ "$PREWARM_AUDIO_MODELS" == "1" ]]; then
  echo "[deploy-audio-host] prewarming NFA + emotion2vec+ ..."
  export HOME="/opt/clypt-audio-host"
  export CLYPT_PHASE1_CACHE_HOME="$PHASE1_CACHE_HOME"
  export XDG_CACHE_HOME="$PHASE1_CACHE_HOME"
  export TORCH_HOME="$PHASE1_CACHE_HOME/torch"
  export HF_HOME="$PHASE1_CACHE_HOME/huggingface"
  export FUNASR_MODEL_SOURCE="${FUNASR_MODEL_SOURCE:-hf}"
  prewarm_attempt=1
  while true; do
    if timeout "${PREWARM_TIMEOUT_S}s" python - <<'PY'
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.forced_aligner import ForcedAlignmentProvider

aligner = ForcedAlignmentProvider()
if aligner._check_available():  # noqa: SLF001
    aligner._ensure_model(device="cpu")  # noqa: SLF001
    print("[deploy-audio-host] NFA prewarm complete")
else:
    print("[deploy-audio-host] NFA prewarm skipped (forced aligner unavailable)")

emotion = Emotion2VecPlusProvider()
emotion._ensure_model()  # noqa: SLF001
print("[deploy-audio-host] audio host prewarm complete ✓")
PY
    then
      break
    fi
    if (( prewarm_attempt >= PREWARM_RETRIES )); then
      echo "[deploy-audio-host] ERROR: prewarm failed after ${PREWARM_RETRIES} attempts." >&2
      exit 1
    fi
    echo "[deploy-audio-host] prewarm attempt ${prewarm_attempt} failed; retrying in ${PREWARM_RETRY_BACKOFF_S}s ..."
    sleep "$PREWARM_RETRY_BACKOFF_S"
    prewarm_attempt=$((prewarm_attempt + 1))
  done
else
  echo "[deploy-audio-host] skipping prewarm (PREWARM_AUDIO_MODELS=$PREWARM_AUDIO_MODELS)"
fi

install -d -m 0755 /var/log/clypt/audio-host

install -D -m 0644 \
  scripts/do_phase1_audio/systemd/clypt-audio-host.service \
  /etc/systemd/system/clypt-audio-host.service

systemctl daemon-reload
systemctl enable clypt-audio-host.service
systemctl restart clypt-audio-host.service

sleep 3
systemctl --no-pager --full status clypt-audio-host.service | head -n 25

echo "[deploy-audio-host] deployment complete."
echo "[deploy-audio-host] Smoke-test health:"
echo "    curl -fsS http://127.0.0.1:${CLYPT_PHASE1_AUDIO_HOST_PORT}/health"
