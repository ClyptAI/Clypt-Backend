#!/usr/bin/env bash
# Deploy script for the H200 Phase 1 VISUAL + Phase 2-4 host.
#
# What this host runs:
#   * Phase 1 visual chain: RF-DETR + ByteTrack (TensorRT FP16 fast path)
#   * Phase 1 audio post-processing: NFA + emotion2vec+ + YAMNet (in-process)
#   * Phase 1 API + worker (orchestrator; calls the RTX 6000 Ada VibeVoice ASR host over HTTP)
#   * Phase 2-4 local SQLite queue worker (calls the RTX host for node-media prep)
#   * SGLang Qwen service is installed/started separately via deploy_sglang_qwen_service.sh
#
# What this host does NOT run:
#   * VibeVoice vLLM ASR (on RTX 6000 Ada)
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
PREWARM_AUDIO_MODELS="${PREWARM_AUDIO_MODELS:-1}"
PREWARM_RETRIES="${PREWARM_RETRIES:-3}"
PREWARM_RETRY_BACKOFF_S="${PREWARM_RETRY_BACKOFF_S:-20}"
PREWARM_TIMEOUT_S="${PREWARM_TIMEOUT_S:-1800}"

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

# Required env vars for the H200 worker. Accept either the new
# CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_* names or the deprecated
# CLYPT_PHASE1_AUDIO_HOST_* aliases (dropped in next release).
for required in \
  GOOGLE_CLOUD_PROJECT \
  GCS_BUCKET \
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

# Require at least one of the VibeVoice-ASR URL/token pairs (new or deprecated name).
for pair in URL TOKEN; do
  new_name="CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_${pair}"
  old_name="CLYPT_PHASE1_AUDIO_HOST_${pair}"
  if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$new_name:-} || -n \${$old_name:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-visual] ERROR: set $new_name (or legacy $old_name) in $ENV_FILE." >&2
    exit 1
  fi
done

# Hard-fail if VibeVoice vLLM envs are still on the H200 — vLLM runs on the RTX
# box, not here.
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

# Validate torchaudio (NFA + emotion2vec+ depend on it).
python - <<'PY'
from backend.providers.audio_runtime import validate_torchaudio_runtime
meta = validate_torchaudio_runtime()
print(f"[deploy-visual] torchaudio OK: {meta['torchaudio_version']}")
PY

# --- 3a. Prewarm NFA + emotion2vec+ model caches on the H200 ---------------
install -d -m 0755 "$PHASE1_CACHE_HOME"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch/kernels"
install -d -m 0755 "$PHASE1_CACHE_HOME/huggingface"

# Stop API/worker first so they do not contend on cache locks while prewarm
# is trying to populate shared paths.
systemctl stop clypt-v31-phase1-worker.service clypt-v31-phase1-api.service 2>/dev/null || true

if [[ "$PREWARM_AUDIO_MODELS" == "1" ]]; then
  echo "[deploy-visual] prewarming NFA + emotion2vec+ ..."
  export HOME="/opt/clypt-phase1"
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
    print("[deploy-visual] NFA prewarm complete")
else:
    print("[deploy-visual] NFA prewarm skipped (forced aligner unavailable)")

emotion = Emotion2VecPlusProvider()
emotion._ensure_model()  # noqa: SLF001
print("[deploy-visual] NFA + emotion2vec+ prewarm complete")
PY
    then
      break
    fi
    if (( prewarm_attempt >= PREWARM_RETRIES )); then
      echo "[deploy-visual] ERROR: prewarm failed after ${PREWARM_RETRIES} attempts." >&2
      exit 1
    fi
    echo "[deploy-visual] prewarm attempt ${prewarm_attempt} failed; retrying in ${PREWARM_RETRY_BACKOFF_S}s ..."
    sleep "$PREWARM_RETRY_BACKOFF_S"
    prewarm_attempt=$((prewarm_attempt + 1))
  done
else
  echo "[deploy-visual] skipping audio-model prewarm (PREWARM_AUDIO_MODELS=$PREWARM_AUDIO_MODELS)"
fi

# --- 4. Ping the RTX VibeVoice-ASR host before starting services -----------
asr_url="${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL:-${CLYPT_PHASE1_AUDIO_HOST_URL:-}}"
asr_token="${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN:-${CLYPT_PHASE1_AUDIO_HOST_TOKEN:-}}"
echo "[deploy-visual] probing VibeVoice-ASR host at $asr_url ..."
if ! curl -fsS \
  -H "Authorization: Bearer ${asr_token}" \
  "${asr_url%/}/health" >/dev/null; then
  echo "[deploy-visual] WARN: VibeVoice-ASR host /health did not respond. Workers will still start but will fail on first job." >&2
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
