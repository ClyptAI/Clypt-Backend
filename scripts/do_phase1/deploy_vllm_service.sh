#!/usr/bin/env bash
# Full vLLM-path deployment script for the GPU droplet.
#
# This is the single script to run on a fresh droplet (or after an rsync)
# when deploying with VIBEVOICE_BACKEND=vllm.
#
# What it does:
#   1. Install host prerequisites (ffmpeg/git/python/curl/unzip)
#   2. Install Docker if not present (needed for the vLLM container)
#   3. Install/configure nvidia-container-toolkit for --gpus all
#   4. Validate env-file syntax and required vars
#   5. pip-install requirements in the main worker venv (with known gotcha handling)
#   6. Validate torchaudio (used by NFA + emotion2vec+ in the main venv)
#   7. Prewarm NFA + emotion2vec+ models in persistent host cache
#   8. Install/refresh the API and worker systemd units
#   9. Clone/update the VibeVoice repo mount for the container
#   10. Build the vLLM Docker image
#   11. Install the vLLM systemd unit and start the container
#   12. Wait for the health endpoint to respond
#   13. Restart the Phase 1 API + worker so they pick up new code
#
# What it does NOT do (intentional):
#   - install any legacy/native/non-vLLM ASR environment
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
#   VLLM_READY_TIMEOUT_S — seconds to wait for healthy (default: 2400)
#   HF_CACHE_DIR         — HuggingFace model cache dir (default: /opt/clypt-phase1/hf-cache)
#   VIBEVOICE_REPO_DIR   — host path mounted to /app in the vLLM container
#   VIBEVOICE_REPO_URL   — VibeVoice git source URL
#   VIBEVOICE_REPO_REF   — branch/tag/commit to deploy from VIBEVOICE_REPO_URL
#   PIP_FALLBACK_LEGACY_RESOLVER — set 0 to disable pip legacy-resolver fallback
#   PREWARM_PHASE1_MODELS — set 0 to skip eager NFA/emotion2vec+ downloads
#   PHASE1_CACHE_HOME     — host cache root used by worker/prewarm (default: /opt/clypt-phase1/.cache)
#   PREWARM_RETRIES       — retry count for model prewarm step (default: 3)
#   PREWARM_RETRY_BACKOFF_S — sleep between failed prewarm attempts (default: 20)
#   PREWARM_TIMEOUT_S     — timeout per prewarm attempt in seconds (default: 1800)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1.txt}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-vllm-vibevoice:latest}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:${VLLM_HOST_PORT}/health}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-2400}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/clypt-phase1/hf-cache}"
VIBEVOICE_REPO_DIR="${VIBEVOICE_REPO_DIR:-/opt/clypt-phase1/vibevoice-repo}"
VIBEVOICE_REPO_URL="${VIBEVOICE_REPO_URL:-https://github.com/microsoft/VibeVoice.git}"
VIBEVOICE_REPO_REF="${VIBEVOICE_REPO_REF:-main}"
PIP_FALLBACK_LEGACY_RESOLVER="${PIP_FALLBACK_LEGACY_RESOLVER:-1}"
PREWARM_PHASE1_MODELS="${PREWARM_PHASE1_MODELS:-1}"
PHASE1_CACHE_HOME="${PHASE1_CACHE_HOME:-}"
PREWARM_RETRIES="${PREWARM_RETRIES:-3}"
PREWARM_RETRY_BACKOFF_S="${PREWARM_RETRY_BACKOFF_S:-20}"
PREWARM_TIMEOUT_S="${PREWARM_TIMEOUT_S:-1800}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-vllm] ERROR: run this script as root." >&2
  exit 1
fi

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

# --- 1. Host prerequisites -------------------------------------------------
echo "[deploy-vllm] ensuring host prerequisites are installed ..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  gnupg \
  lsb-release \
  python3 \
  python3-pip \
  python3-venv \
  unzip

# --- 2. Install Docker if not present -------------------------------------
if ! command -v docker &>/dev/null; then
  echo "[deploy-vllm] Docker not found — installing ..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
  echo "[deploy-vllm] Docker installed: $(docker --version)"
else
  echo "[deploy-vllm] Docker already present: $(docker --version)"
fi

# --- 3. Install/configure nvidia-container-toolkit ------------------------
if ! command -v nvidia-ctk &>/dev/null; then
  echo "[deploy-vllm] nvidia-ctk not found — installing nvidia-container-toolkit ..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
fi
echo "[deploy-vllm] configuring docker nvidia runtime ..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# --- 4. Validate env file before deployment --------------------------------
python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
text = env_path.read_text(encoding="utf-8")
had_trailing_newline = text.endswith("\n")
lines = text.splitlines()
changed = False
updated_lines = []

for line in lines:
    if line.startswith("VIBEVOICE_HOTWORDS_CONTEXT="):
        key, value = line.split("=", 1)
        stripped = value.strip()
        quoted = (
            (stripped.startswith('"') and stripped.endswith('"'))
            or (stripped.startswith("'") and stripped.endswith("'"))
        )
        # Shell `source` breaks on unquoted spaces (e.g. comma+space hotwords list).
        if stripped and (" " in stripped) and not quoted:
            escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
            line = f'{key}="{escaped}"'
            changed = True
    updated_lines.append(line)

if changed:
    env_path.write_text(
        "\n".join(updated_lines) + ("\n" if had_trailing_newline else ""),
        encoding="utf-8",
    )
    print("[deploy-vllm] normalized VIBEVOICE_HOTWORDS_CONTEXT quoting in env file.")
PY

if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a" >/dev/null 2>&1; then
  echo "[deploy-vllm] ERROR: env file is not shell-sourceable: $ENV_FILE" >&2
  echo "[deploy-vllm] Check for unquoted spaces or malformed assignments." >&2
  exit 1
fi

for required in VIBEVOICE_BACKEND VIBEVOICE_VLLM_BASE_URL GOOGLE_CLOUD_PROJECT GCS_BUCKET; do
  if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ -n \${$required:-} ]]" >/dev/null 2>&1; then
    echo "[deploy-vllm] ERROR: required env var missing/empty in $ENV_FILE: $required" >&2
    exit 1
  fi
done
if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ \${VIBEVOICE_BACKEND:-} == vllm ]]" >/dev/null 2>&1; then
  echo "[deploy-vllm] ERROR: VIBEVOICE_BACKEND must be set to 'vllm' in $ENV_FILE for this deploy path." >&2
  exit 1
fi
if ! /usr/bin/env -i bash -c "set -a; source '$ENV_FILE'; set +a; [[ \${VIBEVOICE_VLLM_MODEL:-vibevoice} == vibevoice ]]" >/dev/null 2>&1; then
  echo "[deploy-vllm] ERROR: VIBEVOICE_VLLM_MODEL must be 'vibevoice' (served-model-name)." >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "$PHASE1_CACHE_HOME" ]]; then
  PHASE1_CACHE_HOME="${CLYPT_PHASE1_CACHE_HOME:-/opt/clypt-phase1/.cache}"
fi

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" && ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  echo "[deploy-vllm] ERROR: GOOGLE_APPLICATION_CREDENTIALS points to a missing file: ${GOOGLE_APPLICATION_CREDENTIALS}" >&2
  exit 1
fi

for legacy_var in \
  VIBEVOICE_OUTPUT_MODE \
  VIBEVOICE_WORD_TURN_GAP_MS \
  VIBEVOICE_WORD_TIME_TOKEN_MODE \
  VIBEVOICE_WORD_CHUNK_SECONDS \
  VIBEVOICE_WORD_STREAMING_SEGMENT_DURATION_S; do
  if [[ -n "${!legacy_var:-}" ]]; then
    echo "[deploy-vllm] WARN: $legacy_var is set but unused by the main vLLM pipeline."
  fi
done

# --- 5. Main worker venv — pip install (no native venv, no flash-attn) ----
echo "[deploy-vllm] installing main worker requirements ($REQUIREMENTS_FILE) ..."
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1" "Cython<3.1"
# youtokentome can fail to build in isolated mode without Cython in the build env.
python -m pip install --no-build-isolation youtokentome
if ! python -m pip install -r "$REQUIREMENTS_FILE"; then
  if [[ "$PIP_FALLBACK_LEGACY_RESOLVER" != "1" ]]; then
    echo "[deploy-vllm] ERROR: pip install failed and legacy-resolver fallback is disabled." >&2
    exit 1
  fi
  echo "[deploy-vllm] standard pip resolver failed; retrying with legacy-resolver ..."
  PIP_USE_DEPRECATED=legacy-resolver python -m pip install -r "$REQUIREMENTS_FILE"
fi
# Keep tensorflow-hub compatible on fresh images that bootstrap newer setuptools.
python -m pip install "setuptools==69.5.1"

# --- 6. Validate torchaudio (used by NFA and emotion2vec+ in main venv) ---
python - <<'PY'
from backend.providers.audio_runtime import validate_torchaudio_runtime
meta = validate_torchaudio_runtime()
print(f"[deploy-vllm] main venv torchaudio OK: {meta['torchaudio_version']}")
PY

# --- 7. Prewarm NFA + emotion2vec+ model caches ----------------------------
# Stop API/worker first so they do not contend on cache locks while prewarm
# is trying to populate shared paths.
echo "[deploy-vllm] stopping Phase 1 API/worker before prewarm ..."
systemctl stop clypt-v31-phase1-worker.service clypt-v31-phase1-api.service || true

install -d -m 0755 "$PHASE1_CACHE_HOME"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch"
install -d -m 0755 "$PHASE1_CACHE_HOME/torch/kernels"
install -d -m 0755 "$PHASE1_CACHE_HOME/huggingface"
if [[ "$PREWARM_PHASE1_MODELS" == "1" ]]; then
  echo "[deploy-vllm] prewarming Phase 1 models (NFA + emotion2vec+) ..."
  export HOME="/opt/clypt-phase1"
  export CLYPT_PHASE1_CACHE_HOME="$PHASE1_CACHE_HOME"
  export XDG_CACHE_HOME="$PHASE1_CACHE_HOME"
  export TORCH_HOME="$PHASE1_CACHE_HOME/torch"
  export HF_HOME="$PHASE1_CACHE_HOME/huggingface"
  export FUNASR_MODEL_SOURCE="${FUNASR_MODEL_SOURCE:-hf}"
  prewarm_attempt=1
  while true; do
    echo "[deploy-vllm] prewarm attempt ${prewarm_attempt}/${PREWARM_RETRIES} ..."
    if timeout "${PREWARM_TIMEOUT_S}s" python - <<'PY'
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.forced_aligner import ForcedAlignmentProvider

aligner = ForcedAlignmentProvider()
if aligner._check_available():  # noqa: SLF001
    aligner._ensure_model(device="cpu")  # noqa: SLF001
    print("[deploy-vllm] NFA prewarm complete")
else:
    print("[deploy-vllm] NFA prewarm skipped (forced aligner unavailable in this env)")

emotion = Emotion2VecPlusProvider()
emotion._ensure_model()  # noqa: SLF001
print("[deploy-vllm] Phase 1 model prewarm complete ✓")
PY
    then
      break
    fi
    if [[ $? -eq 124 ]]; then
      echo "[deploy-vllm] prewarm attempt timed out after ${PREWARM_TIMEOUT_S}s (possible stale lock)." >&2
    fi
    if (( prewarm_attempt >= PREWARM_RETRIES )); then
      echo "[deploy-vllm] ERROR: Phase 1 model prewarm failed after ${PREWARM_RETRIES} attempts." >&2
      exit 1
    fi
    echo "[deploy-vllm] prewarm attempt ${prewarm_attempt} failed; retrying in ${PREWARM_RETRY_BACKOFF_S}s ..."
    sleep "$PREWARM_RETRY_BACKOFF_S"
    prewarm_attempt=$((prewarm_attempt + 1))
  done
else
  echo "[deploy-vllm] skipping Phase 1 model prewarm (PREWARM_PHASE1_MODELS=$PREWARM_PHASE1_MODELS)"
fi

# --- 8. Install / refresh API and worker systemd units --------------------
install -d -m 0755 /var/lib/clypt/v3_1_phase1_service
install -d -m 0755 /var/log/clypt/v3_1_phase1
install -d -m 0755 "$PHASE1_CACHE_HOME/torch/kernels"

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-v31-phase1-api.service \
  /etc/systemd/system/clypt-v31-phase1-api.service

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-v31-phase1-worker.service \
  /etc/systemd/system/clypt-v31-phase1-worker.service

systemctl daemon-reload
systemctl enable clypt-v31-phase1-api.service clypt-v31-phase1-worker.service

# --- 9. Clone/update VibeVoice repo for container mount --------------------
if [[ -d "$VIBEVOICE_REPO_DIR/.git" ]]; then
  echo "[deploy-vllm] updating VibeVoice repo at $VIBEVOICE_REPO_DIR ..."
  git -C "$VIBEVOICE_REPO_DIR" fetch --depth 1 origin "$VIBEVOICE_REPO_REF"
  git -C "$VIBEVOICE_REPO_DIR" checkout --force FETCH_HEAD
elif [[ -d "$VIBEVOICE_REPO_DIR" ]]; then
  echo "[deploy-vllm] ERROR: $VIBEVOICE_REPO_DIR exists but is not a git repo." >&2
  echo "[deploy-vllm] Move it aside or remove it, then re-run." >&2
  exit 1
else
  echo "[deploy-vllm] cloning VibeVoice repo to $VIBEVOICE_REPO_DIR ..."
  git clone --depth 1 --branch "$VIBEVOICE_REPO_REF" "$VIBEVOICE_REPO_URL" "$VIBEVOICE_REPO_DIR"
fi

if [[ ! -f "$VIBEVOICE_REPO_DIR/vllm_plugin/scripts/start_server.py" ]]; then
  echo "[deploy-vllm] ERROR: missing start_server.py in $VIBEVOICE_REPO_DIR" >&2
  exit 1
fi

# --- 10. Build vLLM Docker image -------------------------------------------
echo "[deploy-vllm] building Docker image: $VLLM_IMAGE_TAG ..."
docker build -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/
echo "[deploy-vllm] image built: $VLLM_IMAGE_TAG"

# --- 11. HF cache dir + vLLM systemd unit ----------------------------------
install -d -m 0755 "$HF_CACHE_DIR"

install -D -m 0644 \
  scripts/do_phase1/systemd/clypt-vllm-vibevoice.service \
  /etc/systemd/system/clypt-vllm-vibevoice.service

systemctl daemon-reload
systemctl enable clypt-vllm-vibevoice.service

echo "[deploy-vllm] starting clypt-vllm-vibevoice.service ..."
systemctl restart clypt-vllm-vibevoice.service

# --- 12. Wait for vLLM health ----------------------------------------------
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
    logs="$(docker logs --tail 200 clypt-vllm-vibevoice 2>&1 || true)"
    printf '%s\n' "$logs" >&2
    if grep -q "can't open file '/app/vllm_plugin/scripts/start_server.py'" <<<"$logs"; then
      echo "[deploy-vllm] hint: verify $VIBEVOICE_REPO_DIR is present and mounted to /app." >&2
    fi
    if grep -q "unknown runtime" <<<"$logs"; then
      echo "[deploy-vllm] hint: nvidia Docker runtime is not configured correctly." >&2
    fi
    exit 1
  fi
  sleep 5
done

echo "[deploy-vllm] verifying served model id ..."
models_json="$(curl -fsS "http://127.0.0.1:${VLLM_HOST_PORT}/v1/models" || true)"
if ! grep -Eq '"id"[[:space:]]*:[[:space:]]*"vibevoice"' <<<"$models_json"; then
  echo "[deploy-vllm] ERROR: /v1/models did not report model id 'vibevoice'." >&2
  printf '%s\n' "$models_json" >&2
  exit 1
fi

# --- 13. Restart Phase 1 API + worker with new code ------------------------
echo "[deploy-vllm] restarting Phase 1 API and worker ..."
systemctl restart clypt-v31-phase1-api.service
systemctl restart clypt-v31-phase1-worker.service
systemctl is-active --quiet clypt-v31-phase1-api.service
systemctl is-active --quiet clypt-v31-phase1-worker.service

systemctl --no-pager --full status clypt-vllm-vibevoice.service || true
systemctl --no-pager --full status clypt-v31-phase1-worker.service || true
echo ""
echo "[deploy-vllm] deployment complete."
echo "[deploy-vllm] Ensure /etc/clypt-phase1/v3_1_phase1.env contains:"
echo "               VIBEVOICE_BACKEND=vllm"
echo "               VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:${VLLM_HOST_PORT}"
