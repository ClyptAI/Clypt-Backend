#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/preamble.sh
source "$SCRIPT_DIR/../lib/preamble.sh"

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/phase1.env}"
PHASE1_VENV_DIR="${PHASE1_VENV_DIR:-/opt/clypt-phase1/venvs/phase1}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-phase1-vllm-vibevoice:latest}"

require_root
require_dir "$REPO_DIR"
require_file "$ENV_FILE"
fail_if_repo_local_env_present "$REPO_DIR" "$ENV_FILE"
cd "$REPO_DIR"
load_env_file "$ENV_FILE"

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[deploy-phase1] ERROR: GOOGLE_APPLICATION_CREDENTIALS must point to a signing-capable service-account key." >&2
  exit 1
fi
require_google_service_account_key "$GOOGLE_APPLICATION_CREDENTIALS"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl docker.io ffmpeg git gnupg lsb-release python3 python3-pip python3-venv unzip

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

python3 -m venv "$PHASE1_VENV_DIR"
. "$PHASE1_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1" "Cython<4"
python -m pip install --no-build-isolation "youtokentome>=1.0.5"
python -m pip install -r requirements-do-phase1-h200.txt

PHASE1_RUNTIME_HOME="${PHASE1_RUNTIME_HOME:-/opt/clypt-phase1}"
export HOME="$PHASE1_RUNTIME_HOME"
export CLYPT_PHASE1_CACHE_HOME="${CLYPT_PHASE1_CACHE_HOME:-$PHASE1_RUNTIME_HOME/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CLYPT_PHASE1_CACHE_HOME}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"

install -d -m 0755 \
  "$PHASE1_RUNTIME_HOME" \
  "$CLYPT_PHASE1_CACHE_HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  /opt/clypt-phase1/hf-cache

if [[ "${CLYPT_PHASE1_VISUAL_BACKEND:-}" == tensorrt* ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y libnvinfer-bin
  python -m pip install "tensorrt-cu13"
  command -v trtexec >/dev/null
  python - <<'PY'
import tensorrt

print(f"[deploy-phase1] TensorRT Python runtime ready: {tensorrt.__version__}")
PY
fi

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/VibeVoice-ASR",
    cache_dir="/opt/clypt-phase1/hf-cache",
)
PY

python - <<'PY'
import os

from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.forced_aligner import ForcedAlignmentProvider
from backend.providers.yamnet import YAMNetProvider

aligner = ForcedAlignmentProvider()
if not aligner._check_available():
    raise SystemExit("[deploy-phase1] forced aligner deps are unavailable after install")
aligner._ensure_model(aligner._resolve_device())

emotion_provider = Emotion2VecPlusProvider()
emotion_provider._ensure_model()

yamnet_provider = YAMNetProvider(
    device=os.environ.get("CLYPT_PHASE1_YAMNET_DEVICE", "cpu"),
)
yamnet_provider._ensure_runner()

print("[deploy-phase1] prewarmed NFA, emotion2vec+, and YAMNet")
PY

NFA_MODEL_PATH="$TORCH_HOME/NeMo/NeMo_1.23.0/stt_en_fastconformer_hybrid_large_pc/465b32000fc320f5905fda11a1866ef6/stt_en_fastconformer_hybrid_large_pc.nemo"
if [[ ! -f "$NFA_MODEL_PATH" ]]; then
  echo "[deploy-phase1] ERROR: expected NFA model cache missing at $NFA_MODEL_PATH" >&2
  exit 1
fi

install -d -m 0755 /etc/clypt
install -D -m 0644 scripts/do_phase1/clypt-phase1-runtime.env /etc/clypt/clypt-phase1-runtime.env
install -D -m 0755 scripts/do_phase1/run_vllm_vibevoice_container.sh /usr/local/bin/clypt-phase1-run-vllm-vibevoice

docker build -t "$VLLM_IMAGE_TAG" docker/vibevoice-vllm/

install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-vibevoice.service /etc/systemd/system/clypt-phase1-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-vllm-vibevoice.service /etc/systemd/system/clypt-phase1-vllm-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-visual.service /etc/systemd/system/clypt-phase1-visual.service

systemctl daemon-reload
systemctl enable \
  clypt-phase1-vllm-vibevoice.service \
  clypt-phase1-vibevoice.service \
  clypt-phase1-visual.service \
  clypt-phase1-api.service \
  clypt-phase1-worker.service

systemctl restart clypt-phase1-vllm-vibevoice.service
systemctl restart clypt-phase1-vibevoice.service
systemctl restart clypt-phase1-visual.service
systemctl restart clypt-phase1-api.service
systemctl restart clypt-phase1-worker.service

echo "[deploy-phase1] done."
