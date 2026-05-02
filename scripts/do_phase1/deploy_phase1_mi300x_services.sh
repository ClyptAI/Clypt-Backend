#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/preamble.sh
source "$SCRIPT_DIR/../lib/preamble.sh"

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/phase1.env}"
PHASE1_VENV_DIR="${PHASE1_VENV_DIR:-/opt/clypt-phase1/venvs/phase1}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-phase1-vllm-vibevoice-rocm:latest}"
VLLM_ROCM_BASE_IMAGE="${VLLM_ROCM_BASE_IMAGE:-}"
VIBEVOICE_MODEL_ID="${VIBEVOICE_MODEL_ID:-microsoft/VibeVoice-ASR}"
VIBEVOICE_MODEL_REVISION="${VIBEVOICE_MODEL_REVISION:-main}"
VIBEVOICE_MODEL_ENV_FILE="${VIBEVOICE_MODEL_ENV_FILE:-/etc/clypt-phase1/vibevoice-model.env}"

wait_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local delay_s="${4:-5}"
  local i
  for ((i = 1; i <= attempts; i += 1)); do
    if curl -sf "$url" >/dev/null; then
      echo "[deploy-phase1-mi300x] ready: $label"
      return 0
    fi
    sleep "$delay_s"
  done
  echo "[deploy-phase1-mi300x] ERROR: timed out waiting for $label at $url" >&2
  return 1
}

require_root
require_dir "$REPO_DIR"
require_file "$ENV_FILE"
fail_if_repo_local_env_present "$REPO_DIR" "$ENV_FILE"
cd "$REPO_DIR"
load_env_file "$ENV_FILE"

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: GOOGLE_APPLICATION_CREDENTIALS must point to a signing-capable service-account key." >&2
  exit 1
fi
require_google_service_account_key "$GOOGLE_APPLICATION_CREDENTIALS"

for required_path in /dev/kfd /dev/dri; do
  if [[ ! -e "$required_path" ]]; then
    echo "[deploy-phase1-mi300x] ERROR: missing ROCm device path: $required_path" >&2
    exit 1
  fi
done
if ! compgen -G "/dev/dri/renderD*" >/dev/null; then
  echo "[deploy-phase1-mi300x] ERROR: no /dev/dri/renderD* device found for VAAPI." >&2
  exit 1
fi
command -v rocm-smi >/dev/null
command -v amd-smi >/dev/null
rocm-smi >/dev/null
amd-smi static >/dev/null

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl docker.io ffmpeg git gnupg lsb-release \
  libva2 mesa-va-drivers vainfo python3 python3-pip python3-venv unzip
systemctl enable --now docker

python3 -m venv "$PHASE1_VENV_DIR"
. "$PHASE1_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==69.5.1" "Cython<4"
python -m pip install --no-build-isolation "youtokentome>=1.0.5"
python -m pip install -r requirements-do-phase1-mi300x.txt

PHASE1_RUNTIME_HOME="${PHASE1_RUNTIME_HOME:-/opt/clypt-phase1}"
export HOME="$PHASE1_RUNTIME_HOME"
export CLYPT_PHASE1_CACHE_HOME="${CLYPT_PHASE1_CACHE_HOME:-$PHASE1_RUNTIME_HOME/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CLYPT_PHASE1_CACHE_HOME}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export VIBEVOICE_CACHE_DIR="${VIBEVOICE_CACHE_DIR:-$HF_HOME}"

export CLYPT_ACCELERATOR="${CLYPT_ACCELERATOR:-rocm}"
export CLYPT_PHASE1_VISUAL_MODEL="${CLYPT_PHASE1_VISUAL_MODEL:-nano}"
export CLYPT_PHASE1_VISUAL_BACKEND="${CLYPT_PHASE1_VISUAL_BACKEND:-rfdetr_rocm_fp16}"
export CLYPT_PHASE1_VISUAL_BATCH_SIZE="${CLYPT_PHASE1_VISUAL_BATCH_SIZE:-16}"
export CLYPT_PHASE1_VISUAL_THRESHOLD="${CLYPT_PHASE1_VISUAL_THRESHOLD:-0.35}"
export CLYPT_PHASE1_VISUAL_SHAPE="${CLYPT_PHASE1_VISUAL_SHAPE:-640}"
export CLYPT_PHASE1_VISUAL_DECODE="${CLYPT_PHASE1_VISUAL_DECODE:-gpu}"
export CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND="${CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND:-vaapi}"
export CLYPT_PHASE1_NFA_DEVICE="${CLYPT_PHASE1_NFA_DEVICE:-cuda}"
export CLYPT_PHASE1_EMOTION2VEC_DEVICE="${CLYPT_PHASE1_EMOTION2VEC_DEVICE:-cuda}"
export CLYPT_PHASE1_YAMNET_DEVICE="${CLYPT_PHASE1_YAMNET_DEVICE:-cpu}"

install -d -m 0755 \
  "$PHASE1_RUNTIME_HOME" \
  "$CLYPT_PHASE1_CACHE_HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$VIBEVOICE_CACHE_DIR"

if [[ "$CLYPT_ACCELERATOR" != "rocm" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: CLYPT_ACCELERATOR must be rocm for MI300X." >&2
  exit 1
fi
if [[ -z "$VLLM_ROCM_BASE_IMAGE" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: VLLM_ROCM_BASE_IMAGE must be set to the accepted ROCm vLLM image tag." >&2
  echo "[deploy-phase1-mi300x] Official ROCm vLLM images use the vllm/vllm-openai-rocm repository; pin the exact tag accepted by the MI300X canary." >&2
  exit 1
fi
if [[ "$VLLM_ROCM_BASE_IMAGE" == *":latest" || "$VLLM_ROCM_BASE_IMAGE" != *":"* ]]; then
  echo "[deploy-phase1-mi300x] ERROR: VLLM_ROCM_BASE_IMAGE must be an immutable accepted tag, not latest or an untagged image." >&2
  exit 1
fi
if [[ "$CLYPT_PHASE1_NFA_DEVICE" != "cuda" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: CLYPT_PHASE1_NFA_DEVICE must stay cuda on ROCm PyTorch; no CPU downgrade." >&2
  exit 1
fi
if [[ "$CLYPT_PHASE1_EMOTION2VEC_DEVICE" != "cuda" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: CLYPT_PHASE1_EMOTION2VEC_DEVICE must stay cuda on ROCm PyTorch; no CPU downgrade." >&2
  exit 1
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[deploy-phase1-mi300x] PyTorch ROCm GPU unavailable")
if not getattr(torch.version, "hip", None):
    raise SystemExit("[deploy-phase1-mi300x] torch.version.hip is empty; this is not a ROCm wheel")
device_name = torch.cuda.get_device_name(0)
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f"[deploy-phase1-mi300x] PyTorch ROCm ready: hip={torch.version.hip} gpu={device_name} sum={float(y.sum().item())}")
PY

python - <<'PY'
from backend.phase1_runtime.frame_decode import (
    discover_vaapi_render_node,
    validate_ffmpeg_vaapi_support,
    validate_vaapi_render_node,
)
from backend.phase1_runtime.visual_config import VisualPipelineConfig

config = VisualPipelineConfig.from_env()
if config.detector_backend != "rfdetr_rocm_fp16":
    raise SystemExit(
        "[deploy-phase1-mi300x] visual backend must be rfdetr_rocm_fp16, "
        f"got {config.detector_backend!r}"
    )
render_node = discover_vaapi_render_node()
validate_vaapi_render_node(render_node)
validate_ffmpeg_vaapi_support()
print(f"[deploy-phase1-mi300x] VAAPI ready on {render_node}")
PY

python - <<'PY'
import os
import shlex
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

model = os.environ["VIBEVOICE_MODEL_ID"]
requested_revision = os.environ["VIBEVOICE_MODEL_REVISION"]
info = HfApi().model_info(model, revision=requested_revision)
resolved_revision = info.sha
path = snapshot_download(
    repo_id=model,
    revision=resolved_revision,
    cache_dir=os.environ["VIBEVOICE_CACHE_DIR"],
)
container_path = path.replace(os.environ["VIBEVOICE_CACHE_DIR"], "/root/.cache/huggingface", 1)
if container_path == path:
    raise SystemExit(
        f"[deploy-phase1-mi300x] snapshot path {path!r} is outside VIBEVOICE_CACHE_DIR"
    )
with open(os.environ["VIBEVOICE_MODEL_ENV_FILE"], "w", encoding="utf-8") as fh:
    fh.write(f"VIBEVOICE_MODEL_ID={shlex.quote(model)}\n")
    fh.write(f"VIBEVOICE_MODEL_PATH={shlex.quote(container_path)}\n")
    fh.write(f"VIBEVOICE_MODEL_REVISION_RESOLVED={shlex.quote(resolved_revision)}\n")
print(f"[deploy-phase1-mi300x] prewarmed VibeVoice HF snapshot {model}@{resolved_revision} at {path}")
PY

python - <<'PY'
from backend.phase1_runtime.rfdetr_detector import RFDETRPersonDetector
from backend.phase1_runtime.visual_config import VisualPipelineConfig

detector = RFDETRPersonDetector(VisualPipelineConfig.from_env())
detector.load()
detector.unload()
print("[deploy-phase1-mi300x] prewarmed RF-DETR ROCm detector")
PY

python - <<'PY'
import os

from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.forced_aligner import ForcedAlignmentProvider
from backend.providers.yamnet import YAMNetProvider

aligner = ForcedAlignmentProvider(
    device=os.environ.get("CLYPT_PHASE1_NFA_DEVICE", "cuda"),
)
if not aligner._check_available():
    raise SystemExit("[deploy-phase1-mi300x] forced aligner deps are unavailable after install")
aligner._ensure_model(aligner._resolve_device())

emotion_provider = Emotion2VecPlusProvider(
    device=os.environ.get("CLYPT_PHASE1_EMOTION2VEC_DEVICE", "cuda"),
)
emotion_provider._ensure_model()

yamnet_provider = YAMNetProvider(
    device=os.environ.get("CLYPT_PHASE1_YAMNET_DEVICE", "cpu"),
)
yamnet_provider._ensure_runner()

print("[deploy-phase1-mi300x] prewarmed NFA, emotion2vec+, and YAMNet")
PY

NFA_MODEL_PATH="$TORCH_HOME/NeMo/NeMo_1.23.0/stt_en_fastconformer_hybrid_large_pc/465b32000fc320f5905fda11a1866ef6/stt_en_fastconformer_hybrid_large_pc.nemo"
if [[ ! -f "$NFA_MODEL_PATH" ]]; then
  echo "[deploy-phase1-mi300x] ERROR: expected NFA model cache missing at $NFA_MODEL_PATH" >&2
  exit 1
fi

install -d -m 0755 /etc/clypt
install -D -m 0644 scripts/do_phase1/clypt-phase1-runtime.env /etc/clypt/clypt-phase1-runtime.env
install -D -m 0755 scripts/do_phase1/run_vllm_vibevoice_rocm_container.sh /usr/local/bin/clypt-phase1-run-vllm-vibevoice-rocm

docker build \
  --build-arg VLLM_ROCM_BASE_IMAGE="$VLLM_ROCM_BASE_IMAGE" \
  -t "$VLLM_IMAGE_TAG" \
  docker/vibevoice-vllm-rocm/

install -D -m 0644 scripts/do_phase1/systemd/amd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/amd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service
install -D -m 0644 scripts/do_phase1/systemd/amd/clypt-phase1-vibevoice.service /etc/systemd/system/clypt-phase1-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/amd/clypt-phase1-vllm-vibevoice.service /etc/systemd/system/clypt-phase1-vllm-vibevoice.service
install -D -m 0644 scripts/do_phase1/systemd/amd/clypt-phase1-visual.service /etc/systemd/system/clypt-phase1-visual.service

systemctl daemon-reload
systemctl enable \
  clypt-phase1-vllm-vibevoice.service \
  clypt-phase1-vibevoice.service \
  clypt-phase1-visual.service \
  clypt-phase1-api.service \
  clypt-phase1-worker.service

systemctl restart clypt-phase1-vllm-vibevoice.service
wait_url "http://127.0.0.1:8000/v1/models" "VibeVoice vLLM sidecar /v1/models" 120 5
curl -sf http://127.0.0.1:8000/v1/models | grep -q '"id":"vibevoice"\|"id": "vibevoice"'

systemctl restart clypt-phase1-vibevoice.service
systemctl restart clypt-phase1-visual.service
systemctl restart clypt-phase1-api.service
systemctl restart clypt-phase1-worker.service

wait_url "http://127.0.0.1:9100/health" "VibeVoice service" 60 3
wait_url "http://127.0.0.1:9200/health" "visual service" 60 3
wait_url "http://127.0.0.1:8080/healthz" "Phase1 API" 60 3

echo "[deploy-phase1-mi300x] done."
