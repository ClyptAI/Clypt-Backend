#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
NATIVE_VENV_DIR="${NATIVE_VENV_DIR:-/opt/clypt-phase1/.venv-vibevoice-native}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Missing repo dir: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -d "$CUDA_HOME" ]]; then
  echo "Missing CUDA toolkit path: $CUDA_HOME" >&2
  echo "Install cuda-toolkit-12-4 before building the native VibeVoice env." >&2
  exit 1
fi

python3 -m venv "$NATIVE_VENV_DIR"
. "$NATIVE_VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install ninja packaging
python -m pip install -r "$REPO_DIR/requirements-vibevoice-native.txt"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE

python -m pip install \
  --no-build-isolation \
  --no-binary flash-attn \
  --no-cache-dir \
  "flash-attn==${FLASH_ATTN_VERSION}"

python - <<'PY'
import flash_attn
import flash_attn_2_cuda
import torch

print("torch", torch.__version__)
print("flash_attn", flash_attn.__version__)
print("flash_attn_2_cuda", flash_attn_2_cuda)
PY
