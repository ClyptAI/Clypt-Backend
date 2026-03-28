#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
BRANCH="${BRANCH:-codex/balanced-hybrid-phase1-contract}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/do-phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1.txt}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-0}"
BUILD_CUDA_DECORD="${BUILD_CUDA_DECORD:-1}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Expected a repo checkout at $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
if [[ "$SKIP_GIT_SYNC" != "1" ]]; then
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Expected a git checkout at $REPO_DIR when SKIP_GIT_SYNC=0" >&2
    exit 1
  fi
  git fetch origin
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Missing requirements file: $REQUIREMENTS_FILE" >&2
  exit 1
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"

if [[ "$BUILD_CUDA_DECORD" == "1" ]]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  CUDACXX_BIN="${CUDACXX_BIN:-$CUDA_HOME/bin/nvcc}"
  if [[ ! -x "$CUDACXX_BIN" ]]; then
    echo "Expected CUDA compiler at $CUDACXX_BIN" >&2
    exit 1
  fi
  export CUDA_HOME
  export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
  export CUDAToolkit_ROOT="$CUDA_HOME"
  export CUDACXX="$CUDACXX_BIN"
  export PATH="$CUDA_HOME/bin:$PATH"
  apt-get update
  apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev
  python -m pip uninstall -y decord || true
  rm -rf /tmp/decord-src
  git clone --recursive https://github.com/dmlc/decord.git /tmp/decord-src
  pushd /tmp/decord-src >/dev/null
  mkdir -p build
  pushd build >/dev/null
  cmake .. \
    -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX_BIN" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
  make -j"$(nproc)"
  popd >/dev/null
  pushd python >/dev/null
  python setup.py install
  popd >/dev/null
  popd >/dev/null
fi

# InsightFace can pull in the CPU-only onnxruntime package transitively. Keep
# only the GPU build so CUDAExecutionProvider is available at runtime.
python -m pip uninstall -y onnxruntime || true
python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2

PYTHON_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PACKAGES="$REPO_DIR/.venv/lib/python${PYTHON_MM}/site-packages"
NVIDIA_LIB_PATHS=(
  "$SITE_PACKAGES/nvidia/cudnn/lib"
  "$SITE_PACKAGES/nvidia/cublas/lib"
  "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
  "$SITE_PACKAGES/nvidia/cufft/lib"
  "$SITE_PACKAGES/nvidia/curand/lib"
  "$SITE_PACKAGES/nvidia/cusolver/lib"
  "$SITE_PACKAGES/nvidia/cusparse/lib"
  "$SITE_PACKAGES/nvidia/nvjitlink/lib"
  "/usr/local/cuda/targets/x86_64-linux/lib"
  "/usr/local/cuda/lib64"
)
LD_LIBRARY_PATH_VALUE="$(IFS=:; echo "${NVIDIA_LIB_PATHS[*]}")"

# Cache the Phase 1 model/artifact bundle that the DO worker requires before
# it can accept jobs.
python - <<'PY'
from backend.do_phase1_worker import (
    download_asr_model,
    download_insightface_model,
    download_lrasd_model,
    download_yolo_model,
)

download_asr_model()
download_yolo_model()
download_lrasd_model()
download_insightface_model()
PY

install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service
install -d /etc/systemd/system/clypt-phase1-worker.service.d
cat >/etc/systemd/system/clypt-phase1-worker.service.d/10-venv-libs.conf <<EOF
[Service]
Environment="LD_LIBRARY_PATH=${LD_LIBRARY_PATH_VALUE}"
EOF

systemctl daemon-reload
systemctl enable clypt-phase1-api.service
systemctl enable clypt-phase1-worker.service
systemctl restart clypt-phase1-api.service
systemctl restart clypt-phase1-worker.service

systemctl --no-pager --full status clypt-phase1-api.service || true
systemctl --no-pager --full status clypt-phase1-worker.service || true
