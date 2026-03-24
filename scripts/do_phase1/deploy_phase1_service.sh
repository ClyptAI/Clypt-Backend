#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
BRANCH="${BRANCH:-codex/balanced-hybrid-phase1-contract}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/do-phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1.txt}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-0}"
REQUIRE_TENSORRT="${REQUIRE_TENSORRT:-1}"

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

python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"

if [[ "$REQUIRE_TENSORRT" == "1" ]]; then
  if ! command -v trtexec >/dev/null 2>&1; then
    echo "Missing TensorRT binary: trtexec" >&2
    exit 1
  fi
  python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("tensorrt") is None:
    print("Missing TensorRT Python bindings in deploy environment.", file=sys.stderr)
    raise SystemExit(1)
PY
fi

# Cache the Phase 1 model/artifact bundle that the DO worker requires before
# it can accept jobs. TensorRT export is best effort; the worker falls back to
# PyTorch weights when no engine is present.
python - <<'PY'
from backend.do_phase1_worker import (
    download_asr_model,
    download_insightface_model,
    download_lrasd_model,
    download_yolo_model,
    prepare_yolo_onnx_tensorrt,
)

download_asr_model()
download_yolo_model()
download_lrasd_model()
download_insightface_model()
prepare_yolo_onnx_tensorrt()
PY

if [[ "$REQUIRE_TENSORRT" == "1" ]]; then
  if [[ ! -f /root/.cache/clypt/yolo26s.engine ]]; then
    echo "TensorRT engine build did not produce /root/.cache/clypt/yolo26s.engine" >&2
    exit 1
  fi
fi

install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service

systemctl daemon-reload
systemctl enable clypt-phase1-api.service
systemctl enable clypt-phase1-worker.service
systemctl restart clypt-phase1-api.service
systemctl restart clypt-phase1-worker.service

systemctl --no-pager --full status clypt-phase1-api.service || true
systemctl --no-pager --full status clypt-phase1-worker.service || true
