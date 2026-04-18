#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/preamble.sh
source "$SCRIPT_DIR/../lib/preamble.sh"

REPO_DIR="${REPO_DIR:-/opt/clypt-phase26/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase26/phase26.env}"
PHASE26_VENV_DIR="${PHASE26_VENV_DIR:-/opt/clypt-phase26/venvs/phase26}"
SGLANG_VENV_DIR="${SGLANG_VENV_DIR:-/opt/clypt-phase26/venvs/sglang}"
SGLANG_MODEL="${SGLANG_MODEL:-${GENAI_GENERATION_MODEL:-Qwen/Qwen3.6-35B-A3B}}"
SGLANG_HF_HOME="${SGLANG_HF_HOME:-${HF_HOME:-/opt/clypt-phase26/hf-cache}}"

require_root
require_dir "$REPO_DIR"
require_file "$ENV_FILE"
fail_if_repo_local_env_present "$REPO_DIR" "$ENV_FILE"
cd "$REPO_DIR"
load_env_file "$ENV_FILE"

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[deploy-phase26] ERROR: GOOGLE_APPLICATION_CREDENTIALS must point to a service-account key." >&2
  exit 1
fi
require_google_service_account_key "$GOOGLE_APPLICATION_CREDENTIALS"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl ffmpeg git gnupg lsb-release python3 python3-pip python3-venv unzip

python3 -m venv "$PHASE26_VENV_DIR"
. "$PHASE26_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-do-phase26-h200.txt

python3 -m venv "$SGLANG_VENV_DIR"
"$SGLANG_VENV_DIR/bin/pip" install --upgrade pip wheel
"$SGLANG_VENV_DIR/bin/pip" install "${SG_PACKAGE_SPEC:-sglang[all]}"

install -d -m 0755 "$SGLANG_HF_HOME"
HF_HOME="$SGLANG_HF_HOME" HF_HUB_ENABLE_HF_TRANSFER=1 HOME=/opt/clypt-phase26 \
  "$SGLANG_VENV_DIR/bin/python" - <<PY
from huggingface_hub import snapshot_download

path = snapshot_download("${SGLANG_MODEL}")
print(f"[deploy-phase26] prewarmed model cache at {path}")
PY

install -d -m 0755 /etc/clypt
install -D -m 0644 scripts/do_phase26/clypt-phase26-runtime.env /etc/clypt/clypt-phase26-runtime.env
install -D -m 0644 scripts/do_phase26/systemd/clypt-phase26-dispatch.service /etc/systemd/system/clypt-phase26-dispatch.service
install -D -m 0644 scripts/do_phase26/systemd/clypt-phase26-worker.service /etc/systemd/system/clypt-phase26-worker.service
install -D -m 0644 scripts/do_phase26/systemd/clypt-phase26-sglang-qwen.service /etc/systemd/system/clypt-phase26-sglang-qwen.service

systemctl daemon-reload
systemctl enable clypt-phase26-sglang-qwen.service clypt-phase26-dispatch.service clypt-phase26-worker.service
systemctl restart clypt-phase26-sglang-qwen.service
systemctl restart clypt-phase26-dispatch.service
systemctl restart clypt-phase26-worker.service

echo "[deploy-phase26] done."
