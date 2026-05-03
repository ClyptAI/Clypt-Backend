#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/preamble.sh
source "$SCRIPT_DIR/../lib/preamble.sh"

REPO_DIR="${REPO_DIR:-/opt/clypt-phase26/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase26/phase26.env}"
PHASE26_VENV_DIR="${PHASE26_VENV_DIR:-/opt/clypt-phase26/venvs/phase26}"
SG_MODEL_ENV_FILE="${SG_MODEL_ENV_FILE:-/etc/clypt-phase26/sg-model.env}"
SG_STAGE_DROPIN_DIR="/etc/systemd/system/clypt-phase26-sglang-qwen.service.d"
SG_STAGE_DROPIN_FILE="$SG_STAGE_DROPIN_DIR/10-launch-profile.conf"

_wait_for_models() {
  local served_model="${SG_SERVED_MODEL_NAME:-$SG_MODEL}"
  local deadline=$((SECONDS + ${SG_MODELS_TIMEOUT_S:-1800}))
  until curl -fsS "$SG_BASE_URL/models" >/tmp/clypt-phase26-models.json; do
    if (( SECONDS >= deadline )); then
      echo "[deploy-phase26-mi300x] ERROR: timed out waiting for $SG_BASE_URL/models" >&2
      journalctl -u clypt-phase26-sglang-qwen.service -n 200 --no-pager >&2 || true
      exit 1
    fi
    sleep 5
  done
  if ! grep -q "$served_model" /tmp/clypt-phase26-models.json; then
    echo "[deploy-phase26-mi300x] ERROR: /v1/models did not include $served_model" >&2
    cat /tmp/clypt-phase26-models.json >&2
    exit 1
  fi
}

_wait_for_url() {
  local url="$1"
  local label="$2"
  local timeout_s="${3:-120}"
  local deadline=$((SECONDS + timeout_s))
  until curl -fsS "$url" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "[deploy-phase26-mi300x] ERROR: timed out waiting for $label at $url" >&2
      exit 1
    fi
    sleep 2
  done
}

_strict_json_smoke() {
  local payload_file
  local response_file
  payload_file="$(mktemp)"
  response_file="$(mktemp)"

  python - <<PY >"$payload_file"
import json

payload = {
    "model": "${SG_SERVED_MODEL_NAME:-$SG_MODEL}",
    "messages": [
        {
            "role": "user",
            "content": "Return strict JSON with ok=true and label='phase26_rocm_smoke'.",
        }
    ],
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 128,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "phase26_rocm_smoke",
            "schema": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "label": {"type": "string"},
                },
                "required": ["ok", "label"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    "chat_template_kwargs": {"enable_thinking": False},
}
print(json.dumps(payload))
PY

  curl -fsS \
    -H "Content-Type: application/json" \
    --data-binary "@$payload_file" \
    "$SG_BASE_URL/chat/completions" >"$response_file"
  python - <<PY "$response_file"
import json
import sys

response = json.load(open(sys.argv[1], encoding="utf-8"))
content = response["choices"][0]["message"]["content"]
parsed = json.loads(content)
if parsed.get("ok") is not True:
    raise SystemExit("strict JSON smoke did not return ok=true")
print("[deploy-phase26-mi300x] strict JSON smoke passed")
PY
  rm -f "$payload_file" "$response_file"
}

_local_openai_client_smoke() {
  python - <<PY
from backend.providers.config import LocalGenerationSettings
from backend.providers.openai_local import LocalOpenAIQwenClient

client = LocalOpenAIQwenClient(
    settings=LocalGenerationSettings(
        base_url="${SG_BASE_URL}",
        model="${SG_SERVED_MODEL_NAME:-$SG_MODEL}",
        timeout_s=600.0,
        max_retries=0,
    )
)
payload = client.generate_json(
    prompt="Return ok=true and stage='local_openai_client'.",
    response_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "stage": {"type": "string"},
        },
        "required": ["ok", "stage"],
        "additionalProperties": False,
    },
    max_output_tokens=128,
)
if payload.get("ok") is not True:
    raise SystemExit("LocalOpenAIQwenClient smoke did not return ok=true")
print("[deploy-phase26-mi300x] LocalOpenAIQwenClient smoke passed")
PY
}

_long_context_smoke() {
  local payload_file
  local response_file
  payload_file="$(mktemp)"
  response_file="$(mktemp)"

  python - <<PY >"$payload_file"
import json

prompt = "Count the marker occurrences and return strict JSON. " + ("marker " * 4096)
payload = {
    "model": "${SG_SERVED_MODEL_NAME:-$SG_MODEL}",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "max_tokens": 128,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "phase26_rocm_long_context",
            "schema": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "marker_count": {"type": "integer"},
                },
                "required": ["ok", "marker_count"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    "chat_template_kwargs": {"enable_thinking": False},
}
print(json.dumps(payload))
PY

  curl -fsS \
    -H "Content-Type: application/json" \
    --data-binary "@$payload_file" \
    "$SG_BASE_URL/chat/completions" >"$response_file"
  python - <<PY "$response_file"
import json
import sys

response = json.load(open(sys.argv[1], encoding="utf-8"))
content = response["choices"][0]["message"]["content"]
parsed = json.loads(content)
if parsed.get("ok") is not True:
    raise SystemExit("long-context smoke did not return ok=true")
print("[deploy-phase26-mi300x] long-context smoke returned", parsed)
PY
  rm -f "$payload_file" "$response_file"
}

_write_sglang_profile_dropin() {
  local profile="$1"
  install -d -m 0755 "$SG_STAGE_DROPIN_DIR"
  cat >"$SG_STAGE_DROPIN_FILE" <<EOF
[Service]
Environment=SG_LAUNCH_PROFILE_OVERRIDE=$profile
EOF
  systemctl daemon-reload
}

_stop_sglang_service() {
  docker rm -f "${SG_CONTAINER_NAME:-clypt-phase26-sglang-qwen}" >/dev/null 2>&1 || true
  systemctl stop clypt-phase26-sglang-qwen.service >/dev/null 2>&1 || true
  systemctl reset-failed clypt-phase26-sglang-qwen.service >/dev/null 2>&1 || true
}

_restart_sglang_for_profile() {
  local profile="$1"
  echo "[deploy-phase26-mi300x] validating SGLang profile: $profile"
  _stop_sglang_service
  _write_sglang_profile_dropin "$profile"
  systemctl start clypt-phase26-sglang-qwen.service
  _wait_for_models
  case "$profile" in
    strict_json|fp8_kv|scheduler_cache|speculative|final)
      _strict_json_smoke
      ;;
  esac
}

wait_for_apt_locks() {
  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi
  local waited_s=0
  local max_wait_s="${APT_LOCK_WAIT_S:-600}"
  local locks=(
    /var/lib/dpkg/lock-frontend
    /var/lib/dpkg/lock
    /var/cache/apt/archives/lock
    /var/lib/apt/lists/lock
  )
  while fuser "${locks[@]}" >/dev/null 2>&1; do
    if (( waited_s >= max_wait_s )); then
      echo "[deploy-phase26-mi300x] ERROR: timed out waiting for apt/dpkg locks." >&2
      return 1
    fi
    echo "[deploy-phase26-mi300x] waiting for apt/dpkg locks..."
    sleep 5
    waited_s=$((waited_s + 5))
  done
}

require_root
exec 9>/var/lock/clypt-phase26-mi300x-deploy.lock
if ! flock -n 9; then
  echo "[deploy-phase26-mi300x] ERROR: another deploy_phase26_mi300x_services.sh run is already active." >&2
  exit 1
fi

require_dir "$REPO_DIR"
require_file "$ENV_FILE"
fail_if_repo_local_env_present "$REPO_DIR" "$ENV_FILE"
cd "$REPO_DIR"
load_env_file "$ENV_FILE"

SG_MODEL="${SG_MODEL:-${GENAI_GENERATION_MODEL:-Qwen/Qwen3.6-35B-A3B}}"
SG_MODEL_REVISION="${SG_MODEL_REVISION:-${SGLANG_MODEL_REVISION:-main}}"
SG_DOCKER_BASE_IMAGE="${SG_DOCKER_BASE_IMAGE:-lmsysorg/sglang:v0.5.10-rocm720-mi30x}"
SG_DOCKER_IMAGE="${SG_DOCKER_IMAGE:-clypt/sglang:v0.5.10-rocm720-mi30x-clypt1}"
SG_ACCEPTANCE_PROFILES="${SG_ACCEPTANCE_PROFILES:-minimal strict_json fp8_kv scheduler_cache speculative}"
HF_HOME="${HF_HOME:-/opt/clypt-phase26/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-/opt/clypt-phase26/.cache/torch}"
PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-/opt/clypt-phase26/.cache/torch/kernels}"
SG_BASE_URL="${SG_BASE_URL:-${CLYPT_LOCAL_LLM_BASE_URL:-http://127.0.0.1:8001/v1}}"

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[deploy-phase26-mi300x] ERROR: GOOGLE_APPLICATION_CREDENTIALS must point to a service-account key." >&2
  exit 1
fi
require_google_service_account_key "$GOOGLE_APPLICATION_CREDENTIALS"

for device_path in /dev/kfd /dev/dri; do
  if [[ ! -e "$device_path" ]]; then
    echo "[deploy-phase26-mi300x] ERROR: missing required ROCm device path: $device_path" >&2
    exit 1
  fi
done

wait_for_apt_locks
apt-get update
wait_for_apt_locks
packages=(
  build-essential ca-certificates curl ffmpeg git jq lsb-release
  python3 python3-pip python3-venv unzip
)
if ! command -v docker >/dev/null 2>&1; then
  packages+=(docker.io)
fi
wait_for_apt_locks
DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"

systemctl enable --now docker

python3 -m venv "$PHASE26_VENV_DIR"
. "$PHASE26_VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-do-phase26-mi300x.txt
python -m pip install -r requirements-phase1-orchestrator.txt

install -d -m 0755 "$HF_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
install -d -m 0755 /opt/clypt-phase26/test-bank-cache/audio /opt/clypt-phase26/test-bank-cache/videos
install -d -m 0755 /var/lib/clypt/phase1 /var/lib/clypt/phase1/work /var/log/clypt/phase1/logs
HF_HOME="$HF_HOME" HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_OFFLINE=0 HOME=/opt/clypt-phase26 \
  SG_MODEL="$SG_MODEL" SG_MODEL_REVISION="$SG_MODEL_REVISION" SG_MODEL_ENV_FILE="$SG_MODEL_ENV_FILE" \
  python - <<'PY'
import os
import shlex
from huggingface_hub import HfApi, snapshot_download

model = os.environ["SG_MODEL"]
requested_revision = os.environ["SG_MODEL_REVISION"]
hf_home = os.environ["HF_HOME"]
model_env_file = os.environ["SG_MODEL_ENV_FILE"]
info = HfApi().model_info(model, revision=requested_revision)
resolved_revision = info.sha
path = snapshot_download(model, revision=resolved_revision)
container_path = path.replace(hf_home, "/root/.cache/huggingface", 1)
if container_path == path:
    raise SystemExit(
        f"[deploy-phase26-mi300x] snapshot path {path!r} is outside HF_HOME={hf_home!r}"
    )
print(f"[deploy-phase26-mi300x] prewarmed {model}@{resolved_revision} at {path}")
with open(model_env_file, "w", encoding="utf-8") as fh:
    fh.write(f"SG_MODEL={shlex.quote(model)}\n")
    fh.write(f"SG_MODEL_PATH={shlex.quote(container_path)}\n")
    fh.write(f"SG_MODEL_REVISION_RESOLVED={shlex.quote(resolved_revision)}\n")
    fh.write(f"SG_SERVED_MODEL_NAME={shlex.quote(model)}\n")
PY

docker pull "$SG_DOCKER_BASE_IMAGE"
SG_DOCKER_BASE_IMAGE="$SG_DOCKER_BASE_IMAGE" SG_DOCKER_IMAGE="$SG_DOCKER_IMAGE" \
  bash scripts/do_phase26/build_sglang_rocm_mi300x_image.sh
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "$SG_DOCKER_IMAGE" python - <<'PY'
import re
import sglang
from sglang.srt.layers.quantization import QUANTIZATION_METHODS

version = getattr(sglang, "__version__", "")
match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
if not match:
    raise SystemExit(f"[deploy-phase26-mi300x] cannot parse sglang.__version__={version!r}")
parsed = tuple(int(part) for part in match.groups())
if parsed < (0, 5, 10):
    raise SystemExit(f"[deploy-phase26-mi300x] SGLang {version} is too old; need >=0.5.10")
if "quark" in QUANTIZATION_METHODS or "quark_int4fp8_moe" in QUANTIZATION_METHODS:
    raise SystemExit("[deploy-phase26-mi300x] Quark quantization is unexpectedly enabled")
print(f"[deploy-phase26-mi300x] SGLang image version accepted: {version}")
PY

# shellcheck disable=SC1090
source "$SG_MODEL_ENV_FILE"

install -d -m 0755 /etc/clypt
install -D -m 0644 scripts/do_phase26/clypt-phase26-runtime.env /etc/clypt/clypt-phase26-runtime.env
install -D -m 0755 scripts/do_phase26/run_sglang_qwen_rocm_container.sh /opt/clypt-phase26/run_sglang_qwen_rocm_container.sh
install -D -m 0644 scripts/do_phase26/systemd/amd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase26/systemd/amd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service
install -D -m 0644 scripts/do_phase26/systemd/amd/clypt-phase26-dispatch.service /etc/systemd/system/clypt-phase26-dispatch.service
install -D -m 0644 scripts/do_phase26/systemd/amd/clypt-phase26-worker.service /etc/systemd/system/clypt-phase26-worker.service
install -D -m 0644 scripts/do_phase26/systemd/amd/clypt-phase26-sglang-qwen.service /etc/systemd/system/clypt-phase26-sglang-qwen.service

systemctl daemon-reload
systemctl enable clypt-phase26-sglang-qwen.service clypt-phase26-dispatch.service clypt-phase26-worker.service clypt-phase1-api.service clypt-phase1-worker.service
systemctl stop clypt-phase1-worker.service clypt-phase1-api.service clypt-phase26-worker.service clypt-phase26-dispatch.service || true
for profile in $SG_ACCEPTANCE_PROFILES; do
  HF_HUB_OFFLINE=1 _restart_sglang_for_profile "$profile"
  _stop_sglang_service
done
HF_HUB_OFFLINE=1 _restart_sglang_for_profile final
_local_openai_client_smoke
_long_context_smoke

systemctl restart clypt-phase26-dispatch.service
_wait_for_url http://127.0.0.1:9300/health "Phase26 dispatch health"
systemctl restart clypt-phase1-api.service
_wait_for_url http://127.0.0.1:8080/health "Phase1 API health"
systemctl restart clypt-phase26-worker.service
systemctl restart clypt-phase1-worker.service

echo "[deploy-phase26-mi300x] done."
