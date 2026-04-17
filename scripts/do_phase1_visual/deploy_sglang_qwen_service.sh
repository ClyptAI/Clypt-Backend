#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
SGLANG_VENV_DIR="${SGLANG_VENV_DIR:-/opt/clypt-phase1/venvs/sglang}"
SG_PACKAGE_SPEC="${SG_PACKAGE_SPEC:-sglang[all]==0.5.10}"
SG_BASE_URL="${SG_BASE_URL:-http://127.0.0.1:8001}"
SG_PORT="${SG_PORT:-8001}"
SG_MODEL="${SG_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SG_GRAMMAR_BACKEND="${SG_GRAMMAR_BACKEND:-xgrammar}"
SG_SCHEDULE_POLICY="${SG_SCHEDULE_POLICY:-lpm}"
SG_CHUNKED_PREFILL_SIZE="${SG_CHUNKED_PREFILL_SIZE:-8192}"
SG_MEM_FRACTION_STATIC="${SG_MEM_FRACTION_STATIC:-0.78}"
SG_CONTEXT_LENGTH="${SG_CONTEXT_LENGTH:-131072}"
SG_KV_CACHE_DTYPE="${SG_KV_CACHE_DTYPE:-fp8_e4m3}"
SG_ENABLE_RADIX_CACHE="${SG_ENABLE_RADIX_CACHE:-1}"
SG_SPECULATIVE_MODE="${SG_SPECULATIVE_MODE:-nextn}"
SG_SPECULATIVE_NUM_STEPS="${SG_SPECULATIVE_NUM_STEPS:-3}"
SG_SPECULATIVE_TOPK="${SG_SPECULATIVE_TOPK:-1}"
SG_SPECULATIVE_DRAFT_TOKENS="${SG_SPECULATIVE_DRAFT_TOKENS:-4}"
SG_EXTRA_ARGS="${SG_EXTRA_ARGS:-}"
SG_READY_TIMEOUT_S="${SG_READY_TIMEOUT_S:-1800}"
SG_SYSTEMD_UNIT="${SG_SYSTEMD_UNIT:-clypt-sglang-qwen.service}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[deploy-sglang-qwen] ERROR: run this script as root." >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[deploy-sglang-qwen] ERROR: repo dir not found: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[deploy-sglang-qwen] ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

echo "[deploy-sglang-qwen] loading runtime env from ${ENV_FILE} ..."
set -a
. "$ENV_FILE"
set +a

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
SGLANG_VENV_DIR="${SGLANG_VENV_DIR:-/opt/clypt-phase1/venvs/sglang}"
SG_PACKAGE_SPEC="${SG_PACKAGE_SPEC:-sglang[all]==0.5.10}"
SG_BASE_URL="${SG_BASE_URL:-http://127.0.0.1:8001}"
SG_PORT="${SG_PORT:-8001}"
SG_MODEL="${SG_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SG_GRAMMAR_BACKEND="${SG_GRAMMAR_BACKEND:-xgrammar}"
SG_SCHEDULE_POLICY="${SG_SCHEDULE_POLICY:-lpm}"
SG_CHUNKED_PREFILL_SIZE="${SG_CHUNKED_PREFILL_SIZE:-8192}"
SG_MEM_FRACTION_STATIC="${SG_MEM_FRACTION_STATIC:-0.78}"
SG_CONTEXT_LENGTH="${SG_CONTEXT_LENGTH:-131072}"
SG_KV_CACHE_DTYPE="${SG_KV_CACHE_DTYPE:-fp8_e4m3}"
SG_ENABLE_RADIX_CACHE="${SG_ENABLE_RADIX_CACHE:-1}"
SG_SPECULATIVE_MODE="${SG_SPECULATIVE_MODE:-nextn}"
SG_SPECULATIVE_NUM_STEPS="${SG_SPECULATIVE_NUM_STEPS:-3}"
SG_SPECULATIVE_TOPK="${SG_SPECULATIVE_TOPK:-1}"
SG_SPECULATIVE_DRAFT_TOKENS="${SG_SPECULATIVE_DRAFT_TOKENS:-4}"
SG_EXTRA_ARGS="${SG_EXTRA_ARGS:-}"
SG_READY_TIMEOUT_S="${SG_READY_TIMEOUT_S:-1800}"
SG_SYSTEMD_UNIT="${SG_SYSTEMD_UNIT:-clypt-sglang-qwen.service}"

echo "[deploy-sglang-qwen] ensuring host prerequisites are installed ..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ninja-build

echo "[deploy-sglang-qwen] ensuring dedicated SGLang venv exists at ${SGLANG_VENV_DIR} ..."
install -d -m 0755 "$(dirname "$SGLANG_VENV_DIR")"
python3 -m venv "$SGLANG_VENV_DIR"
. "$SGLANG_VENV_DIR/bin/activate"

echo "[deploy-sglang-qwen] installing SGLang mainline ..."
python -m pip install --upgrade pip wheel
python -m pip install "$SG_PACKAGE_SPEC"

echo "[deploy-sglang-qwen] installing systemd unit ..."
install -D -m 0644 \
  scripts/do_phase1_visual/systemd/clypt-sglang-qwen.service \
  "/etc/systemd/system/${SG_SYSTEMD_UNIT}"

echo "[deploy-sglang-qwen] updating unit with configured runtime values ..."
python3 - <<'PY' "$SG_SYSTEMD_UNIT" "$SGLANG_VENV_DIR" "$SG_MODEL" "$SG_PORT" "$SG_GRAMMAR_BACKEND" "$SG_SCHEDULE_POLICY" "$SG_CHUNKED_PREFILL_SIZE" "$SG_MEM_FRACTION_STATIC" "$SG_CONTEXT_LENGTH" "$SG_EXTRA_ARGS" "$SG_KV_CACHE_DTYPE" "$SG_ENABLE_RADIX_CACHE" "$SG_SPECULATIVE_MODE" "$SG_SPECULATIVE_NUM_STEPS" "$SG_SPECULATIVE_TOPK" "$SG_SPECULATIVE_DRAFT_TOKENS"
from pathlib import Path
import re
import sys

(
    unit_name,
    sglang_venv_dir,
    model,
    port,
    grammar_backend,
    schedule_policy,
    chunked_prefill_size,
    mem_fraction,
    context_len,
    extra_args,
    kv_cache_dtype,
    enable_radix_cache,
    speculative_mode,
    speculative_num_steps,
    speculative_topk,
    speculative_draft_tokens,
) = sys.argv[1:]
unit_path = Path("/etc/systemd/system") / unit_name
text = unit_path.read_text(encoding="utf-8")
optional_flags: list[str] = []
if schedule_policy.strip():
    optional_flags.extend(["--schedule-policy", schedule_policy.strip()])
if chunked_prefill_size.strip() and chunked_prefill_size.strip() != "0":
    optional_flags.extend(["--chunked-prefill-size", chunked_prefill_size.strip()])
if kv_cache_dtype.strip():
    optional_flags.extend(["--kv-cache-dtype", kv_cache_dtype.strip()])
_radix = enable_radix_cache.strip().lower()
if _radix in {"0", "false", "off", "no", "disable"}:
    optional_flags.append("--disable-radix-cache")
if speculative_mode.strip().lower() == "nextn":
    # Qwen3.6-35B-A3B is a hybrid Mamba/Attention MoE. Pair NextN MTP with the
    # extra_buffer mamba-scheduler-strategy; the systemd unit separately exports
    # SGLANG_ENABLE_SPEC_V2=1 so radix cache + MTP can coexist.
    optional_flags.extend([
        "--mamba-scheduler-strategy", "extra_buffer",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", speculative_num_steps.strip(),
        "--speculative-eagle-topk", speculative_topk.strip(),
        "--speculative-num-draft-tokens", speculative_draft_tokens.strip(),
    ])
if extra_args.strip():
    optional_flags.append(extra_args.strip())
exec_line = (
    "ExecStart="
    f"{sglang_venv_dir}/bin/python -m sglang.launch_server "
    f"--model-path {model} --host 127.0.0.1 --port {port} "
    f"--reasoning-parser qwen3 --grammar-backend {grammar_backend} "
    f"{' '.join(optional_flags)} "
    f"--mem-fraction-static {mem_fraction} --context-length {context_len}"
)
exec_line = re.sub(r"\s+", " ", exec_line).strip()
text = re.sub(r"^ExecStart=.*$", exec_line, text, flags=re.MULTILINE)
unit_path.write_text(text, encoding="utf-8")
PY

echo "[deploy-sglang-qwen] stopping legacy Qwen vLLM service (if present) ..."
systemctl stop clypt-vllm-qwen.service 2>/dev/null || true
systemctl disable clypt-vllm-qwen.service 2>/dev/null || true

echo "[deploy-sglang-qwen] reloading and starting ${SG_SYSTEMD_UNIT} ..."
systemctl daemon-reload
systemctl enable "$SG_SYSTEMD_UNIT"
systemctl restart "$SG_SYSTEMD_UNIT"

echo "[deploy-sglang-qwen] waiting for ${SG_BASE_URL}/health ..."
deadline=$(( $(date +%s) + SG_READY_TIMEOUT_S ))
while true; do
  if curl -fsS "${SG_BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  now=$(date +%s)
  if (( now >= deadline )); then
    echo "[deploy-sglang-qwen] ERROR: service did not become healthy in ${SG_READY_TIMEOUT_S}s" >&2
    systemctl --no-pager --full status "$SG_SYSTEMD_UNIT" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "[deploy-sglang-qwen] validating /v1/models on ${SG_BASE_URL} ..."
models_json="$(curl -fsS "${SG_BASE_URL}/v1/models" || true)"
if ! python3 - "$models_json" "$SG_MODEL" <<'PY'; then
import json
import sys

raw = sys.argv[1]
target = sys.argv[2]
try:
    payload = json.loads(raw)
except Exception:
    raise SystemExit(1)
ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
if target not in ids:
    raise SystemExit(1)
PY
  echo "[deploy-sglang-qwen] ERROR: expected model id '${SG_MODEL}' not found in /v1/models" >&2
  printf '%s\n' "$models_json" >&2
  exit 1
fi

echo "[deploy-sglang-qwen] restarting phase services for fresh config pick-up ..."
systemctl restart clypt-v31-phase1-api.service
systemctl restart clypt-v31-phase1-worker.service
systemctl restart clypt-v31-phase24-local-worker.service

echo "[deploy-sglang-qwen] done."
echo "[deploy-sglang-qwen] ensure env contains:"
echo "  GENAI_GENERATION_BACKEND=local_openai"
echo "  CLYPT_LOCAL_LLM_BASE_URL=${SG_BASE_URL}/v1"
echo "  CLYPT_LOCAL_LLM_MODEL=${SG_MODEL}"
echo "  SG_SCHEDULE_POLICY=${SG_SCHEDULE_POLICY}"
echo "  SG_CHUNKED_PREFILL_SIZE=${SG_CHUNKED_PREFILL_SIZE}"
echo "  SG_MEM_FRACTION_STATIC=${SG_MEM_FRACTION_STATIC}"
echo "  SG_CONTEXT_LENGTH=${SG_CONTEXT_LENGTH}"
echo "  SG_KV_CACHE_DTYPE=${SG_KV_CACHE_DTYPE}"
echo "  SG_ENABLE_RADIX_CACHE=${SG_ENABLE_RADIX_CACHE}"
echo "  SG_SPECULATIVE_MODE=${SG_SPECULATIVE_MODE}"
echo "  SG_SPECULATIVE_NUM_STEPS=${SG_SPECULATIVE_NUM_STEPS}"
echo "  SG_SPECULATIVE_TOPK=${SG_SPECULATIVE_TOPK}"
echo "  SG_SPECULATIVE_DRAFT_TOKENS=${SG_SPECULATIVE_DRAFT_TOKENS}"
echo "  CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0"
echo "  CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1"
