#!/usr/bin/env bash
set -euo pipefail

SG_DOCKER_IMAGE="${SG_DOCKER_IMAGE:-lmsysorg/sglang:v0.5.10-rocm720-mi30x}"
SG_CONTAINER_NAME="${SG_CONTAINER_NAME:-clypt-phase26-sglang-qwen}"
SG_MODEL="${SG_MODEL:-${GENAI_GENERATION_MODEL:-Qwen/Qwen3.6-35B-A3B}}"
SG_MODEL_PATH="${SG_MODEL_PATH:-$SG_MODEL}"
SG_SERVED_MODEL_NAME="${SG_SERVED_MODEL_NAME:-$SG_MODEL}"
SG_HOST="${SG_HOST:-127.0.0.1}"
SG_PORT="${SG_PORT:-8001}"
SG_LAUNCH_PROFILE="${SG_LAUNCH_PROFILE:-final}"
SG_CONTEXT_LENGTH="${SG_CONTEXT_LENGTH:-65536}"
SG_MEM_FRACTION_STATIC="${SG_MEM_FRACTION_STATIC:-0.78}"
SG_KV_CACHE_DTYPE="${SG_KV_CACHE_DTYPE:-fp8_e4m3}"
SG_GRAMMAR_BACKEND="${SG_GRAMMAR_BACKEND:-xgrammar}"
SG_REASONING_PARSER="${SG_REASONING_PARSER:-qwen3}"
SG_ATTENTION_BACKEND="${SG_ATTENTION_BACKEND:-triton}"
SG_SCHEDULE_POLICY="${SG_SCHEDULE_POLICY:-lpm}"
SG_CHUNKED_PREFILL_SIZE="${SG_CHUNKED_PREFILL_SIZE:-8192}"
SG_MAMBA_SCHEDULER_STRATEGY="${SG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
SG_SPECULATIVE_ALGORITHM="${SG_SPECULATIVE_ALGORITHM:-NEXTN}"
SG_SPECULATIVE_NUM_STEPS="${SG_SPECULATIVE_NUM_STEPS:-3}"
SG_SPECULATIVE_EAGLE_TOPK="${SG_SPECULATIVE_EAGLE_TOPK:-1}"
SG_SPECULATIVE_NUM_DRAFT_TOKENS="${SG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
SG_ENABLE_SPEC_V2="${SG_ENABLE_SPEC_V2:-1}"
HF_HOME="${HF_HOME:-/opt/clypt-phase26/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-/opt/clypt-phase26/.cache/torch}"
PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-/opt/clypt-phase26/.cache/torch/kernels}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
  echo "[sglang-rocm] ERROR: required ROCm device paths /dev/kfd and /dev/dri are not available." >&2
  exit 1
fi

install -d "$HF_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
docker rm -f "$SG_CONTAINER_NAME" >/dev/null 2>&1 || true

declare -a args=(
  --rm
  --name "$SG_CONTAINER_NAME"
  --network host
  --device=/dev/kfd
  --device=/dev/dri
  --group-add video
  --ipc=host
  --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined
  -e "HF_HOME=/root/.cache/huggingface"
  -e "TORCH_HOME=/root/.cache/torch"
  -e "PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels"
  -e "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  -e "HF_HUB_ENABLE_HF_TRANSFER=1"
  -e "SGLANG_ENABLE_SPEC_V2=$SG_ENABLE_SPEC_V2"
  -e "TOKENIZERS_PARALLELISM=false"
  -v "$HF_HOME:/root/.cache/huggingface"
  -v "$TORCH_HOME:/root/.cache/torch"
  "$SG_DOCKER_IMAGE"
  python -m sglang.launch_server
  --model-path "$SG_MODEL_PATH"
  --served-model-name "$SG_SERVED_MODEL_NAME"
  --host "$SG_HOST"
  --port "$SG_PORT"
  --trust-remote-code
  --attention-backend "$SG_ATTENTION_BACKEND"
  --mem-fraction-static "$SG_MEM_FRACTION_STATIC"
  --context-length "$SG_CONTEXT_LENGTH"
)

case "$SG_LAUNCH_PROFILE" in
  minimal)
    ;;
  strict_json)
    args+=(
      --reasoning-parser "$SG_REASONING_PARSER"
      --grammar-backend "$SG_GRAMMAR_BACKEND"
    )
    ;;
  fp8_kv)
    args+=(
      --reasoning-parser "$SG_REASONING_PARSER"
      --grammar-backend "$SG_GRAMMAR_BACKEND"
      --kv-cache-dtype "$SG_KV_CACHE_DTYPE"
    )
    ;;
  scheduler_cache)
    args+=(
      --reasoning-parser "$SG_REASONING_PARSER"
      --grammar-backend "$SG_GRAMMAR_BACKEND"
      --kv-cache-dtype "$SG_KV_CACHE_DTYPE"
      --schedule-policy "$SG_SCHEDULE_POLICY"
      --chunked-prefill-size "$SG_CHUNKED_PREFILL_SIZE"
      --mamba-scheduler-strategy "$SG_MAMBA_SCHEDULER_STRATEGY"
    )
    ;;
  speculative|final)
    args+=(
      --reasoning-parser "$SG_REASONING_PARSER"
      --grammar-backend "$SG_GRAMMAR_BACKEND"
      --kv-cache-dtype "$SG_KV_CACHE_DTYPE"
      --schedule-policy "$SG_SCHEDULE_POLICY"
      --chunked-prefill-size "$SG_CHUNKED_PREFILL_SIZE"
      --mamba-scheduler-strategy "$SG_MAMBA_SCHEDULER_STRATEGY"
      --speculative-algorithm "$SG_SPECULATIVE_ALGORITHM"
      --speculative-num-steps "$SG_SPECULATIVE_NUM_STEPS"
      --speculative-eagle-topk "$SG_SPECULATIVE_EAGLE_TOPK"
      --speculative-num-draft-tokens "$SG_SPECULATIVE_NUM_DRAFT_TOKENS"
    )
    ;;
  *)
    echo "[sglang-rocm] ERROR: unsupported SG_LAUNCH_PROFILE=$SG_LAUNCH_PROFILE" >&2
    exit 1
    ;;
esac

if [[ -n "${SG_EXTRA_ARGS:-}" ]]; then
  echo "[sglang-rocm] ERROR: SG_EXTRA_ARGS is intentionally unsupported; use SG_EXTRA_ARGS_FILE with one argv token per line." >&2
  exit 1
fi

if [[ -n "${SG_EXTRA_ARGS_FILE:-}" ]]; then
  if [[ ! -f "$SG_EXTRA_ARGS_FILE" ]]; then
    echo "[sglang-rocm] ERROR: SG_EXTRA_ARGS_FILE does not exist: $SG_EXTRA_ARGS_FILE" >&2
    exit 1
  fi
  while IFS= read -r extra_arg || [[ -n "$extra_arg" ]]; do
    [[ -z "$extra_arg" || "$extra_arg" == \#* ]] && continue
    args+=("$extra_arg")
  done <"$SG_EXTRA_ARGS_FILE"
fi

exec docker run "${args[@]}"
