#!/usr/bin/env bash
set -euo pipefail

VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-phase1-vllm-vibevoice:latest}"
VIBEVOICE_MODEL_ID="${VIBEVOICE_MODEL_ID:-microsoft/VibeVoice-ASR}"
VIBEVOICE_CACHE_DIR="${VIBEVOICE_CACHE_DIR:-/opt/clypt-phase1/hf-cache}"
VIBEVOICE_VLLM_DTYPE="${VIBEVOICE_VLLM_DTYPE:-bfloat16}"
VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION="${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VIBEVOICE_VLLM_MAX_NUM_SEQS="${VIBEVOICE_VLLM_MAX_NUM_SEQS:-4}"

exec /usr/bin/docker run \
  --name clypt-phase1-vllm-vibevoice \
  --gpus all \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e VIBEVOICE_MODEL_ID="${VIBEVOICE_MODEL_ID}" \
  -e VIBEVOICE_VLLM_DTYPE="${VIBEVOICE_VLLM_DTYPE}" \
  -e VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION="${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION}" \
  -e VIBEVOICE_VLLM_MAX_NUM_SEQS="${VIBEVOICE_VLLM_MAX_NUM_SEQS}" \
  -v "${VIBEVOICE_CACHE_DIR}:/root/.cache/huggingface" \
  -w /opt/vibevoice-baked \
  --entrypoint bash \
  "${VLLM_IMAGE_TAG}" \
  -lc '
set -euo pipefail
MODEL_PATH="$(python3 -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ[\"VIBEVOICE_MODEL_ID\"]))")"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "[phase1-vllm] snapshot_download returned an empty model path" >&2
  exit 1
fi
python3 -m vllm_plugin.tools.generate_tokenizer_files --output "${MODEL_PATH}" || true
exec vllm serve "${MODEL_PATH}" \
  --served-model-name vibevoice \
  --trust-remote-code \
  --dtype "${VIBEVOICE_VLLM_DTYPE}" \
  --gpu-memory-utilization "${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VIBEVOICE_VLLM_MAX_NUM_SEQS}" \
  --port 8000 \
  --host 0.0.0.0
'
