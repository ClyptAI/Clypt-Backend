#!/usr/bin/env bash
set -euo pipefail

VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-clypt-phase1-vllm-vibevoice-rocm:latest}"
VIBEVOICE_MODEL_ID="${VIBEVOICE_MODEL_ID:-microsoft/VibeVoice-ASR}"
VIBEVOICE_MODEL_PATH="${VIBEVOICE_MODEL_PATH:-}"
VIBEVOICE_CACHE_DIR="${VIBEVOICE_CACHE_DIR:-/opt/clypt-phase1/.cache/huggingface}"
VIBEVOICE_VLLM_DTYPE="${VIBEVOICE_VLLM_DTYPE:-bfloat16}"
VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION="${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
VIBEVOICE_VLLM_MAX_NUM_SEQS="${VIBEVOICE_VLLM_MAX_NUM_SEQS:-4}"
VIBEVOICE_FFMPEG_MAX_CONCURRENCY="${VIBEVOICE_FFMPEG_MAX_CONCURRENCY:-24}"

for required_path in /dev/kfd /dev/dri; do
  if [[ ! -e "$required_path" ]]; then
    echo "[phase1-vllm-rocm] ERROR: missing ROCm device path: $required_path" >&2
    exit 1
  fi
done

exec /usr/bin/docker run \
  --name clypt-phase1-vllm-vibevoice \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 16g \
  --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY="${VIBEVOICE_FFMPEG_MAX_CONCURRENCY}" \
  -e HF_HOME="/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e VIBEVOICE_MODEL_ID="${VIBEVOICE_MODEL_ID}" \
  -e VIBEVOICE_MODEL_PATH="${VIBEVOICE_MODEL_PATH}" \
  -e VIBEVOICE_VLLM_DTYPE="${VIBEVOICE_VLLM_DTYPE}" \
  -e VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION="${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION}" \
  -e VIBEVOICE_VLLM_MAX_NUM_SEQS="${VIBEVOICE_VLLM_MAX_NUM_SEQS}" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -v "${VIBEVOICE_CACHE_DIR}:/root/.cache/huggingface" \
  -w /opt/vibevoice-baked \
  --entrypoint bash \
  "${VLLM_IMAGE_TAG}" \
  -lc '
set -euo pipefail
if [[ -n "${VIBEVOICE_MODEL_PATH:-}" ]]; then
  MODEL_PATH="${VIBEVOICE_MODEL_PATH}"
else
  if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    echo "[phase1-vllm-rocm] HF_HUB_OFFLINE=1 requires VIBEVOICE_MODEL_PATH from deploy prewarm." >&2
    exit 1
  fi
  MODEL_PATH="$(python3 -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ[\"VIBEVOICE_MODEL_ID\"]))")"
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "[phase1-vllm-rocm] snapshot_download returned an empty model path" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[phase1-vllm-rocm] model path does not exist inside container: ${MODEL_PATH}" >&2
  exit 1
fi
python3 -m vllm_plugin.tools.generate_tokenizer_files --output "${MODEL_PATH}"
exec vllm serve "${MODEL_PATH}" \
  --served-model-name vibevoice \
  --trust-remote-code \
  --dtype "${VIBEVOICE_VLLM_DTYPE}" \
  --gpu-memory-utilization "${VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VIBEVOICE_VLLM_MAX_NUM_SEQS}" \
  --port 8000 \
  --host 0.0.0.0
'
