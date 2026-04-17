#!/usr/bin/env bash
set -euo pipefail

# Deploy the combined Cloud Run L4 ASR + node-media-prep service.
# This intentionally reuses the existing Dockerfile/app surface while the
# combined topology is rolled out.

PROJECT="${PROJECT:-clypt-v3}"
REGION="${REGION:-us-east4}"
SERVICE="${SERVICE:-clypt-phase1-l4-combined}"
REPO_IMAGE_BASE="${REPO_IMAGE_BASE:-us-east4-docker.pkg.dev/clypt-v3/cloud-run-source-deploy/clypt-phase1-l4-combined}"
TAG="${TAG:-manual-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${REPO_IMAGE_BASE}:${TAG}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-32Gi}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
SET_ENV_VARS="${SET_ENV_VARS:-}"
VIBEVOICE_REPO_URL="${VIBEVOICE_REPO_URL:-https://github.com/microsoft/VibeVoice.git}"
VIBEVOICE_REPO_REF="${VIBEVOICE_REPO_REF:-main}"

# GCS-FUSE-backed persistent HuggingFace cache. Belt-and-suspenders alongside
# the in-image model bake: if a future rebuild adds new models (e.g. tokenizers,
# aux weights) the bucket retains them across revisions. Mounted at a
# NON-overlapping path (/mnt/hf-persist, not /root/.cache/huggingface) so the
# baked model layer remains the primary read path for VibeVoice-ASR weights.
HF_CACHE_BUCKET="${HF_CACHE_BUCKET:-clypt-v3-hf-cache}"
HF_CACHE_MOUNT_PATH="${HF_CACHE_MOUNT_PATH:-/mnt/hf-persist}"

echo "[l4-combined-deploy] ensuring GCS bucket gs://${HF_CACHE_BUCKET} (${REGION})"
if ! gcloud storage buckets describe "gs://${HF_CACHE_BUCKET}" \
     --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${HF_CACHE_BUCKET}" \
    --project "${PROJECT}" \
    --location "${REGION}" \
    --uniform-bucket-level-access
fi

echo "[l4-combined-deploy] building image: ${IMAGE}"
TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT
cat > "${TMP_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: [
      "build",
      "-f", "docker/phase24-media-prep/Dockerfile",
      "--build-arg", "VIBEVOICE_REPO_URL=${VIBEVOICE_REPO_URL}",
      "--build-arg", "VIBEVOICE_REPO_REF=${VIBEVOICE_REPO_REF}",
      "-t", "${IMAGE}",
      "."
    ]
images:
  - "${IMAGE}"
options:
  machineType: E2_HIGHCPU_32
  diskSizeGb: 200
timeout: 3600s
EOF
gcloud builds submit \
  --project "${PROJECT}" \
  --config "${TMP_CONFIG}" \
  .

echo "[l4-combined-deploy] deploying service: ${SERVICE} (${REGION})"
DEPLOY_CMD=(
  gcloud run deploy "${SERVICE}"
  --project "${PROJECT}"
  --region "${REGION}"
  --image "${IMAGE}"
  --gpu 1
  --gpu-type "${GPU_TYPE}"
  --cpu "${CPU}"
  --memory "${MEMORY}"
  --max-instances "${MAX_INSTANCES}"
  --concurrency 1
  --timeout "${TIMEOUT}"
  --execution-environment gen2
  --no-gpu-zonal-redundancy
  --no-allow-unauthenticated
  --add-volume "name=hf-persist,type=cloud-storage,bucket=${HF_CACHE_BUCKET}"
  --add-volume-mount "volume=hf-persist,mount-path=${HF_CACHE_MOUNT_PATH}"
  # With the VibeVoice-ASR model baked into the image, cold start is dominated
  # by image pull (~1-2 min from in-region Artifact Registry) + vLLM GPU load
  # (~2-5 min). Budget ~14 min for the probe window: 240 (cap) + 10 * 60.
  --startup-probe "tcpSocket.port=8080,initialDelaySeconds=240,periodSeconds=60,timeoutSeconds=10,failureThreshold=10"
  --quiet
)

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  DEPLOY_CMD+=(--service-account "${SERVICE_ACCOUNT}")
fi

if [[ -n "${SET_ENV_VARS}" ]]; then
  DEPLOY_CMD+=(--set-env-vars "${SET_ENV_VARS}")
fi

"${DEPLOY_CMD[@]}"

echo "[l4-combined-deploy] done"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format='value(status.latestReadyRevisionName,spec.template.spec.containers[0].image)'
