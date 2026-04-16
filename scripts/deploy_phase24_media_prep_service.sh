#!/usr/bin/env bash
set -euo pipefail

# Deploy the Cloud Run L4 media prep service from the dedicated Dockerfile.

PROJECT="${PROJECT:-clypt-v3}"
REGION="${REGION:-us-east4}"
SERVICE="${SERVICE:-clypt-phase24-media-prep}"
REPO_IMAGE_BASE="${REPO_IMAGE_BASE:-us-east4-docker.pkg.dev/clypt-v3/cloud-run-source-deploy/clypt-phase24-media-prep}"
TAG="${TAG:-manual-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${REPO_IMAGE_BASE}:${TAG}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-32Gi}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
SET_ENV_VARS="${SET_ENV_VARS:-}"

echo "[phase24-media-prep-deploy] building image: ${IMAGE}"
TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT
cat > "${TMP_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-f", "docker/phase24-media-prep/Dockerfile", "-t", "${IMAGE}", "."]
images:
  - "${IMAGE}"
EOF
gcloud builds submit \
  --project "${PROJECT}" \
  --config "${TMP_CONFIG}" \
  .

echo "[phase24-media-prep-deploy] deploying service: ${SERVICE} (${REGION})"
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
  --no-gpu-zonal-redundancy
  --no-allow-unauthenticated
  --quiet
)

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  DEPLOY_CMD+=(--service-account "${SERVICE_ACCOUNT}")
fi

if [[ -n "${SET_ENV_VARS}" ]]; then
  DEPLOY_CMD+=(--set-env-vars "${SET_ENV_VARS}")
fi

"${DEPLOY_CMD[@]}"

echo "[phase24-media-prep-deploy] done"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format='value(status.latestReadyRevisionName,spec.template.spec.containers[0].image)'
