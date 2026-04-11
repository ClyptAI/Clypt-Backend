#!/usr/bin/env bash
set -euo pipefail

# Deploy the Phase 2-4 Cloud Run worker from the dedicated Dockerfile.
# This path is required to keep ffmpeg installed in the runtime image.

PROJECT="${PROJECT:-clypt-v3}"
REGION="${REGION:-us-east4}"
SERVICE="${SERVICE:-clypt-phase24-worker}"
REPO_IMAGE_BASE="${REPO_IMAGE_BASE:-us-east4-docker.pkg.dev/clypt-v3/cloud-run-source-deploy/clypt-phase24-worker}"
TAG="${TAG:-manual-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${REPO_IMAGE_BASE}:${TAG}"

echo "[phase24-deploy] building image: ${IMAGE}"
TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT
cat > "${TMP_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-f", "docker/phase24-worker/Dockerfile", "-t", "${IMAGE}", "."]
images:
  - "${IMAGE}"
EOF
gcloud builds submit \
  --project "${PROJECT}" \
  --config "${TMP_CONFIG}" \
  .

echo "[phase24-deploy] deploying service: ${SERVICE} (${REGION})"
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --quiet

echo "[phase24-deploy] done"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format='value(status.latestReadyRevisionName,spec.template.spec.containers[0].image)'
