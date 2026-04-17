#!/usr/bin/env bash
set -euo pipefail

# Deploy the combined L4 ASR + node-media-prep service on a GCE L4 VM.
#
# Replaces the Cloud Run L4 path (scripts/deploy_l4_combined_service.sh) once
# we hit the 24 GB VRAM ceiling on Cloud Run: VibeVoice's fp32 audio-encoder
# upcast starves vLLM's profile_run. On GCE we get the same L4, but with a
# persistent host FS, full control over env knobs, and we pair it with a
# bf16 audio-encoder sed patch in the Dockerfile.
#
# Networking: firewall restricts port 8080 to the DigitalOcean droplet IP so
# we can run the service with AUTH_MODE=none.

PROJECT="${PROJECT:-clypt-v3}"
ZONE="${ZONE:-us-central1-a}"
REGION_AR="${REGION_AR:-us-east4}"  # Artifact Registry region (existing images live here).
VM_NAME="${VM_NAME:-clypt-phase1-l4-gce}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
FIREWALL_RULE="${FIREWALL_RULE:-clypt-l4-combined-ingress}"
NETWORK_TAG="${NETWORK_TAG:-clypt-l4-combined}"
DROPLET_IP="${DROPLET_IP:-165.245.141.217}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"  # Empty => default compute SA.

REPO_IMAGE_BASE="${REPO_IMAGE_BASE:-${REGION_AR}-docker.pkg.dev/${PROJECT}/cloud-run-source-deploy/clypt-phase1-l4-combined}"
TAG="${TAG:-gce-bf16-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${REPO_IMAGE_BASE}:${TAG}"

CONTAINER_NAME="${CONTAINER_NAME:-clypt-l4-combined}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/var/clypt/hf-cache}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"

echo "[gce-deploy] project=${PROJECT} zone=${ZONE} vm=${VM_NAME} image=${IMAGE}"

#------------------------------------------------------------------------------
# 1) Build and push the bf16-patched image.
#------------------------------------------------------------------------------
echo "[gce-deploy] building image: ${IMAGE}"
TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT
cat > "${TMP_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: [
      "build",
      "-f", "docker/phase24-media-prep/Dockerfile",
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

#------------------------------------------------------------------------------
# 2) Ensure the firewall rule exists and is narrow.
#------------------------------------------------------------------------------
echo "[gce-deploy] ensuring firewall rule: ${FIREWALL_RULE}"
if gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud compute firewall-rules update "${FIREWALL_RULE}" \
    --project "${PROJECT}" \
    --source-ranges="${DROPLET_IP}/32" \
    --rules="tcp:${CONTAINER_PORT}" \
    --target-tags="${NETWORK_TAG}"
else
  gcloud compute firewall-rules create "${FIREWALL_RULE}" \
    --project "${PROJECT}" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules="tcp:${CONTAINER_PORT}" \
    --source-ranges="${DROPLET_IP}/32" \
    --target-tags="${NETWORK_TAG}"
fi

#------------------------------------------------------------------------------
# 3) Create (or recreate) the VM.
#------------------------------------------------------------------------------
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/usr/bin/env bash
set -euo pipefail

HOST_HF_CACHE="__HOST_HF_CACHE__"
CONTAINER_NAME="__CONTAINER_NAME__"
IMAGE="__IMAGE__"
CONTAINER_PORT="__CONTAINER_PORT__"
REGION_AR="__REGION_AR__"

# Wait for NVIDIA driver + container toolkit to be ready. The Deep Learning VM
# image runs an install-gpu-driver service on first boot; nvidia-smi is the
# canonical ready signal.
for i in $(seq 1 60); do
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    break
  fi
  echo "[startup] waiting for nvidia-smi (attempt ${i}/60)"
  sleep 10
done

# Ensure docker + nvidia-container-toolkit are present and configured.
if ! command -v docker >/dev/null; then
  echo "[startup] installing docker"
  apt-get update
  apt-get install -y docker.io
  systemctl enable --now docker
fi

if ! docker info --format '{{ json .Runtimes }}' 2>/dev/null | grep -q nvidia; then
  echo "[startup] configuring nvidia runtime"
  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  # --batch --yes --output is required here: GCE startup scripts have no
  # controlling TTY, so plain `gpg --dearmor -o ...` fails with
  # "cannot open '/dev/tty': No such device or address" and aborts the script
  # under `set -e`. See docs/ERROR_LOG.md 2026-04-16.
  rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor --output /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p "${HOST_HF_CACHE}"

# Auth to Artifact Registry using the VM's attached service account.
gcloud auth configure-docker "${REGION_AR}-docker.pkg.dev" --quiet

# Pull the image (retry on transient failures).
for i in $(seq 1 5); do
  if docker pull "${IMAGE}"; then
    break
  fi
  echo "[startup] pull retry ${i}/5"
  sleep 15
done

# Stop any prior container before starting a fresh one (idempotent re-run).
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Start the combined service.
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart=always \
  --gpus all \
  --network=host \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  -e PORT="${CONTAINER_PORT}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  "${IMAGE}"

echo "[startup] container ${CONTAINER_NAME} started; streaming boot logs"
timeout 30 docker logs -f "${CONTAINER_NAME}" || true
STARTUP_EOF
)

# Substitute in per-deploy values.
STARTUP_SCRIPT="${STARTUP_SCRIPT//__HOST_HF_CACHE__/${HOST_HF_CACHE}}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__CONTAINER_NAME__/${CONTAINER_NAME}}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__IMAGE__/${IMAGE}}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__CONTAINER_PORT__/${CONTAINER_PORT}}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__REGION_AR__/${REGION_AR}}"

STARTUP_FILE="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}" "${STARTUP_FILE}"' EXIT
printf "%s" "${STARTUP_SCRIPT}" > "${STARTUP_FILE}"

# Nuke any prior VM with the same name so the deploy is idempotent.
if gcloud compute instances describe "${VM_NAME}" --zone "${ZONE}" --project "${PROJECT}" >/dev/null 2>&1; then
  echo "[gce-deploy] deleting existing VM: ${VM_NAME}"
  gcloud compute instances delete "${VM_NAME}" --zone "${ZONE}" --project "${PROJECT}" --quiet
fi

CREATE_CMD=(
  gcloud compute instances create "${VM_NAME}"
  --project "${PROJECT}"
  --zone "${ZONE}"
  --machine-type "${MACHINE_TYPE}"
  --accelerator "type=${GPU_TYPE},count=1"
  --maintenance-policy=TERMINATE
  --restart-on-failure
  --image-family "${IMAGE_FAMILY}"
  --image-project "${IMAGE_PROJECT}"
  --boot-disk-size "${BOOT_DISK_SIZE}"
  --boot-disk-type "${BOOT_DISK_TYPE}"
  --metadata "install-nvidia-driver=True,enable-oslogin=TRUE"
  --metadata-from-file "startup-script=${STARTUP_FILE}"
  --tags "${NETWORK_TAG}"
  --scopes=cloud-platform
)

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  CREATE_CMD+=(--service-account "${SERVICE_ACCOUNT}")
fi

echo "[gce-deploy] creating VM: ${VM_NAME}"
"${CREATE_CMD[@]}"

EXTERNAL_IP="$(gcloud compute instances describe "${VM_NAME}" \
  --zone "${ZONE}" --project "${PROJECT}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo ""
echo "[gce-deploy] done."
echo "[gce-deploy] VM:           ${VM_NAME} (${ZONE})"
echo "[gce-deploy] external IP:  ${EXTERNAL_IP}"
echo "[gce-deploy] service URL:  http://${EXTERNAL_IP}:${CONTAINER_PORT}"
echo ""
echo "[gce-deploy] Startup script is running asynchronously; container boot"
echo "[gce-deploy] (image pull + vLLM GPU load) typically takes 4-8 minutes."
echo "[gce-deploy] Watch progress with:"
echo "    gcloud compute ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} --command 'sudo journalctl -u google-startup-scripts.service -n 200 --no-pager'"
echo "    gcloud compute ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} --command 'docker logs -f ${CONTAINER_NAME}'"
