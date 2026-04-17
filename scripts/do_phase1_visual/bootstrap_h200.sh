#!/usr/bin/env bash
# Bootstrap the H200 Phase 1 VISUAL + Phase 2-4 host.
#
# This droplet runs the visual chain (RF-DETR + ByteTrack), the Phase 1
# audio post-processing chain (NFA + emotion2vec+ + YAMNet), the Phase 1
# orchestrator, the SGLang Qwen service, and the Phase 2-4 local worker.
# VibeVoice vLLM ASR and ffmpeg NVENC node-media prep live on the
# RTX 6000 Ada host — see scripts/do_phase1_audio/.
#
# Run as root on a fresh NVIDIA AI/ML base image droplet.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  gnupg \
  lsb-release \
  ninja-build \
  python3 \
  unzip \
  python3-venv \
  python3-pip

install -d /opt/clypt-phase1
install -d /opt/clypt-phase1/hf-cache
install -d /opt/clypt-phase1/venvs
install -d /opt/clypt-phase1/videos
install -d /opt/clypt-phase1/.cache/torch/kernels
install -d /opt/clypt-phase1/.cache/huggingface
install -d /etc/clypt-phase1
install -d /var/lib/clypt/v3_1_phase1_service
install -d /var/log/clypt/v3_1_phase1

if [[ ! -x /root/.bun/bin/bun ]]; then
  curl -fsSL https://bun.sh/install | bash
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This host may not be a GPU base image." >&2
fi

# Pre-cache the SGLang Qwen weights into /opt/clypt-phase1/hf-cache. The
# committed clypt-sglang-qwen.service runs with HF_HUB_OFFLINE=1 in its
# systemd Environment=, which tells huggingface_hub to skip all network
# calls and use only the local cache. If the model isn't resident here
# before the unit starts, SGLang will crash on first boot with a
# "local files only" / missing-revision error. Idempotent — skips the
# download if a snapshot is already present. Set SG_MODEL to override.
SG_MODEL="${SG_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SG_HF_HOME="/opt/clypt-phase1/hf-cache"
# HF cache layout: $HF_HOME/hub/models--<org>--<repo>/snapshots/<rev>/...
SG_MODEL_CACHE_DIR="${SG_HF_HOME}/hub/models--${SG_MODEL/\//--}"
if [[ -d "$SG_MODEL_CACHE_DIR/snapshots" ]] \
  && [[ -n "$(find "$SG_MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
  echo "[bootstrap-h200] ${SG_MODEL} already cached at ${SG_MODEL_CACHE_DIR} — skipping pre-download."
else
  echo "[bootstrap-h200] Pre-caching ${SG_MODEL} into ${SG_HF_HOME} (one-time, ~72 GB) ..."
  echo "[bootstrap-h200] If ${SG_MODEL} is a gated repo, export HF_TOKEN before running this script."
  pip3 install --break-system-packages --quiet "huggingface_hub>=0.28"
  HF_HOME="$SG_HF_HOME" python3 - "$SG_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1])
PY
  echo "[bootstrap-h200] ${SG_MODEL} cached."
fi

echo "Bootstrap complete. Next steps:"
echo "  1. rsync the repo to /opt/clypt-phase1/repo"
echo "  2. create /etc/clypt-phase1/v3_1_phase1.env (see docs/runtime/known-good.env)"
echo "  3. bash scripts/do_phase1_visual/deploy_visual_service.sh"
echo "  4. bash scripts/do_phase1_visual/deploy_sglang_qwen_service.sh"
echo ""
echo "To refresh the Qwen weights later, either: (a) rerun this script after"
echo "deleting ${SG_MODEL_CACHE_DIR}, or (b) temporarily unset HF_HUB_OFFLINE"
echo "in clypt-sglang-qwen.service, restart to let SGLang pull the new"
echo "revision, then re-set HF_HUB_OFFLINE=1 and restart again."
