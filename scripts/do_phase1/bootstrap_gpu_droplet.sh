#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y \
  build-essential \
  cmake \
  ffmpeg \
  git \
  jq \
  libavcodec-dev \
  libavdevice-dev \
  libavfilter-dev \
  libavformat-dev \
  libavutil-dev \
  libgl1 \
  libglib2.0-0 \
  libsndfile1 \
  python3 \
  python3-pip \
  python3-venv \
  rsync

mkdir -p /opt/clypt-phase1
mkdir -p /etc/clypt-phase1
mkdir -p /var/lib/clypt/do_phase1_service

chmod 755 /opt/clypt-phase1
chmod 755 /etc/clypt-phase1
chmod 755 /var/lib/clypt/do_phase1_service

echo "Bootstrap complete."
echo "Next:"
echo "1. Copy backend/do_phase1_service/.env.example to /etc/clypt-phase1/do-phase1.env and fill it in."
echo "   Ensure these face-pipeline defaults are present:"
echo "   - CLYPT_FACE_PIPELINE_START_FRAME=0"
echo "   - CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE=1280"
echo "   - CLYPT_FULLFRAME_FACE_MIN_SIZE=14"
echo "   - CLYPT_FACE_PIPELINE_GPU=1"
echo "2. Copy GCP service-account JSON to /etc/clypt-phase1/gcp-sa.json."
echo "3. Run scripts/do_phase1/deploy_phase1_service.sh from the repo checkout on the droplet."
