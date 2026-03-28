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
  libgl1 \
  libgles2 \
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
echo "2. Copy GCP service-account JSON to /etc/clypt-phase1/gcp-sa.json."
echo "3. Run scripts/do_phase1/deploy_phase1_service.sh from the repo checkout on the droplet."
