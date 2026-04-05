#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/v3_1_phase1.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-do-phase1.txt}"
BRANCH="${BRANCH:-v3.1-refactor}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-0}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Missing repo dir: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_DIR"

if [[ "$SKIP_GIT_SYNC" != "1" ]]; then
  git fetch origin
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"

install -D -m 0644 scripts/do_phase1/systemd/clypt-v31-phase1-api.service /etc/systemd/system/clypt-v31-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-v31-phase1-worker.service /etc/systemd/system/clypt-v31-phase1-worker.service

systemctl daemon-reload
systemctl enable clypt-v31-phase1-api.service
systemctl enable clypt-v31-phase1-worker.service
systemctl restart clypt-v31-phase1-api.service
systemctl restart clypt-v31-phase1-worker.service

systemctl --no-pager --full status clypt-v31-phase1-api.service || true
systemctl --no-pager --full status clypt-v31-phase1-worker.service || true
