#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
BRANCH="${BRANCH:-codex/balanced-hybrid-phase1-contract}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/do-phase1.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Expected a git checkout at $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-api.service /etc/systemd/system/clypt-phase1-api.service
install -D -m 0644 scripts/do_phase1/systemd/clypt-phase1-worker.service /etc/systemd/system/clypt-phase1-worker.service

systemctl daemon-reload
systemctl enable --now clypt-phase1-api.service
systemctl enable --now clypt-phase1-worker.service

systemctl --no-pager --full status clypt-phase1-api.service || true
systemctl --no-pager --full status clypt-phase1-worker.service || true
