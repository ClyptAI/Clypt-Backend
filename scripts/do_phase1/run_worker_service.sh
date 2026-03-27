#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/clypt-phase1/repo}"
ENV_FILE="${ENV_FILE:-/etc/clypt-phase1/do-phase1.env}"
IDLE_SLEEP_SECONDS="${IDLE_SLEEP_SECONDS:-2}"
ERROR_SLEEP_SECONDS="${ERROR_SLEEP_SECONDS:-2}"

cd "$REPO_DIR"
set -a
. "$ENV_FILE"
set +a

while true; do
  if .venv/bin/python - <<'PY'
from backend.do_phase1_service.state_store import SQLiteJobStore
from backend.do_phase1_service.worker import (
    DEFAULT_DB_PATH,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNNING_STALE_AFTER_SECONDS,
    run_worker_once,
)

processed = run_worker_once(
    SQLiteJobStore(DEFAULT_DB_PATH),
    output_root=DEFAULT_OUTPUT_ROOT,
    stale_after_seconds=DEFAULT_RUNNING_STALE_AFTER_SECONDS,
    heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
)
raise SystemExit(0 if processed else 3)
PY
  then
    continue
  fi

  exit_code=$?
  if [[ "$exit_code" -eq 3 ]]; then
    sleep "$IDLE_SLEEP_SECONDS"
    continue
  fi

  sleep "$ERROR_SLEEP_SECONDS"
done
