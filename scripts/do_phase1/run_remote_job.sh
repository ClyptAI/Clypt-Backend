#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing repo virtualenv python at $PYTHON_BIN" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$ROOT/scripts/do_phase1/run_remote_job.py" "$@"
