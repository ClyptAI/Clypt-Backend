#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${DO_PHASE1_BASE_URL:-}"
SOURCE_URL="${1:-}"

if [[ -z "$BASE_URL" ]]; then
  echo "Set DO_PHASE1_BASE_URL to the Phase 1 service base URL, e.g. http://HOST:8080" >&2
  exit 1
fi

if [[ -z "$SOURCE_URL" ]]; then
  echo "Usage: DO_PHASE1_BASE_URL=http://HOST:8080 bash scripts/do_phase1/run_remote_job.sh \"https://youtube.com/watch?v=...\"" >&2
  exit 1
fi

python -m scripts.do_phase1.run_remote_job --base-url "$BASE_URL" --source-url "$SOURCE_URL"
