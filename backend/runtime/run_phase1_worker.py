from __future__ import annotations

import argparse
from pathlib import Path

from backend.phase1_runtime.factory import build_default_phase1_job_runner
from backend.phase1_runtime.state_store import SQLiteJobStore
from backend.phase1_runtime.worker import Phase1Worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V3.1 Phase 1 worker loop.")
    parser.add_argument("--db-path", default="backend/outputs/v3_1_phase1_service/jobs.db")
    parser.add_argument("--logs-root", default="backend/outputs/v3_1_phase1_service/logs")
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = SQLiteJobStore(args.db_path)
    job_runner = build_default_phase1_job_runner()
    worker = Phase1Worker(
        store=store,
        run_job=job_runner.run_job,
        logs_root=Path(args.logs_root),
    )
    worker.run_forever(poll_interval_s=args.poll_interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
