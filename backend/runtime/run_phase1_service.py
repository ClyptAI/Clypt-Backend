from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from backend.phase1_runtime.app import create_app
from backend.phase1_runtime.state_store import SQLiteJobStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V3.1 Phase 1 FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db-path", default="backend/outputs/v3_1_phase1_service/jobs.db")
    parser.add_argument("--logs-root", default="backend/outputs/v3_1_phase1_service/logs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = create_app(
        store=SQLiteJobStore(args.db_path),
        logs_root=Path(args.logs_root),
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
