from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.phase1_runtime.factory import build_default_phase1_job_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V3.1 Phase 1 runtime locally.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-url", help="YouTube or direct source URL")
    source.add_argument("--source-path", help="Local source video path")
    parser.add_argument("--job-id", required=True, help="Stable run/job id")
    parser.add_argument("--run-phase14", action="store_true", help="Continue into live Phases 2-4 after Phase 1")
    parser.add_argument("--working-root", default=None, help="Optional workspace root override")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    runner = build_default_phase1_job_runner(working_root=args.working_root)
    result = runner.run_job(
        job_id=args.job_id,
        source_url=args.source_url,
        source_path=args.source_path,
        runtime_controls={"run_phase14": bool(args.run_phase14)},
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
