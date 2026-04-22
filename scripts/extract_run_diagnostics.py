"""Extract a unified per-run diagnostics bundle.

Pulls together the durable Phase 1 / Phase 2-4 state for a ``run_id`` from:

- Spanner (`runs`, `phase24_jobs`, `phase_metrics`, `phase_substeps`, node/edge/candidate counts)
- Phase 1 API SQLite job store
- Phase26 local SQLite queue
- optional Phase 1 log file tail
- optional filtered journald snippets for the relevant systemd units

This is intended to be the canonical operator/debug script when tracing why a
run failed or where wall time went.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from typing import Any

from backend.providers import load_phase26_host_settings
from backend.repository import SpannerPhase14Repository

logger = logging.getLogger("extract_run_diagnostics")
UTC = timezone.utc

_DEFAULT_PHASE1_DB_PATH = Path("/var/lib/clypt/phase1/jobs.db")
_DEFAULT_PHASE1_UNITS = (
    "clypt-phase1-worker.service",
    "clypt-phase1-vibevoice.service",
)
_DEFAULT_PHASE26_UNITS = (
    "clypt-phase26-worker.service",
    "clypt-phase26-dispatch.service",
)
_KNOWN_NOISE_SUBSTRINGS = (
    "Failed to export metrics to Cloud Monitoring",
    "monitoring.timeSeries.create",
    "Permission monitoring.timeSeries.create denied",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _phase_metric_to_json(record: Any) -> dict[str, Any]:
    return {
        "phase_name": record.phase_name,
        "status": record.status,
        "started_at": _json_ready(record.started_at),
        "ended_at": _json_ready(record.ended_at),
        "duration_ms": record.duration_ms,
        "query_version": getattr(record, "query_version", None),
        "metadata": _json_ready(getattr(record, "metadata", None)),
        "error_payload": _json_ready(getattr(record, "error_payload", None)),
    }


def _phase_substep_to_json(record: Any) -> dict[str, Any]:
    return {
        "phase_name": record.phase_name,
        "step_name": record.step_name,
        "step_key": record.step_key,
        "status": record.status,
        "started_at": _json_ready(record.started_at),
        "ended_at": _json_ready(record.ended_at),
        "duration_ms": record.duration_ms,
        "query_version": getattr(record, "query_version", None),
        "metadata": _json_ready(getattr(record, "metadata", None)),
        "error_payload": _json_ready(getattr(record, "error_payload", None)),
    }


def _read_phase1_job(db_path: Path, run_id: str) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    payload = dict(row)
    for key in ("runtime_controls_json", "result_json", "failure_json"):
        raw = payload.pop(key, None)
        payload[key.removesuffix("_json")] = json.loads(raw) if raw else None
    return payload


def _read_phase24_queue_row(queue_path: Path, run_id: str) -> dict[str, Any] | None:
    if not queue_path.exists():
        return None
    connection = sqlite3.connect(str(queue_path))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM phase24_jobs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    payload = dict(row)
    payload["payload"] = json.loads(payload.pop("payload_json"))
    return payload


def _tail_file(path: Path, *, line_count: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if line_count <= 0:
        return lines
    return lines[-line_count:]


def _is_noise_line(line: str) -> bool:
    return any(needle in line for needle in _KNOWN_NOISE_SUBSTRINGS)


def _run_journalctl(
    *,
    unit: str,
    run_id: str,
    since: str | None,
    line_count: int,
    include_noise: bool,
) -> dict[str, Any]:
    if shutil.which("journalctl") is None:
        return {"unit": unit, "available": False, "reason": "journalctl_not_found", "lines": []}
    command = ["journalctl", "-u", unit, "--no-pager", "-o", "cat"]
    if since:
        command.extend(["--since", since])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"unit": unit, "available": False, "reason": str(exc), "lines": []}
    if completed.returncode not in {0, 1}:
        return {
            "unit": unit,
            "available": False,
            "reason": completed.stderr.strip() or f"journalctl_exit_{completed.returncode}",
            "lines": [],
        }
    matching_lines = [line for line in completed.stdout.splitlines() if run_id in line]
    if not include_noise:
        matching_lines = [line for line in matching_lines if not _is_noise_line(line)]
    if line_count > 0:
        matching_lines = matching_lines[-line_count:]
    return {"unit": unit, "available": True, "line_count": len(matching_lines), "lines": matching_lines}


def _format_duration_ms(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value / 1000.0:,.1f}s"


def _phase_metric_index(metrics: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        indexed[str(metric.get("phase_name") or "")] = metric
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Clypt per-run diagnostics from Spanner, SQLite, and journald.")
    parser.add_argument("--run-id", required=True, help="Phase 1 / Phase24 run_id to inspect.")
    parser.add_argument(
        "--phase1-db-path",
        default=str(_DEFAULT_PHASE1_DB_PATH),
        help="Phase 1 API SQLite job store path (default: /var/lib/clypt/phase1/jobs.db).",
    )
    parser.add_argument(
        "--phase24-queue-path",
        default=None,
        help="Phase26 local queue SQLite path. Defaults to load_provider_settings().phase24_local_queue.path.",
    )
    parser.add_argument(
        "--phase1-log-lines",
        type=int,
        default=80,
        help="Tail line count for the Phase 1 job log file when available.",
    )
    parser.add_argument(
        "--journal-lines",
        type=int,
        default=120,
        help="Tail line count per systemd unit after filtering to the run_id.",
    )
    parser.add_argument(
        "--journal-unit",
        action="append",
        default=[],
        help="Additional systemd unit to inspect. May be passed multiple times.",
    )
    parser.add_argument(
        "--no-journal",
        action="store_true",
        help="Skip journald collection entirely.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Keep known noisy Cloud Monitoring exporter lines in the journal output.",
    )
    parser.add_argument(
        "--journal-since",
        default=None,
        help="Explicit journalctl --since value. Defaults to the run created_at time when available.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the full machine-readable diagnostics JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    settings = load_phase26_host_settings()
    queue_path = (
        Path(args.phase24_queue_path)
        if args.phase24_queue_path
        else Path(settings.phase24_local_queue.path)
    )
    repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)

    run_record = repository.get_run(args.run_id)
    phase24_job = repository.get_phase24_job(args.run_id)
    phase_metrics = [_phase_metric_to_json(item) for item in repository.list_phase_metrics(run_id=args.run_id)]
    phase_substeps = [
        _phase_substep_to_json(item)
        for item in repository.list_phase_substeps(run_id=args.run_id)
    ]
    nodes = repository.list_nodes(run_id=args.run_id)
    edges = repository.list_edges(run_id=args.run_id)
    candidates = repository.list_candidates(run_id=args.run_id)

    phase1_job = _read_phase1_job(Path(args.phase1_db_path), args.run_id)
    phase24_queue_row = _read_phase24_queue_row(queue_path, args.run_id)

    phase1_log_tail: list[str] = []
    if phase1_job and phase1_job.get("log_path"):
        phase1_log_tail = _tail_file(Path(str(phase1_job["log_path"])), line_count=args.phase1_log_lines)

    journal_since = args.journal_since
    if journal_since is None and run_record is not None and getattr(run_record, "created_at", None) is not None:
        journal_since = run_record.created_at.isoformat()

    journal_units = list(dict.fromkeys([*_DEFAULT_PHASE1_UNITS, *_DEFAULT_PHASE26_UNITS, *args.journal_unit]))
    journal: list[dict[str, Any]] = []
    if not args.no_journal:
        journal = [
            _run_journalctl(
                unit=unit,
                run_id=args.run_id,
                since=journal_since,
                line_count=args.journal_lines,
                include_noise=args.include_noise,
            )
            for unit in journal_units
        ]

    phase_metric_map = _phase_metric_index(phase_metrics)
    summary = {
        "run_id": args.run_id,
        "run_record": _json_ready(run_record),
        "phase24_job": _json_ready(phase24_job),
        "phase_counts": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "candidate_count": len(candidates),
            "phase_metric_count": len(phase_metrics),
            "phase_substep_count": len(phase_substeps),
        },
        "phase_metrics": phase_metrics,
        "phase_substeps": phase_substeps,
        "phase1_job": phase1_job,
        "phase24_queue_row": phase24_queue_row,
        "phase1_log_tail": phase1_log_tail,
        "journal_since": journal_since,
        "journal": journal,
        "noise_filter": {
            "enabled": not args.include_noise,
            "known_noise_substrings": list(_KNOWN_NOISE_SUBSTRINGS),
        },
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        logger.info("wrote diagnostics JSON to %s", output_path)

    print()
    print("=" * 80)
    print(f"Run diagnostics: {args.run_id}")
    print(f"  run status      : {(getattr(run_record, 'status', None) or 'missing')!s}")
    print(f"  phase24 status  : {(getattr(phase24_job, 'status', None) or 'missing')!s}")
    print(f"  nodes / edges   : {len(nodes)} / {len(edges)}")
    print(f"  candidates      : {len(candidates)}")
    print(f"  phase1 sqlite   : {'present' if phase1_job else 'missing'}")
    print(f"  phase24 queue   : {'present' if phase24_queue_row else 'missing'}")
    print()
    print("Per-phase metrics:")
    if not phase_metrics:
        print("  (none)")
    else:
        for phase_name in ("phase1_media_prep", "phase1_vibevoice_asr", "phase1_forced_alignment", "phase1_emotion2vec", "phase1_yamnet", "phase2", "phase3", "phase4", "phase24"):
            metric = phase_metric_map.get(phase_name)
            if metric is None:
                continue
            print(
                f"  {phase_name:<24} {str(metric.get('status') or '?'):<10} {_format_duration_ms(metric.get('duration_ms')):>10}"
            )
    if phase1_job:
        print()
        print("Phase1 API job:")
        print(f"  status          : {phase1_job.get('status')}")
        print(f"  current_step    : {phase1_job.get('current_step')}")
        print(f"  progress        : {phase1_job.get('progress_message')}")
        print(f"  log_path        : {phase1_job.get('log_path')}")
    if phase24_queue_row:
        print()
        print("Phase26 local queue:")
        print(f"  status          : {phase24_queue_row.get('status')}")
        print(f"  attempt_count   : {phase24_queue_row.get('attempt_count')}")
        print(f"  worker_id       : {phase24_queue_row.get('worker_id')}")
        print(f"  last_error      : {phase24_queue_row.get('last_error')}")
    if phase1_log_tail:
        print()
        print("Phase1 log tail:")
        for line in phase1_log_tail:
            print(f"  {line}")
    if journal:
        print()
        print("Filtered journald:")
        for item in journal:
            print(f"  [{item['unit']}] {item.get('line_count', 0)} matching lines")
            for line in item.get("lines", []):
                print(f"    {line}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
