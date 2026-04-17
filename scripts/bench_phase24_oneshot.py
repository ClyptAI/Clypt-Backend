"""One-shot Phase 2-4 replay bench.

Reuses an existing PHASE24_DONE run's source_video_gcs_uri + phase1_outputs_gcs_uri
from Spanner, rebuilds a fresh run_id under it, and runs
``Phase24WorkerService.handle_task`` inline. Captures wall-clock and Spanner
phase_metrics/substeps breakdown for latency analysis.

Intended for ad-hoc benchmarking on the Phase 2-4 host (currently the H200
droplet). Node-media prep runs in-process on the same host — Phase 2-4 is a
single-host workload.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from google.cloud import spanner  # noqa: E402

from backend.providers import load_provider_settings  # noqa: E402
from backend.runtime.phase24_worker_app import (  # noqa: E402
    Phase24TaskPayload,
    build_default_phase24_worker_service,
)

logger = logging.getLogger("bench_phase24")


def _lookup_source_run(
    *,
    project: str,
    instance: str,
    database: str,
    run_id: str,
) -> dict[str, Any]:
    client = spanner.Client(project=project)
    db = client.instance(instance).database(database)
    with db.snapshot() as snap:
        rows = list(
            snap.execute_sql(
                "SELECT run_id, status, source_url, source_video_gcs_uri, metadata_json "
                "FROM runs WHERE run_id = @r",
                params={"r": run_id},
                param_types={"r": spanner.param_types.STRING},
            )
        )
    if not rows:
        raise SystemExit(f"source run_id {run_id!r} not found in Spanner")
    row = rows[0]
    meta: dict[str, Any] = {}
    if row[4]:
        try:
            meta = json.loads(row[4])
        except Exception:
            meta = {}
    p1_uri = meta.get("phase1_outputs_gcs_uri") if isinstance(meta, dict) else None
    if not p1_uri:
        raise SystemExit(
            f"source run_id {run_id!r} has no phase1_outputs_gcs_uri in metadata_json; "
            "pick a run that completed Phase 1 (status PHASE24_DONE or PHASE24_QUEUED)."
        )
    return {
        "run_id": row[0],
        "status": row[1],
        "source_url": row[2],
        "source_video_gcs_uri": row[3],
        "phase1_outputs_gcs_uri": p1_uri,
        "query_version": meta.get("query_version"),
    }


def _format_metric(rec: Any) -> str:
    duration = rec.duration_ms
    dur_s = f"{duration / 1000.0:,.1f}s" if duration is not None else "?"
    return f"  {rec.phase_name:<10} {rec.status:<10} {dur_s:>10}"


def _format_substep(rec: Any) -> str:
    duration = rec.duration_ms
    dur_s = f"{duration / 1000.0:,.1f}s" if duration is not None else "?"
    return f"    {rec.phase_name}/{rec.step_name}/{rec.step_key:<30} {rec.status:<10} {dur_s:>10}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-run-id",
        required=True,
        help="Existing Spanner run_id to replay Phase 2-4 against (needs phase1_outputs_gcs_uri).",
    )
    ap.add_argument(
        "--bench-run-id",
        default=None,
        help="Optional explicit bench run_id. Defaults to bench_<ts>_<src-prefix>.",
    )
    ap.add_argument(
        "--phase3-long-range-top-k",
        type=int,
        default=None,
        help="Override runtime control. Defaults to env.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    settings = load_provider_settings()

    src = _lookup_source_run(
        project=settings.spanner.project,
        instance=settings.spanner.instance,
        database=settings.spanner.database,
        run_id=args.source_run_id,
    )
    logger.info("source run metadata: %s", json.dumps(src, indent=2, default=str))

    now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = args.source_run_id.split("_")[-1][:12] or uuid.uuid4().hex[:8]
    bench_run_id = args.bench_run_id or f"bench_{now_ts}_{short}"

    payload = Phase24TaskPayload(
        run_id=bench_run_id,
        source_url=src["source_url"],
        source_video_gcs_uri=src["source_video_gcs_uri"],
        phase1_outputs_gcs_uri=src["phase1_outputs_gcs_uri"],
        phase3_long_range_top_k=(
            args.phase3_long_range_top_k
            if args.phase3_long_range_top_k is not None
            else int(os.environ.get("CLYPT_PHASE3_LONG_RANGE_TOP_K") or 2)
        ),
        query_version=src["query_version"] or None,
    )
    job_id = f"bench-job-{uuid.uuid4().hex[:12]}"

    service = build_default_phase24_worker_service()

    logger.info("starting handle_task bench: run_id=%s job_id=%s", bench_run_id, job_id)
    t0 = time.perf_counter()
    wall_start = datetime.now(timezone.utc)
    try:
        result = service.handle_task(payload, job_id=job_id, attempt=1)
        final_status = "ok"
        err_msg: str | None = None
    except Exception as exc:
        result = {"run_id": bench_run_id, "status": "exception"}
        final_status = "exception"
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("handle_task raised")
    wall_end = datetime.now(timezone.utc)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics = service.repository.list_phase_metrics(run_id=bench_run_id)
    substeps = service.repository.list_phase_substeps(run_id=bench_run_id)

    print()
    print("=" * 80)
    print(f"Phase 2-4 bench result: run_id={bench_run_id}")
    print(f"  status          : {result.get('status')!r}")
    print(f"  wall clock      : {wall_ms/1000.0:,.1f}s  (start {wall_start.isoformat()} end {wall_end.isoformat()})")
    print(f"  final           : {final_status}")
    if err_msg:
        print(f"  error           : {err_msg}")
    print()
    print("Per-phase metrics:")
    if metrics:
        for m in metrics:
            print(_format_metric(m))
    else:
        print("  (none written — worker short-circuited)")
    print()
    if substeps:
        print("Substeps:")
        for s in substeps:
            print(_format_substep(s))
    print("=" * 80)

    summary_md = result.get("summary") or {}
    meta = summary_md.get("metadata") if isinstance(summary_md, dict) else None
    if isinstance(meta, dict):
        print("\nSummary metadata (truncated):")
        print(json.dumps({k: v for k, v in meta.items() if k != "full_debug"}, indent=2, default=str)[:4000])

    return 0 if final_status == "ok" and result.get("status") not in {"exception", "max_attempts_exceeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
