"""Ad-hoc Phase 4 replay helper for a persisted run.

This replays Phase 4 under a fresh run id by:

1. loading the persisted Phase 1 handoff JSON from GCS
2. rebuilding the canonical timeline via ``run_phase_1``
3. loading Phase 2 nodes + Phase 3 edges from the source run
4. optionally cloning those nodes/edges into the replay run id
5. invoking ``run_phase_4`` directly

The intended use case is exactly the failure mode we saw on
``job_5ab9c5a9837b41fe8eb4ddf566c554f8``: Phase 4 failed after Phase 3 had
already succeeded and all required upstream state was durable.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import uuid
from typing import Any

from backend.phase1_runtime.payloads import Phase1SidecarOutputs
from backend.providers import load_phase26_host_settings
from backend.repository import (
    Phase24JobRecord,
    PhaseMetricRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    SpannerPhase14Repository,
)
from backend.runtime.phase24_worker_app import build_default_phase24_worker_service

logger = logging.getLogger("tmp_replay_phase4")
UTC = timezone.utc


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _clone_nodes(*, run_id: str, nodes: list[Any]) -> list[SemanticNodeRecord]:
    return [
        SemanticNodeRecord(
            run_id=run_id,
            node_id=node.node_id,
            node_type=node.node_type,
            start_ms=node.start_ms,
            end_ms=node.end_ms,
            source_turn_ids=list(node.source_turn_ids),
            word_ids=list(node.word_ids),
            transcript_text=node.transcript_text,
            node_flags=list(node.node_flags),
            summary=node.summary,
            evidence=_json_ready(deepcopy(node.evidence)),
            semantic_embedding=list(node.semantic_embedding) if node.semantic_embedding is not None else None,
            multimodal_embedding=list(node.multimodal_embedding) if node.multimodal_embedding is not None else None,
        )
        for node in nodes
    ]


def _clone_edges(*, run_id: str, edges: list[Any]) -> list[SemanticEdgeRecord]:
    return [
        SemanticEdgeRecord(
            run_id=run_id,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            edge_type=edge.edge_type,
            rationale=edge.rationale,
            confidence=edge.confidence,
            support_count=edge.support_count,
            batch_ids=list(edge.batch_ids),
        )
        for edge in edges
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Phase 4 from a persisted source run under a new replay run id.")
    parser.add_argument("--source-run-id", required=True, help="Existing run_id whose Phase 1/2/3 state should be replayed.")
    parser.add_argument(
        "--replay-run-id",
        default=None,
        help="Optional explicit replay run id. Defaults to replay_phase4_<timestamp>_<suffix>.",
    )
    parser.add_argument(
        "--no-clone-phase23-state",
        action="store_true",
        help="Do not write cloned nodes/edges to the replay run id before Phase 4.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the replay summary JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    settings = load_phase26_host_settings()
    repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)
    service = build_default_phase24_worker_service(settings=settings)
    runner = replace(
        service.runner,
        query_version=settings.phase24_worker.query_version,
        log_event=lambda **event: logger.info(json.dumps(event, default=str, separators=(",", ":"))),
    )

    source_run = repository.get_run(args.source_run_id)
    if source_run is None:
        raise SystemExit(f"source run_id {args.source_run_id!r} not found in Spanner runs")
    source_job = repository.get_phase24_job(args.source_run_id)
    source_metadata = dict(source_run.metadata or {})
    phase1_outputs_gcs_uri = str(source_metadata.get("phase1_outputs_gcs_uri") or "").strip()
    if not phase1_outputs_gcs_uri:
        raise SystemExit(
            f"source run_id {args.source_run_id!r} is missing metadata.phase1_outputs_gcs_uri; "
            "cannot replay Phase 4 accurately."
        )

    suffix = args.source_run_id.split("_")[-1][:12] or uuid.uuid4().hex[:8]
    replay_run_id = args.replay_run_id or (
        f"replay_phase4_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{suffix}"
    )
    replay_job_id = f"replay-phase4-{uuid.uuid4().hex[:12]}"

    temp_dir = tempfile.TemporaryDirectory(prefix=f"phase4-replay-{replay_run_id}-")
    handoff_path = Path(temp_dir.name) / "phase1_outputs.json"
    try:
        runner.storage_client.download_file(
            gcs_uri=phase1_outputs_gcs_uri,
            local_path=handoff_path,
        )
        phase1_outputs = Phase1SidecarOutputs.model_validate(
            json.loads(handoff_path.read_text(encoding="utf-8"))
        )
    finally:
        # Keep the temp dir open until the replay finishes; the file may still
        # be useful for debugging and the payload object already lives in memory.
        pass

    nodes = runner._load_resume_nodes(run_id=args.source_run_id)
    edges = runner._load_resume_edges(run_id=args.source_run_id)
    if not nodes:
        raise SystemExit(f"source run_id {args.source_run_id!r} has no persisted Phase 2 nodes")
    if not edges:
        raise SystemExit(f"source run_id {args.source_run_id!r} has no persisted Phase 3 edges")

    now = datetime.now(UTC)
    repository.upsert_run(
        RunRecord(
            run_id=replay_run_id,
            source_url=source_run.source_url,
            source_video_gcs_uri=source_run.source_video_gcs_uri,
            status="PHASE24_RUNNING",
            created_at=now,
            updated_at=now,
            metadata={
                "source_run_id": args.source_run_id,
                "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
                "query_version": settings.phase24_worker.query_version,
                "replay_mode": "phase4_only",
            },
        )
    )
    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id=replay_run_id,
            status="running",
            attempt_count=1,
            last_error=None,
            worker_name=service.service_name,
            task_name=replay_job_id,
            locked_at=now,
            updated_at=now,
            completed_at=None,
            metadata={
                "source_run_id": args.source_run_id,
                "query_version": settings.phase24_worker.query_version,
                "replay_mode": "phase4_only",
            },
        )
    )

    if not args.no_clone_phase23_state:
        repository.write_nodes(run_id=replay_run_id, nodes=_clone_nodes(run_id=replay_run_id, nodes=nodes))
        repository.write_edges(run_id=replay_run_id, edges=_clone_edges(run_id=replay_run_id, edges=edges))

    paths = runner.build_run_paths(run_id=replay_run_id)
    phase1 = runner.run_phase_1(paths=paths, phase1_outputs=phase1_outputs)
    phase4_started_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        phase4_summary = runner.run_phase_4(
            run_id=replay_run_id,
            job_id=replay_job_id,
            attempt=1,
            paths=paths,
            source_url=source_run.source_url or phase1_outputs.phase1_audio.source_audio,
            canonical_timeline=phase1["canonical_timeline"],
            nodes=nodes,
            edges=edges,
            extra_prompt_texts=[],
            signal_output=None,
        )
    except Exception as exc:
        ended_at = datetime.now(UTC)
        repository.write_phase_metric(
            PhaseMetricRecord(
                run_id=replay_run_id,
                phase_name="phase4",
                status="failed",
                started_at=phase4_started_at,
                ended_at=ended_at,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
                query_version=settings.phase24_worker.query_version,
                metadata={"source_run_id": args.source_run_id, "replay_mode": "phase4_only"},
            )
        )
        repository.upsert_phase24_job(
            Phase24JobRecord(
                run_id=replay_run_id,
                status="failed",
                attempt_count=1,
                last_error={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
                worker_name=service.service_name,
                task_name=replay_job_id,
                locked_at=phase4_started_at,
                updated_at=ended_at,
                completed_at=ended_at,
                metadata={
                    "source_run_id": args.source_run_id,
                    "query_version": settings.phase24_worker.query_version,
                    "replay_mode": "phase4_only",
                },
            )
        )
        repository.upsert_run(
            RunRecord(
                run_id=replay_run_id,
                source_url=source_run.source_url,
                source_video_gcs_uri=source_run.source_video_gcs_uri,
                status="FAILED",
                created_at=now,
                updated_at=ended_at,
                metadata={
                    "source_run_id": args.source_run_id,
                    "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
                    "query_version": settings.phase24_worker.query_version,
                    "replay_mode": "phase4_only",
                    "replay_job_id": replay_job_id,
                },
            )
        )
        raise

    ended_at = datetime.now(UTC)
    phase4_duration_ms = (time.perf_counter() - started) * 1000.0
    repository.write_phase_metric(
        PhaseMetricRecord(
            run_id=replay_run_id,
            phase_name="phase4",
            status="succeeded",
            started_at=phase4_started_at,
            ended_at=ended_at,
            duration_ms=phase4_duration_ms,
            error_payload=None,
            query_version=settings.phase24_worker.query_version,
            metadata={
                "source_run_id": args.source_run_id,
                "replay_mode": "phase4_only",
                "candidate_count": int(phase4_summary.get("final_candidate_count") or 0),
                "seed_count": int(phase4_summary.get("seed_count") or 0),
                "subgraph_count": int(phase4_summary.get("subgraph_count") or 0),
            },
        )
    )
    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id=replay_run_id,
            status="succeeded",
            attempt_count=1,
            last_error=None,
            worker_name=service.service_name,
            task_name=replay_job_id,
            locked_at=phase4_started_at,
            updated_at=ended_at,
            completed_at=ended_at,
            metadata={
                "source_run_id": args.source_run_id,
                "query_version": settings.phase24_worker.query_version,
                "replay_mode": "phase4_only",
            },
        )
    )
    repository.upsert_run(
        RunRecord(
            run_id=replay_run_id,
            source_url=source_run.source_url,
            source_video_gcs_uri=source_run.source_video_gcs_uri,
            status="PHASE24_DONE",
            created_at=now,
            updated_at=ended_at,
            metadata={
                "source_run_id": args.source_run_id,
                "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
                "query_version": settings.phase24_worker.query_version,
                "replay_mode": "phase4_only",
                "replay_job_id": replay_job_id,
            },
        )
    )

    summary = {
        "source_run_id": args.source_run_id,
        "source_phase24_job": _json_ready(source_job),
        "replay_run_id": replay_run_id,
        "replay_job_id": replay_job_id,
        "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
        "clone_phase23_state": not args.no_clone_phase23_state,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "phase4_duration_ms": phase4_duration_ms,
        "phase4_summary": _json_ready(phase4_summary),
        "artifacts_dir": str(paths.root / paths.run_id),
    }
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("wrote replay summary JSON to %s", output_path)

    print()
    print("=" * 80)
    print(f"Phase 4 replay succeeded")
    print(f"  source_run_id   : {args.source_run_id}")
    print(f"  replay_run_id   : {replay_run_id}")
    print(f"  replay_job_id   : {replay_job_id}")
    print(f"  node_count      : {len(nodes)}")
    print(f"  edge_count      : {len(edges)}")
    print(f"  candidate_count : {phase4_summary.get('final_candidate_count')}")
    print(f"  phase4_duration : {phase4_duration_ms / 1000.0:,.1f}s")
    print(f"  artifacts_dir   : {paths.root / paths.run_id}")
    print("=" * 80)

    with suppress(Exception):
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
