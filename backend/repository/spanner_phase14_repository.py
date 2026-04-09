from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from .models import (
    ClipCandidateRecord,
    Phase24JobRecord,
    PhaseMetricRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    TimelineTurnRecord,
)
from .phase14_repository import Phase14Repository

try:  # pragma: no cover - import guard for environments without the dependency installed yet
    from google.cloud import spanner as spanner_client_module
except ImportError:  # pragma: no cover
    spanner_client_module = None

try:  # pragma: no cover - import guard for environments without google-api-core
    from google.api_core.exceptions import AlreadyExists
except ImportError:  # pragma: no cover
    class AlreadyExists(Exception):
        pass

try:  # pragma: no cover - import guard for environments without the dependency installed yet
    from google.cloud.spanner_v1 import param_types as spanner_param_types
except ImportError:  # pragma: no cover
    spanner_param_types = None

UTC = timezone.utc
_MAX_DDL_OPERATION_MESSAGES = ("already exists", "already defined")
_STRING_PARAM_TYPE = spanner_param_types.STRING if spanner_param_types is not None else "STRING"


def _json_dumps(value: Any | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: Any | None) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _coerce_datetime(value: Any | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    raise TypeError(f"unsupported datetime value: {value!r}")


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        if hasattr(row, "get"):
            return row.get(key, default)
        return default


def _row_value(row: Any, index: int, key: str) -> Any:
    try:
        return row[index]
    except Exception:
        return _row_get(row, key)


def _row_to_mapping(row: Any, columns: Sequence[str]) -> dict[str, Any]:
    return {column: _row_value(row, index, column) for index, column in enumerate(columns)}


def _stable_clip_id(record: ClipCandidateRecord) -> str:
    basis = "|".join(
        [
            record.run_id,
            ",".join(record.node_ids),
            str(record.start_ms),
            str(record.end_ms),
            record.rationale,
            ",".join(record.source_prompt_ids),
            record.seed_node_id or "",
            record.subgraph_id or "",
        ]
    )
    digest = sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"clip_{digest}"


def _clean_ddl_statements(statements: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for statement in statements:
        normalized = statement.strip().rstrip(";")
        if not normalized:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _is_expected_bootstrap_error(exc: Exception) -> bool:
    if isinstance(exc, AlreadyExists):
        return True
    message = str(exc).lower()
    return any(fragment in message for fragment in _MAX_DDL_OPERATION_MESSAGES)


def build_phase14_bootstrap_ddl() -> list[str]:
    return [
        """
        CREATE TABLE runs (
            run_id STRING(128) NOT NULL,
            source_url STRING(MAX),
            source_video_gcs_uri STRING(MAX),
            status STRING(64) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id)
        """,
        """
        CREATE TABLE timeline_turns (
            run_id STRING(128) NOT NULL,
            turn_id STRING(128) NOT NULL,
            speaker_id STRING(128) NOT NULL,
            start_ms INT64 NOT NULL,
            end_ms INT64 NOT NULL,
            word_ids ARRAY<STRING(128)>,
            transcript_text STRING(MAX) NOT NULL,
            identification_match STRING(MAX)
        ) PRIMARY KEY (run_id, turn_id)
        """,
        """
        CREATE TABLE semantic_nodes (
            run_id STRING(128) NOT NULL,
            node_id STRING(128) NOT NULL,
            node_type STRING(64) NOT NULL,
            start_ms INT64 NOT NULL,
            end_ms INT64 NOT NULL,
            source_turn_ids ARRAY<STRING(128)>,
            word_ids ARRAY<STRING(128)>,
            transcript_text STRING(MAX) NOT NULL,
            node_flags ARRAY<STRING(128)>,
            summary STRING(MAX) NOT NULL,
            evidence_json STRING(MAX),
            semantic_embedding ARRAY<FLOAT32>,
            multimodal_embedding ARRAY<FLOAT32>
        ) PRIMARY KEY (run_id, node_id)
        """,
        """
        CREATE TABLE semantic_edges (
            run_id STRING(128) NOT NULL,
            source_node_id STRING(128) NOT NULL,
            target_node_id STRING(128) NOT NULL,
            edge_type STRING(64) NOT NULL,
            rationale STRING(MAX),
            confidence FLOAT64,
            support_count INT64,
            batch_ids ARRAY<STRING(128)>
        ) PRIMARY KEY (run_id, source_node_id, target_node_id, edge_type)
        """,
        """
        CREATE TABLE clip_candidates (
            run_id STRING(128) NOT NULL,
            clip_id STRING(128) NOT NULL,
            node_ids ARRAY<STRING(128)>,
            start_ms INT64 NOT NULL,
            end_ms INT64 NOT NULL,
            score FLOAT64 NOT NULL,
            rationale STRING(MAX) NOT NULL,
            source_prompt_ids ARRAY<STRING(128)>,
            seed_node_id STRING(128),
            subgraph_id STRING(128),
            query_aligned BOOL,
            pool_rank INT64,
            score_breakdown_json STRING(MAX)
        ) PRIMARY KEY (run_id, clip_id)
        """,
        """
        CREATE TABLE phase_metrics (
            run_id STRING(128) NOT NULL,
            phase_name STRING(128) NOT NULL,
            status STRING(64) NOT NULL,
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            duration_ms FLOAT64,
            error_json STRING(MAX),
            query_version STRING(128),
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id, phase_name)
        """,
        """
        CREATE TABLE phase24_jobs (
            run_id STRING(128) NOT NULL,
            status STRING(64) NOT NULL,
            attempt_count INT64 NOT NULL,
            last_error_json STRING(MAX),
            worker_name STRING(MAX),
            task_name STRING(MAX),
            locked_at TIMESTAMP,
            updated_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id)
        """,
        """
        CREATE OR REPLACE PROPERTY GRAPH ClyptSemanticGraph
          NODE TABLES (semantic_nodes)
          EDGE TABLES (
            semantic_edges
              SOURCE KEY (run_id, source_node_id) REFERENCES semantic_nodes (run_id, node_id)
              DESTINATION KEY (run_id, target_node_id) REFERENCES semantic_nodes (run_id, node_id)
          )
        """,
    ]


def apply_ddl_statements(
    database: Any,
    statements: Iterable[str],
    *,
    operation_id: str = "phase14-bootstrap",
    timeout_s: float = 600.0,
) -> None:
    cleaned = _clean_ddl_statements(statements)
    if not cleaned:
        return
    for index, statement in enumerate(cleaned, start=1):
        try:
            operation = database.update_ddl([statement], operation_id=f"{operation_id}-{index:02d}")
        except Exception as exc:
            if _is_expected_bootstrap_error(exc):
                continue
            raise
        try:
            operation.result(timeout=timeout_s)
        except Exception as exc:
            if _is_expected_bootstrap_error(exc):
                continue
            raise


def bootstrap_phase14_schema(database: Any, *, timeout_s: float = 600.0) -> None:
    apply_ddl_statements(database, build_phase14_bootstrap_ddl(), timeout_s=timeout_s)


@dataclass(slots=True)
class SpannerPhase14Repository(Phase14Repository):
    database: Any
    ddl_operation_timeout_s: float = 600.0

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Any,
        spanner_client: Any | None = None,
    ) -> "SpannerPhase14Repository":
        client = spanner_client
        if client is None:
            if spanner_client_module is None:  # pragma: no cover - depends on optional package
                raise ImportError("google-cloud-spanner is required to create a Spanner repository")
            client = spanner_client_module.Client(project=settings.project)
        database = client.instance(settings.instance).database(settings.database)
        return cls(database=database, ddl_operation_timeout_s=settings.ddl_operation_timeout_s)

    def bootstrap_schema(self) -> None:
        bootstrap_phase14_schema(self.database, timeout_s=self.ddl_operation_timeout_s)

    def upsert_run(self, record: RunRecord) -> RunRecord:
        self._batch_upsert(
            "runs",
            [
                "run_id",
                "source_url",
                "source_video_gcs_uri",
                "status",
                "created_at",
                "updated_at",
                "metadata_json",
            ],
            [
                [
                    record.run_id,
                    record.source_url,
                    record.source_video_gcs_uri,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    _json_dumps(record.metadata),
                ]
            ],
        )
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        rows = self._query(
            """
            SELECT run_id, source_url, source_video_gcs_uri, status, created_at, updated_at, metadata_json
            FROM runs
            WHERE run_id = @run_id
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "source_url",
                "source_video_gcs_uri",
                "status",
                "created_at",
                "updated_at",
                "metadata_json",
            ),
        )
        if not rows:
            return None
        row = _row_to_mapping(
            rows[0],
            (
                "run_id",
                "source_url",
                "source_video_gcs_uri",
                "status",
                "created_at",
                "updated_at",
                "metadata_json",
            ),
        )
        return RunRecord(
            run_id=row["run_id"],
            source_url=row["source_url"],
            source_video_gcs_uri=row["source_video_gcs_uri"],
            status=row["status"],
            created_at=_coerce_datetime(row["created_at"]) or datetime.now(UTC),
            updated_at=_coerce_datetime(row["updated_at"]) or datetime.now(UTC),
            metadata=_json_loads(row["metadata_json"]) or {},
        )

    def write_timeline_turns(self, *, run_id: str, turns: Sequence[TimelineTurnRecord]) -> None:
        for turn in turns:
            _ensure_run_id_match("timeline_turns", run_id, turn.run_id)
        rows = [
            [
                run_id,
                turn.turn_id,
                turn.speaker_id,
                int(turn.start_ms),
                int(turn.end_ms),
                list(turn.word_ids),
                turn.transcript_text,
                turn.identification_match,
            ]
            for turn in turns
        ]
        self._batch_upsert(
            "timeline_turns",
            [
                "run_id",
                "turn_id",
                "speaker_id",
                "start_ms",
                "end_ms",
                "word_ids",
                "transcript_text",
                "identification_match",
            ],
            rows,
        )

    def list_timeline_turns(self, *, run_id: str) -> list[TimelineTurnRecord]:
        rows = self._query(
            """
            SELECT run_id, turn_id, speaker_id, start_ms, end_ms, word_ids, transcript_text, identification_match
            FROM timeline_turns
            WHERE run_id = @run_id
            ORDER BY start_ms ASC, turn_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "turn_id",
                "speaker_id",
                "start_ms",
                "end_ms",
                "word_ids",
                "transcript_text",
                "identification_match",
            ),
        )
        return [
            TimelineTurnRecord(
                run_id=row["run_id"],
                turn_id=row["turn_id"],
                speaker_id=row["speaker_id"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                word_ids=list(row["word_ids"] or []),
                transcript_text=row["transcript_text"],
                identification_match=row["identification_match"],
            )
            for row in rows
        ]

    def write_nodes(self, *, run_id: str, nodes: Sequence[SemanticNodeRecord]) -> None:
        for node in nodes:
            _ensure_run_id_match("semantic_nodes", run_id, node.run_id)
        rows = [
            [
                run_id,
                node.node_id,
                node.node_type,
                int(node.start_ms),
                int(node.end_ms),
                list(node.source_turn_ids),
                list(node.word_ids),
                node.transcript_text,
                list(node.node_flags),
                node.summary,
                _json_dumps(node.evidence),
                list(node.semantic_embedding) if node.semantic_embedding is not None else None,
                list(node.multimodal_embedding) if node.multimodal_embedding is not None else None,
            ]
            for node in nodes
        ]
        self._batch_upsert(
            "semantic_nodes",
            [
                "run_id",
                "node_id",
                "node_type",
                "start_ms",
                "end_ms",
                "source_turn_ids",
                "word_ids",
                "transcript_text",
                "node_flags",
                "summary",
                "evidence_json",
                "semantic_embedding",
                "multimodal_embedding",
            ],
            rows,
        )

    def list_nodes(self, *, run_id: str) -> list[SemanticNodeRecord]:
        rows = self._query(
            """
            SELECT run_id, node_id, node_type, start_ms, end_ms, source_turn_ids, word_ids,
                   transcript_text, node_flags, summary, evidence_json, semantic_embedding, multimodal_embedding
            FROM semantic_nodes
            WHERE run_id = @run_id
            ORDER BY start_ms ASC, node_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "node_id",
                "node_type",
                "start_ms",
                "end_ms",
                "source_turn_ids",
                "word_ids",
                "transcript_text",
                "node_flags",
                "summary",
                "evidence_json",
                "semantic_embedding",
                "multimodal_embedding",
            ),
        )
        return [
            SemanticNodeRecord(
                run_id=row["run_id"],
                node_id=row["node_id"],
                node_type=row["node_type"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                source_turn_ids=list(row["source_turn_ids"] or []),
                word_ids=list(row["word_ids"] or []),
                transcript_text=row["transcript_text"],
                node_flags=list(row["node_flags"] or []),
                summary=row["summary"],
                evidence=_json_loads(row["evidence_json"]) or {},
                semantic_embedding=list(row["semantic_embedding"] or [])
                if row["semantic_embedding"] is not None
                else None,
                multimodal_embedding=list(row["multimodal_embedding"] or [])
                if row["multimodal_embedding"] is not None
                else None,
            )
            for row in rows
        ]

    def write_edges(self, *, run_id: str, edges: Sequence[SemanticEdgeRecord]) -> None:
        for edge in edges:
            _ensure_run_id_match("semantic_edges", run_id, edge.run_id)
        rows = [
            [
                run_id,
                edge.source_node_id,
                edge.target_node_id,
                edge.edge_type,
                edge.rationale,
                edge.confidence,
                edge.support_count,
                list(edge.batch_ids),
            ]
            for edge in edges
        ]
        self._batch_upsert(
            "semantic_edges",
            [
                "run_id",
                "source_node_id",
                "target_node_id",
                "edge_type",
                "rationale",
                "confidence",
                "support_count",
                "batch_ids",
            ],
            rows,
        )

    def list_edges(self, *, run_id: str) -> list[SemanticEdgeRecord]:
        rows = self._query(
            """
            SELECT run_id, source_node_id, target_node_id, edge_type, rationale, confidence, support_count, batch_ids
            FROM semantic_edges
            WHERE run_id = @run_id
            ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "source_node_id",
                "target_node_id",
                "edge_type",
                "rationale",
                "confidence",
                "support_count",
                "batch_ids",
            ),
        )
        return [
            SemanticEdgeRecord(
                run_id=row["run_id"],
                source_node_id=row["source_node_id"],
                target_node_id=row["target_node_id"],
                edge_type=row["edge_type"],
                rationale=row["rationale"],
                confidence=row["confidence"],
                support_count=row["support_count"],
                batch_ids=list(row["batch_ids"] or []),
            )
            for row in rows
        ]

    def write_candidates(self, *, run_id: str, candidates: Sequence[ClipCandidateRecord]) -> None:
        for candidate in candidates:
            _ensure_run_id_match("clip_candidates", run_id, candidate.run_id)
        normalized_candidates = [
            candidate if candidate.clip_id else candidate.model_copy(update={"clip_id": _stable_clip_id(candidate)})
            for candidate in candidates
        ]
        rows = [
            [
                run_id,
                candidate.clip_id,
                list(candidate.node_ids),
                int(candidate.start_ms),
                int(candidate.end_ms),
                float(candidate.score),
                candidate.rationale,
                list(candidate.source_prompt_ids),
                candidate.seed_node_id,
                candidate.subgraph_id,
                candidate.query_aligned,
                candidate.pool_rank,
                _json_dumps(candidate.score_breakdown),
            ]
            for candidate in normalized_candidates
        ]
        self._batch_upsert(
            "clip_candidates",
            [
                "run_id",
                "clip_id",
                "node_ids",
                "start_ms",
                "end_ms",
                "score",
                "rationale",
                "source_prompt_ids",
                "seed_node_id",
                "subgraph_id",
                "query_aligned",
                "pool_rank",
                "score_breakdown_json",
            ],
            rows,
        )

    def list_candidates(self, *, run_id: str) -> list[ClipCandidateRecord]:
        rows = self._query(
            """
            SELECT run_id, clip_id, node_ids, start_ms, end_ms, score, rationale, source_prompt_ids,
                   seed_node_id, subgraph_id, query_aligned, pool_rank, score_breakdown_json
            FROM clip_candidates
            WHERE run_id = @run_id
            ORDER BY COALESCE(pool_rank, 2147483647) ASC, score DESC, clip_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "clip_id",
                "node_ids",
                "start_ms",
                "end_ms",
                "score",
                "rationale",
                "source_prompt_ids",
                "seed_node_id",
                "subgraph_id",
                "query_aligned",
                "pool_rank",
                "score_breakdown_json",
            ),
        )
        return [
            ClipCandidateRecord(
                run_id=row["run_id"],
                clip_id=row["clip_id"],
                node_ids=list(row["node_ids"] or []),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                score=float(row["score"]),
                rationale=row["rationale"],
                source_prompt_ids=list(row["source_prompt_ids"] or []),
                seed_node_id=row["seed_node_id"],
                subgraph_id=row["subgraph_id"],
                query_aligned=row["query_aligned"],
                pool_rank=int(row["pool_rank"]) if row["pool_rank"] is not None else None,
                score_breakdown=_json_loads(row["score_breakdown_json"]),
            )
            for row in rows
        ]

    def write_phase_metric(self, record: PhaseMetricRecord) -> PhaseMetricRecord:
        self._batch_upsert(
            "phase_metrics",
            [
                "run_id",
                "phase_name",
                "status",
                "started_at",
                "ended_at",
                "duration_ms",
                "error_json",
                "query_version",
                "metadata_json",
            ],
            [
                [
                    record.run_id,
                    record.phase_name,
                    record.status,
                    record.started_at,
                    record.ended_at,
                    record.duration_ms,
                    _json_dumps(record.error_payload),
                    record.query_version,
                    _json_dumps(record.metadata),
                ]
            ],
        )
        return record

    def list_phase_metrics(self, *, run_id: str) -> list[PhaseMetricRecord]:
        rows = self._query(
            """
            SELECT run_id, phase_name, status, started_at, ended_at, duration_ms, error_json, query_version, metadata_json
            FROM phase_metrics
            WHERE run_id = @run_id
            ORDER BY started_at ASC, phase_name ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "phase_name",
                "status",
                "started_at",
                "ended_at",
                "duration_ms",
                "error_json",
                "query_version",
                "metadata_json",
            ),
        )
        return [
            PhaseMetricRecord(
                run_id=row["run_id"],
                phase_name=row["phase_name"],
                status=row["status"],
                started_at=_coerce_datetime(row["started_at"]) or datetime.now(UTC),
                ended_at=_coerce_datetime(row["ended_at"]),
                duration_ms=float(row["duration_ms"]) if row["duration_ms"] is not None else None,
                error_payload=_json_loads(row["error_json"]),
                query_version=row["query_version"],
                metadata=_json_loads(row["metadata_json"]) or {},
            )
            for row in rows
        ]

    def upsert_phase24_job(self, record: Phase24JobRecord) -> Phase24JobRecord:
        self._batch_upsert(
            "phase24_jobs",
            [
                "run_id",
                "status",
                "attempt_count",
                "last_error_json",
                "worker_name",
                "task_name",
                "locked_at",
                "updated_at",
                "completed_at",
                "metadata_json",
            ],
            [
                [
                    record.run_id,
                    record.status,
                    int(record.attempt_count),
                    _json_dumps(record.last_error),
                    record.worker_name,
                    record.task_name,
                    record.locked_at,
                    record.updated_at,
                    record.completed_at,
                    _json_dumps(record.metadata),
                ]
            ],
        )
        return record

    def acquire_phase24_job_lease(
        self,
        *,
        run_id: str,
        job_id: str,
        worker_name: str,
        attempt: int,
        query_version: str | None,
        running_timeout_s: int = 1800,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        attempt_count = max(1, int(attempt))
        columns = (
            "run_id",
            "status",
            "attempt_count",
            "last_error_json",
            "worker_name",
            "task_name",
            "locked_at",
            "updated_at",
            "completed_at",
            "metadata_json",
        )
        result: dict[str, Any] = {}

        def _txn(txn: Any) -> None:
            rows = list(
                txn.execute_sql(
                    """
                    SELECT run_id, status, attempt_count, last_error_json, worker_name, task_name, locked_at,
                           updated_at, completed_at, metadata_json
                    FROM phase24_jobs
                    WHERE run_id = @run_id
                    """,
                    params={"run_id": run_id},
                    param_types={"run_id": _STRING_PARAM_TYPE},
                )
            )
            if rows:
                row = _row_to_mapping(rows[0], columns)
                status = str(row["status"] or "")
                if status == "running":
                    locked_at = _coerce_datetime(row["locked_at"])
                    is_stale = locked_at is None or (now - locked_at).total_seconds() > float(running_timeout_s)
                    if not is_stale:
                        result.update(
                            {
                                "acquired": False,
                                "status": "running",
                                "attempt_count": int(row["attempt_count"] or 0),
                                "task_name": row["task_name"],
                            }
                        )
                        return
                elif status == "succeeded":
                    result.update(
                        {
                            "acquired": False,
                            "status": "succeeded",
                            "attempt_count": int(row["attempt_count"] or 0),
                            "task_name": row["task_name"],
                        }
                    )
                    return
                metadata = _json_loads(row["metadata_json"]) or {}
                if query_version:
                    metadata["query_version"] = query_version
                attempt_value = max(attempt_count, int(row["attempt_count"] or 0))
                txn.insert_or_update(
                    table="phase24_jobs",
                    columns=list(columns),
                    values=[
                        [
                            run_id,
                            "running",
                            attempt_value,
                            None,
                            worker_name,
                            job_id,
                            now,
                            now,
                            None,
                            _json_dumps(metadata),
                        ]
                    ],
                )
                result.update(
                    {
                        "acquired": True,
                        "status": "running",
                        "attempt_count": attempt_value,
                        "task_name": job_id,
                    }
                )
                return

            metadata: dict[str, Any] = {}
            if query_version:
                metadata["query_version"] = query_version
            txn.insert_or_update(
                table="phase24_jobs",
                columns=list(columns),
                values=[
                    [
                        run_id,
                        "running",
                        attempt_count,
                        None,
                        worker_name,
                        job_id,
                        now,
                        now,
                        None,
                        _json_dumps(metadata),
                    ]
                ],
            )
            result.update(
                {
                    "acquired": True,
                    "status": "running",
                    "attempt_count": attempt_count,
                    "task_name": job_id,
                }
            )

        run_in_transaction = getattr(self.database, "run_in_transaction", None)
        if callable(run_in_transaction):
            run_in_transaction(_txn)
        else:
            current = self.get_phase24_job(run_id)
            if current is not None and current.status in {"running", "succeeded"}:
                if current.status == "running":
                    locked_at = current.locked_at
                    is_stale = locked_at is None or (now - locked_at).total_seconds() > float(running_timeout_s)
                    if is_stale:
                        current = current.model_copy(update={"status": "failed"})
                    else:
                        return {
                            "acquired": False,
                            "status": current.status,
                            "attempt_count": int(current.attempt_count or 0),
                            "task_name": current.task_name,
                        }
                if current.status == "succeeded":
                    return {
                        "acquired": False,
                        "status": current.status,
                        "attempt_count": int(current.attempt_count or 0),
                        "task_name": current.task_name,
                    }
            metadata = dict(current.metadata if current is not None else {})
            if query_version:
                metadata["query_version"] = query_version
            updated = self.upsert_phase24_job(
                Phase24JobRecord(
                    run_id=run_id,
                    status="running",
                    attempt_count=max(attempt_count, int(current.attempt_count or 0)) if current else attempt_count,
                    last_error=None,
                    worker_name=worker_name,
                    task_name=job_id,
                    locked_at=now,
                    updated_at=now,
                    completed_at=None,
                    metadata=metadata,
                )
            )
            return {
                "acquired": True,
                "status": updated.status,
                "attempt_count": updated.attempt_count,
                "task_name": updated.task_name,
            }

        return result

    def get_phase24_job(self, run_id: str) -> Phase24JobRecord | None:
        rows = self._query(
            """
            SELECT run_id, status, attempt_count, last_error_json, worker_name, task_name, locked_at, updated_at,
                   completed_at, metadata_json
            FROM phase24_jobs
            WHERE run_id = @run_id
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "status",
                "attempt_count",
                "last_error_json",
                "worker_name",
                "task_name",
                "locked_at",
                "updated_at",
                "completed_at",
                "metadata_json",
            ),
        )
        if not rows:
            return None
        row = _row_to_mapping(
            rows[0],
            (
                "run_id",
                "status",
                "attempt_count",
                "last_error_json",
                "worker_name",
                "task_name",
                "locked_at",
                "updated_at",
                "completed_at",
                "metadata_json",
            ),
        )
        return Phase24JobRecord(
            run_id=row["run_id"],
            status=row["status"],
            attempt_count=int(row["attempt_count"] or 0),
            last_error=_json_loads(row["last_error_json"]),
            worker_name=row["worker_name"],
            task_name=row["task_name"],
            locked_at=_coerce_datetime(row["locked_at"]),
            updated_at=_coerce_datetime(row["updated_at"]) or datetime.now(UTC),
            completed_at=_coerce_datetime(row["completed_at"]),
            metadata=_json_loads(row["metadata_json"]) or {},
        )

    def _batch_upsert(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        with self.database.batch() as batch:
            batch.insert_or_update(table=table, columns=list(columns), values=[list(row) for row in rows])

    def _query(
        self,
        statement: str,
        params: dict[str, Any],
        *,
        param_types: dict[str, Any] | None = None,
        columns: Sequence[str] | None = None,
    ) -> list[Any]:
        with self.database.snapshot() as snapshot:
            result = snapshot.execute_sql(statement, params=params, param_types=param_types)
            rows = list(result)
        if columns is None:
            return rows
        return [_row_to_mapping(row, columns) for row in rows]


def _ensure_run_id_match(table_name: str, expected_run_id: str, actual_run_id: str) -> None:
    if expected_run_id != actual_run_id:
        raise ValueError(
            f"run_id mismatch for {table_name}: expected {expected_run_id!r}, got {actual_run_id!r}"
        )


__all__ = [
    "SpannerPhase14Repository",
    "apply_ddl_statements",
    "bootstrap_phase14_schema",
    "build_phase14_bootstrap_ddl",
]
