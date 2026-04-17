from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from .models import (
    CandidateSignalLinkRecord,
    ClipCandidateRecord,
    ExternalSignalClusterRecord,
    ExternalSignalRecord,
    Phase24JobRecord,
    PhaseMetricRecord,
    PhaseSubstepRecord,
    PromptSourceLinkRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    NodeSignalLinkRecord,
    SubgraphProvenanceRecord,
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
_MAX_DDL_OPERATION_MESSAGES = ("already exists", "already defined", "duplicate name in schema")
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
    except KeyError:
        if hasattr(row, "get"):
            return row.get(key, default)
        return default


def _row_value(row: Any, index: int, key: str) -> Any:
    try:
        return row[index]
    except IndexError:
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
            score_breakdown_json STRING(MAX),
            external_signal_score FLOAT64,
            agreement_bonus FLOAT64,
            external_attribution_json STRING(MAX)
        ) PRIMARY KEY (run_id, clip_id)
        """,
        """
        CREATE TABLE external_signals (
            run_id STRING(128) NOT NULL,
            signal_id STRING(128) NOT NULL,
            signal_type STRING(32) NOT NULL,
            source_platform STRING(32) NOT NULL,
            source_id STRING(MAX) NOT NULL,
            author_id STRING(MAX),
            text STRING(MAX) NOT NULL,
            engagement_score FLOAT64 NOT NULL,
            published_at TIMESTAMP,
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id, signal_id)
        """,
        """
        CREATE TABLE external_signal_clusters (
            run_id STRING(128) NOT NULL,
            cluster_id STRING(128) NOT NULL,
            cluster_type STRING(32) NOT NULL,
            summary_text STRING(MAX) NOT NULL,
            member_signal_ids ARRAY<STRING(128)>,
            cluster_weight FLOAT64 NOT NULL,
            embedding ARRAY<FLOAT32>,
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id, cluster_id)
        """,
        """
        CREATE TABLE node_signal_links (
            run_id STRING(128) NOT NULL,
            node_id STRING(128) NOT NULL,
            cluster_id STRING(128) NOT NULL,
            link_type STRING(16) NOT NULL,
            hop_distance INT64 NOT NULL,
            time_offset_ms INT64 NOT NULL,
            similarity FLOAT64 NOT NULL,
            link_score FLOAT64 NOT NULL,
            evidence_json STRING(MAX),
            CONSTRAINT fk_node_signal_links_node FOREIGN KEY (run_id, node_id)
              REFERENCES semantic_nodes (run_id, node_id),
            CONSTRAINT fk_node_signal_links_cluster FOREIGN KEY (run_id, cluster_id)
              REFERENCES external_signal_clusters (run_id, cluster_id)
        ) PRIMARY KEY (run_id, node_id, cluster_id)
        """,
        """
        CREATE TABLE candidate_signal_links (
            run_id STRING(128) NOT NULL,
            clip_id STRING(128) NOT NULL,
            cluster_id STRING(128) NOT NULL,
            cluster_type STRING(32) NOT NULL,
            aggregated_link_score FLOAT64 NOT NULL,
            coverage_ms INT64 NOT NULL,
            direct_node_count INT64 NOT NULL,
            inferred_node_count INT64 NOT NULL,
            agreement_flags ARRAY<STRING(32)>,
            bonus_applied FLOAT64 NOT NULL,
            evidence_json STRING(MAX),
            CONSTRAINT fk_candidate_signal_links_clip FOREIGN KEY (run_id, clip_id)
              REFERENCES clip_candidates (run_id, clip_id),
            CONSTRAINT fk_candidate_signal_links_cluster FOREIGN KEY (run_id, cluster_id)
              REFERENCES external_signal_clusters (run_id, cluster_id)
        ) PRIMARY KEY (run_id, clip_id, cluster_id)
        """,
        """
        CREATE TABLE prompt_source_links (
            run_id STRING(128) NOT NULL,
            prompt_id STRING(128) NOT NULL,
            prompt_source_type STRING(16) NOT NULL,
            source_cluster_id STRING(128),
            source_cluster_type STRING(32),
            metadata_json STRING(MAX),
            CONSTRAINT fk_prompt_source_links_cluster FOREIGN KEY (run_id, source_cluster_id)
              REFERENCES external_signal_clusters (run_id, cluster_id)
        ) PRIMARY KEY (run_id, prompt_id)
        """,
        """
        CREATE TABLE subgraph_provenance (
            run_id STRING(128) NOT NULL,
            subgraph_id STRING(128) NOT NULL,
            seed_source_set ARRAY<STRING(16)>,
            seed_prompt_ids ARRAY<STRING(128)>,
            source_cluster_ids ARRAY<STRING(128)>,
            support_summary_json STRING(MAX),
            canonical_selected BOOL NOT NULL,
            dedupe_overlap_ratio FLOAT64,
            selection_reason STRING(128),
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id, subgraph_id)
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
        CREATE TABLE phase_substeps (
            run_id STRING(128) NOT NULL,
            phase_name STRING(128) NOT NULL,
            step_name STRING(128) NOT NULL,
            step_key STRING(256) NOT NULL,
            status STRING(64) NOT NULL,
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            duration_ms FLOAT64,
            error_json STRING(MAX),
            query_version STRING(128),
            metadata_json STRING(MAX)
        ) PRIMARY KEY (run_id, phase_name, step_name, step_key)
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
            safe_operation_id = f"{operation_id}_{index:02d}".replace("-", "_")
            operation = database.update_ddl([statement], operation_id=safe_operation_id)
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
        normalized_candidates = sorted(
            normalized_candidates,
            key=lambda candidate: (
                candidate.pool_rank if candidate.pool_rank is not None else 2**31 - 1,
                -float(candidate.score),
                candidate.clip_id or "",
            ),
        )
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
                candidate.external_signal_score,
                candidate.agreement_bonus,
                _json_dumps(candidate.external_attribution_json),
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
                "external_signal_score",
                "agreement_bonus",
                "external_attribution_json",
            ],
            rows,
        )

    def list_candidates(self, *, run_id: str) -> list[ClipCandidateRecord]:
        rows = self._query(
            """
            SELECT run_id, clip_id, node_ids, start_ms, end_ms, score, rationale, source_prompt_ids,
                   seed_node_id, subgraph_id, query_aligned, pool_rank, score_breakdown_json,
                   external_signal_score, agreement_bonus, external_attribution_json
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
                "external_signal_score",
                "agreement_bonus",
                "external_attribution_json",
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
                external_signal_score=float(row["external_signal_score"])
                if row["external_signal_score"] is not None
                else None,
                agreement_bonus=float(row["agreement_bonus"]) if row["agreement_bonus"] is not None else None,
                external_attribution_json=_json_loads(row["external_attribution_json"]),
            )
            for row in rows
        ]

    def write_external_signals(self, *, run_id: str, signals: Sequence[ExternalSignalRecord]) -> None:
        for signal in signals:
            _ensure_run_id_match("external_signals", run_id, signal.run_id)
        normalized_signals = sorted(signals, key=lambda signal: signal.signal_id)
        rows = [
            [
                run_id,
                signal.signal_id,
                signal.signal_type,
                signal.source_platform,
                signal.source_id,
                signal.author_id,
                signal.text,
                float(signal.engagement_score),
                signal.published_at,
                _json_dumps(signal.metadata),
            ]
            for signal in normalized_signals
        ]
        self._batch_upsert(
            "external_signals",
            [
                "run_id",
                "signal_id",
                "signal_type",
                "source_platform",
                "source_id",
                "author_id",
                "text",
                "engagement_score",
                "published_at",
                "metadata_json",
            ],
            rows,
        )

    def list_external_signals(self, *, run_id: str) -> list[ExternalSignalRecord]:
        rows = self._query(
            """
            SELECT run_id, signal_id, signal_type, source_platform, source_id, author_id, text,
                   engagement_score, published_at, metadata_json
            FROM external_signals
            WHERE run_id = @run_id
            ORDER BY signal_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "signal_id",
                "signal_type",
                "source_platform",
                "source_id",
                "author_id",
                "text",
                "engagement_score",
                "published_at",
                "metadata_json",
            ),
        )
        return [
            ExternalSignalRecord(
                run_id=row["run_id"],
                signal_id=row["signal_id"],
                signal_type=row["signal_type"],
                source_platform=row["source_platform"],
                source_id=row["source_id"],
                author_id=row["author_id"],
                text=row["text"],
                engagement_score=float(row["engagement_score"]),
                published_at=_coerce_datetime(row["published_at"]),
                metadata=_json_loads(row["metadata_json"]) or {},
            )
            for row in rows
        ]

    def write_external_signal_clusters(
        self, *, run_id: str, clusters: Sequence[ExternalSignalClusterRecord]
    ) -> None:
        for cluster in clusters:
            _ensure_run_id_match("external_signal_clusters", run_id, cluster.run_id)
        normalized_clusters = sorted(clusters, key=lambda cluster: cluster.cluster_id)
        rows = [
            [
                run_id,
                cluster.cluster_id,
                cluster.cluster_type,
                cluster.summary_text,
                list(cluster.member_signal_ids),
                float(cluster.cluster_weight),
                list(cluster.embedding),
                _json_dumps(cluster.metadata),
            ]
            for cluster in normalized_clusters
        ]
        self._batch_upsert(
            "external_signal_clusters",
            [
                "run_id",
                "cluster_id",
                "cluster_type",
                "summary_text",
                "member_signal_ids",
                "cluster_weight",
                "embedding",
                "metadata_json",
            ],
            rows,
        )

    def list_external_signal_clusters(self, *, run_id: str) -> list[ExternalSignalClusterRecord]:
        rows = self._query(
            """
            SELECT run_id, cluster_id, cluster_type, summary_text, member_signal_ids, cluster_weight, embedding,
                   metadata_json
            FROM external_signal_clusters
            WHERE run_id = @run_id
            ORDER BY cluster_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "cluster_id",
                "cluster_type",
                "summary_text",
                "member_signal_ids",
                "cluster_weight",
                "embedding",
                "metadata_json",
            ),
        )
        return [
            ExternalSignalClusterRecord(
                run_id=row["run_id"],
                cluster_id=row["cluster_id"],
                cluster_type=row["cluster_type"],
                summary_text=row["summary_text"],
                member_signal_ids=list(row["member_signal_ids"] or []),
                cluster_weight=float(row["cluster_weight"]),
                embedding=list(row["embedding"] or []),
                metadata=_json_loads(row["metadata_json"]) or {},
            )
            for row in rows
        ]

    def write_node_signal_links(self, *, run_id: str, links: Sequence[NodeSignalLinkRecord]) -> None:
        for link in links:
            _ensure_run_id_match("node_signal_links", run_id, link.run_id)
        cluster_types = self._load_external_signal_cluster_types(run_id=run_id)
        existing_node_ids = {node.node_id for node in self.list_nodes(run_id=run_id)}
        for link in links:
            if link.node_id not in existing_node_ids:
                raise ValueError(f"node_signal_links references missing node_id {link.node_id!r}")
            if link.cluster_id not in cluster_types:
                raise ValueError(f"node_signal_links references missing cluster_id {link.cluster_id!r}")
        normalized_links = sorted(links, key=lambda link: (link.node_id, link.cluster_id))
        rows = [
            [
                run_id,
                link.node_id,
                link.cluster_id,
                link.link_type,
                int(link.hop_distance),
                int(link.time_offset_ms),
                float(link.similarity),
                float(link.link_score),
                _json_dumps(link.evidence),
            ]
            for link in normalized_links
        ]
        self._batch_upsert(
            "node_signal_links",
            [
                "run_id",
                "node_id",
                "cluster_id",
                "link_type",
                "hop_distance",
                "time_offset_ms",
                "similarity",
                "link_score",
                "evidence_json",
            ],
            rows,
        )

    def list_node_signal_links(self, *, run_id: str) -> list[NodeSignalLinkRecord]:
        rows = self._query(
            """
            SELECT run_id, node_id, cluster_id, link_type, hop_distance, time_offset_ms, similarity, link_score,
                   evidence_json
            FROM node_signal_links
            WHERE run_id = @run_id
            ORDER BY node_id ASC, cluster_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "node_id",
                "cluster_id",
                "link_type",
                "hop_distance",
                "time_offset_ms",
                "similarity",
                "link_score",
                "evidence_json",
            ),
        )
        return [
            NodeSignalLinkRecord(
                run_id=row["run_id"],
                node_id=row["node_id"],
                cluster_id=row["cluster_id"],
                link_type=row["link_type"],
                hop_distance=int(row["hop_distance"]),
                time_offset_ms=int(row["time_offset_ms"]),
                similarity=float(row["similarity"]),
                link_score=float(row["link_score"]),
                evidence=_json_loads(row["evidence_json"]) or {},
            )
            for row in rows
        ]

    def write_candidate_signal_links(
        self, *, run_id: str, links: Sequence[CandidateSignalLinkRecord]
    ) -> None:
        for link in links:
            _ensure_run_id_match("candidate_signal_links", run_id, link.run_id)
        cluster_types = self._load_external_signal_cluster_types(run_id=run_id)
        existing_clip_ids = {candidate.clip_id for candidate in self.list_candidates(run_id=run_id)}
        for link in links:
            if link.clip_id not in existing_clip_ids:
                raise ValueError(f"candidate_signal_links references missing clip_id {link.clip_id!r}")
            cluster_type = cluster_types.get(link.cluster_id)
            if cluster_type is None:
                raise ValueError(f"candidate_signal_links references missing cluster_id {link.cluster_id!r}")
            if cluster_type != link.cluster_type:
                raise ValueError(
                    "candidate_signal_links.cluster_type must match linked cluster type "
                    f"for cluster_id {link.cluster_id!r}"
                )
        normalized_links = sorted(links, key=lambda link: (link.clip_id, link.cluster_id))
        rows = [
            [
                run_id,
                link.clip_id,
                link.cluster_id,
                link.cluster_type,
                float(link.aggregated_link_score),
                int(link.coverage_ms),
                int(link.direct_node_count),
                int(link.inferred_node_count),
                list(link.agreement_flags),
                float(link.bonus_applied),
                _json_dumps(link.evidence),
            ]
            for link in normalized_links
        ]
        self._batch_upsert(
            "candidate_signal_links",
            [
                "run_id",
                "clip_id",
                "cluster_id",
                "cluster_type",
                "aggregated_link_score",
                "coverage_ms",
                "direct_node_count",
                "inferred_node_count",
                "agreement_flags",
                "bonus_applied",
                "evidence_json",
            ],
            rows,
        )

    def list_candidate_signal_links(self, *, run_id: str) -> list[CandidateSignalLinkRecord]:
        rows = self._query(
            """
            SELECT run_id, clip_id, cluster_id, cluster_type, aggregated_link_score, coverage_ms,
                   direct_node_count, inferred_node_count, agreement_flags, bonus_applied, evidence_json
            FROM candidate_signal_links
            WHERE run_id = @run_id
            ORDER BY clip_id ASC, cluster_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "clip_id",
                "cluster_id",
                "cluster_type",
                "aggregated_link_score",
                "coverage_ms",
                "direct_node_count",
                "inferred_node_count",
                "agreement_flags",
                "bonus_applied",
                "evidence_json",
            ),
        )
        return [
            CandidateSignalLinkRecord(
                run_id=row["run_id"],
                clip_id=row["clip_id"],
                cluster_id=row["cluster_id"],
                cluster_type=row["cluster_type"],
                aggregated_link_score=float(row["aggregated_link_score"]),
                coverage_ms=int(row["coverage_ms"]),
                direct_node_count=int(row["direct_node_count"]),
                inferred_node_count=int(row["inferred_node_count"]),
                agreement_flags=list(row["agreement_flags"] or []),
                bonus_applied=float(row["bonus_applied"]),
                evidence=_json_loads(row["evidence_json"]) or {},
            )
            for row in rows
        ]

    def write_prompt_source_links(self, *, run_id: str, links: Sequence[PromptSourceLinkRecord]) -> None:
        for link in links:
            _ensure_run_id_match("prompt_source_links", run_id, link.run_id)
        cluster_types = self._load_external_signal_cluster_types(run_id=run_id)
        for link in links:
            if link.source_cluster_id is None:
                continue
            cluster_type = cluster_types.get(link.source_cluster_id)
            if cluster_type is None:
                raise ValueError(
                    f"prompt_source_links references missing source_cluster_id {link.source_cluster_id!r}"
                )
            if link.source_cluster_type != cluster_type:
                raise ValueError(
                    "prompt_source_links.source_cluster_type must match linked cluster type "
                    f"for source_cluster_id {link.source_cluster_id!r}"
                )
        normalized_links = sorted(links, key=lambda link: link.prompt_id)
        rows = [
            [
                run_id,
                link.prompt_id,
                link.prompt_source_type,
                link.source_cluster_id,
                link.source_cluster_type,
                _json_dumps(link.metadata),
            ]
            for link in normalized_links
        ]
        self._batch_upsert(
            "prompt_source_links",
            [
                "run_id",
                "prompt_id",
                "prompt_source_type",
                "source_cluster_id",
                "source_cluster_type",
                "metadata_json",
            ],
            rows,
        )

    def list_prompt_source_links(self, *, run_id: str) -> list[PromptSourceLinkRecord]:
        rows = self._query(
            """
            SELECT run_id, prompt_id, prompt_source_type, source_cluster_id, source_cluster_type, metadata_json
            FROM prompt_source_links
            WHERE run_id = @run_id
            ORDER BY prompt_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "prompt_id",
                "prompt_source_type",
                "source_cluster_id",
                "source_cluster_type",
                "metadata_json",
            ),
        )
        return [
            PromptSourceLinkRecord(
                run_id=row["run_id"],
                prompt_id=row["prompt_id"],
                prompt_source_type=row["prompt_source_type"],
                source_cluster_id=row["source_cluster_id"],
                source_cluster_type=row["source_cluster_type"],
                metadata=_json_loads(row["metadata_json"]) or {},
            )
            for row in rows
        ]

    def write_subgraph_provenance(
        self, *, run_id: str, provenance: Sequence[SubgraphProvenanceRecord]
    ) -> None:
        for record in provenance:
            _ensure_run_id_match("subgraph_provenance", run_id, record.run_id)
        normalized_provenance = sorted(provenance, key=lambda record: record.subgraph_id)
        rows = [
            [
                run_id,
                record.subgraph_id,
                list(record.seed_source_set),
                list(record.seed_prompt_ids),
                list(record.source_cluster_ids),
                _json_dumps(record.support_summary),
                bool(record.canonical_selected),
                record.dedupe_overlap_ratio,
                record.selection_reason,
                _json_dumps(record.metadata),
            ]
            for record in normalized_provenance
        ]
        self._batch_upsert(
            "subgraph_provenance",
            [
                "run_id",
                "subgraph_id",
                "seed_source_set",
                "seed_prompt_ids",
                "source_cluster_ids",
                "support_summary_json",
                "canonical_selected",
                "dedupe_overlap_ratio",
                "selection_reason",
                "metadata_json",
            ],
            rows,
        )

    def list_subgraph_provenance(self, *, run_id: str) -> list[SubgraphProvenanceRecord]:
        rows = self._query(
            """
            SELECT run_id, subgraph_id, seed_source_set, seed_prompt_ids, source_cluster_ids,
                   support_summary_json, canonical_selected, dedupe_overlap_ratio, selection_reason, metadata_json
            FROM subgraph_provenance
            WHERE run_id = @run_id
            ORDER BY subgraph_id ASC
            """,
            {"run_id": run_id},
            param_types={"run_id": _STRING_PARAM_TYPE},
            columns=(
                "run_id",
                "subgraph_id",
                "seed_source_set",
                "seed_prompt_ids",
                "source_cluster_ids",
                "support_summary_json",
                "canonical_selected",
                "dedupe_overlap_ratio",
                "selection_reason",
                "metadata_json",
            ),
        )
        return [
            SubgraphProvenanceRecord(
                run_id=row["run_id"],
                subgraph_id=row["subgraph_id"],
                seed_source_set=list(row["seed_source_set"] or []),
                seed_prompt_ids=list(row["seed_prompt_ids"] or []),
                source_cluster_ids=list(row["source_cluster_ids"] or []),
                support_summary=_json_loads(row["support_summary_json"]) or {},
                canonical_selected=bool(row["canonical_selected"]),
                dedupe_overlap_ratio=float(row["dedupe_overlap_ratio"])
                if row["dedupe_overlap_ratio"] is not None
                else None,
                selection_reason=row["selection_reason"],
                metadata=_json_loads(row["metadata_json"]) or {},
            )
            for row in rows
        ]

    def _load_external_signal_cluster_types(self, *, run_id: str) -> dict[str, str]:
        return {
            cluster.cluster_id: cluster.cluster_type
            for cluster in self.list_external_signal_clusters(run_id=run_id)
        }

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

    def write_phase_substeps(self, *, run_id: str, substeps: Sequence[PhaseSubstepRecord]) -> None:
        for record in substeps:
            _ensure_run_id_match("phase_substeps", run_id, record.run_id)
        normalized_substeps = sorted(
            substeps,
            key=lambda record: (record.phase_name, record.step_name, record.step_key),
        )
        rows = [
            [
                run_id,
                record.phase_name,
                record.step_name,
                record.step_key,
                record.status,
                record.started_at,
                record.ended_at,
                record.duration_ms,
                _json_dumps(record.error_payload),
                record.query_version,
                _json_dumps(record.metadata),
            ]
            for record in normalized_substeps
        ]
        self._batch_upsert(
            "phase_substeps",
            [
                "run_id",
                "phase_name",
                "step_name",
                "step_key",
                "status",
                "started_at",
                "ended_at",
                "duration_ms",
                "error_json",
                "query_version",
                "metadata_json",
            ],
            rows,
        )

    def list_phase_substeps(
        self,
        *,
        run_id: str,
        phase_name: str | None = None,
    ) -> list[PhaseSubstepRecord]:
        where_clause = "WHERE run_id = @run_id"
        params: dict[str, Any] = {"run_id": run_id}
        param_types: dict[str, Any] = {"run_id": _STRING_PARAM_TYPE}
        if phase_name is not None:
            where_clause += " AND phase_name = @phase_name"
            params["phase_name"] = phase_name
            param_types["phase_name"] = _STRING_PARAM_TYPE
        rows = self._query(
            f"""
            SELECT run_id, phase_name, step_name, step_key, status, started_at, ended_at,
                   duration_ms, error_json, query_version, metadata_json
            FROM phase_substeps
            {where_clause}
            ORDER BY started_at ASC, phase_name ASC, step_name ASC, step_key ASC
            """,
            params,
            param_types=param_types,
            columns=(
                "run_id",
                "phase_name",
                "step_name",
                "step_key",
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
            PhaseSubstepRecord(
                run_id=row["run_id"],
                phase_name=row["phase_name"],
                step_name=row["step_name"],
                step_key=row["step_key"],
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

    def delete_run(self, *, run_id: str) -> None:
        delete_order = [
            "candidate_signal_links",
            "node_signal_links",
            "prompt_source_links",
            "subgraph_provenance",
            "clip_candidates",
            "external_signal_clusters",
            "external_signals",
            "semantic_edges",
            "semantic_nodes",
            "timeline_turns",
            "phase_substeps",
            "phase_metrics",
            "phase24_jobs",
            "runs",
        ]

        def _delete_txn(txn: Any) -> None:
            execute_update = getattr(txn, "execute_update", None)
            if callable(execute_update):
                for table in delete_order:
                    execute_update(
                        f"DELETE FROM {table} WHERE run_id = @run_id",
                        params={"run_id": run_id},
                        param_types={"run_id": _STRING_PARAM_TYPE},
                    )
                return

            # Fallback for in-memory fakes used by unit tests.
            storage = getattr(self.database, "storage", None)
            if isinstance(storage, dict):
                for table in delete_order:
                    rows = list(storage.get(table, []))
                    storage[table] = [row for row in rows if row.get("run_id") != run_id]

        run_in_transaction = getattr(self.database, "run_in_transaction", None)
        if callable(run_in_transaction):
            run_in_transaction(_delete_txn)
            return

        # Last-resort fallback for simple database doubles without transactions.
        storage = getattr(self.database, "storage", None)
        if isinstance(storage, dict):
            for table in delete_order:
                rows = list(storage.get(table, []))
                storage[table] = [row for row in rows if row.get("run_id") != run_id]

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
