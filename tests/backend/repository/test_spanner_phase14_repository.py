from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.repository.models import (
    ClipCandidateRecord,
    Phase24JobRecord,
    PhaseMetricRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    TimelineTurnRecord,
)
import backend.repository.spanner_phase14_repository as spanner_repo
from backend.repository.spanner_phase14_repository import (
    SpannerPhase14Repository,
    apply_ddl_statements,
    bootstrap_phase14_schema,
    build_phase14_bootstrap_ddl,
)

UTC = timezone.utc


class _FakeOperation:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def result(self, timeout: float | None = None):
        self.timeouts.append(timeout)
        return None


class _FakeBatch:
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database

    def __enter__(self) -> "_FakeBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def insert_or_update(self, *, table: str, columns: list[str], values: list[list[object]]) -> None:
        self.database.batch_calls.append({"table": table, "columns": columns, "values": values})
        table_rows = self.database.storage.setdefault(table, [])
        pk_fields = self.database.pk_fields[table]
        for value_row in values:
            row = dict(zip(columns, value_row, strict=True))
            table_rows[:] = [existing for existing in table_rows if tuple(existing[field] for field in pk_fields) != tuple(row[field] for field in pk_fields)]
            table_rows.append(row)


class _FakeSnapshot:
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database

    def __enter__(self) -> "_FakeSnapshot":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute_sql(
        self,
        statement: str,
        params: dict[str, object] | None = None,
        param_types: dict[str, object] | None = None,
    ):
        self.database.query_calls.append({"statement": statement, "params": params or {}, "param_types": param_types})
        assert params is not None
        assert "run_id" in params
        assert param_types == {"run_id": spanner_repo._STRING_PARAM_TYPE}
        statement_lower = statement.lower()
        table = None
        for candidate in self.database.storage:
            if f"from {candidate}" in statement_lower:
                table = candidate
                break
        if table is None:
            return []
        rows = list(self.database.storage.get(table, []))
        run_id = (params or {}).get("run_id")
        if run_id is not None:
            rows = [row for row in rows if row.get("run_id") == run_id]
        return rows


class _FakeDatabase:
    pk_fields = {
        "runs": ("run_id",),
        "timeline_turns": ("run_id", "turn_id"),
        "semantic_nodes": ("run_id", "node_id"),
        "semantic_edges": ("run_id", "source_node_id", "target_node_id", "edge_type"),
        "clip_candidates": ("run_id", "clip_id"),
        "phase_metrics": ("run_id", "phase_name"),
        "phase24_jobs": ("run_id",),
    }

    def __init__(self) -> None:
        self.storage: dict[str, list[dict[str, object]]] = {}
        self.batch_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.ddl_calls: list[dict[str, object]] = []
        self.last_operation: _FakeOperation | None = None

    def batch(self) -> _FakeBatch:
        return _FakeBatch(self)

    def snapshot(self) -> _FakeSnapshot:
        return _FakeSnapshot(self)

    def update_ddl(self, statements: list[str], operation_id: str | None = None) -> _FakeOperation:
        self.ddl_calls.append({"statements": statements, "operation_id": operation_id})
        self.last_operation = _FakeOperation()
        return self.last_operation

    class _FakeTransaction:
        def __init__(self, database: "_FakeDatabase") -> None:
            self.database = database

        def execute_sql(
            self,
            statement: str,
            params: dict[str, object] | None = None,
            param_types: dict[str, object] | None = None,
        ):
            snapshot = _FakeSnapshot(self.database)
            return snapshot.execute_sql(statement, params=params, param_types=param_types)

        def insert_or_update(self, *, table: str, columns: list[str], values: list[list[object]]) -> None:
            batch = _FakeBatch(self.database)
            batch.insert_or_update(table=table, columns=columns, values=values)

    def run_in_transaction(self, fn):
        return fn(self._FakeTransaction(self))


def test_build_phase14_bootstrap_ddl_includes_core_tables_and_graph() -> None:
    ddl = build_phase14_bootstrap_ddl()

    assert any("CREATE TABLE runs" in statement for statement in ddl)
    assert any("CREATE TABLE timeline_turns" in statement for statement in ddl)
    assert any("CREATE TABLE semantic_nodes" in statement for statement in ddl)
    assert any("CREATE TABLE clip_candidates" in statement for statement in ddl)
    assert any("CREATE OR REPLACE PROPERTY GRAPH" in statement for statement in ddl)


def test_apply_ddl_statements_cleans_and_dedupes_inputs() -> None:
    database = _FakeDatabase()

    apply_ddl_statements(
        database,
        ["  CREATE TABLE foo (id INT64) PRIMARY KEY (id)  ", "", "CREATE TABLE foo (id INT64) PRIMARY KEY (id);"],
        operation_id="phase14-bootstrap-test",
        timeout_s=3.5,
    )

    assert database.ddl_calls == [
        {
            "statements": ["CREATE TABLE foo (id INT64) PRIMARY KEY (id)"],
            "operation_id": "phase14-bootstrap-test-01",
        }
    ]
    assert database.ddl_calls[0]["statements"] == ["CREATE TABLE foo (id INT64) PRIMARY KEY (id)"]


def test_apply_ddl_statements_ignores_expected_already_exists_errors() -> None:
    class _ExpectedErrorOperation:
        def result(self, timeout: float | None = None):
            raise RuntimeError("Table already exists")

    class _ExpectedErrorDatabase(_FakeDatabase):
        def update_ddl(self, statements: list[str], operation_id: str | None = None):
            self.ddl_calls.append({"statements": statements, "operation_id": operation_id})
            return _ExpectedErrorOperation()

    database = _ExpectedErrorDatabase()

    apply_ddl_statements(database, ["CREATE TABLE foo (id INT64) PRIMARY KEY (id)"])

    assert database.ddl_calls


def test_apply_ddl_statements_continues_past_expected_bootstrap_errors() -> None:
    class _MixedOperation:
        def __init__(self, error: str | None = None):
            self.error = error

        def result(self, timeout: float | None = None):
            if self.error is not None:
                raise RuntimeError(self.error)
            return None

    class _MixedDatabase(_FakeDatabase):
        def update_ddl(self, statements: list[str], operation_id: str | None = None):
            self.ddl_calls.append({"statements": statements, "operation_id": operation_id})
            statement = statements[0]
            if "foo" in statement:
                return _MixedOperation("already defined")
            return _MixedOperation()

    database = _MixedDatabase()

    apply_ddl_statements(
        database,
        [
            "CREATE TABLE foo (id INT64) PRIMARY KEY (id)",
            "CREATE TABLE bar (id INT64) PRIMARY KEY (id)",
        ],
    )

    assert [call["statements"][0] for call in database.ddl_calls] == [
        "CREATE TABLE foo (id INT64) PRIMARY KEY (id)",
        "CREATE TABLE bar (id INT64) PRIMARY KEY (id)",
    ]


def test_apply_ddl_statements_raises_unknown_bootstrap_errors() -> None:
    class _UnexpectedErrorOperation:
        def result(self, timeout: float | None = None):
            raise RuntimeError("disk full")

    class _UnexpectedErrorDatabase(_FakeDatabase):
        def update_ddl(self, statements: list[str], operation_id: str | None = None):
            self.ddl_calls.append({"statements": statements, "operation_id": operation_id})
            return _UnexpectedErrorOperation()

    database = _UnexpectedErrorDatabase()

    with pytest.raises(RuntimeError, match="disk full"):
        apply_ddl_statements(database, ["CREATE TABLE foo (id INT64) PRIMARY KEY (id)"])


def test_spanner_phase14_repository_round_trips_core_records() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database, ddl_operation_timeout_s=15.0)

    run = RunRecord(
        run_id="run_001",
        source_url="https://example.com/video",
        source_video_gcs_uri="gs://bucket/source.mp4",
        status="PHASE24_RUNNING",
        created_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 8, 18, 5, tzinfo=UTC),
        metadata={"source": "test"},
    )
    repository.upsert_run(run)

    turns = [
        TimelineTurnRecord(
            run_id="run_001",
            turn_id="turn_1",
            speaker_id="SPEAKER_0",
            start_ms=0,
            end_ms=1000,
            word_ids=["word_1", "word_2"],
            transcript_text="hello world",
            identification_match=None,
        ),
        TimelineTurnRecord(
            run_id="run_001",
            turn_id="turn_2",
            speaker_id="SPEAKER_1",
            start_ms=1000,
            end_ms=1500,
            word_ids=["word_3"],
            transcript_text="right",
            identification_match="match",
        ),
    ]
    repository.write_timeline_turns(run_id="run_001", turns=turns)

    nodes = [
        SemanticNodeRecord(
            run_id="run_001",
            node_id="node_1",
            node_type="claim",
            start_ms=0,
            end_ms=1000,
            source_turn_ids=["turn_1"],
            word_ids=["word_1", "word_2"],
            transcript_text="hello world",
            node_flags=["callback_candidate"],
            summary="A basic claim",
            evidence={"emotion_labels": ["neutral"], "audio_events": ["laughter"]},
            semantic_embedding=[0.1, 0.2],
            multimodal_embedding=[0.3, 0.4],
        ),
        SemanticNodeRecord(
            run_id="run_001",
            node_id="node_2",
            node_type="reveal",
            start_ms=1000,
            end_ms=2000,
            source_turn_ids=["turn_2"],
            word_ids=["word_3"],
            transcript_text="right",
            node_flags=[],
            summary="A reveal",
            evidence={},
            semantic_embedding=None,
            multimodal_embedding=None,
        ),
    ]
    repository.write_nodes(run_id="run_001", nodes=nodes)

    edges = [
        SemanticEdgeRecord(
            run_id="run_001",
            source_node_id="node_2",
            target_node_id="node_1",
            edge_type="payoff_of",
            rationale="The second node pays off the first.",
            confidence=0.91,
            support_count=3,
            batch_ids=["batch_1"],
        )
    ]
    repository.write_edges(run_id="run_001", edges=edges)

    candidates = [
        ClipCandidateRecord(
            run_id="run_001",
            clip_id=None,
            node_ids=["node_1", "node_2"],
            start_ms=0,
            end_ms=2000,
            score=8.4,
            rationale="complete setup and reveal",
            source_prompt_ids=["prompt_1"],
            seed_node_id="node_1",
            subgraph_id="sg_1",
            query_aligned=True,
            pool_rank=1,
            score_breakdown={"overall_clip_quality": 8.4},
        )
    ]
    repository.write_candidates(run_id="run_001", candidates=candidates)

    metric = PhaseMetricRecord(
        run_id="run_001",
        phase_name="phase2",
        status="succeeded",
        started_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
        ended_at=datetime(2026, 4, 8, 18, 1, tzinfo=UTC),
        duration_ms=60000.0,
        error_payload=None,
        query_version="graph-v1",
        metadata={"rows_written": 2},
    )
    repository.write_phase_metric(metric)

    job = Phase24JobRecord(
        run_id="run_001",
        status="running",
        attempt_count=1,
        last_error=None,
        worker_name="clypt-phase24-worker",
        task_name="projects/clypt-v3/locations/us-central1/queues/clypt-phase24/tasks/phase24-run-001",
        locked_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
        completed_at=None,
        metadata={"query_version": "graph-v1"},
    )
    repository.upsert_phase24_job(job)

    assert repository.get_run("run_001") == run
    assert repository.list_timeline_turns(run_id="run_001") == turns
    assert repository.list_nodes(run_id="run_001") == nodes
    assert repository.list_edges(run_id="run_001") == edges
    listed_candidates = repository.list_candidates(run_id="run_001")
    assert len(listed_candidates) == 1
    assert listed_candidates[0].clip_id is not None and listed_candidates[0].clip_id.startswith("clip_")
    assert listed_candidates[0].score_breakdown == {"overall_clip_quality": 8.4}
    assert repository.list_phase_metrics(run_id="run_001") == [metric]
    assert repository.get_phase24_job("run_001") == job

    stored_clip_row = database.storage["clip_candidates"][0]
    assert isinstance(stored_clip_row["clip_id"], str)
    assert stored_clip_row["clip_id"].startswith("clip_")
    assert database.batch_calls
    assert database.query_calls


def test_spanner_phase14_repository_rejects_mismatched_run_ids() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database)

    turn = TimelineTurnRecord(
        run_id="run_002",
        turn_id="turn_1",
        speaker_id="SPEAKER_0",
        start_ms=0,
        end_ms=1000,
        word_ids=[],
        transcript_text="hello",
    )
    node = SemanticNodeRecord(
        run_id="run_002",
        node_id="node_1",
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        source_turn_ids=[],
        word_ids=[],
        transcript_text="hello",
        node_flags=[],
        summary="summary",
        evidence={},
    )
    edge = SemanticEdgeRecord(
        run_id="run_002",
        source_node_id="node_1",
        target_node_id="node_2",
        edge_type="supports",
    )
    candidate = ClipCandidateRecord(
        run_id="run_002",
        clip_id=None,
        node_ids=["node_1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="rationale",
    )

    with pytest.raises(ValueError, match="timeline_turns"):
        repository.write_timeline_turns(run_id="run_001", turns=[turn])
    with pytest.raises(ValueError, match="semantic_nodes"):
        repository.write_nodes(run_id="run_001", nodes=[node])
    with pytest.raises(ValueError, match="semantic_edges"):
        repository.write_edges(run_id="run_001", edges=[edge])
    with pytest.raises(ValueError, match="clip_candidates"):
        repository.write_candidates(run_id="run_001", candidates=[candidate])


def test_spanner_phase14_repository_decodes_tuple_like_rows_by_column_order() -> None:
    class _TupleRowDatabase(_FakeDatabase):
        class _TupleSnapshot(_FakeSnapshot):
            def execute_sql(
                self,
                statement: str,
                params: dict[str, object] | None = None,
                param_types: dict[str, object] | None = None,
            ):
                statement_lower = statement.lower()
                if "from runs" in statement_lower:
                    return [
                        (
                            "run_001",
                            "https://example.com/video",
                            "gs://bucket/source.mp4",
                            "PHASE24_RUNNING",
                            datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
                            datetime(2026, 4, 8, 18, 5, tzinfo=UTC),
                            {"source": "tuple"},
                        )
                    ]
                return super().execute_sql(statement, params=params, param_types=param_types)

        def snapshot(self) -> "_TupleRowDatabase._TupleSnapshot":
            return self._TupleSnapshot(self)

    repository = SpannerPhase14Repository(database=_TupleRowDatabase())

    run = repository.get_run("run_001")

    assert run is not None
    assert run.run_id == "run_001"
    assert run.metadata == {"source": "tuple"}


def test_spanner_phase14_repository_bootstrap_schema_uses_configured_timeout() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database, ddl_operation_timeout_s=7.25)

    repository.bootstrap_schema()

    assert database.ddl_calls[0]["operation_id"].startswith("phase14-bootstrap-")
    assert database.ddl_calls[0]["statements"]
    assert database.ddl_calls and database.ddl_calls[0]
    assert database.last_operation is not None
    assert database.last_operation.timeouts == [7.25]


def test_spanner_phase14_repository_acquire_phase24_job_lease_short_circuits_running_and_succeeded() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database)
    now = datetime.now(UTC)

    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id="run_001",
            status="running",
            attempt_count=2,
            last_error=None,
            worker_name="worker",
            task_name="task-1",
            locked_at=now,
            updated_at=now,
            completed_at=None,
            metadata={"query_version": "graph-v1"},
        )
    )
    running = repository.acquire_phase24_job_lease(
        run_id="run_001",
        job_id="task-2",
        worker_name="worker",
        attempt=3,
        query_version="graph-v2",
    )
    assert running == {
        "acquired": False,
        "status": "running",
        "attempt_count": 2,
        "task_name": "task-1",
    }

    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id="run_001",
            status="succeeded",
            attempt_count=2,
            last_error=None,
            worker_name="worker",
            task_name="task-1",
            locked_at=now,
            updated_at=now,
            completed_at=now,
            metadata={"query_version": "graph-v1"},
        )
    )
    succeeded = repository.acquire_phase24_job_lease(
        run_id="run_001",
        job_id="task-3",
        worker_name="worker",
        attempt=4,
        query_version="graph-v2",
    )
    assert succeeded == {
        "acquired": False,
        "status": "succeeded",
        "attempt_count": 2,
        "task_name": "task-1",
    }


def test_spanner_phase14_repository_acquire_phase24_job_lease_reclaims_stale_running_lock() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database)
    stale_time = datetime(2026, 4, 8, 16, 0, tzinfo=UTC)

    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id="run_001",
            status="running",
            attempt_count=1,
            last_error=None,
            worker_name="worker",
            task_name="task-old",
            locked_at=stale_time,
            updated_at=stale_time,
            completed_at=None,
            metadata={"query_version": "graph-v1"},
        )
    )

    lease = repository.acquire_phase24_job_lease(
        run_id="run_001",
        job_id="task-new",
        worker_name="worker-new",
        attempt=2,
        query_version="graph-v2",
        running_timeout_s=1,
    )
    assert lease["acquired"] is True
    assert lease["status"] == "running"
    row = repository.get_phase24_job("run_001")
    assert row is not None
    assert row.task_name == "task-new"
    assert row.worker_name == "worker-new"


def test_spanner_phase14_repository_acquire_phase24_job_lease_sets_running_state() -> None:
    database = _FakeDatabase()
    repository = SpannerPhase14Repository(database=database)
    now = datetime(2026, 4, 8, 18, 0, tzinfo=UTC)

    repository.upsert_phase24_job(
        Phase24JobRecord(
            run_id="run_001",
            status="failed",
            attempt_count=1,
            last_error={"code": "x"},
            worker_name="worker",
            task_name="task-1",
            locked_at=now,
            updated_at=now,
            completed_at=now,
            metadata={"query_version": "graph-v1"},
        )
    )

    lease = repository.acquire_phase24_job_lease(
        run_id="run_001",
        job_id="task-2",
        worker_name="worker-2",
        attempt=3,
        query_version="graph-v2",
    )
    assert lease["acquired"] is True
    assert lease["status"] == "running"
    assert lease["attempt_count"] == 3
    row = repository.get_phase24_job("run_001")
    assert row is not None
    assert row.status == "running"
    assert row.task_name == "task-2"
    assert row.worker_name == "worker-2"
    assert row.metadata["query_version"] == "graph-v2"
