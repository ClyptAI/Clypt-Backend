"""Integration tests for the /v1/ API endpoints.

Uses an in-memory mock repository to verify the full request -> response flow
without needing Spanner or GCP credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.v1.app import create_app
from backend.repository.models import (
    ClipCandidateRecord,
    PhaseMetricRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    TimelineTurnRecord,
)
from backend.repository.phase14_repository import Phase14Repository


# ── In-memory mock repository ─────────────────────────────────────────────────

class InMemoryRepository(Phase14Repository):
    """Minimal in-memory implementation for testing the API layer."""

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}
        self._nodes: dict[str, list[SemanticNodeRecord]] = {}
        self._edges: dict[str, list[SemanticEdgeRecord]] = {}
        self._candidates: dict[str, list[ClipCandidateRecord]] = {}
        self._turns: dict[str, list[TimelineTurnRecord]] = {}
        self._metrics: dict[str, list[PhaseMetricRecord]] = {}

    def upsert_run(self, record):
        self._runs[record.run_id] = record
        return record

    def get_run(self, run_id):
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())

    def write_timeline_turns(self, *, run_id, turns):
        self._turns[run_id] = list(turns)

    def list_timeline_turns(self, *, run_id):
        return self._turns.get(run_id, [])

    def write_nodes(self, *, run_id, nodes):
        self._nodes[run_id] = list(nodes)

    def list_nodes(self, *, run_id):
        return self._nodes.get(run_id, [])

    def write_edges(self, *, run_id, edges):
        self._edges[run_id] = list(edges)

    def list_edges(self, *, run_id):
        return self._edges.get(run_id, [])

    def write_candidates(self, *, run_id, candidates):
        self._candidates[run_id] = list(candidates)

    def list_candidates(self, *, run_id):
        return self._candidates.get(run_id, [])

    def write_phase_metric(self, record):
        self._metrics.setdefault(record.run_id, []).append(record)
        return record

    def list_phase_metrics(self, *, run_id):
        return self._metrics.get(run_id, [])

    # Stubs for methods we don't use in the API tests
    def write_external_signals(self, *, run_id, signals): pass
    def list_external_signals(self, *, run_id): return []
    def write_external_signal_clusters(self, *, run_id, clusters): pass
    def list_external_signal_clusters(self, *, run_id): return []
    def write_node_signal_links(self, *, run_id, links): pass
    def list_node_signal_links(self, *, run_id): return []
    def write_candidate_signal_links(self, *, run_id, links): pass
    def list_candidate_signal_links(self, *, run_id): return []
    def write_prompt_source_links(self, *, run_id, links): pass
    def list_prompt_source_links(self, *, run_id): return []
    def write_subgraph_provenance(self, *, run_id, provenance): pass
    def list_subgraph_provenance(self, *, run_id): return []
    def write_phase_substeps(self, *, run_id, substeps): pass
    def list_phase_substeps(self, *, run_id, phase_name=None): return []
    def upsert_phase24_job(self, record): return record
    def get_phase24_job(self, run_id): return None
    def acquire_phase24_job_lease(self, **kwargs): return {}
    def delete_run(self, *, run_id): self._runs.pop(run_id, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)

SAMPLE_RUN = RunRecord(
    run_id="run_test_001",
    source_url="https://youtube.com/watch?v=abc123",
    status="PHASE24_DONE",
    created_at=NOW,
    updated_at=NOW,
    metadata={"display_name": "Test Run"},
)

SAMPLE_NODE = SemanticNodeRecord(
    run_id="run_test_001",
    node_id="node_1",
    node_type="claim",
    start_ms=0,
    end_ms=5000,
    source_turn_ids=["t1"],
    word_ids=["w1", "w2"],
    transcript_text="This is a test claim",
    node_flags=["topic_pivot"],
    summary="Test claim summary",
    evidence={"emotion_labels": ["happy"], "audio_events": ["laughter"]},
)

SAMPLE_EDGE = SemanticEdgeRecord(
    run_id="run_test_001",
    source_node_id="node_1",
    target_node_id="node_2",
    edge_type="supports",
    rationale="Node 1 supports node 2",
    confidence=0.85,
)

SAMPLE_CLIP = ClipCandidateRecord(
    run_id="run_test_001",
    clip_id="clip_1",
    node_ids=["node_1"],
    start_ms=0,
    end_ms=30000,
    score=0.92,
    rationale="Strong claim with engagement",
    seed_node_id="node_1",
)

SAMPLE_TURN = TimelineTurnRecord(
    run_id="run_test_001",
    turn_id="t1",
    speaker_id="spk_0",
    start_ms=0,
    end_ms=5000,
    transcript_text="Hello everyone",
)


@pytest.fixture
def repo() -> InMemoryRepository:
    r = InMemoryRepository()
    r.upsert_run(SAMPLE_RUN)
    r.write_nodes(run_id="run_test_001", nodes=[SAMPLE_NODE])
    r.write_edges(run_id="run_test_001", edges=[SAMPLE_EDGE])
    r.write_candidates(run_id="run_test_001", candidates=[SAMPLE_CLIP])
    r.write_timeline_turns(run_id="run_test_001", turns=[SAMPLE_TURN])
    return r


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def app(repo, artifact_root):
    return create_app(repo=repo, artifact_root=artifact_root)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_list_runs(client):
    r = await client.get("/v1/runs")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "run_test_001"
    assert data[0]["latest_phase"] == 4
    assert data[0]["latest_status"] == "completed"
    assert data[0]["display_name"] == "Test Run"


@pytest.mark.anyio
async def test_get_run_detail(client):
    r = await client.get("/v1/runs/run_test_001")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "run_test_001"
    assert data["node_count"] == 1
    assert data["edge_count"] == 1
    assert data["clip_count"] == 1


@pytest.mark.anyio
async def test_get_run_not_found(client):
    r = await client.get("/v1/runs/nonexistent")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_create_run(client):
    r = await client.post("/v1/runs", json={
        "source_url": "https://youtube.com/watch?v=new",
        "display_name": "New Run",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["source_url"] == "https://youtube.com/watch?v=new"
    assert data["display_name"] == "New Run"
    assert "run_id" in data
    assert "created_at" in data


@pytest.mark.anyio
async def test_list_nodes(client):
    r = await client.get("/v1/runs/run_test_001/nodes")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["node_id"] == "node_1"
    assert data[0]["node_type"] == "claim"
    assert data[0]["evidence"]["emotion_labels"] == ["happy"]


@pytest.mark.anyio
async def test_get_node(client):
    r = await client.get("/v1/runs/run_test_001/nodes/node_1")
    assert r.status_code == 200
    assert r.json()["node_id"] == "node_1"


@pytest.mark.anyio
async def test_get_node_not_found(client):
    r = await client.get("/v1/runs/run_test_001/nodes/nonexistent")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_list_edges(client):
    r = await client.get("/v1/runs/run_test_001/edges")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["edge_type"] == "supports"


@pytest.mark.anyio
async def test_list_clips(client):
    r = await client.get("/v1/runs/run_test_001/clips")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["clip_id"] == "clip_1"
    assert data[0]["score"] == 0.92


@pytest.mark.anyio
async def test_get_clip(client):
    r = await client.get("/v1/runs/run_test_001/clips/clip_1")
    assert r.status_code == 200
    assert r.json()["clip_id"] == "clip_1"


@pytest.mark.anyio
async def test_approve_clip(client):
    r = await client.post("/v1/runs/run_test_001/clips/clip_1/approve")
    assert r.status_code == 200
    assert r.json()["clip_id"] == "clip_1"


@pytest.mark.anyio
async def test_reject_clip(client):
    r = await client.post("/v1/runs/run_test_001/clips/clip_1/reject")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_grounding_default(client):
    r = await client.get("/v1/runs/run_test_001/clips/clip_1/grounding")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "run_test_001"
    assert data["clip_id"] == "clip_1"
    assert data["shots"] == []


@pytest.mark.anyio
async def test_grounding_roundtrip(client):
    state = {
        "run_id": "run_test_001",
        "clip_id": "clip_1",
        "shots": [{
            "shot_idx": 0,
            "rects": {"trk_a": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}},
            "user_tracklets": [],
            "hidden_tracklet_ids": [],
        }],
        "updated_at": "2026-04-17T12:00:00Z",
    }
    r = await client.put(
        "/v1/runs/run_test_001/clips/clip_1/grounding",
        json=state,
    )
    assert r.status_code == 200
    saved = r.json()
    assert len(saved["shots"]) == 1
    assert saved["shots"][0]["rects"]["trk_a"]["x"] == 0.1

    # Read it back
    r2 = await client.get("/v1/runs/run_test_001/clips/clip_1/grounding")
    assert r2.status_code == 200
    assert r2.json()["shots"][0]["rects"]["trk_a"]["x"] == 0.1


@pytest.mark.anyio
async def test_render_presets(client):
    r = await client.get("/v1/render/presets")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 4
    ids = {p["id"] for p in data}
    assert "tiktok_9x16" in ids


@pytest.mark.anyio
async def test_render_submit(client):
    r = await client.post(
        "/v1/runs/run_test_001/clips/clip_1/render",
        json={"preset_id": "tiktok_9x16"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


@pytest.mark.anyio
async def test_timeline_no_artifacts(client):
    # No artifacts on disk and repo has turns -> should use repo fallback
    r = await client.get("/v1/runs/run_test_001/timeline")
    assert r.status_code == 200
    data = r.json()
    assert data["duration_ms"] == 5000
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["speaker_id"] == "spk_0"
