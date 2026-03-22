from pathlib import Path

from fastapi.testclient import TestClient

from backend.do_phase1_service.app import create_app
from backend.do_phase1_service.state_store import SQLiteJobStore


def test_post_jobs_returns_202(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.post("/jobs", json={"source_url": "https://youtube.com/watch?v=x"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]


def test_healthz_reports_ok(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
