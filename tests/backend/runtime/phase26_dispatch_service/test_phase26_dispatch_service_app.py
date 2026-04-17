from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_phase26_dispatch_health_ok(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue
    from backend.runtime.phase26_dispatch_service import app as app_module

    queue = Phase24LocalQueue(tmp_path / "phase26.sqlite")
    app = app_module.create_app(queue=queue, expected_auth_token="dispatch-token")
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_phase26_dispatch_requires_bearer(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue
    from backend.runtime.phase26_dispatch_service import app as app_module

    queue = Phase24LocalQueue(tmp_path / "phase26.sqlite")
    app = app_module.create_app(queue=queue, expected_auth_token="dispatch-token")
    with TestClient(app) as client:
        resp = client.post("/tasks/phase26-enqueue", json={"run_id": "run-1", "payload": {}})

    assert resp.status_code == 401


def test_phase26_dispatch_enqueues_into_local_sqlite_queue(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue
    from backend.runtime.phase26_dispatch_service import app as app_module

    queue_path = tmp_path / "phase26.sqlite"
    queue = Phase24LocalQueue(queue_path)
    app = app_module.create_app(queue=queue, expected_auth_token="dispatch-token")

    payload = {
        "run_id": "run-42",
        "payload": {
            "run_id": "run-42",
            "source_url": "https://example.com/watch?v=42",
            "phase1_outputs_gcs_uri": "gs://bucket/phase1/42.json",
        },
    }

    with TestClient(app) as client:
        resp = client.post(
            "/tasks/phase26-enqueue",
            json=payload,
            headers={"Authorization": "Bearer dispatch-token"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == "run-42"
    assert body["status"] == "queued"
    assert body["task_name"].startswith("local-sqlite:")
    job_id = body["task_name"].split(":", 1)[1]
    row = queue.get_job(job_id)
    assert row is not None
    assert row["run_id"] == "run-42"
