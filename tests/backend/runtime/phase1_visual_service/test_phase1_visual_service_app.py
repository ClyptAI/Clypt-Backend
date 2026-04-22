from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


class _FakeVisualExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object | None]] = []

    def extract(self, *, video_path: Path, workspace=None) -> dict:
        self.calls.append((video_path, workspace))
        return {
            "video_metadata": {"fps": 30.0, "duration_ms": 1000},
            "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
            "tracks": [],
            "person_detections": [],
            "face_detections": [],
            "visual_identities": [],
            "mask_stability_signals": [],
            "tracking_metrics": {"tracker_backend": "rfdetr_nano_bytetrack"},
        }


def test_phase1_visual_health_ok() -> None:
    from backend.runtime.phase1_visual_service import app as app_module

    fake = _FakeVisualExtractor()
    app = app_module.create_app(visual_extractor=fake, expected_auth_token="token")
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_phase1_visual_extract_requires_bearer() -> None:
    from backend.runtime.phase1_visual_service import app as app_module

    app = app_module.create_app(
        visual_extractor=_FakeVisualExtractor(),
        expected_auth_token="token",
    )
    with TestClient(app) as client:
        resp = client.post("/tasks/visual-extract", json={"video_path": "/tmp/input.mp4"})

    assert resp.status_code == 401


def test_phase1_visual_extract_returns_visual_payload(tmp_path: Path) -> None:
    from backend.runtime.phase1_visual_service import app as app_module

    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    fake = _FakeVisualExtractor()
    app = app_module.create_app(visual_extractor=fake, expected_auth_token="token")

    with TestClient(app) as client:
        resp = client.post(
            "/tasks/visual-extract",
            json={"run_id": "run-1", "video_path": str(video_path)},
            headers={"Authorization": "Bearer token"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tracking_metrics"]["tracker_backend"] == "rfdetr_nano_bytetrack"
    assert fake.calls == [(video_path, None)]
