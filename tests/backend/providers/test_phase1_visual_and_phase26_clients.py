from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest


class _FakeHTTPResponse:
    def __init__(self, *, body: dict, status: int = 200) -> None:
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_remote_phase1_visual_client_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from backend.providers.config import Phase1VisualServiceSettings
    from backend.providers.visual_service_client import RemotePhase1VisualClient

    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            body={
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [],
                "tracks": [],
                "person_detections": [],
                "face_detections": [],
                "visual_identities": [],
                "mask_stability_signals": [],
                "tracking_metrics": {"tracker_backend": "rfdetr_small_bytetrack"},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemotePhase1VisualClient(
        settings=Phase1VisualServiceSettings(
            service_url="http://127.0.0.1:9200",
            auth_token="visual-token",
        )
    )

    payload = client.extract(video_path=video_path, workspace=None)

    assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_small_bytetrack"
    assert captured["url"] == "http://127.0.0.1:9200/tasks/visual-extract"
    assert captured["body"] == {"video_path": str(video_path)}


def test_remote_phase26_dispatch_client_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.providers.config import Phase26DispatchServiceSettings
    from backend.providers.phase26_dispatch_client import RemotePhase26DispatchClient

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            body={"run_id": "run-1", "status": "queued", "task_name": "local-sqlite:job-1"}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemotePhase26DispatchClient(
        settings=Phase26DispatchServiceSettings(
            service_url="http://10.0.0.8:9300",
            auth_token="dispatch-token",
        )
    )

    task_name = client.enqueue_phase24(run_id="run-1", payload={"run_id": "run-1"})

    assert task_name == "local-sqlite:job-1"
    assert captured["url"] == "http://10.0.0.8:9300/tasks/phase26-enqueue"
    assert captured["body"] == {"run_id": "run-1", "payload": {"run_id": "run-1"}}
