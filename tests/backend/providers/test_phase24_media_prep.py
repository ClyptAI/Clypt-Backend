from __future__ import annotations

import io
import json
from types import SimpleNamespace
import sys
import types
from urllib.error import HTTPError

import pytest


def _make_nodes():
    return [
        SimpleNamespace(node_id="node_a", start_ms=0, end_ms=1000),
        SimpleNamespace(node_id="node_b", start_ms=1000, end_ms=2000),
    ]


def _make_paths():
    return SimpleNamespace(run_id="run-123")


def _make_phase1_outputs():
    return SimpleNamespace(phase1_audio={"video_gcs_uri": "gs://bucket/source.mp4"})


def test_cloud_run_media_prep_client_posts_ordered_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.providers.config import Phase24MediaPrepSettings
    from backend.providers.phase24_media_prep import CloudRunMediaPrepClient

    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "run_id": "run-123",
                    "items": [
                        {"node_id": "node_a", "file_uri": "gs://bucket/node_a.mp4", "mime_type": "video/mp4"},
                        {"node_id": "node_b", "file_uri": "gs://bucket/node_b.mp4", "mime_type": "video/mp4"},
                    ],
                }
            ).encode("utf-8")

    def fake_fetch_id_token(_req, audience):
        captured["audience"] = audience
        return "token-123"

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    google_module = types.ModuleType("google")
    google_auth_module = types.ModuleType("google.auth")
    google_auth_transport_module = types.ModuleType("google.auth.transport")
    google_auth_transport_requests_module = types.ModuleType("google.auth.transport.requests")
    google_auth_transport_requests_module.Request = lambda: object()
    google_oauth2_module = types.ModuleType("google.oauth2")
    google_oauth2_id_token_module = types.ModuleType("google.oauth2.id_token")
    google_oauth2_id_token_module.fetch_id_token = fake_fetch_id_token
    google_oauth2_module.id_token = google_oauth2_id_token_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", google_auth_transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", google_auth_transport_requests_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", google_oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.id_token", google_oauth2_id_token_module)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = CloudRunMediaPrepClient(
        settings=Phase24MediaPrepSettings(
            backend="cloud_run_l4",
            service_url="https://media-prep.example.com",
            auth_mode="id_token",
            timeout_s=123,
        )
    )

    result = client.prepare_node_media(
        nodes=_make_nodes(),
        paths=_make_paths(),
        phase1_outputs=_make_phase1_outputs(),
    )

    assert captured["audience"] == "https://media-prep.example.com"
    assert captured["url"] == "https://media-prep.example.com/tasks/node-media-prep"
    assert captured["timeout"] == 123
    assert captured["headers"]["Authorization"] == "Bearer token-123"
    assert captured["body"] == {
        "run_id": "run-123",
        "source_video_gcs_uri": "gs://bucket/source.mp4",
        "object_prefix": "phase14/run-123/node_media",
        "items": [
            {"node_id": "node_a", "start_ms": 0, "end_ms": 1000},
            {"node_id": "node_b", "start_ms": 1000, "end_ms": 2000},
        ],
    }
    assert [item["node_id"] for item in result] == ["node_a", "node_b"]


def test_cloud_run_media_prep_client_rejects_reordered_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.providers.config import Phase24MediaPrepSettings
    from backend.providers.phase24_media_prep import CloudRunMediaPrepClient

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "run_id": "run-123",
                    "items": [
                        {"node_id": "node_b", "file_uri": "gs://bucket/node_b.mp4", "mime_type": "video/mp4"},
                        {"node_id": "node_a", "file_uri": "gs://bucket/node_a.mp4", "mime_type": "video/mp4"},
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _FakeResponse())

    client = CloudRunMediaPrepClient(
        settings=Phase24MediaPrepSettings(
            backend="cloud_run_l4",
            service_url="https://media-prep.example.com",
            auth_mode="none",
        )
    )

    with pytest.raises(RuntimeError, match="unexpected order"):
        client.prepare_node_media(
            nodes=_make_nodes(),
            paths=_make_paths(),
            phase1_outputs=_make_phase1_outputs(),
        )


def test_cloud_run_media_prep_client_surfaces_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.providers.config import Phase24MediaPrepSettings
    from backend.providers.phase24_media_prep import CloudRunMediaPrepClient

    def fake_urlopen(req, timeout):
        raise HTTPError(
            req.full_url,
            503,
            "unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"backend unavailable"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = CloudRunMediaPrepClient(
        settings=Phase24MediaPrepSettings(
            backend="cloud_run_l4",
            service_url="https://media-prep.example.com",
            auth_mode="none",
        )
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.prepare_node_media(
            nodes=_make_nodes(),
            paths=_make_paths(),
            phase1_outputs=_make_phase1_outputs(),
        )


def test_cloud_run_media_prep_client_short_circuits_empty_nodes() -> None:
    from backend.providers.config import Phase24MediaPrepSettings
    from backend.providers.phase24_media_prep import CloudRunMediaPrepClient

    client = CloudRunMediaPrepClient(
        settings=Phase24MediaPrepSettings(
            backend="cloud_run_l4",
            service_url="https://media-prep.example.com",
            auth_mode="none",
        )
    )

    result = client.prepare_node_media(
        nodes=[],
        paths=_make_paths(),
        phase1_outputs=_make_phase1_outputs(),
    )

    assert result == []
