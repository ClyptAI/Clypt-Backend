from __future__ import annotations

import json
import sys
import types

from fastapi.testclient import TestClient


def _load_app_module():
    if "scripts.modal.node_media_prep_app" in sys.modules:
        del sys.modules["scripts.modal.node_media_prep_app"]

    class _FakeImage:
        @staticmethod
        def debian_slim():
            return _FakeImage()

        def apt_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def pip_install_from_requirements(self, *_args, **_kwargs):
            return self

    class _FakeSecret:
        @staticmethod
        def from_name(_name: str):
            return object()

    class _FakeApp:
        def __init__(self, _name: str):
            pass

        def function(self, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    fake_modal = types.SimpleNamespace(
        App=_FakeApp,
        Image=_FakeImage,
        Secret=_FakeSecret,
        asgi_app=lambda: (lambda fn: fn),
    )
    sys.modules["modal"] = fake_modal

    from scripts.modal import node_media_prep_app

    return node_media_prep_app


def _payload() -> dict:
    return {
        "run_id": "run-123",
        "video_gcs_uri": "gs://bucket/source.mp4",
        "object_prefix": "phase14/run-123/node_media",
        "nodes": [{"node_id": "n_1", "start_ms": 0, "end_ms": 1000}],
    }


def test_health_returns_ok(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    client = TestClient(node_media_prep_app.web_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_node_media_prep_requires_bearer_token(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    monkeypatch.setenv("NODE_MEDIA_PREP_AUTH_TOKEN", "secret-token")
    client = TestClient(node_media_prep_app.web_app)

    response = client.post("/tasks/node-media-prep", json=_payload())
    assert response.status_code == 401


def test_node_media_prep_runs_with_authorized_request(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    monkeypatch.setenv("NODE_MEDIA_PREP_AUTH_TOKEN", "secret-token")

    captured: dict[str, object] = {}

    def fake_build_storage_client():
        captured["storage_client"] = "built"
        return object()

    def fake_run_node_media_prep(*, request, storage_client, scratch_root):
        captured["request"] = request
        captured["storage_client"] = storage_client
        captured["scratch_root"] = scratch_root
        return {"media": [{"node_id": "n_1", "file_uri": "gs://bucket/n_1.mp4"}]}

    monkeypatch.setattr(node_media_prep_app, "_build_storage_client", fake_build_storage_client)
    monkeypatch.setattr(node_media_prep_app, "run_node_media_prep", fake_run_node_media_prep)

    client = TestClient(node_media_prep_app.web_app)
    response = client.post(
        "/tasks/node-media-prep",
        json=_payload(),
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"media": [{"node_id": "n_1", "file_uri": "gs://bucket/n_1.mp4"}]}
    assert getattr(captured["request"], "run_id") == "run-123"


def test_build_storage_client_supports_json_credentials(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    calls: dict[str, object] = {}
    monkeypatch.setenv("GCS_BUCKET", "bucket")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "cid",
                "client_secret": "secret",
                "refresh_token": "refresh",
            }
        ),
    )
    node_media_prep_app._build_storage_client.cache_clear()

    class _FakeStorageModule:
        class Client:
            def __init__(self, *, project=None, credentials=None):
                calls["project"] = project
                calls["credentials"] = credentials

    def fake_load_credentials_from_dict(info, scopes):
        calls["info"] = info
        calls["scopes"] = scopes
        return object(), "project-123"

    import google.cloud as google_cloud

    monkeypatch.setattr("google.auth.load_credentials_from_dict", fake_load_credentials_from_dict)
    monkeypatch.setattr(google_cloud, "storage", _FakeStorageModule(), raising=False)

    client = node_media_prep_app._build_storage_client()
    assert client.settings.gcs_bucket == "bucket"
    assert calls["project"] == "project-123"
    assert calls["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
