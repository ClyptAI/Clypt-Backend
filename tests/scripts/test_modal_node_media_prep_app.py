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

    class _FakeFunctionCall:
        def __init__(self, object_id: str):
            self.object_id = object_id

        @classmethod
        def from_id(cls, object_id: str):
            return cls(object_id)

        def get(self, timeout=0):  # noqa: ARG002
            raise TimeoutError

    class _FakeExceptionModule:
        class OutputExpiredError(Exception):
            pass

        class NotFoundError(Exception):
            pass

    fake_modal = types.SimpleNamespace(
        App=_FakeApp,
        Image=_FakeImage,
        Secret=_FakeSecret,
        FunctionCall=_FakeFunctionCall,
        exception=_FakeExceptionModule,
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


def test_node_media_prep_submit_returns_call_id(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    monkeypatch.setenv("NODE_MEDIA_PREP_AUTH_TOKEN", "secret-token")

    captured: dict[str, object] = {}

    class _FakeSpawnedCall:
        object_id = "fc-123"

    class _FakeJob:
        @staticmethod
        def spawn(payload):
            captured["payload"] = payload
            return _FakeSpawnedCall()

    monkeypatch.setattr(node_media_prep_app, "node_media_prep_job", _FakeJob)

    client = TestClient(node_media_prep_app.web_app)
    response = client.post(
        "/tasks/node-media-prep",
        json=_payload(),
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "call_id": "fc-123",
        "status": "submitted",
        "result_path": "/tasks/node-media-prep/result/fc-123",
    }
    assert captured["payload"]["run_id"] == "run-123"


def test_node_media_prep_result_returns_pending(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    monkeypatch.setenv("NODE_MEDIA_PREP_AUTH_TOKEN", "secret-token")

    class _PendingCall:
        def __init__(self, object_id: str):
            self.object_id = object_id

        def get(self, timeout=0):  # noqa: ARG002
            raise TimeoutError

    monkeypatch.setattr(
        node_media_prep_app.modal.FunctionCall,
        "from_id",
        classmethod(lambda cls, object_id: _PendingCall(object_id)),
    )

    client = TestClient(node_media_prep_app.web_app)
    response = client.get(
        "/tasks/node-media-prep/result/fc-123",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 202
    assert response.json() == {"call_id": "fc-123", "status": "pending"}


def test_node_media_prep_result_returns_completed_payload(monkeypatch) -> None:
    node_media_prep_app = _load_app_module()
    monkeypatch.setattr(node_media_prep_app, "_require_codec", lambda *_args: None)
    monkeypatch.setenv("NODE_MEDIA_PREP_AUTH_TOKEN", "secret-token")

    class _CompleteCall:
        def __init__(self, object_id: str):
            self.object_id = object_id

        def get(self, timeout=0):  # noqa: ARG002
            return {
                "run_id": "run-123",
                "batch_id": "batch_0000",
                "batch_start_ms": 0,
                "batch_end_ms": 1000,
                "node_count": 1,
                "ffmpeg_mode": "hybrid_batch_gpu",
                "download_ms": 10.0,
                "extract_ms": 20.0,
                "upload_ms": 30.0,
                "total_ms": 60.0,
                "media": [{"node_id": "n_1", "file_uri": "gs://bucket/n_1.mp4"}],
            }

    monkeypatch.setattr(
        node_media_prep_app.modal.FunctionCall,
        "from_id",
        classmethod(lambda cls, object_id: _CompleteCall(object_id)),
    )

    client = TestClient(node_media_prep_app.web_app)
    response = client.get(
        "/tasks/node-media-prep/result/fc-123",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "batch_id": "batch_0000",
        "batch_start_ms": 0,
        "batch_end_ms": 1000,
        "node_count": 1,
        "ffmpeg_mode": "hybrid_batch_gpu",
        "download_ms": 10.0,
        "extract_ms": 20.0,
        "upload_ms": 30.0,
        "total_ms": 60.0,
        "media": [{"node_id": "n_1", "file_uri": "gs://bucket/n_1.mp4"}],
        "call_id": "fc-123",
        "status": "succeeded",
    }


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
