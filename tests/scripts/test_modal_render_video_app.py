from __future__ import annotations

import json
import sys
import types

from fastapi.testclient import TestClient


def _load_app_module():
    if "scripts.modal.render_video_app" in sys.modules:
        del sys.modules["scripts.modal.render_video_app"]

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
                setattr(fn, "_modal_function_kwargs", dict(_kwargs))
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

    from scripts.modal import render_video_app

    return render_video_app


def _payload() -> dict:
    return {
        "run_id": "run-123",
        "source_video_gcs_uri": "gs://bucket/source.mp4",
        "artifact_gcs_uris": {
            "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
            "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
        },
        "clips": [{"clip_id": "clip_001", "clip_start_ms": 0, "clip_end_ms": 1000}],
    }


def test_render_video_health_returns_ok(monkeypatch) -> None:
    render_video_app = _load_app_module()
    monkeypatch.setattr(render_video_app, "_require_ffmpeg", lambda: None)
    client = TestClient(render_video_app.web_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_render_video_submit_returns_call_id(monkeypatch) -> None:
    render_video_app = _load_app_module()
    monkeypatch.setattr(render_video_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setenv("PHASE6_RENDER_AUTH_TOKEN", "render-token")

    class _FakeSpawnedCall:
        object_id = "fc-render"

    class _FakeJob:
        @staticmethod
        def spawn(payload):
            assert payload["run_id"] == "run-123"
            return _FakeSpawnedCall()

    monkeypatch.setattr(render_video_app, "render_video_job", _FakeJob)

    client = TestClient(render_video_app.web_app)
    response = client.post(
        "/tasks/render-video",
        json=_payload(),
        headers={"Authorization": "Bearer render-token"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "call_id": "fc-render",
        "status": "submitted",
        "result_path": "/tasks/render-video/result/fc-render",
    }


def test_render_video_result_returns_completed_payload(monkeypatch) -> None:
    render_video_app = _load_app_module()
    monkeypatch.setattr(render_video_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setenv("PHASE6_RENDER_AUTH_TOKEN", "render-token")

    class _CompleteCall:
        def __init__(self, object_id: str):
            self.object_id = object_id

        def get(self, timeout=0):  # noqa: ARG002
            return {
                "run_id": "run-123",
                "outputs": [{"clip_id": "clip_001", "video_gcs_uri": "gs://bucket/render/clip_001.mp4"}],
            }

    monkeypatch.setattr(
        render_video_app.modal.FunctionCall,
        "from_id",
        classmethod(lambda cls, object_id: _CompleteCall(object_id)),
    )

    client = TestClient(render_video_app.web_app)
    response = client.get(
        "/tasks/render-video/result/fc-render",
        headers={"Authorization": "Bearer render-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "outputs": [{"clip_id": "clip_001", "video_gcs_uri": "gs://bucket/render/clip_001.mp4"}],
        "call_id": "fc-render",
        "status": "succeeded",
    }
