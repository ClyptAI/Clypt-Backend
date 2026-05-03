from __future__ import annotations

import sys
import types

from fastapi.testclient import TestClient


def _load_app_module():
    if "scripts.modal.visual_extract_app" in sys.modules:
        del sys.modules["scripts.modal.visual_extract_app"]

    class _FakeImage:
        commands: list[str]

        def __init__(self):
            self.commands = []

        @staticmethod
        def debian_slim(*_args, **_kwargs):
            return _FakeImage()

        def apt_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def pip_install_from_requirements(self, *_args, **_kwargs):
            return self

        def run_commands(self, *commands, **_kwargs):
            self.commands.extend(commands)
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

    from scripts.modal import visual_extract_app

    return visual_extract_app


def _payload() -> dict:
    return {
        "run_id": "run-visual",
        "video_gcs_uri": "gs://bucket/source.mp4",
        "source_video_sha256": "sha256:abc",
    }


def test_visual_extract_submit_returns_call_id(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setenv("VISUAL_EXTRACT_AUTH_TOKEN", "visual-token")

    captured: dict[str, object] = {}

    class _FakeSpawnedCall:
        object_id = "fc-visual"

    class _FakeJob:
        @staticmethod
        def spawn(payload):
            captured["payload"] = payload
            return _FakeSpawnedCall()

    monkeypatch.setattr(visual_extract_app, "visual_extract_job", _FakeJob)

    client = TestClient(visual_extract_app.web_app)
    response = client.post(
        "/tasks/visual-extract",
        json=_payload(),
        headers={"Authorization": "Bearer visual-token"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "call_id": "fc-visual",
        "status": "submitted",
        "result_path": "/tasks/visual-extract/result/fc-visual",
    }
    assert captured["payload"]["video_gcs_uri"] == "gs://bucket/source.mp4"


def test_visual_extract_result_returns_completed_payload(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setenv("VISUAL_EXTRACT_AUTH_TOKEN", "visual-token")

    class _CompleteCall:
        def __init__(self, object_id: str):
            self.object_id = object_id

        def get(self, timeout=0):  # noqa: ARG002
            return {
                "run_id": "run-visual",
                "phase1_visual": {"shot_changes": [], "tracks": []},
            }

    monkeypatch.setattr(
        visual_extract_app.modal.FunctionCall,
        "from_id",
        classmethod(lambda cls, object_id: _CompleteCall(object_id)),
    )

    client = TestClient(visual_extract_app.web_app)
    response = client.get(
        "/tasks/visual-extract/result/fc-visual",
        headers={"Authorization": "Bearer visual-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-visual",
        "phase1_visual": {"shot_changes": [], "tracks": []},
        "call_id": "fc-visual",
        "status": "succeeded",
    }


def test_visual_extract_gpu_function_is_dedicated_l40s() -> None:
    visual_extract_app = _load_app_module()
    kwargs = visual_extract_app.visual_extract_job._modal_function_kwargs
    assert kwargs["gpu"] == "L40S"
    assert kwargs["min_containers"] == 1
    assert kwargs["max_containers"] == 1


def test_visual_extract_image_installs_trtexec() -> None:
    visual_extract_app = _load_app_module()
    commands = "\n".join(visual_extract_app.image.commands)
    assert "libnvinfer-bin" in commands
    assert "trtexec" in commands


def test_require_visual_runtime_uses_trtexec_help(monkeypatch) -> None:
    visual_extract_app = _load_app_module()

    monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
        ),
    )

    check_output_calls: list[list[str]] = []

    def _fake_check_output(cmd, **_kwargs):
        check_output_calls.append(cmd)
        if cmd == ["ffmpeg", "-hwaccels"]:
            return "cuda\n"
        if cmd == ["ffmpeg", "-filters"]:
            return "scale_cuda\n"
        raise AssertionError(f"unexpected check_output command: {cmd}")

    run_calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        run_calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(visual_extract_app.subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(visual_extract_app.subprocess, "run", _fake_run)

    visual_extract_app._require_visual_runtime()

    assert check_output_calls == [["ffmpeg", "-hwaccels"], ["ffmpeg", "-filters"]]
    assert run_calls == [["trtexec", "--help"]]
