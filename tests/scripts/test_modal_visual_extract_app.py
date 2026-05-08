from __future__ import annotations

import sys
import types
from pathlib import Path

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


def test_visual_defaults_use_modal_segmentation_model_env(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setenv("CLYPT_MODAL_VISUAL_MODEL", "seg_nano")
    monkeypatch.setenv("CLYPT_MODAL_VISUAL_BATCH_SIZE", "8")
    monkeypatch.delenv("CLYPT_PHASE1_VISUAL_MODEL", raising=False)
    monkeypatch.delenv("CLYPT_PHASE1_VISUAL_BATCH_SIZE", raising=False)

    visual_extract_app._set_visual_defaults()

    import os

    assert os.environ["CLYPT_PHASE1_VISUAL_MODEL"] == "seg_nano"
    assert os.environ["CLYPT_PHASE1_VISUAL_BATCH_SIZE"] == "8"


def test_upload_mask_artifacts_replaces_local_paths_with_gcs_uri(tmp_path: Path) -> None:
    visual_extract_app = _load_app_module()
    artifact_path = tmp_path / "visual_masks_lowres_v1.npz"
    artifact_path.write_bytes(b"npz")
    phase1_visual = {
        "mask_artifacts": [
            {
                "artifact_id": "visual_masks_lowres_v1",
                "local_path": str(artifact_path),
                "encoding": "npz_compressed_lowres_binary_v1",
            }
        ],
        "tracking_metrics": {},
    }

    class _Storage:
        def upload_file(self, *, local_path, object_name):
            assert local_path == artifact_path
            assert object_name == "phase14/run-visual/visual/visual_masks_lowres_v1.npz"
            return f"gs://bucket/{object_name}"

    visual_extract_app._upload_mask_artifacts(
        phase1_visual=phase1_visual,
        run_id="run-visual",
        storage_client=_Storage(),
    )

    artifact = phase1_visual["mask_artifacts"][0]
    assert "local_path" not in artifact
    assert artifact["gcs_uri"] == "gs://bucket/phase14/run-visual/visual/visual_masks_lowres_v1.npz"
    assert phase1_visual["tracking_metrics"]["mask_artifacts"] == phase1_visual["mask_artifacts"]


def test_upload_phase1_visual_artifact_writes_gzipped_json(tmp_path: Path) -> None:
    visual_extract_app = _load_app_module()
    phase1_visual = {"shot_changes": [], "tracks": [{"track_id": "p1"}]}
    uploaded: dict[str, object] = {}

    class _Storage:
        def upload_file(self, *, local_path, object_name):
            uploaded["local_path"] = local_path
            uploaded["object_name"] = object_name
            assert local_path.exists()
            return f"gs://bucket/{object_name}"

    artifact = visual_extract_app._upload_phase1_visual_artifact(
        phase1_visual_payload=phase1_visual,
        run_id="run-visual",
        storage_client=_Storage(),
    )

    assert artifact["artifact_id"] == "phase1_visual_v1"
    assert artifact["encoding"] == "json_gzip_v1"
    assert artifact["gcs_uri"] == "gs://bucket/phase14/run-visual/visual/phase1_visual.json.gz"
    assert uploaded["object_name"] == "phase14/run-visual/visual/phase1_visual.json.gz"


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


def test_ready_returns_503_when_gpu_probe_is_cold(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setattr(visual_extract_app, "_build_storage_client", lambda: object())
    monkeypatch.setattr(visual_extract_app, "_read_visual_readiness_state", lambda _client: None)
    monkeypatch.setattr(visual_extract_app, "_compute_visual_runtime_fingerprint", lambda: "fp-1")
    monkeypatch.setattr(
        visual_extract_app,
        "visual_extract_job",
        types.SimpleNamespace(
            remote=staticmethod(
                lambda payload: {  # noqa: ARG005
                    "status": "cold",
                    "reason": "gpu_worker_not_warmed",
                    "runtime_fingerprint": "fp-1",
                }
            )
        ),
    )

    client = TestClient(visual_extract_app.web_app)
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "cold"


def test_ready_returns_200_when_gpu_probe_is_ready(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setattr(visual_extract_app, "_build_storage_client", lambda: object())
    monkeypatch.setattr(visual_extract_app, "_read_visual_readiness_state", lambda _client: None)
    monkeypatch.setattr(visual_extract_app, "_compute_visual_runtime_fingerprint", lambda: "fp-1")
    monkeypatch.setattr(
        visual_extract_app,
        "visual_extract_job",
        types.SimpleNamespace(
            remote=staticmethod(
                lambda payload: {  # noqa: ARG005
                    "status": "ready",
                    "reason": "gpu_worker_warm",
                    "runtime_fingerprint": "fp-1",
                    "gpu_container_boot_id": "boot-1",
                }
            )
        ),
    )

    client = TestClient(visual_extract_app.web_app)
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["persisted_state_status"] == "missing"


def test_visual_warmup_submit_returns_call_id_and_records_warming_state(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_ffmpeg", lambda: None)
    monkeypatch.setenv("VISUAL_EXTRACT_AUTH_TOKEN", "visual-token")

    captured: dict[str, object] = {}

    class _FakeSpawnedCall:
        object_id = "fc-warmup"

    class _FakeJob:
        @staticmethod
        def spawn(payload):
            captured["payload"] = payload
            return _FakeSpawnedCall()

    monkeypatch.setattr(visual_extract_app, "visual_extract_job", _FakeJob)
    monkeypatch.setattr(visual_extract_app, "_build_storage_client", lambda: object())
    monkeypatch.setattr(
        visual_extract_app,
        "_write_visual_readiness_state",
        lambda **kwargs: captured.setdefault("ready_state", kwargs["payload"]),
    )
    monkeypatch.setattr(visual_extract_app, "_compute_visual_runtime_fingerprint", lambda: "fp-1")

    client = TestClient(visual_extract_app.web_app)
    response = client.post(
        "/tasks/visual-warmup",
        headers={"Authorization": "Bearer visual-token"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "call_id": "fc-warmup",
        "status": "submitted",
        "result_path": "/tasks/visual-warmup/result/fc-warmup",
    }
    assert captured["payload"]["job_kind"] == "warmup"
    assert captured["ready_state"]["status"] == "warming"


def test_visual_extract_job_ready_probe_returns_cold_without_warm_state(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_visual_runtime", lambda: None)
    monkeypatch.setattr(visual_extract_app, "_build_storage_client", lambda: object())
    monkeypatch.setattr(visual_extract_app, "_compute_visual_runtime_fingerprint", lambda: "fp-1")
    monkeypatch.setattr(visual_extract_app, "_GPU_CONTAINER_BOOT_ID", "boot-1")
    monkeypatch.setattr(visual_extract_app, "_GPU_READY_STATE", None)

    response = visual_extract_app.visual_extract_job({"job_kind": "ready_probe"})

    assert response == {
        "status": "cold",
        "reason": "gpu_worker_not_warmed",
        "runtime_fingerprint": "fp-1",
        "gpu_container_boot_id": "boot-1",
    }


def test_visual_extract_job_warmup_marks_gpu_ready(monkeypatch) -> None:
    visual_extract_app = _load_app_module()
    monkeypatch.setattr(visual_extract_app, "_require_visual_runtime", lambda: None)
    monkeypatch.setattr(visual_extract_app, "_build_storage_client", lambda: object())
    monkeypatch.setattr(visual_extract_app, "_compute_visual_runtime_fingerprint", lambda: "fp-1")
    monkeypatch.setattr(visual_extract_app, "_GPU_CONTAINER_BOOT_ID", "boot-1")
    monkeypatch.setattr(visual_extract_app, "_GPU_READY_STATE", None)

    written: dict[str, object] = {}

    monkeypatch.setattr(
        visual_extract_app,
        "_extract_phase1_visual",
        lambda **kwargs: {  # noqa: ARG005
            "tracking_metrics": {
                "emitted_track_rows": 4,
                "pose_validated_tracklets": 2,
                "pose_auto_follow_eligible_tracklets": 1,
            },
            "tracks": [],
            "shot_changes": [],
        },
    )
    monkeypatch.setattr(
        visual_extract_app,
        "VisualPayload",
        types.SimpleNamespace(model_validate=staticmethod(lambda payload: payload)),
    )
    monkeypatch.setattr(
        visual_extract_app,
        "_write_visual_readiness_state",
        lambda **kwargs: written.setdefault("payload", kwargs["payload"]),
    )

    response = visual_extract_app.visual_extract_job(
        {"job_kind": "warmup", "submitted_at_ms": 0.0}
    )

    assert response["status"] == "succeeded"
    assert response["warmup"]["pose_validated_tracklets"] == 2
    assert response["readiness"]["gpu_container_boot_id"] == "boot-1"
    assert written["payload"]["status"] == "ready"
    assert visual_extract_app._GPU_READY_STATE["status"] == "ready"
