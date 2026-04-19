from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


class _FakeStorageClient:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, Path]] = []
        self.uploads: list[tuple[Path, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> None:
        self.downloads.append((gcs_uri, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake canonical audio")

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((local_path, object_name))
        return f"gs://bucket/{object_name}"


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, *, audio_path, audio_gcs_uri, context_info=None):  # noqa: ARG002
        self.calls.append(audio_gcs_uri)
        if audio_gcs_uri.endswith("shard_000_of_002.wav"):
            return [
                {"Speaker": 3, "Start": 10.0, "End": 12.0, "Content": "host intro"},
                {"Speaker": 7, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
            ]
        return [
            {"Speaker": 1, "Start": 5.0, "End": 7.0, "Content": "host follow-up"},
            {"Speaker": 2, "Start": 9.0, "End": 11.0, "Content": "guest follow-up"},
        ]


class _FakeVerifier:
    def similarity(self, left: tuple[int, int], right: tuple[int, int]) -> float:
        scores = {
            ((0, 3), (1, 1)): 0.93,
            ((0, 7), (1, 2)): 0.91,
        }
        return scores.get((left, right), 0.0)


class _FakeLongFormSettings:
    enabled = True
    single_pass_max_minutes = 60
    two_shard_max_minutes = 120
    three_shard_max_minutes = 180
    max_shards = 3
    speaker_match_threshold = 0.85
    representative_clip_min_seconds = 15.0
    representative_clip_max_seconds = 30.0


class _FakeDeps:
    def __init__(self, scratch_root: Path) -> None:
        self.vibevoice_provider = _FakeProvider()
        self.storage_client = _FakeStorageClient()
        self.scratch_root = scratch_root
        self.expected_auth_token = "token"
        self.speaker_verifier = _FakeVerifier()
        self.longform_settings = _FakeLongFormSettings()


def test_phase1_vibevoice_health_ok(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime.phase1_vibevoice_service import app as app_module

    deps = _FakeDeps(tmp_path)
    monkeypatch.setattr(app_module, "get_app_deps", lambda: deps)

    with TestClient(app_module.create_app()) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_phase1_vibevoice_asr_longform_merges_two_shards(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime.phase1_vibevoice_service import app as app_module

    deps = _FakeDeps(tmp_path)
    monkeypatch.setattr(app_module, "get_app_deps", lambda: deps)
    monkeypatch.setattr(app_module, "_probe_audio_duration_s", lambda path: 91 * 60, raising=False)
    monkeypatch.setattr(
        app_module,
        "_extract_shard_audio",
        lambda *, source_audio_path, output_audio_path, start_s, end_s: output_audio_path.write_bytes(b"shard"),  # noqa: ARG005
        raising=False,
    )

    with TestClient(app_module.create_app()) as client:
        resp = client.post(
            "/tasks/vibevoice-asr",
            json={
                "run_id": "run-1",
                "audio_gcs_uri": "gs://bucket/canonical.wav",
            },
            headers={"Authorization": "Bearer token"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["turns"] == [
        {"Speaker": 0, "Start": 10.0, "End": 12.0, "Content": "host intro"},
        {"Speaker": 1, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
        {"Speaker": 0, "Start": 2735.0, "End": 2737.0, "Content": "host follow-up"},
        {"Speaker": 1, "Start": 2739.0, "End": 2741.0, "Content": "guest follow-up"},
    ]
    assert len(deps.vibevoice_provider.calls) == 2
    assert len(deps.storage_client.uploads) == 2
    assert any(event["stage_name"] == "vibevoice_longform_merge" for event in body["stage_events"])
