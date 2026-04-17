"""Smoke tests for the RTX 6000 Ada Phase 1 audio host FastAPI app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.runtime.phase1_audio_service import app as app_module
from backend.runtime.phase1_audio_service.deps import AppDeps


class _FakeStorage:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.downloads: list[tuple[str, Path]] = []
        self.uploads: list[tuple[Path, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"audio-bytes")
        self.downloads.append((gcs_uri, local_path))
        return local_path

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((local_path, object_name))
        return f"gs://bucket/{object_name}"


class _FakeVibeVoice:
    def run(self, *, audio_path: str, audio_gcs_uri: str | None = None):
        return [{"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"}]


class _FakeAligner:
    def run(self, *, audio_path: str, turns: list[dict[str, Any]]):
        return [
            {
                "word_id": "w_000001",
                "text": "hello",
                "start_ms": 0,
                "end_ms": 300,
                "speaker_id": "SPEAKER_0",
            }
        ]


class _FakeEmotion:
    def run(self, *, audio_path: str, turns: list[dict[str, Any]]):
        return {"segments": []}


class _FakeYamnet:
    def run(self, *, audio_path: str):
        return {"events": []}


@pytest.fixture()
def deps(tmp_path: Path) -> AppDeps:
    return AppDeps(
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeAligner(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        storage_client=_FakeStorage(tmp_path),
        scratch_root=tmp_path,
        expected_auth_token="test-token",
    )


@pytest.fixture()
def client(deps: AppDeps):
    app = app_module.create_app()
    app.dependency_overrides[app_module.get_app_deps] = lambda: deps
    with TestClient(app) as c:
        yield c


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["audio_chain_ready"] is True


def test_phase1_audio_requires_bearer(client: TestClient) -> None:
    resp = client.post(
        "/tasks/phase1-audio",
        json={"audio_gcs_uri": "gs://bucket/audio.wav"},
    )
    assert resp.status_code == 401


def test_phase1_audio_rejects_wrong_bearer(client: TestClient) -> None:
    resp = client.post(
        "/tasks/phase1-audio",
        json={"audio_gcs_uri": "gs://bucket/audio.wav"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 403


def test_phase1_audio_returns_merged_payload(client: TestClient, deps: AppDeps) -> None:
    resp = client.post(
        "/tasks/phase1-audio",
        json={
            "audio_gcs_uri": "gs://bucket/audio.wav",
            "run_id": "run_001",
            "source_url": "https://youtu.be/demo",
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == "run_001"
    assert body["turns"][0]["Content"] == "hello"
    assert body["diarization_payload"]["words"][0]["text"] == "hello"
    assert body["emotion2vec_payload"] == {"segments": []}
    assert body["yamnet_payload"] == {"events": []}
    stage_names = {ev["stage_name"] for ev in body["stage_events"]}
    assert {"vibevoice_asr", "forced_alignment", "emotion2vec", "yamnet"} <= stage_names
    # Audio was downloaded from GCS exactly once.
    downloads = deps.storage_client.downloads  # type: ignore[attr-defined]
    assert len(downloads) == 1
    assert downloads[0][0] == "gs://bucket/audio.wav"


def test_phase1_audio_rejects_non_200_on_download_failure(
    client: TestClient, deps: AppDeps
) -> None:
    def _boom(*, gcs_uri: str, local_path: Path) -> Path:
        raise RuntimeError("gcs 404")

    deps.storage_client.download_file = _boom  # type: ignore[attr-defined]
    resp = client.post(
        "/tasks/phase1-audio",
        json={"audio_gcs_uri": "gs://bucket/missing.wav"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 502
    assert "gcs 404" in resp.text


def test_node_media_prep_rejects_non_json_body(client: TestClient) -> None:
    resp = client.post(
        "/tasks/node-media-prep",
        content=b"not-json",
        headers={"Authorization": "Bearer test-token", "content-type": "application/json"},
    )
    assert resp.status_code == 400


def test_node_media_prep_validates_request(client: TestClient) -> None:
    resp = client.post(
        "/tasks/node-media-prep",
        json={"run_id": "run_001"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400
    assert "video_gcs_uri" in resp.text
