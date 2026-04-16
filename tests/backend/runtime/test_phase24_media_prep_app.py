from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


class _FakeStorageClient:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        self.downloads.append((gcs_uri, str(local_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("video", encoding="utf-8")
        return local_path

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        return f"gs://bucket/{object_name}"


def _payload() -> dict[str, object]:
    return {
        "run_id": "run-123",
        "source_video_gcs_uri": "gs://bucket/source.mp4",
        "object_prefix": "phase14/run-123/node_media",
        "items": [
            {"node_id": "node_a", "start_ms": 0, "end_ms": 1000},
            {"node_id": "node_b", "start_ms": 1000, "end_ms": 2000},
        ],
    }


def test_phase24_media_prep_app_prepares_and_returns_ordered_descriptors(
    monkeypatch,
) -> None:
    import backend.runtime.phase24_media_prep_app as media_prep_app

    captured: dict[str, object] = {}

    def fake_prepare_node_media_embeddings(
        *,
        nodes,
        source_video_path,
        clips_dir,
        storage_client,
        object_prefix,
        max_concurrent=None,
    ):
        captured["node_ids"] = [node.node_id for node in nodes]
        captured["source_video_path"] = str(source_video_path)
        captured["clips_dir"] = str(clips_dir)
        captured["object_prefix"] = object_prefix
        return [
            {
                "node_id": "node_a",
                "file_uri": "gs://bucket/phase14/run-123/node_media/node_a.mp4",
                "mime_type": "video/mp4",
                "local_path": "/tmp/node_a.mp4",
            },
            {
                "node_id": "node_b",
                "file_uri": "gs://bucket/phase14/run-123/node_media/node_b.mp4",
                "mime_type": "video/mp4",
                "local_path": "/tmp/node_b.mp4",
            },
        ]

    monkeypatch.setattr(media_prep_app, "prepare_node_media_embeddings", fake_prepare_node_media_embeddings)

    storage_client = _FakeStorageClient()
    client = TestClient(
        media_prep_app.create_app(
            service=media_prep_app.Phase24MediaPrepService(storage_client=storage_client)
        )
    )

    response = client.post("/tasks/node-media-prep", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "items": [
            {"node_id": "node_a", "file_uri": "gs://bucket/phase14/run-123/node_media/node_a.mp4", "mime_type": "video/mp4"},
            {"node_id": "node_b", "file_uri": "gs://bucket/phase14/run-123/node_media/node_b.mp4", "mime_type": "video/mp4"},
        ],
    }
    assert storage_client.downloads[0][0] == "gs://bucket/source.mp4"
    assert captured["node_ids"] == ["node_a", "node_b"]
    assert captured["object_prefix"] == "phase14/run-123/node_media"


def test_phase24_media_prep_app_exposes_healthz() -> None:
    from backend.runtime.phase24_media_prep_app import Phase24MediaPrepService, create_app

    client = TestClient(create_app(service=Phase24MediaPrepService(storage_client=_FakeStorageClient())))
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_phase24_media_prep_app_exposes_asr_route() -> None:
    from backend.runtime.phase24_media_prep_app import Phase24MediaPrepService, create_app

    class _FakeASRService:
        def handle_request(self, payload):
            assert payload.audio_gcs_uri == "gs://bucket/source_audio.wav"
            assert payload.context_info == "custom hotwords"
            return {
                "turns": [
                    {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "hello world"},
                ]
            }

    client = TestClient(
        create_app(
            service=Phase24MediaPrepService(storage_client=_FakeStorageClient()),
            asr_service=_FakeASRService(),
        )
    )
    response = client.post(
        "/tasks/asr",
        json={
            "audio_gcs_uri": "gs://bucket/source_audio.wav",
            "context_info": "custom hotwords",
            "generation_config": {
                "repetition_penalty": 1.03,
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "turns": [
            {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "hello world"},
        ]
    }
