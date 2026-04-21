"""Tests for the remote node-media-prep runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.runtime.node_media_prep import (
    NodeMediaPrepRequest,
    run_node_media_prep,
)


class _FakeStorage:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, Path]] = []
        self.uploads: list[tuple[Path, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"video-bytes")
        self.downloads.append((gcs_uri, local_path))
        return local_path

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((local_path, object_name))
        return f"gs://bucket/{object_name}"


def test_from_payload_rejects_non_gcs_uri() -> None:
    with pytest.raises(ValueError, match="gs://"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "r",
                "video_gcs_uri": "http://example.com/video.mp4",
                "nodes": [],
            }
        )


def test_from_payload_rejects_missing_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "",
                "video_gcs_uri": "gs://b/v.mp4",
                "nodes": [],
            }
        )


def test_from_payload_rejects_inverted_time() -> None:
    with pytest.raises(ValueError, match="start_ms > end_ms"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "r",
                "video_gcs_uri": "gs://b/v.mp4",
                "nodes": [{"node_id": "n1", "start_ms": 2000, "end_ms": 1000}],
            }
        )


def test_from_payload_defaults_object_prefix() -> None:
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_123",
            "video_gcs_uri": "gs://b/v.mp4",
            "nodes": [{"node_id": "n1", "start_ms": 0, "end_ms": 1000}],
        }
    )
    assert req.object_prefix == "phase14/run_123/node_media"
    assert req.max_concurrency == 12


def test_run_node_media_prep_returns_empty_for_empty_nodes(tmp_path: Path) -> None:
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_empty",
            "video_gcs_uri": "gs://b/v.mp4",
            "nodes": [],
        }
    )
    result = run_node_media_prep(
        request=req,
        storage_client=_FakeStorage(),
        scratch_root=tmp_path,
    )
    assert result == {"run_id": "run_empty", "media": []}


def test_run_node_media_prep_returns_batch_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_prepare_node_media_embeddings(
        *,
        nodes,
        source_video_path,
        clips_dir,
        storage_client,
        object_prefix,
        max_concurrent,
        return_diagnostics,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        assert return_diagnostics is True
        return (
            [
                {
                    "node_id": "n1",
                    "file_uri": "gs://bucket/phase14/run_001/node_media/batches/batch_0000/n1.mp4",
                    "mime_type": "video/mp4",
                    "local_path": str(clips_dir / "n1.mp4"),
                },
                {
                    "node_id": "n2",
                    "file_uri": "gs://bucket/phase14/run_001/node_media/batches/batch_0000/n2.mp4",
                    "mime_type": "video/mp4",
                    "local_path": str(clips_dir / "n2.mp4"),
                },
            ],
            {
                "ffmpeg_mode": "hybrid_batch_gpu",
                "extract_ms": 25.0,
                "upload_ms": 15.0,
                "upload_bytes": 4096,
            },
        )

    monkeypatch.setattr(
        "backend.runtime.node_media_prep.prepare_node_media_embeddings",
        fake_prepare_node_media_embeddings,
    )

    storage = _FakeStorage()
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_001",
            "video_gcs_uri": "gs://bucket/source.mp4",
            "object_prefix": "phase14/run_001/node_media/batches/batch_0000",
            "max_concurrency": 2,
            "submitted_at_ms": 0.0,
            "nodes": [
                {"node_id": "n1", "start_ms": 0, "end_ms": 1000},
                {"node_id": "n2", "start_ms": 2000, "end_ms": 3000},
            ],
        }
    )
    result = run_node_media_prep(
        request=req,
        storage_client=storage,
        scratch_root=tmp_path,
    )

    assert result["run_id"] == "run_001"
    assert result["batch_id"] == "batch_0000"
    assert result["batch_start_ms"] == 0
    assert result["batch_end_ms"] == 3000
    assert result["node_count"] == 2
    assert result["ffmpeg_mode"] == "hybrid_batch_gpu"
    assert result["queue_wait_ms"] is not None
    assert result["download_ms"] >= 0.0
    assert result["download_bytes"] == len(b"video-bytes")
    assert result["extract_ms"] == 25.0
    assert result["upload_ms"] == 15.0
    assert result["upload_bytes"] == 4096
    assert result["total_ms"] >= 0.0
    assert [item["node_id"] for item in result["media"]] == ["n1", "n2"]
    assert len(storage.downloads) == 1
