from __future__ import annotations

import threading
import time
from pathlib import Path
import subprocess

import pytest


def _make_node(node_id: str, start_ms: int, end_ms: int):
    from backend.pipeline.contracts import SemanticGraphNode

    return SemanticGraphNode(
        node_id=node_id,
        node_type="claim",
        start_ms=start_ms,
        end_ms=end_ms,
        transcript_text=f"node {node_id}",
        summary=f"summary {node_id}",
    )


def test_prepare_node_media_embeddings_preserves_input_order_and_uploads_all_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    nodes = [
        _make_node("node_a", 0, 1000),
        _make_node("node_b", 1000, 2000),
        _make_node("node_c", 2000, 3000),
    ]

    active_workers = 0
    max_seen_workers = 0
    lock = threading.Lock()
    uploaded_objects: list[str] = []
    delays = {"node_a": 0.06, "node_b": 0.01, "node_c": 0.03}

    def fake_extract_node_clip(*, source_video_path, output_path, start_ms, end_ms):
        nonlocal active_workers, max_seen_workers
        with lock:
            active_workers += 1
            max_seen_workers = max(max_seen_workers, active_workers)
        try:
            time.sleep(delays[output_path.stem])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{source_video_path}:{start_ms}:{end_ms}", encoding="utf-8")
            return output_path
        finally:
            with lock:
                active_workers -= 1

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            uploaded_objects.append(object_name)
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "extract_node_clip", fake_extract_node_clip)

    descriptors = media_embeddings.prepare_node_media_embeddings(
        nodes=nodes,
        source_video_path=tmp_path / "source.mp4",
        clips_dir=tmp_path / "clips",
        storage_client=_FakeStorageClient(),
        object_prefix="phase14/run-1/node_media",
        max_concurrent=2,
    )

    assert [descriptor["node_id"] for descriptor in descriptors] == [node.node_id for node in nodes]
    assert [descriptor["file_uri"] for descriptor in descriptors] == [
        "gs://bucket/phase14/run-1/node_media/node_a.mp4",
        "gs://bucket/phase14/run-1/node_media/node_b.mp4",
        "gs://bucket/phase14/run-1/node_media/node_c.mp4",
    ]
    assert sorted(uploaded_objects) == [
        "phase14/run-1/node_media/node_a.mp4",
        "phase14/run-1/node_media/node_b.mp4",
        "phase14/run-1/node_media/node_c.mp4",
    ]
    assert max_seen_workers == 2


def test_prepare_node_media_embeddings_uses_configured_concurrency_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    nodes = [
        _make_node("node_a", 0, 1000),
        _make_node("node_b", 1000, 2000),
        _make_node("node_c", 2000, 3000),
        _make_node("node_d", 3000, 4000),
    ]

    active_workers = 0
    max_seen_workers = 0
    lock = threading.Lock()

    def fake_extract_node_clip(*, source_video_path, output_path, start_ms, end_ms):
        nonlocal active_workers, max_seen_workers
        with lock:
            active_workers += 1
            max_seen_workers = max(max_seen_workers, active_workers)
        try:
            time.sleep(0.05)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("clip", encoding="utf-8")
            return output_path
        finally:
            with lock:
                active_workers -= 1

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "extract_node_clip", fake_extract_node_clip)

    descriptors = media_embeddings.prepare_node_media_embeddings(
        nodes=nodes,
        source_video_path=tmp_path / "source.mp4",
        clips_dir=tmp_path / "clips",
        storage_client=_FakeStorageClient(),
        object_prefix="phase14/run-1/node_media",
        max_concurrent=2,
    )

    assert len(descriptors) == len(nodes)
    assert max_seen_workers == 2


def test_prepare_node_media_embeddings_fails_fast_with_node_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    nodes = [
        _make_node("node_a", 0, 1000),
        _make_node("node_b", 1000, 2000),
    ]

    def fake_extract_node_clip(*, source_video_path, output_path, start_ms, end_ms):
        if output_path.stem == "node_b":
            raise RuntimeError("ffmpeg failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("clip", encoding="utf-8")
        return output_path

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "extract_node_clip", fake_extract_node_clip)

    with pytest.raises(RuntimeError, match=r"node_id=node_b.*ffmpeg failed"):
        media_embeddings.prepare_node_media_embeddings(
            nodes=nodes,
            source_video_path=tmp_path / "source.mp4",
            clips_dir=tmp_path / "clips",
            storage_client=_FakeStorageClient(),
            object_prefix="phase14/run-1/node_media",
            max_concurrent=2,
        )


def test_extract_node_clip_uses_480p_gpu_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    commands: list[list[str]] = []

    def fake_run(cmd, check, stdout, stderr):
        commands.append(list(cmd))
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("CLYPT_PHASE24_FFMPEG_DEVICE", "gpu")
    monkeypatch.setattr(media_embeddings.subprocess, "run", fake_run)

    output_path = media_embeddings.extract_node_clip(
        source_video_path=tmp_path / "source.mp4",
        output_path=tmp_path / "clips" / "node.mp4",
        start_ms=0,
        end_ms=1000,
    )

    assert output_path.exists()
    assert len(commands) == 1
    cmd = commands[0]
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1] == "scale_cuda=-2:480"


def test_extract_node_clip_uses_480p_cpu_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    commands: list[list[str]] = []

    def fake_run(cmd, check, stdout, stderr):
        commands.append(list(cmd))
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("CLYPT_PHASE24_FFMPEG_DEVICE", "cpu")
    monkeypatch.setattr(media_embeddings.subprocess, "run", fake_run)

    output_path = media_embeddings.extract_node_clip(
        source_video_path=tmp_path / "source.mp4",
        output_path=tmp_path / "clips" / "node.mp4",
        start_ms=0,
        end_ms=1000,
    )

    assert output_path.exists()
    assert len(commands) == 1
    cmd = commands[0]
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1] == "scale=-2:480"
