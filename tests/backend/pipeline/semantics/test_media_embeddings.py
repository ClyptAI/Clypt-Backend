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


def test_plan_node_media_batches_groups_adjacent_nodes() -> None:
    from backend.pipeline.semantics.media_embeddings import plan_node_media_batches

    batches = plan_node_media_batches(
        nodes=[
            _make_node("node_a", 0, 1000),
            _make_node("node_b", 1500, 2500),
            _make_node("node_c", 4300, 5200),
        ]
    )

    assert len(batches) == 1
    assert batches[0].batch_id == "batch_0000"
    assert [node.node_id for node in batches[0].nodes] == ["node_a", "node_b", "node_c"]
    assert batches[0].start_ms == 0
    assert batches[0].end_ms == 5200


def test_plan_node_media_batches_splits_sparse_nodes_and_caps_span_and_count() -> None:
    from backend.pipeline.semantics.media_embeddings import plan_node_media_batches

    nodes = [
        _make_node(f"node_{idx}", idx * 1000, idx * 1000 + 900)
        for idx in range(9)
    ]
    nodes.append(_make_node("node_far", 200000, 201000))

    batches = plan_node_media_batches(nodes=nodes)

    assert [batch.batch_id for batch in batches] == ["batch_0000", "batch_0001", "batch_0002"]
    assert [node.node_id for node in batches[0].nodes] == [f"node_{idx}" for idx in range(8)]
    assert [node.node_id for node in batches[1].nodes] == ["node_8"]
    assert [node.node_id for node in batches[2].nodes] == ["node_far"]


def test_plan_node_media_batches_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.pipeline.semantics.media_embeddings import plan_node_media_batches

    monkeypatch.setenv("CLYPT_PHASE24_NODE_MEDIA_BATCH_GAP_MS", "500")
    monkeypatch.setenv("CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_NODES", "2")
    monkeypatch.setenv("CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_SPAN_MS", "1500")

    batches = plan_node_media_batches(
        nodes=[
            _make_node("node_a", 0, 400),
            _make_node("node_b", 700, 1100),
            _make_node("node_c", 1400, 1800),
        ]
    )

    assert [batch.batch_id for batch in batches] == ["batch_0000", "batch_0001"]
    assert [node.node_id for node in batches[0].nodes] == ["node_a", "node_b"]
    assert [node.node_id for node in batches[1].nodes] == ["node_c"]


def test_prepare_node_media_embeddings_preserves_input_order_and_uploads_all_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    nodes = [
        _make_node("node_a", 12000, 15000),
        _make_node("node_b", 15500, 18000),
        _make_node("node_c", 18500, 21000),
    ]

    batch_calls: list[dict[str, int]] = []
    trim_calls: list[dict[str, int | str]] = []
    uploaded_objects: list[str] = []

    def fake_extract_batch_window(*, source_video_path, output_path, coarse_start_ms, padded_end_ms):
        batch_calls.append(
            {
                "coarse_start_ms": coarse_start_ms,
                "padded_end_ms": padded_end_ms,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"batch")
        return output_path

    def fake_trim_node_clip_from_batch(*, batch_video_path, output_path, local_start_ms, local_end_ms):
        trim_calls.append(
            {
                "name": output_path.stem,
                "local_start_ms": local_start_ms,
                "local_end_ms": local_end_ms,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return output_path

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            uploaded_objects.append(object_name)
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "_source_duration_ms", lambda **kwargs: 120_000)
    monkeypatch.setattr(media_embeddings, "_extract_batch_window", fake_extract_batch_window)
    monkeypatch.setattr(media_embeddings, "_trim_node_clip_from_batch", fake_trim_node_clip_from_batch)

    descriptors, diagnostics = media_embeddings.prepare_node_media_embeddings(
        nodes=nodes,
        source_video_path=tmp_path / "source.mp4",
        clips_dir=tmp_path / "clips",
        storage_client=_FakeStorageClient(),
        object_prefix="phase14/run-1/node_media/batches/batch_0000",
        max_concurrent=2,
        return_diagnostics=True,
    )

    assert len(batch_calls) == 1
    assert batch_calls[0] == {"coarse_start_ms": 0, "padded_end_ms": 23000}
    assert trim_calls == [
        {"name": "node_a", "local_start_ms": 12000, "local_end_ms": 15000},
        {"name": "node_b", "local_start_ms": 15500, "local_end_ms": 18000},
        {"name": "node_c", "local_start_ms": 18500, "local_end_ms": 21000},
    ]
    assert [descriptor["node_id"] for descriptor in descriptors] == ["node_a", "node_b", "node_c"]
    assert [descriptor["file_uri"] for descriptor in descriptors] == [
        "gs://bucket/phase14/run-1/node_media/batches/batch_0000/node_a.mp4",
        "gs://bucket/phase14/run-1/node_media/batches/batch_0000/node_b.mp4",
        "gs://bucket/phase14/run-1/node_media/batches/batch_0000/node_c.mp4",
    ]
    assert sorted(uploaded_objects) == [
        "phase14/run-1/node_media/batches/batch_0000/node_a.mp4",
        "phase14/run-1/node_media/batches/batch_0000/node_b.mp4",
        "phase14/run-1/node_media/batches/batch_0000/node_c.mp4",
    ]
    assert diagnostics["batch_start_ms"] == 12000
    assert diagnostics["batch_end_ms"] == 21000
    assert diagnostics["coarse_start_ms"] == 0
    assert diagnostics["upload_bytes"] == len(b"clip") * 3


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

    def fake_extract_batch_window(*, source_video_path, output_path, coarse_start_ms, padded_end_ms):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"batch")
        return output_path

    def fake_trim_node_clip_from_batch(*, batch_video_path, output_path, local_start_ms, local_end_ms):
        nonlocal active_workers, max_seen_workers
        with lock:
            active_workers += 1
            max_seen_workers = max(max_seen_workers, active_workers)
        try:
            time.sleep(0.05)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"clip")
            return output_path
        finally:
            with lock:
                active_workers -= 1

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "_source_duration_ms", lambda **kwargs: 120_000)
    monkeypatch.setattr(media_embeddings, "_extract_batch_window", fake_extract_batch_window)
    monkeypatch.setattr(media_embeddings, "_trim_node_clip_from_batch", fake_trim_node_clip_from_batch)

    descriptors = media_embeddings.prepare_node_media_embeddings(
        nodes=nodes,
        source_video_path=tmp_path / "source.mp4",
        clips_dir=tmp_path / "clips",
        storage_client=_FakeStorageClient(),
        object_prefix="phase14/run-1/node_media/batches/batch_0000",
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

    def fake_extract_batch_window(*, source_video_path, output_path, coarse_start_ms, padded_end_ms):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"batch")
        return output_path

    def fake_trim_node_clip_from_batch(*, batch_video_path, output_path, local_start_ms, local_end_ms):
        if output_path.stem == "node_b":
            raise RuntimeError("ffmpeg failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return output_path

    class _FakeStorageClient:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    monkeypatch.setattr(media_embeddings, "_source_duration_ms", lambda **kwargs: 120_000)
    monkeypatch.setattr(media_embeddings, "_extract_batch_window", fake_extract_batch_window)
    monkeypatch.setattr(media_embeddings, "_trim_node_clip_from_batch", fake_trim_node_clip_from_batch)

    with pytest.raises(RuntimeError, match=r"node_id=node_b.*ffmpeg failed"):
        media_embeddings.prepare_node_media_embeddings(
            nodes=nodes,
            source_video_path=tmp_path / "source.mp4",
            clips_dir=tmp_path / "clips",
            storage_client=_FakeStorageClient(),
            object_prefix="phase14/run-1/node_media/batches/batch_0000",
            max_concurrent=2,
        )


def test_extract_node_clip_uses_480p_gpu_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    commands: list[list[str]] = []

    def fake_run(cmd, check, stdout=None, stderr=None, capture_output=False, text=False):
        commands.append(list(cmd))
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"streams":[{"width":1280,"height":720}],"format":{"duration":"10.0"}}',
                stderr="",
            )
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
    assert len(commands) == 2
    cmd = commands[1]
    resize_index = cmd.index("-resize")
    assert cmd[resize_index + 1] == "854x480"


def test_extract_node_clip_uses_480p_cpu_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.semantics import media_embeddings

    commands: list[list[str]] = []

    def fake_run(cmd, check, stdout=None, stderr=None, capture_output=False, text=False):
        commands.append(list(cmd))
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"streams":[{"width":1280,"height":720}],"format":{"duration":"10.0"}}',
                stderr="",
            )
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
    assert len(commands) == 2
    cmd = commands[1]
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1] == "scale=854:480"
