from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess
import time
from typing import Any

from ..contracts import SemanticGraphNode

logger = logging.getLogger(__name__)
_NODE_MEDIA_TARGET_HEIGHT = 480
_DEFAULT_NODE_MEDIA_BATCH_GAP_MS = 2000
_DEFAULT_NODE_MEDIA_BATCH_MAX_NODES = 8
_DEFAULT_NODE_MEDIA_BATCH_MAX_SPAN_MS = 120000
_DEFAULT_NODE_MEDIA_BATCH_PAD_MS = 2000
_DEFAULT_NODE_MEDIA_BATCH_COARSE_SEEK_PAD_MS = 10000


@dataclass(slots=True)
class NodeMediaBatchPlan:
    batch_id: str
    nodes: list[Any]
    start_ms: int
    end_ms: int


def _format_seconds(value_ms: int) -> str:
    return f"{max(value_ms, 0) / 1000.0:.3f}"


def _node_window(node: Any) -> tuple[int, int]:
    start_ms = int(getattr(node, "start_ms", 0))
    end_ms = int(getattr(node, "end_ms", 0))
    return start_ms, end_ms


@lru_cache(maxsize=32)
def _probe_video_metadata(*, video_path: str) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_s = float((payload.get("format") or {}).get("duration") or 0.0)
    duration_ms = max(1, int(round(duration_s * 1000.0)))
    if width < 1 or height < 1:
        raise RuntimeError(f"Could not probe video dimensions for {video_path}.")
    return width, height, duration_ms


def _nearest_even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _target_dimensions(*, source_video_path: Path) -> tuple[int, int]:
    source_width, source_height, _duration_ms = _probe_video_metadata(video_path=str(source_video_path))
    if source_height <= _NODE_MEDIA_TARGET_HEIGHT:
        return _nearest_even(source_width), _nearest_even(source_height)
    scale = _NODE_MEDIA_TARGET_HEIGHT / float(source_height)
    return _nearest_even(source_width * scale), _NODE_MEDIA_TARGET_HEIGHT


def _source_duration_ms(*, source_video_path: Path) -> int:
    _width, _height, duration_ms = _probe_video_metadata(video_path=str(source_video_path))
    return duration_ms


def _read_max_concurrency(*, node_count: int, max_concurrent: int | None) -> int:
    if os.getenv("CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY") is not None:
        raise ValueError(
            "CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY has been renamed to "
            "CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT; update your env file."
        )
    if max_concurrent is None:
        raw_value = os.getenv("CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT") or "12"
        try:
            max_concurrent = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                "CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT must be a positive integer."
            ) from exc
    if max_concurrent < 1:
        raise ValueError("CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT must be >= 1.")
    return min(max_concurrent, node_count) if node_count > 0 else 1


def _read_positive_int_env(name: str, *, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1.")
    return parsed


def _read_batch_gap_ms() -> int:
    return _read_positive_int_env(
        "CLYPT_PHASE24_NODE_MEDIA_BATCH_GAP_MS",
        default=_DEFAULT_NODE_MEDIA_BATCH_GAP_MS,
    )


def _read_batch_max_nodes() -> int:
    return _read_positive_int_env(
        "CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_NODES",
        default=_DEFAULT_NODE_MEDIA_BATCH_MAX_NODES,
    )


def _read_batch_max_span_ms() -> int:
    return _read_positive_int_env(
        "CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_SPAN_MS",
        default=_DEFAULT_NODE_MEDIA_BATCH_MAX_SPAN_MS,
    )


def _read_batch_pad_ms() -> int:
    return _read_positive_int_env(
        "CLYPT_PHASE24_NODE_MEDIA_BATCH_PAD_MS",
        default=_DEFAULT_NODE_MEDIA_BATCH_PAD_MS,
    )


def _read_batch_coarse_seek_pad_ms() -> int:
    return _read_positive_int_env(
        "CLYPT_PHASE24_NODE_MEDIA_BATCH_COARSE_SEEK_PAD_MS",
        default=_DEFAULT_NODE_MEDIA_BATCH_COARSE_SEEK_PAD_MS,
    )


def plan_node_media_batches(
    *,
    nodes: list[Any],
    gap_ms: int | None = None,
    max_nodes: int | None = None,
    max_span_ms: int | None = None,
) -> list[NodeMediaBatchPlan]:
    if not nodes:
        return []
    gap_ms = _read_batch_gap_ms() if gap_ms is None else int(gap_ms)
    max_nodes = _read_batch_max_nodes() if max_nodes is None else int(max_nodes)
    max_span_ms = _read_batch_max_span_ms() if max_span_ms is None else int(max_span_ms)

    sorted_nodes = sorted(
        nodes,
        key=lambda node: (
            int(getattr(node, "start_ms", 0)),
            int(getattr(node, "end_ms", 0)),
            str(getattr(node, "node_id", "")),
        ),
    )
    batches: list[NodeMediaBatchPlan] = []
    current_nodes: list[Any] = []
    current_start_ms = 0
    current_end_ms = 0

    for node in sorted_nodes:
        node_start_ms, node_end_ms = _node_window(node)
        if not current_nodes:
            current_nodes = [node]
            current_start_ms = node_start_ms
            current_end_ms = node_end_ms
            continue

        next_end_ms = max(current_end_ms, node_end_ms)
        span_would_be_ms = next_end_ms - current_start_ms
        should_split = (
            len(current_nodes) >= max_nodes
            or node_start_ms > current_end_ms + gap_ms
            or span_would_be_ms > max_span_ms
        )
        if should_split:
            batches.append(
                NodeMediaBatchPlan(
                    batch_id=f"batch_{len(batches):04d}",
                    nodes=list(current_nodes),
                    start_ms=current_start_ms,
                    end_ms=current_end_ms,
                )
            )
            current_nodes = [node]
            current_start_ms = node_start_ms
            current_end_ms = node_end_ms
            continue

        current_nodes.append(node)
        current_end_ms = next_end_ms

    if current_nodes:
        batches.append(
            NodeMediaBatchPlan(
                batch_id=f"batch_{len(batches):04d}",
                nodes=list(current_nodes),
                start_ms=current_start_ms,
                end_ms=current_end_ms,
            )
        )
    return batches


def _read_ffmpeg_device() -> str:
    ffmpeg_device = (os.getenv("CLYPT_PHASE24_FFMPEG_DEVICE") or "auto").strip().lower()
    if ffmpeg_device not in {"auto", "gpu", "cpu"}:
        raise ValueError(
            f"Unsupported CLYPT_PHASE24_FFMPEG_DEVICE={ffmpeg_device!r}; expected auto|gpu|cpu."
        )
    return ffmpeg_device


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_with_device_fallback(*, gpu_cmd: list[str], cpu_cmd: list[str], output_path: Path) -> None:
    ffmpeg_device = _read_ffmpeg_device()
    if ffmpeg_device == "cpu":
        _run_ffmpeg(cpu_cmd)
        return
    if ffmpeg_device == "gpu":
        _run_ffmpeg(gpu_cmd)
        return
    try:
        _run_ffmpeg(gpu_cmd)
    except subprocess.CalledProcessError:
        logger.warning(
            "[phase2] GPU ffmpeg unavailable for %s; falling back to CPU encoder.",
            output_path.name,
        )
        _run_ffmpeg(cpu_cmd)


def extract_node_clip(
    *,
    source_video_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
) -> Path:
    duration_ms = max(end_ms - start_ms, 100)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_width, target_height = _target_dimensions(source_video_path=source_video_path)
    gpu_cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-resize",
        f"{target_width}x{target_height}",
        "-i",
        str(source_video_path),
        "-ss",
        _format_seconds(start_ms),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-b:v",
        "0",
        "-an",
        str(output_path),
    ]
    cpu_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video_path),
        "-ss",
        _format_seconds(start_ms),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "libx264",
        "-vf",
        f"scale={target_width}:{target_height}",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(output_path),
    ]
    _run_with_device_fallback(gpu_cmd=gpu_cmd, cpu_cmd=cpu_cmd, output_path=output_path)
    return output_path


def _extract_batch_window(
    *,
    source_video_path: Path,
    output_path: Path,
    coarse_start_ms: int,
    padded_end_ms: int,
) -> Path:
    duration_ms = max(padded_end_ms - coarse_start_ms, 100)
    target_width, target_height = _target_dimensions(source_video_path=source_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        _format_seconds(coarse_start_ms),
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-resize",
        f"{target_width}x{target_height}",
        "-i",
        str(source_video_path),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-b:v",
        "0",
        "-an",
        str(output_path),
    ]
    cpu_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        _format_seconds(coarse_start_ms),
        "-i",
        str(source_video_path),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "libx264",
        "-vf",
        f"scale={target_width}:{target_height}",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(output_path),
    ]
    _run_with_device_fallback(gpu_cmd=gpu_cmd, cpu_cmd=cpu_cmd, output_path=output_path)
    return output_path


def _trim_node_clip_from_batch(
    *,
    batch_video_path: Path,
    output_path: Path,
    local_start_ms: int,
    local_end_ms: int,
) -> Path:
    duration_ms = max(local_end_ms - local_start_ms, 100)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-i",
        str(batch_video_path),
        "-ss",
        _format_seconds(local_start_ms),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-b:v",
        "0",
        "-an",
        str(output_path),
    ]
    cpu_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(batch_video_path),
        "-ss",
        _format_seconds(local_start_ms),
        "-t",
        _format_seconds(duration_ms),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(output_path),
    ]
    _run_with_device_fallback(gpu_cmd=gpu_cmd, cpu_cmd=cpu_cmd, output_path=output_path)
    return output_path


def prepare_node_media_embeddings(
    *,
    nodes: list[SemanticGraphNode],
    source_video_path: Path,
    clips_dir: Path,
    storage_client,
    object_prefix: str,
    max_concurrent: int | None = None,
    return_diagnostics: bool = False,
) -> list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, Any]]:
    if not nodes:
        empty_diagnostics = {
            "ffmpeg_mode": "hybrid_batch_gpu",
            "node_count": 0,
            "extract_ms": 0.0,
            "upload_ms": 0.0,
            "total_ms": 0.0,
        }
        if return_diagnostics:
            return [], empty_diagnostics
        return []

    worker_count = _read_max_concurrency(
        node_count=len(nodes),
        max_concurrent=max_concurrent,
    )
    total_started = time.perf_counter()
    source_duration_ms = _source_duration_ms(source_video_path=source_video_path)
    batch_start_ms = min(int(node.start_ms) for node in nodes)
    batch_end_ms = max(int(node.end_ms) for node in nodes)
    batch_pad_ms = _read_batch_pad_ms()
    coarse_seek_pad_ms = _read_batch_coarse_seek_pad_ms()
    padded_start_ms = max(0, batch_start_ms - batch_pad_ms)
    padded_end_ms = min(source_duration_ms, batch_end_ms + batch_pad_ms)
    coarse_start_ms = max(0, padded_start_ms - coarse_seek_pad_ms)
    batch_window_path = clips_dir / "_batch" / "window.mp4"

    extract_started = time.perf_counter()
    _extract_batch_window(
        source_video_path=source_video_path,
        output_path=batch_window_path,
        coarse_start_ms=coarse_start_ms,
        padded_end_ms=padded_end_ms,
    )

    upload_ms_total = 0.0
    upload_bytes_total = 0

    def _prepare_node_media(node: SemanticGraphNode) -> tuple[dict[str, str], float]:
        node_start_ms = int(node.start_ms)
        node_end_ms = int(node.end_ms)
        clip_path = _trim_node_clip_from_batch(
            batch_video_path=batch_window_path,
            output_path=clips_dir / f"{node.node_id}.mp4",
            local_start_ms=max(0, node_start_ms - coarse_start_ms),
            local_end_ms=max(0, node_end_ms - coarse_start_ms),
        )
        clip_size_bytes = clip_path.stat().st_size if clip_path.exists() else 0
        upload_started = time.perf_counter()
        file_uri = storage_client.upload_file(
            local_path=clip_path,
            object_name=f"{object_prefix}/{node.node_id}.mp4",
        )
        upload_ms = (time.perf_counter() - upload_started) * 1000.0
        return (
            {
                "node_id": node.node_id,
                "file_uri": file_uri,
                "mime_type": "video/mp4",
                "local_path": str(clip_path),
            },
            upload_ms,
            clip_size_bytes,
        )

    descriptors: list[dict[str, str] | None] = [None] * len(nodes)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_prepare_node_media, node): (index, node)
            for index, node in enumerate(nodes)
        }
        for future in as_completed(futures):
            index, node = futures[future]
            try:
                descriptor, upload_ms, upload_bytes = future.result()
                descriptors[index] = descriptor
                upload_ms_total += upload_ms
                upload_bytes_total += int(upload_bytes)
            except Exception as exc:
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError(
                    "Failed to prepare node media embedding for "
                    f"node_id={node.node_id} start_ms={node.start_ms} end_ms={node.end_ms}: {exc}"
                ) from exc

    total_duration_ms = (time.perf_counter() - total_started) * 1000.0
    extract_duration_ms = (time.perf_counter() - extract_started) * 1000.0 - upload_ms_total
    diagnostics = {
        "ffmpeg_mode": "hybrid_batch_cpu"
        if _read_ffmpeg_device() == "cpu"
        else "hybrid_batch_gpu",
        "node_count": len(nodes),
        "batch_start_ms": batch_start_ms,
        "batch_end_ms": batch_end_ms,
        "padded_start_ms": padded_start_ms,
        "padded_end_ms": padded_end_ms,
        "coarse_start_ms": coarse_start_ms,
        "batch_gap_ms": _read_batch_gap_ms(),
        "batch_max_nodes": _read_batch_max_nodes(),
        "batch_max_span_ms": _read_batch_max_span_ms(),
        "batch_pad_ms": batch_pad_ms,
        "batch_coarse_seek_pad_ms": coarse_seek_pad_ms,
        "extract_ms": max(0.0, extract_duration_ms),
        "upload_ms": upload_ms_total,
        "upload_bytes": upload_bytes_total,
        "total_ms": total_duration_ms,
    }
    ordered = [descriptor for descriptor in descriptors if descriptor is not None]
    if return_diagnostics:
        return ordered, diagnostics
    return ordered


__all__ = [
    "NodeMediaBatchPlan",
    "extract_node_clip",
    "plan_node_media_batches",
    "prepare_node_media_embeddings",
]
