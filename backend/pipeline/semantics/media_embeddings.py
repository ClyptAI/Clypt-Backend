from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..contracts import SemanticGraphNode

logger = logging.getLogger(__name__)
_NODE_MEDIA_TARGET_HEIGHT = 480


def _format_seconds(value_ms: int) -> str:
    return f"{max(value_ms, 0) / 1000.0:.3f}"


@lru_cache(maxsize=32)
def _probe_video_dimensions(*, video_path: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
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
    if width < 1 or height < 1:
        raise RuntimeError(f"Could not probe video dimensions for {video_path}.")
    return width, height


def _nearest_even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _target_dimensions(*, source_video_path: Path) -> tuple[int, int]:
    source_width, source_height = _probe_video_dimensions(video_path=str(source_video_path))
    if source_height <= _NODE_MEDIA_TARGET_HEIGHT:
        return _nearest_even(source_width), _nearest_even(source_height)
    scale = _NODE_MEDIA_TARGET_HEIGHT / float(source_height)
    return _nearest_even(source_width * scale), _NODE_MEDIA_TARGET_HEIGHT


def _read_max_concurrency(*, node_count: int, max_concurrent: int | None) -> int:
    if os.getenv("CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY") is not None:
        raise ValueError(
            "CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY has been renamed to "
            "CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT; update your env file."
        )
    if max_concurrent is None:
        raw_value = os.getenv("CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT") or "16"
        try:
            max_concurrent = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                "CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT must be a positive integer."
            ) from exc
    if max_concurrent < 1:
        raise ValueError("CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT must be >= 1.")
    return min(max_concurrent, node_count) if node_count > 0 else 1


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
    ffmpeg_device = (os.getenv("CLYPT_PHASE24_FFMPEG_DEVICE") or "auto").strip().lower()
    if ffmpeg_device not in {"auto", "gpu", "cpu"}:
        raise ValueError(
            f"Unsupported CLYPT_PHASE24_FFMPEG_DEVICE={ffmpeg_device!r}; expected auto|gpu|cpu."
        )

    # Full GPU pipeline: h264_cuvid (NVDEC) decode + h264_nvenc encode.
    # -c:v h264_cuvid forces NVDEC explicitly; -hwaccel cuda alone falls back
    # to native (CPU) decode on ffmpeg 4.4 even when VRAM is available.
    # Modal L4 is the current node-media-prep baseline, so the default worker
    # cap is 16 concurrent clip jobs unless an env override lowers it.
    # Downscale embedding clips to 480p so Vertex multimodal embeddings do not
    # pay 1080p prep/upload cost. Use decoder-side resize on the GPU path:
    # Modal L4 accepted h264_cuvid `-resize WxH`, while the earlier
    # `scale_cuda=-2:480` graph broke format negotiation and forced CPU
    # fallback for every clip.
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
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
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
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    def _run(cmd: list[str]) -> None:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if ffmpeg_device == "cpu":
        _run(cpu_cmd)
        return output_path

    if ffmpeg_device == "gpu":
        _run(gpu_cmd)
        return output_path

    # auto mode prefers GPU but gracefully falls back to CPU.
    try:
        _run(gpu_cmd)
    except subprocess.CalledProcessError:
        logger.warning(
            "[phase2] GPU ffmpeg unavailable for %s; falling back to CPU encoder.",
            output_path.name,
        )
        _run(cpu_cmd)
    return output_path


def prepare_node_media_embeddings(
    *,
    nodes: list[SemanticGraphNode],
    source_video_path: Path,
    clips_dir: Path,
    storage_client,
    object_prefix: str,
    max_concurrent: int | None = None,
) -> list[dict[str, str]]:
    if not nodes:
        return []

    worker_count = _read_max_concurrency(
        node_count=len(nodes),
        max_concurrent=max_concurrent,
    )

    def _prepare_node_media(node: SemanticGraphNode) -> dict[str, str]:
        clip_path = extract_node_clip(
            source_video_path=source_video_path,
            output_path=clips_dir / f"{node.node_id}.mp4",
            start_ms=int(node.start_ms),
            end_ms=int(node.end_ms),
        )
        file_uri = storage_client.upload_file(
            local_path=clip_path,
            object_name=f"{object_prefix}/{node.node_id}.mp4",
        )
        return {
            "node_id": node.node_id,
            "file_uri": file_uri,
            "mime_type": "video/mp4",
            "local_path": str(clip_path),
        }

    descriptors: list[dict[str, str] | None] = [None] * len(nodes)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_prepare_node_media, node): (index, node)
            for index, node in enumerate(nodes)
        }
        for future in as_completed(futures):
            index, node = futures[future]
            try:
                descriptors[index] = future.result()
            except Exception as exc:
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError(
                    "Failed to prepare node media embedding for "
                    f"node_id={node.node_id} start_ms={node.start_ms} end_ms={node.end_ms}: {exc}"
                ) from exc

    return [descriptor for descriptor in descriptors if descriptor is not None]


__all__ = [
    "extract_node_clip",
    "prepare_node_media_embeddings",
]
