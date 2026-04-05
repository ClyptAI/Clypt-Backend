from __future__ import annotations

from pathlib import Path
import subprocess

from ..contracts import SemanticGraphNode


def _format_seconds(value_ms: int) -> str:
    return f"{max(value_ms, 0) / 1000.0:.3f}"


def extract_node_clip(
    *,
    source_video_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
) -> Path:
    duration_ms = max(end_ms - start_ms, 100)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def prepare_node_media_embeddings(
    *,
    nodes: list[SemanticGraphNode],
    source_video_path: Path,
    clips_dir: Path,
    storage_client,
    object_prefix: str,
) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    for node in nodes:
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
        descriptors.append(
            {
                "node_id": node.node_id,
                "file_uri": file_uri,
                "mime_type": "video/mp4",
                "local_path": str(clip_path),
            }
        )
    return descriptors


__all__ = [
    "extract_node_clip",
    "prepare_node_media_embeddings",
]
