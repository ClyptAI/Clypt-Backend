"""GPU decode for Phase 1 visual extraction.

Responsibilities:
- Decode frames with ffmpeg NVDEC (`-hwaccel cuda`)
- Yield batches of (frame_idx, rgb_array) tuples
- Preserve frame index bookkeeping
- Fail fast when GPU decode is unavailable
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_idx: int
    rgb: np.ndarray
    source_width: int
    source_height: int


def _probe_frame_dimensions(*, video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = (result.stdout or "").strip()
    if "x" not in raw:
        raise RuntimeError(f"Could not probe frame dimensions for {video_path}.")
    width_raw, height_raw = raw.split("x", 1)
    width = int(width_raw)
    height = int(height_raw)
    if width < 1 or height < 1:
        raise RuntimeError(f"Invalid frame dimensions for {video_path}: {width}x{height}.")
    return width, height


def decode_video_frames(
    *,
    video_path: Path,
    decode_backend: str = "gpu",
    stride: int = 1,
    target_width: int | None = None,
    target_height: int | None = None,
) -> Iterator[DecodedFrame]:
    """Yield all frames from the video as RGB numpy arrays.

    Args:
        video_path: path to the source video file.
        decode_backend: frame decode backend. Only "gpu" is supported.
        stride: decode every Nth frame. 1 = every frame (default).
    """
    backend = (decode_backend or "gpu").strip().lower()
    if backend != "gpu":
        raise ValueError(
            f"Unsupported decode backend {decode_backend!r}; only 'gpu' is supported."
        )
    if stride < 1:
        raise ValueError("stride must be >= 1")

    width, height = _probe_frame_dimensions(video_path=video_path)
    output_width = int(target_width or width)
    output_height = int(target_height or height)
    if output_width < 1 or output_height < 1:
        raise ValueError("target_width and target_height must be >= 1 when provided")
    frame_bytes = output_width * output_height * 3
    if frame_bytes <= 0:
        raise RuntimeError(
            f"Invalid decode frame size for {video_path}: {width}x{height}."
        )

    vf_chain = "hwdownload,format=nv12,format=rgb24"
    if output_width != width or output_height != height:
        vf_chain = f"scale_cuda={output_width}:{output_height},{vf_chain}"

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(video_path),
        "-vf",
        # Resize on-GPU when requested, then download as NV12 before RGB conversion.
        # Some ffmpeg/NVDEC builds reject direct hwdownload->rgb24 conversion.
        vf_chain,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for GPU frame decoding."
        ) from exc

    frame_idx = 0
    try:
        while True:
            assert process.stdout is not None
            raw_frame = process.stdout.read(frame_bytes)
            if not raw_frame:
                break
            if len(raw_frame) != frame_bytes:
                raise RuntimeError(
                    "Truncated frame payload from ffmpeg GPU decode: "
                    f"expected {frame_bytes} bytes, got {len(raw_frame)}."
                )
            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                continue
            rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
            yield DecodedFrame(
                frame_idx=frame_idx,
                rgb=rgb,
                source_width=width,
                source_height=height,
            )
            frame_idx += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.wait()
    stderr_text = ""
    if process.stderr is not None:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
    if process.returncode not in (0, None):
        raise RuntimeError(
            "[visual] GPU frame decode failed "
            f"(exit={process.returncode}) for {video_path}: {stderr_text[-1200:]}"
        )

    logger.info(
        "Decoded %d frames from %s (backend=%s)",
        frame_idx,
        video_path.name,
        backend,
    )


def batch_frames(
    frames: Iterator[DecodedFrame],
    *,
    batch_size: int,
) -> Iterator[list[DecodedFrame]]:
    """Collect decoded frames into batches of the given size."""
    batch: list[DecodedFrame] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["DecodedFrame", "batch_frames", "decode_video_frames"]
