"""GPU-resident frame decode for Phase 1 visual extraction.

Responsibilities:
- Decode frames with NVDEC into CUDA tensors via TorchAudio StreamReader
- Resize in the hardware decoder when requested
- Convert YUV CUDA tensors to RGB CUDA tensors without a host round-trip
- Yield either frame batches or legacy per-frame wrappers
- Fail fast when GPU decode is unavailable
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

_HW_DECODER_BY_CODEC = {
    "av1": "av1_cuvid",
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "wmv3": "wmv3_cuvid",
}


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_idx: int
    source_width: int
    source_height: int
    rgb: np.ndarray | None = None
    tensor: Any | None = None


@dataclass(frozen=True, slots=True)
class DecodedFrameBatch:
    frame_indices: tuple[int, ...]
    frames: Any
    source_width: int
    source_height: int


def _probe_video_stream_info(*, video_path: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
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
    codec_name = str(stream.get("codec_name") or "").strip().lower()
    if width < 1 or height < 1:
        raise RuntimeError(f"Invalid frame dimensions for {video_path}: {width}x{height}.")
    if not codec_name:
        raise RuntimeError(f"Could not probe video codec for {video_path}.")
    return width, height, codec_name


def _probe_frame_dimensions(*, video_path: Path) -> tuple[int, int]:
    width, height, _ = _probe_video_stream_info(video_path=video_path)
    return width, height


def _select_hw_video_decoder(*, codec_name: str) -> str:
    decoder = _HW_DECODER_BY_CODEC.get(codec_name.strip().lower())
    if not decoder:
        raise RuntimeError(
            "Unsupported video codec for GPU decode: "
            f"{codec_name!r}. Expected one of: {', '.join(sorted(_HW_DECODER_BY_CODEC))}."
        )
    return decoder


def _yuv_to_rgb_cuda(frames):
    """Convert StreamReader NVDEC output (T, C, H, W) to RGB on GPU."""
    import torch

    if getattr(frames, "ndim", 0) != 4 or int(frames.shape[1]) != 3:
        raise RuntimeError(
            "Expected NVDEC chunk with shape (frames, 3, H, W); "
            f"got {tuple(getattr(frames, 'shape', ()))!r}."
        )
    work = frames.to(dtype=torch.float32)
    y = work[:, 0, :, :].div_(255.0)
    u = work[:, 1, :, :].div_(255.0).sub_(0.5)
    v = work[:, 2, :, :].div_(255.0).sub_(0.5)
    r = y + (1.14 * v)
    g = y + (-0.396 * u) + (-0.581 * v)
    b = y + (2.029 * u)
    rgb = torch.stack((r, g, b), dim=1)
    return rgb.mul_(255.0).clamp_(0.0, 255.0).to(dtype=torch.uint8)


def decode_video_frame_batches(
    *,
    video_path: Path,
    batch_size: int,
    decode_backend: str = "gpu",
    stride: int = 1,
    target_width: int | None = None,
    target_height: int | None = None,
    buffer_chunk_size: int = 3,
    hw_accel_device: str = "cuda:0",
    hw_device_index: int = 0,
) -> Iterator[DecodedFrameBatch]:
    """Yield GPU-resident RGB frame batches from the source video."""
    backend = (decode_backend or "gpu").strip().lower()
    if backend != "gpu":
        raise ValueError(
            f"Unsupported decode backend {decode_backend!r}; only 'gpu' is supported."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if buffer_chunk_size < 1:
        raise ValueError("buffer_chunk_size must be >= 1")

    width, height, codec_name = _probe_video_stream_info(video_path=video_path)
    output_width = int(target_width or width)
    output_height = int(target_height or height)
    if output_width < 1 or output_height < 1:
        raise ValueError("target_width and target_height must be >= 1 when provided")

    try:
        from torchaudio.io import StreamReader
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio with StreamReader support is required for GPU frame decoding."
        ) from exc

    decoder = _select_hw_video_decoder(codec_name=codec_name)
    decoder_option = {"gpu": str(hw_device_index)}
    if output_width != width or output_height != height:
        decoder_option["resize"] = f"{output_width}x{output_height}"

    reader = StreamReader(str(video_path))
    reader.add_video_stream(
        batch_size,
        buffer_chunk_size=buffer_chunk_size,
        decoder=decoder,
        decoder_option=decoder_option,
        hw_accel=hw_accel_device,
    )

    raw_frame_idx = 0
    emitted_frames = 0
    while True:
        status = reader.fill_buffer()
        chunks = reader.pop_chunks()
        chunk = chunks[0] if chunks else None
        if chunk is not None:
            if getattr(chunk, "device", None) is None or getattr(chunk.device, "type", "") != "cuda":
                raise RuntimeError(
                    "GPU decode returned a non-CUDA chunk; "
                    "NVDEC direct-CUDA decode is required for Phase1 visual extraction."
                )
            raw_chunk_len = int(chunk.shape[0])
            selected_offsets = list(range(0, raw_chunk_len, stride))
            if selected_offsets:
                chunk = chunk[selected_offsets]
                rgb_chunk = _yuv_to_rgb_cuda(chunk)
                frame_indices = tuple(raw_frame_idx + offset for offset in selected_offsets)
                emitted_frames += len(frame_indices)
                yield DecodedFrameBatch(
                    frame_indices=frame_indices,
                    frames=rgb_chunk,
                    source_width=width,
                    source_height=height,
                )
            raw_frame_idx += raw_chunk_len
        if status == 1 and chunk is None:
            break

    logger.info(
        "Decoded %d frames from %s (backend=%s, decoder=%s, device=%s)",
        emitted_frames,
        video_path.name,
        backend,
        decoder,
        hw_accel_device,
    )


def decode_video_frames(
    *,
    video_path: Path,
    decode_backend: str = "gpu",
    stride: int = 1,
    target_width: int | None = None,
    target_height: int | None = None,
    batch_size: int = 1,
    buffer_chunk_size: int = 3,
    hw_accel_device: str = "cuda:0",
    hw_device_index: int = 0,
) -> Iterator[DecodedFrame]:
    """Legacy wrapper yielding per-frame objects for tests and compatibility."""
    for batch in decode_video_frame_batches(
        video_path=video_path,
        batch_size=batch_size,
        decode_backend=decode_backend,
        stride=stride,
        target_width=target_width,
        target_height=target_height,
        buffer_chunk_size=buffer_chunk_size,
        hw_accel_device=hw_accel_device,
        hw_device_index=hw_device_index,
    ):
        for index, frame_idx in enumerate(batch.frame_indices):
            yield DecodedFrame(
                frame_idx=frame_idx,
                tensor=batch.frames[index],
                source_width=batch.source_width,
                source_height=batch.source_height,
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


__all__ = [
    "DecodedFrame",
    "DecodedFrameBatch",
    "batch_frames",
    "decode_video_frame_batches",
    "decode_video_frames",
]
