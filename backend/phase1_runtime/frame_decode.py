"""CPU-based video frame decoder for Phase 1 visual extraction.

Responsibilities:
- Decode frames from video via OpenCV
- Yield batches of (frame_idx, rgb_array) tuples
- Preserve frame index bookkeeping
- Keep decode off the GPU lane
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_idx: int
    rgb: np.ndarray


def decode_video_frames(
    *,
    video_path: Path,
    stride: int = 1,
) -> Iterator[DecodedFrame]:
    """Yield all frames from the video as RGB numpy arrays.

    Args:
        video_path: path to the source video file.
        stride: decode every Nth frame. 1 = every frame (default).
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for frame decoding."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield DecodedFrame(frame_idx=frame_idx, rgb=rgb)
            frame_idx += 1
    finally:
        cap.release()

    logger.info("Decoded %d frames from %s", frame_idx, video_path.name)


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
