"""AMD VAAPI GPU decode for Phase 1 visual extraction.

Responsibilities:
- Decode frames with ffmpeg VAAPI (`-hwaccel vaapi`)
- Yield batches of (frame_idx, rgb_array) tuples
- Preserve frame index bookkeeping
- Fail fast when GPU decode is unavailable
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_RENDER_NODE_RE = re.compile(r"^renderD\d+$")


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


def discover_vaapi_render_node(*, dev_dri: Path = Path("/dev/dri")) -> Path:
    """Return the first DRM render node suitable for VAAPI decode."""
    if not dev_dri.exists():
        raise RuntimeError(
            f"No VAAPI render node found: {dev_dri} does not exist. "
            "AMD GPU decode requires /dev/dri/renderD*."
        )
    render_nodes = sorted(
        path for path in dev_dri.iterdir() if _RENDER_NODE_RE.match(path.name)
    )
    if not render_nodes:
        raise RuntimeError(
            f"No VAAPI render node found under {dev_dri}. "
            "AMD GPU decode requires /dev/dri/renderD* and must not fall back to software decode."
        )
    return render_nodes[0]


def validate_vaapi_render_node(render_node: Path) -> None:
    """Validate the selected render node with vainfo before ffmpeg starts."""
    if not render_node.exists():
        raise RuntimeError(f"VAAPI render node does not exist: {render_node}")
    try:
        result = subprocess.run(
            ["vainfo", "--display", "drm", "--device", str(render_node)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "vainfo is required to validate AMD VAAPI decode. "
            "Install vainfo/libva-utils on the Phase1 MI300X host."
        ) from exc
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout, result.stderr] if part)
        raise RuntimeError(
            "VAAPI validation failed for "
            f"{render_node}: {details[-1200:]}"
        )


def validate_ffmpeg_vaapi_support() -> None:
    """Fail before decode if ffmpeg lacks VAAPI or scale_vaapi support."""
    try:
        hwaccels = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
        )
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for AMD VAAPI frame decoding."
        ) from exc
    if hwaccels.returncode != 0 or "vaapi" not in (hwaccels.stdout or ""):
        raise RuntimeError(
            "ffmpeg does not report VAAPI hwaccel support; "
            "GPU decode is required and software decode is not allowed."
        )
    filter_output = "\n".join(part for part in [filters.stdout, filters.stderr] if part)
    if filters.returncode != 0 or "scale_vaapi" not in filter_output:
        raise RuntimeError(
            "ffmpeg does not expose scale_vaapi; "
            "AMD GPU resize/decode is required and software decode is not allowed."
        )


def decode_video_frames(
    *,
    video_path: Path,
    decode_backend: str = "gpu",
    gpu_decode_backend: str = "vaapi",
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
    accelerator_backend = (gpu_decode_backend or "vaapi").strip().lower()
    if accelerator_backend != "vaapi":
        raise ValueError(
            f"Unsupported GPU decode backend {gpu_decode_backend!r}; only AMD VAAPI is supported."
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

    render_node = discover_vaapi_render_node()
    validate_vaapi_render_node(render_node)
    validate_ffmpeg_vaapi_support()

    vf_chain = "hwdownload,format=nv12,format=rgb24"
    if output_width != width or output_height != height:
        vf_chain = f"scale_vaapi={output_width}:{output_height},{vf_chain}"

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-vaapi_device",
        str(render_node),
        "-hwaccel",
        "vaapi",
        "-hwaccel_output_format",
        "vaapi",
        "-i",
        str(video_path),
        "-vf",
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
            "ffmpeg is required for AMD VAAPI frame decoding."
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
            "[visual] AMD VAAPI GPU frame decode failed "
            f"(exit={process.returncode}) for {video_path}: {stderr_text[-1200:]}"
        )

    logger.info(
        "Decoded %d frames from %s (backend=%s)",
        frame_idx,
        video_path.name,
        accelerator_backend,
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
    "batch_frames",
    "decode_video_frames",
    "discover_vaapi_render_node",
    "validate_ffmpeg_vaapi_support",
    "validate_vaapi_render_node",
]
