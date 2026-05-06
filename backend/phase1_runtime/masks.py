from __future__ import annotations

from typing import Any

import numpy as np

MASK_RLE_ENCODING = "rle_row_major_v1"
MASK_RLE_SOURCE = "rfdetr_seg_nano_tensorrt"
MASK_THRESHOLD = 0.5


def encode_mask_rle(
    mask: np.ndarray,
    *,
    threshold: float = MASK_THRESHOLD,
    source: str = MASK_RLE_SOURCE,
) -> dict[str, Any]:
    """Encode a source-frame mask as row-major run-length counts."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape!r}")
    binary = array > float(threshold) if array.dtype != np.bool_ else array
    flat = binary.reshape(-1)
    counts: list[int] = []
    current_value = False
    run_length = 0
    for value in flat:
        bit = bool(value)
        if bit == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        current_value = bit
        run_length = 1
    counts.append(run_length)
    return {
        "encoding": MASK_RLE_ENCODING,
        "size": [int(array.shape[0]), int(array.shape[1])],
        "counts": counts,
        "threshold": float(threshold),
        "source": source,
    }


def resize_mask_nearest(mask: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize a model-space mask to source-frame dimensions without soft blending."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape!r}")
    target_size = (int(height), int(width))
    if array.shape == target_size:
        return array
    try:
        import cv2
    except ImportError:
        y_idx = np.linspace(0, array.shape[0] - 1, int(height)).round().astype(np.int64)
        x_idx = np.linspace(0, array.shape[1] - 1, int(width)).round().astype(np.int64)
        return array[y_idx[:, None], x_idx[None, :]]
    resized = cv2.resize(
        array.astype(np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


__all__ = [
    "MASK_RLE_ENCODING",
    "MASK_RLE_SOURCE",
    "MASK_THRESHOLD",
    "encode_mask_rle",
    "resize_mask_nearest",
]
