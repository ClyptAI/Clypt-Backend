from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MASK_RLE_ENCODING = "rle_row_major_v1"
MASK_RLE_SOURCE = "rfdetr_seg_nano_tensorrt"
MASK_THRESHOLD = 0.5
LOWRES_MASK_REF_ENCODING = "lowres_mask_ref_v1"
LOWRES_MASK_ARTIFACT_ID = "visual_masks_lowres_v1"
LOWRES_MASK_ARTIFACT_ENCODING = "npz_compressed_lowres_binary_v1"


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


@dataclass(slots=True)
class MaskArtifactWriter:
    """Collect low-resolution instance masks into one compressed sidecar artifact."""

    artifact_path: Path
    artifact_id: str = LOWRES_MASK_ARTIFACT_ID
    threshold: float = MASK_THRESHOLD
    _masks: list[np.ndarray] = field(default_factory=list)
    _index: list[dict[str, Any]] = field(default_factory=list)
    write_ms: float = 0.0
    artifact_bytes: int = 0

    def add(
        self,
        *,
        frame_idx: int,
        detection_id: str,
        mask: np.ndarray,
        bbox_xyxy: list[float],
        source_size: tuple[int, int],
    ) -> dict[str, Any]:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D low-res mask, got shape {array.shape!r}")
        binary = (array > float(self.threshold)).astype(np.uint8, copy=False)
        packed = np.packbits(binary.reshape(-1))
        mask_index = len(self._masks)
        self._masks.append(packed.copy())
        source_height, source_width = source_size
        entry = {
            "mask_index": int(mask_index),
            "frame_idx": int(frame_idx),
            "detection_id": str(detection_id),
            "bbox_xyxy": [float(value) for value in bbox_xyxy],
            "source_size": [int(source_height), int(source_width)],
            "mask_shape": [int(binary.shape[0]), int(binary.shape[1])],
            "packed_length": int(packed.shape[0]),
            "threshold": float(self.threshold),
            "source": MASK_RLE_SOURCE,
        }
        self._index.append(entry)
        return {
            "encoding": LOWRES_MASK_REF_ENCODING,
            "artifact_id": self.artifact_id,
            "mask_index": int(mask_index),
            "frame_idx": int(frame_idx),
            "detection_id": str(detection_id),
            "bbox_xyxy": entry["bbox_xyxy"],
            "source_size": entry["source_size"],
            "mask_shape": entry["mask_shape"],
            "packed_length": entry["packed_length"],
            "threshold": float(self.threshold),
            "source": MASK_RLE_SOURCE,
        }

    @property
    def mask_count(self) -> int:
        return len(self._masks)

    def finalize(self) -> dict[str, Any] | None:
        if not self._masks:
            return None
        t0 = time.perf_counter()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        max_len = max(int(mask.shape[0]) for mask in self._masks)
        packed_masks = np.zeros((len(self._masks), max_len), dtype=np.uint8)
        packed_lengths = np.zeros((len(self._masks),), dtype=np.int32)
        for index, mask in enumerate(self._masks):
            packed_masks[index, : mask.shape[0]] = mask
            packed_lengths[index] = int(mask.shape[0])
        np.savez_compressed(
            self.artifact_path,
            packed_masks=packed_masks,
            packed_lengths=packed_lengths,
            index_json=np.array(json.dumps(self._index, separators=(",", ":"))),
            encoding=np.array(LOWRES_MASK_ARTIFACT_ENCODING),
            threshold=np.array(float(self.threshold), dtype=np.float32),
        )
        self.write_ms = (time.perf_counter() - t0) * 1000.0
        self.artifact_bytes = int(self.artifact_path.stat().st_size)
        return {
            "artifact_id": self.artifact_id,
            "encoding": LOWRES_MASK_ARTIFACT_ENCODING,
            "local_path": str(self.artifact_path),
            "mask_count": int(self.mask_count),
            "mask_shape": list(self._index[0]["mask_shape"]),
            "threshold": float(self.threshold),
            "bytes": self.artifact_bytes,
            "write_ms": round(self.write_ms, 2),
        }


@dataclass(slots=True)
class MaskArtifactReader:
    """Load compressed low-resolution instance masks from the sidecar artifact."""

    artifact_path: Path
    artifact_id: str = LOWRES_MASK_ARTIFACT_ID
    _packed_masks: np.ndarray | None = None
    _packed_lengths: np.ndarray | None = None
    _index_by_mask_index: dict[int, dict[str, Any]] | None = None
    _cache: dict[int, np.ndarray] = field(default_factory=dict)

    def _ensure_loaded(self) -> None:
        if self._packed_masks is not None:
            return
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"visual mask artifact not found at {self.artifact_path}"
            )
        with np.load(self.artifact_path, allow_pickle=False) as payload:
            encoding = str(np.asarray(payload["encoding"]).item())
            if encoding != LOWRES_MASK_ARTIFACT_ENCODING:
                raise RuntimeError(
                    "visual mask artifact encoding mismatch: "
                    f"expected {LOWRES_MASK_ARTIFACT_ENCODING!r}, got {encoding!r}"
                )
            self._packed_masks = np.asarray(payload["packed_masks"], dtype=np.uint8)
            self._packed_lengths = np.asarray(payload["packed_lengths"], dtype=np.int32)
            index_json = str(np.asarray(payload["index_json"]).item())
        index = json.loads(index_json)
        self._index_by_mask_index = {
            int(entry["mask_index"]): dict(entry)
            for entry in index
        }

    def get(self, mask_ref: dict[str, Any]) -> np.ndarray:
        self._ensure_loaded()
        if str(mask_ref.get("encoding") or "") != LOWRES_MASK_REF_ENCODING:
            raise RuntimeError(
                "visual mask ref encoding mismatch: "
                f"expected {LOWRES_MASK_REF_ENCODING!r}, got {mask_ref.get('encoding')!r}"
            )
        if str(mask_ref.get("artifact_id") or "") != self.artifact_id:
            raise RuntimeError(
                "visual mask ref artifact mismatch: "
                f"expected {self.artifact_id!r}, got {mask_ref.get('artifact_id')!r}"
            )
        mask_index = int(mask_ref["mask_index"])
        cached = self._cache.get(mask_index)
        if cached is not None:
            return cached
        if self._packed_masks is None or self._packed_lengths is None or self._index_by_mask_index is None:
            raise RuntimeError("visual mask artifact reader failed to initialize")
        if mask_index not in self._index_by_mask_index:
            raise KeyError(f"mask_index {mask_index} missing from visual mask artifact index")
        entry = self._index_by_mask_index[mask_index]
        packed_length = int(self._packed_lengths[mask_index])
        packed = self._packed_masks[mask_index, :packed_length]
        mask_shape = entry["mask_shape"]
        height = int(mask_shape[0])
        width = int(mask_shape[1])
        unpacked = np.unpackbits(packed)[: height * width]
        mask = unpacked.reshape((height, width)).astype(bool, copy=False)
        self._cache[mask_index] = mask
        return mask


__all__ = [
    "LOWRES_MASK_ARTIFACT_ENCODING",
    "LOWRES_MASK_ARTIFACT_ID",
    "LOWRES_MASK_REF_ENCODING",
    "MaskArtifactReader",
    "MASK_RLE_ENCODING",
    "MASK_RLE_SOURCE",
    "MASK_THRESHOLD",
    "MaskArtifactWriter",
    "encode_mask_rle",
    "resize_mask_nearest",
]
