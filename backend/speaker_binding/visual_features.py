from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrackletReIDSample:
    frame_idx: int
    embedding: np.ndarray
    quality: float = 1.0


@dataclass(frozen=True)
class TrackletReIDEvidence:
    centroid: np.ndarray | None
    sample_count: int
    quality: float
    frame_indices: tuple[int, ...]


def _normalize_embedding(embedding: np.ndarray | Iterable[float] | None) -> np.ndarray | None:
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return None
    return (arr / norm).astype(np.float32)


def build_tracklet_reid_evidence(
    samples: Iterable[TrackletReIDSample | dict],
) -> TrackletReIDEvidence:
    usable_vectors: list[np.ndarray] = []
    weights: list[float] = []
    frame_indices: list[int] = []

    for sample in samples:
        if isinstance(sample, dict):
            frame_idx = int(sample.get("frame_idx", -1))
            embedding = sample.get("embedding")
            quality = float(sample.get("quality", 1.0) or 0.0)
        else:
            frame_idx = int(sample.frame_idx)
            embedding = sample.embedding
            quality = float(sample.quality or 0.0)
        normalized = _normalize_embedding(embedding)
        if normalized is None:
            continue
        usable_vectors.append(normalized)
        weights.append(max(0.0, quality))
        if frame_idx >= 0:
            frame_indices.append(frame_idx)

    if not usable_vectors:
        return TrackletReIDEvidence(
            centroid=None,
            sample_count=0,
            quality=0.0,
            frame_indices=tuple(),
        )

    weight_arr = np.asarray(weights, dtype=np.float32)
    if float(weight_arr.sum()) <= 1e-9:
        weight_arr = np.ones(len(usable_vectors), dtype=np.float32)
    stacked = np.stack(usable_vectors, axis=0)
    centroid = np.average(stacked, axis=0, weights=weight_arr).astype(np.float32)
    centroid = _normalize_embedding(centroid)
    quality = float(np.clip(float(weight_arr.mean()), 0.0, 1.0))
    return TrackletReIDEvidence(
        centroid=centroid,
        sample_count=len(usable_vectors),
        quality=round(quality, 3),
        frame_indices=tuple(sorted(set(frame_indices))),
    )


def cosine_similarity(
    left: np.ndarray | Iterable[float] | None,
    right: np.ndarray | Iterable[float] | None,
) -> float:
    left_arr = _normalize_embedding(left)
    right_arr = _normalize_embedding(right)
    if left_arr is None or right_arr is None:
        return 0.0
    return float(np.clip(float(np.dot(left_arr, right_arr)), -1.0, 1.0))
