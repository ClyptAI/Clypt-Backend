from __future__ import annotations

import hashlib
import math


EMBEDDING_DIMENSIONS = 16


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [component / norm for component in vector]


def embed_text(
    *,
    text: str,
    dimensions: int = EMBEDDING_DIMENSIONS,
    namespace: str = "text",
) -> list[float]:
    cleaned = " ".join((text or "").lower().split())
    if not cleaned:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in cleaned.split():
        digest = hashlib.sha256(f"{namespace}:{token}".encode("utf-8")).digest()
        for idx in range(dimensions):
            byte_value = digest[idx % len(digest)]
            centered = (byte_value / 255.0) - 0.5
            vector[idx] += centered
    return _normalize(vector)


def embed_media_descriptor(*, descriptor: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    return embed_text(text=descriptor, dimensions=dimensions, namespace="media")


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return float("-inf")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return float("-inf")
    return dot / (left_norm * right_norm)


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "cosine_similarity",
    "embed_media_descriptor",
    "embed_text",
]
