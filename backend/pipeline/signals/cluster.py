from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from backend.pipeline._embedding_utils import cosine_similarity

from .contracts import ExternalSignal, ExternalSignalCluster


def cluster_signals(
    *,
    signals: list[ExternalSignal],
    embeddings: list[list[float]],
    cluster_type: str,
    similarity_threshold: float,
) -> list[ExternalSignalCluster]:
    if not signals:
        return []
    if len(signals) != len(embeddings):
        raise ValueError("signals and embeddings length mismatch")

    parents = list(range(len(signals)))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parents[rj] = ri

    threshold = float(similarity_threshold)
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim != float("-inf") and sim >= threshold:
                _union(i, j)

    buckets: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(signals)):
        buckets[_find(idx)].append(idx)

    clusters: list[ExternalSignalCluster] = []
    for cluster_index, member_indices in enumerate(
        sorted(buckets.values(), key=lambda members: min(members)),
        start=1,
    ):
        member_signals = [signals[i] for i in member_indices]
        member_embeddings = [embeddings[i] for i in member_indices]
        avg_embedding = _average_embedding(member_embeddings)
        summary = _cluster_summary(member_signals)
        clusters.append(
            ExternalSignalCluster(
                cluster_id=f"{cluster_type}_cluster_{cluster_index:03d}",
                cluster_type=cluster_type,  # type: ignore[arg-type]
                summary_text=summary,
                member_signal_ids=[signal.signal_id for signal in member_signals],
                cluster_weight=0.0,
                embedding=avg_embedding,
                metadata={
                    "member_count": len(member_signals),
                },
            )
        )
    return clusters


def _average_embedding(vectors: Iterable[list[float]]) -> list[float]:
    vectors_list = [list(vector) for vector in vectors]
    if not vectors_list:
        return []
    size = len(vectors_list[0])
    totals = [0.0] * size
    for vector in vectors_list:
        if len(vector) != size:
            raise ValueError("embedding size mismatch within cluster")
        for idx, value in enumerate(vector):
            totals[idx] += float(value)
    return [value / len(vectors_list) for value in totals]


def _cluster_summary(signals: list[ExternalSignal]) -> str:
    by_engagement = sorted(signals, key=lambda signal: (-signal.engagement_score, signal.signal_id))
    preview = [signal.text.strip() for signal in by_engagement[:3] if signal.text.strip()]
    if not preview:
        return "No textual summary available."
    return " | ".join(preview)


__all__ = ["cluster_signals"]
