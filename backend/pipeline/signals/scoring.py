from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from backend.pipeline.contracts import ClipCandidate, SemanticGraphNode
from backend.pipeline.config import SignalConfig

from .contracts import (
    CandidateSignalLink,
    ExternalSignal,
    ExternalSignalCluster,
    NodeSignalLink,
    PromptSourceType,
    SignalPromptSpec,
)


@dataclass(slots=True)
class SignalScoringResult:
    candidates: list[ClipCandidate]
    candidate_signal_links: list[CandidateSignalLink]


def apply_signal_scoring(
    *,
    candidates: list[ClipCandidate],
    nodes: list[SemanticGraphNode],
    signals: list[ExternalSignal],
    clusters: list[ExternalSignalCluster],
    node_links: list[NodeSignalLink],
    prompt_specs: list[SignalPromptSpec],
    cfg: SignalConfig,
) -> SignalScoringResult:
    if not candidates or not clusters or not node_links:
        return SignalScoringResult(candidates=list(candidates), candidate_signal_links=[])

    node_by_id = {node.node_id: node for node in nodes}
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}

    signal_by_id: dict[str, ExternalSignal] = {signal.signal_id: signal for signal in signals}
    signal_score_by_id = _compute_signal_scores(signals=signals, cfg=cfg)

    # 8.3.2 cluster weight (engagement + frequency)
    cluster_weight_by_id: dict[str, float] = {}
    for cluster in clusters:
        signal_scores = [
            float(signal_score_by_id[signal_id])
            for signal_id in cluster.member_signal_ids
            if signal_id in signal_score_by_id
        ]
        if signal_scores:
            mean_eng = sum(signal_scores) / len(signal_scores)
            max_eng = max(signal_scores)
        else:
            mean_eng = 0.0
            max_eng = 0.0
        freq_ref = max(1.0, float(cfg.cluster_freq_ref))
        freq_term = _clip(_safe_log1p(len(cluster.member_signal_ids)) / _safe_log1p(freq_ref), 0.0, 1.0)
        cluster_weight = _clip(
            (float(cfg.cluster_mean_weight) * mean_eng)
            + (float(cfg.cluster_max_weight) * max_eng)
            + (float(cfg.cluster_freq_weight) * freq_term),
            0.0,
            1.0,
        )
        cluster_weight_by_id[cluster.cluster_id] = cluster_weight

    # 8.3.3 node link score
    scored_links: list[NodeSignalLink] = []
    for link in node_links:
        hop_decay = 1.0
        if link.hop_distance == 1:
            hop_decay = float(cfg.hop_decay_1)
        elif link.hop_distance >= 2:
            hop_decay = float(cfg.hop_decay_2)
        time_decay = max(0.0, 1.0 - (abs(int(link.time_offset_ms)) / max(float(cfg.time_window_ms), float(cfg.epsilon))))
        direct_score = float(cluster_weight_by_id.get(link.cluster_id, 0.0)) * max(0.0, float(link.similarity))
        link_score = direct_score * hop_decay * time_decay
        scored_links.append(link.model_copy(update={"link_score": float(link_score)}))

    links_by_cluster_and_node: dict[tuple[str, str], NodeSignalLink] = {
        (link.cluster_id, link.node_id): link for link in scored_links
    }
    prompt_source_by_id: dict[str, PromptSourceType] = {
        prompt.prompt_id: prompt.prompt_source_type for prompt in prompt_specs
    }

    scored_candidates: list[ClipCandidate] = []
    candidate_signal_links: list[CandidateSignalLink] = []

    for candidate in candidates:
        candidate_duration_ms = max(int(candidate.end_ms) - int(candidate.start_ms), 0)
        cluster_contribs: dict[str, float] = {}
        source_contribs: dict[str, list[tuple[str, float]]] = defaultdict(list)
        source_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        candidate_link_rows: list[CandidateSignalLink] = []

        for cluster in clusters:
            linked_nodes = [
                node_by_id[node_id]
                for node_id in candidate.node_ids
                if node_id in node_by_id and (cluster.cluster_id, node_id) in links_by_cluster_and_node
            ]
            if not linked_nodes:
                continue

            weighted_sum = 0.0
            weighted_den = 0.0
            total_overlap_ms = 0
            direct_overlap_ms = 0
            direct_count = 0
            inferred_count = 0
            for node in linked_nodes:
                overlap_ms = _interval_overlap_ms(candidate.start_ms, candidate.end_ms, node.start_ms, node.end_ms)
                if overlap_ms <= 0:
                    continue
                link = links_by_cluster_and_node[(cluster.cluster_id, node.node_id)]
                total_overlap_ms += overlap_ms
                weighted_sum += float(overlap_ms) * float(link.link_score)
                weighted_den += float(overlap_ms)
                if link.link_type == "direct":
                    direct_overlap_ms += overlap_ms
                    direct_count += 1
                else:
                    inferred_count += 1
                source_intervals[cluster.cluster_type].append((max(candidate.start_ms, node.start_ms), min(candidate.end_ms, node.end_ms)))

            agg_ck = 0.0 if weighted_den <= float(cfg.epsilon) else (weighted_sum / weighted_den)
            if candidate_duration_ms <= float(cfg.epsilon):
                coverage_ck = 0.0
            else:
                coverage_ck = _clip(total_overlap_ms / candidate_duration_ms, 0.0, 1.0)
            direct_ratio_ck = 0.0 if total_overlap_ms <= 0 else (direct_overlap_ms / total_overlap_ms)

            cluster_contrib = (
                agg_ck
                * ((1.0 - float(cfg.coverage_weight)) + (float(cfg.coverage_weight) * coverage_ck))
                * ((1.0 - float(cfg.direct_ratio_weight)) + (float(cfg.direct_ratio_weight) * direct_ratio_ck))
            )
            cluster_contrib = min(float(cluster_contrib), float(cfg.cluster_cap))
            cluster_contribs[cluster.cluster_id] = cluster_contrib
            source_contribs[cluster.cluster_type].append((cluster.cluster_id, cluster_contrib))
            candidate_link_rows.append(
                CandidateSignalLink(
                    clip_id=candidate.clip_id or "",
                    cluster_id=cluster.cluster_id,
                    cluster_type=cluster.cluster_type,
                    aggregated_link_score=float(cluster_contrib),
                    coverage_ms=int(total_overlap_ms),
                    direct_node_count=direct_count,
                    inferred_node_count=inferred_count,
                    agreement_flags=[],
                    bonus_applied=0.0,
                    evidence={
                        "coverage": coverage_ck,
                        "direct_ratio": direct_ratio_ck,
                        "cluster_weight": float(cluster_weight_by_id.get(cluster.cluster_id, 0.0)),
                        "signal_scores": {
                            signal_id: float(signal_score_by_id.get(signal_id, 0.0))
                            for signal_id in cluster.member_signal_ids
                        },
                        "signal_types": {
                            signal_id: signal_by_id[signal_id].signal_type
                            for signal_id in cluster.member_signal_ids
                            if signal_id in signal_by_id
                        },
                    },
                )
            )

        external_signal_score = min(sum(cluster_contribs.values()), float(cfg.total_cap))

        has_general = any(
            prompt_source_by_id.get(prompt_id) == "general"
            for prompt_id in candidate.source_prompt_ids
        )
        source_meaningful: dict[str, bool] = {}
        for source in ("comment", "trend"):
            contrib_list = source_contribs.get(source) or []
            max_contrib = max((value for _, value in contrib_list), default=0.0)
            coverage = _source_coverage(intervals=source_intervals.get(source) or [], candidate_duration_ms=candidate_duration_ms, eps=float(cfg.epsilon))
            source_meaningful[source] = (
                max_contrib >= float(cfg.meaningful_min_cluster_contrib)
                and coverage >= float(cfg.meaningful_min_source_coverage)
            )

        agreement_bonus = 0.0
        agreement_flags: list[str] = []
        if has_general:
            agreement_flags.append("general")
        for source in ("comment", "trend"):
            if source_meaningful[source]:
                agreement_flags.append(source)

        if has_general and source_meaningful["comment"] and source_meaningful["trend"]:
            agreement_bonus = float(cfg.agreement_bonus_tier2)
        elif has_general and source_meaningful["comment"]:
            agreement_bonus = float(cfg.agreement_bonus_tier1)

        agreement_bonus = min(agreement_bonus, float(cfg.agreement_cap))
        candidate_link_rows = [
            row.model_copy(
                update={
                    "agreement_flags": list(agreement_flags),
                    "bonus_applied": float(agreement_bonus),
                }
            )
            for row in candidate_link_rows
        ]
        candidate_signal_links.extend(candidate_link_rows)

        final_score = float(candidate.score) + float(external_signal_score) + float(agreement_bonus)
        attribution_json = None
        if candidate_link_rows:
            attribution_json = {
                "cluster_links": [row.model_dump(mode="json") for row in candidate_link_rows],
                "agreement_flags": list(agreement_flags),
                "external_signal_score": float(external_signal_score),
                "agreement_bonus": float(agreement_bonus),
            }
        scored_candidates.append(
            candidate.model_copy(
                update={
                    "score": float(final_score),
                    "score_breakdown": {
                        **(candidate.score_breakdown or {}),
                        "base_score": float(candidate.score),
                        "external_signal_score": float(external_signal_score),
                        "agreement_bonus": float(agreement_bonus),
                    },
                    "external_signal_score": float(external_signal_score) if candidate_link_rows else None,
                    "agreement_bonus": float(agreement_bonus) if candidate_link_rows else None,
                    "external_attribution_json": attribution_json,
                }
            )
        )

    scored_candidates.sort(key=lambda item: (-(item.score or 0.0), item.start_ms, item.end_ms, item.clip_id or ""))
    for rank, candidate in enumerate(scored_candidates, start=1):
        scored_candidates[rank - 1] = candidate.model_copy(update={"pool_rank": rank})

    return SignalScoringResult(candidates=scored_candidates, candidate_signal_links=candidate_signal_links)


def _interval_overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(int(end_a), int(end_b)) - max(int(start_a), int(start_b)))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _safe_log1p(value: float) -> float:
    import math

    return math.log1p(max(0.0, float(value)))


def _compute_signal_scores(*, signals: list[ExternalSignal], cfg: SignalConfig) -> dict[str, float]:
    raw_by_id: dict[str, float] = {}
    raw_nonspam: list[float] = []
    for signal in signals:
        quality = str((signal.metadata or {}).get("quality") or "").strip().lower()
        quality_mult = _quality_multiplier(quality)
        raw = _raw_signal_engagement(signal=signal, cfg=cfg, quality_mult=quality_mult)
        raw_by_id[signal.signal_id] = raw
        if quality != "spam" and raw > 0.0:
            raw_nonspam.append(raw)
    denom = max(_percentile(raw_nonspam, 95.0), float(cfg.epsilon))
    if denom <= float(cfg.epsilon):
        return {signal.signal_id: 0.0 for signal in signals}
    return {
        signal.signal_id: _clip(raw_by_id.get(signal.signal_id, 0.0) / denom, 0.0, 1.0)
        for signal in signals
    }


def _quality_multiplier(quality: str) -> float:
    quality_key = (quality or "").strip().lower()
    if quality_key == "high_signal":
        return 1.0
    if quality_key == "contextual":
        return 0.75
    if quality_key == "low_signal":
        return 0.30
    if quality_key == "spam":
        return 0.0
    return 1.0


def _raw_signal_engagement(*, signal: ExternalSignal, cfg: SignalConfig, quality_mult: float) -> float:
    metadata = signal.metadata or {}
    like_count = max(0.0, float(metadata.get("like_count") or 0.0))
    likes_term = _safe_log1p(like_count)
    if signal.signal_type == "comment_top":
        reply_count = max(0.0, float(metadata.get("reply_count") or 0.0))
        reply_term = _safe_log1p(reply_count)
        return quality_mult * (
            float(cfg.engagement_top_like_weight) * likes_term
            + float(cfg.engagement_top_reply_weight) * reply_term
        )
    if signal.signal_type == "comment_reply":
        parent_reply_count = max(0.0, float(metadata.get("parent_reply_count") or 0.0))
        parent_term = _safe_log1p(parent_reply_count)
        return quality_mult * (
            float(cfg.engagement_reply_like_weight) * likes_term
            + float(cfg.engagement_reply_parent_weight) * parent_term
        )
    # Trend signals do not have the same engagement fields; use normalized payload score fallback.
    return quality_mult * max(0.0, float(signal.engagement_score or 0.0))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pct = _clip(percentile, 0.0, 100.0) / 100.0
    rank = pct * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ((1.0 - weight) * ordered[lower]) + (weight * ordered[upper])


def _source_coverage(*, intervals: list[tuple[int, int]], candidate_duration_ms: int, eps: float) -> float:
    if candidate_duration_ms <= eps:
        return 0.0
    if not intervals:
        return 0.0
    ordered = sorted((max(0, int(start)), max(0, int(end))) for start, end in intervals if int(end) > int(start))
    if not ordered:
        return 0.0
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    union_overlap = sum(max(0, end - start) for start, end in merged)
    return _clip(union_overlap / max(float(candidate_duration_ms), eps), 0.0, 1.0)


__all__ = ["SignalScoringResult", "apply_signal_scoring"]
