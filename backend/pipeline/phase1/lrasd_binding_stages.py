"""LR-ASD speaker binding sub-stages: turn aggregation, calibration, policy, smoothing."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def bind_audio_turns_to_local_tracks(
    turns: list[dict],
    local_candidate_evidence: list[dict],
    *,
    ambiguity_margin: float = 0.05,
    support_tiebreak_margin: float = 0.1,
) -> list[dict]:
    """Aggregate LR-ASD (and blended) local-track scores across each diarized turn."""
    def _as_int_ms(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _as_score(candidate: dict) -> float | None:
        for field_name in ("score", "total", "prob", "body_prior", "confidence"):
            value = candidate.get(field_name)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    normalized_evidence: list[dict] = []
    for evidence in local_candidate_evidence or []:
        start_time_ms = _as_int_ms(evidence.get("start_time_ms"), default=0)
        end_time_ms = _as_int_ms(evidence.get("end_time_ms"), default=start_time_ms)
        if end_time_ms < start_time_ms:
            start_time_ms, end_time_ms = end_time_ms, start_time_ms
        if end_time_ms <= start_time_ms:
            continue

        candidates = evidence.get("candidates")
        if not isinstance(candidates, list):
            candidates = [evidence]

        normalized_candidates: list[dict] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or bool(candidate.get("hard_reject", False)):
                continue
            local_track_id = (
                candidate.get("local_track_id")
                or candidate.get("local_tid")
                or candidate.get("track_id")
            )
            if local_track_id in (None, ""):
                continue
            score = _as_score(candidate)
            if score is None:
                continue
            normalized_candidates.append(
                {
                    "local_track_id": str(local_track_id),
                    "score": float(score),
                }
            )

        if normalized_candidates:
            normalized_evidence.append(
                {
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "candidates": normalized_candidates,
                }
            )

    bindings: list[dict] = []
    for turn in turns or []:
        start_time_ms = _as_int_ms(turn.get("start_time_ms"), default=0)
        end_time_ms = _as_int_ms(turn.get("end_time_ms"), default=start_time_ms)
        if end_time_ms < start_time_ms:
            start_time_ms, end_time_ms = end_time_ms, start_time_ms
        turn_duration_ms = max(1, end_time_ms - start_time_ms)
        overlap_present = bool(turn.get("overlap", False))
        explicit_exclusive = turn.get("exclusive")
        high_ambiguity_turn = bool(
            overlap_present
            or explicit_exclusive is False
        )

        weighted_score_ms_by_track: dict[str, float] = defaultdict(float)
        support_ms_by_track: dict[str, int] = defaultdict(int)
        clean_weighted_score_ms_by_track: dict[str, float] = defaultdict(float)
        clean_support_ms_by_track: dict[str, int] = defaultdict(int)
        overlapping_evidence: list[dict] = []
        slice_boundaries_ms = {start_time_ms, end_time_ms}
        max_visible_candidates = 0

        for evidence in normalized_evidence:
            overlap_start_ms = max(start_time_ms, int(evidence["start_time_ms"]))
            overlap_end_ms = min(end_time_ms, int(evidence["end_time_ms"]))
            if overlap_end_ms <= overlap_start_ms:
                continue
            overlapping_evidence.append(
                {
                    "start_time_ms": overlap_start_ms,
                    "end_time_ms": overlap_end_ms,
                    "candidates": evidence["candidates"],
                }
            )
            slice_boundaries_ms.add(overlap_start_ms)
            slice_boundaries_ms.add(overlap_end_ms)

        ordered_boundaries_ms = sorted(slice_boundaries_ms)
        for slice_start_ms, slice_end_ms in zip(ordered_boundaries_ms, ordered_boundaries_ms[1:]):
            slice_duration_ms = slice_end_ms - slice_start_ms
            if slice_duration_ms <= 0:
                continue
            active_by_track: dict[str, list[float]] = defaultdict(list)
            for evidence in overlapping_evidence:
                if int(evidence["end_time_ms"]) <= slice_start_ms or int(evidence["start_time_ms"]) >= slice_end_ms:
                    continue
                for candidate in evidence["candidates"]:
                    active_by_track[str(candidate["local_track_id"])].append(float(candidate["score"]))
            visible_candidate_count = len(active_by_track)
            max_visible_candidates = max(max_visible_candidates, visible_candidate_count)

            for local_track_id, active_scores in active_by_track.items():
                if not active_scores:
                    continue
                avg_score = (sum(active_scores) / len(active_scores))
                weighted_score_ms_by_track[local_track_id] += (avg_score * slice_duration_ms)
                support_ms_by_track[local_track_id] += slice_duration_ms
                if visible_candidate_count <= 2:
                    clean_weighted_score_ms_by_track[local_track_id] += (avg_score * slice_duration_ms)
                    clean_support_ms_by_track[local_track_id] += slice_duration_ms

        binding: dict[str, Any] = {
            "speaker_id": str(turn.get("speaker_id", "") or ""),
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "local_track_id": None,
            "ambiguous": False,
            "winning_score": None,
            "winning_margin": None,
            "support_ratio": 0.0,
        }
        if max_visible_candidates > 2:
            binding["max_visible_candidates"] = int(max_visible_candidates)
        if not weighted_score_ms_by_track:
            if high_ambiguity_turn:
                binding["ambiguous"] = True
            bindings.append(binding)
            continue

        ranked = sorted(
            (
                (
                    float(weighted_score_ms / turn_duration_ms),
                    float(support_ms_by_track[local_track_id] / turn_duration_ms),
                    str(local_track_id),
                )
                for local_track_id, weighted_score_ms in weighted_score_ms_by_track.items()
            ),
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        )
        best_score, best_support_ratio, best_local_track_id = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        second_support_ratio = ranked[1][1] if len(ranked) > 1 else None
        winning_margin = (
            float(best_score)
            if second_score is None
            else float(best_score - second_score)
        )
        support_margin = (
            float(best_support_ratio)
            if second_support_ratio is None
            else float(best_support_ratio - second_support_ratio)
        )

        if high_ambiguity_turn:
            best_score *= 0.5
            winning_margin *= 0.5

        binding["winning_score"] = float(best_score)
        binding["winning_margin"] = float(winning_margin)
        binding["support_ratio"] = float(best_support_ratio)
        crowded_turn = max_visible_candidates > 2
        if crowded_turn:
            clean_ranked = sorted(
                (
                    (
                        float(clean_weighted_score_ms_by_track[local_track_id] / clean_support_ms_by_track[local_track_id]),
                        float(clean_support_ms_by_track[local_track_id] / turn_duration_ms),
                        str(local_track_id),
                    )
                    for local_track_id in clean_support_ms_by_track.keys()
                    if int(clean_support_ms_by_track[local_track_id]) > 0
                ),
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            if clean_ranked:
                clean_best_score, clean_best_support_ratio, clean_best_local_track_id = clean_ranked[0]
                clean_best_support_ms = int(clean_support_ms_by_track.get(clean_best_local_track_id, 0))
                clean_second_score = clean_ranked[1][0] if len(clean_ranked) > 1 else None
                binding["clean_local_track_id"] = str(clean_best_local_track_id)
                binding["clean_support_ms"] = clean_best_support_ms
                binding["clean_support_ratio"] = float(clean_best_support_ratio)
                binding["clean_winning_score"] = float(clean_best_score)
                binding["clean_winning_margin"] = (
                    float(clean_best_score)
                    if clean_second_score is None
                    else float(clean_best_score - clean_second_score)
                )

        if high_ambiguity_turn:
            binding["ambiguous"] = True
        elif (
            second_score is not None
            and winning_margin < float(ambiguity_margin)
            and support_margin < float(support_tiebreak_margin)
        ):
            binding["ambiguous"] = True
        elif best_score > 0.0 and best_support_ratio > 0.0:
            binding["local_track_id"] = str(best_local_track_id)

        bindings.append(binding)

    return bindings


def calibrate_lrasd_word_confidence(
    *,
    best_prob: float | None,
    best_body: float,
    top_margin: float | None,
    min_assignment_margin: float,
) -> float | None:
    """Map raw LR-ASD / blend signals to a single [0, 1] confidence (v3 calibrated_confidence)."""
    margin = float(top_margin or 0.0)
    scale = max(min_assignment_margin * 2.0, 1e-6)
    norm_margin = max(0.0, min(1.0, margin / scale))
    if best_prob is not None:
        p = max(0.0, min(1.0, float(best_prob)))
        b = max(0.0, min(1.0, float(best_body)))
        return max(0.0, min(1.0, 0.50 * p + 0.30 * b + 0.20 * norm_margin))
    b_only = max(0.0, min(1.0, float(best_body)))
    return max(0.0, min(1.0, 0.62 * b_only + 0.28 * norm_margin + 0.10))


def evaluate_lrasd_assignment_policy(
    *,
    best_prob: float | None,
    best_total: float,
    best_body: float,
    second_total: float | None,
    min_lrasd_prob: float,
    min_assignment_margin: float,
    min_body_fallback_score: float,
    audio_prior_applied: bool,
) -> tuple[bool, float | None]:
    """Decision policy: assign vs unknown from threshold gates (matches legacy worker logic)."""
    top_margin = None if second_total is None else float(best_total - second_total)
    if best_prob is not None:
        confident_pick = bool(
            float(best_prob) >= min_lrasd_prob
            and (
                second_total is None
                or (best_total - second_total) >= min_assignment_margin
                or best_body >= 0.80
                or audio_prior_applied
            )
        )
    else:
        confident_pick = bool(
            best_body >= min_body_fallback_score
            and (
                second_total is None
                or (best_total - second_total) >= (0.5 * min_assignment_margin)
            )
        )
    return confident_pick, top_margin


def lrasd_abstention_reason(
    *,
    confident_pick: bool,
    no_candidates: bool,
    audio_prior_abstain: bool,
    audio_prior_mismatch: bool,
    best_prob: float | None,
    best_body: float,
    second_total: float | None,
    best_total: float,
    min_lrasd_prob: float,
    min_assignment_margin: float,
    min_body_fallback_score: float,
    audio_prior_applied: bool,
) -> str | None:
    """Human-readable abstention label when assignment is unknown."""
    if confident_pick:
        return None
    if no_candidates:
        return "no_candidates"
    if audio_prior_abstain:
        return "audio_turn_abstain"
    if audio_prior_mismatch:
        return "audio_prior_mismatch"
    if best_prob is None:
        if float(best_body) < min_body_fallback_score:
            return "below_body_fallback"
        if second_total is not None and (best_total - second_total) < (0.5 * min_assignment_margin):
            return "below_min_assignment_margin"
        return "no_asd_score"
    if float(best_prob) < min_lrasd_prob:
        return "below_min_lrasd_prob"
    if (
        second_total is not None
        and (best_total - second_total) < min_assignment_margin
        and float(best_body) < 0.80
        and not audio_prior_applied
    ):
        return "below_min_assignment_margin"
    return "low_confidence"


def apply_turn_consistency_smoothing(
    words: list[dict],
    *,
    protected_unknown_key: str,
    window: int = 2,
    min_neighbor_votes: int = 2,
    suppress_singleton_switches: bool = False,
) -> None:
    """Neighborhood majority smoothing plus optional singleton / one-word switch suppression."""
    seq = [w.get("speaker_track_id") for w in words]
    smoothed = seq[:]
    local_seq = [w.get("speaker_local_track_id") for w in words]
    local_smoothed = local_seq[:]
    for i in range(len(seq)):
        if bool(words[i].get(protected_unknown_key, False)):
            smoothed[i] = None
            local_smoothed[i] = None
            continue
        lo = max(0, i - window)
        hi = min(len(seq), i + window + 1)
        neigh = [t for t in seq[lo:hi] if t]
        if not neigh:
            local_neigh = [t for t in local_seq[lo:hi] if t]
            if local_neigh:
                local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
                if local_cnt >= min_neighbor_votes:
                    local_smoothed[i] = local_major
            continue
        major, cnt = Counter(neigh).most_common(1)[0]
        if cnt >= min_neighbor_votes:
            smoothed[i] = major
        local_neigh = [t for t in local_seq[lo:hi] if t]
        if local_neigh:
            local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
            if local_cnt >= min_neighbor_votes:
                local_smoothed[i] = local_major

    if suppress_singleton_switches and len(smoothed) >= 3:
        for i in range(1, len(smoothed) - 1):
            if bool(words[i].get(protected_unknown_key, False)):
                continue
            a, b, c = smoothed[i - 1], smoothed[i], smoothed[i + 1]
            if a is None or c is None:
                continue
            if a == c and b is not None and b != a:
                smoothed[i] = a
            la, lb, lc = local_smoothed[i - 1], local_smoothed[i], local_smoothed[i + 1]
            if la is not None and lc is not None and la == lc and lb is not None and lb != la:
                local_smoothed[i] = la

    for w, tid, local_tid in zip(words, smoothed, local_smoothed):
        if bool(w.get(protected_unknown_key, False)):
            w["speaker_track_id"] = None
            w["speaker_tag"] = "unknown"
            w["speaker_local_track_id"] = None
            w["speaker_local_tag"] = "unknown"
            continue
        w["speaker_track_id"] = tid
        w["speaker_tag"] = tid or "unknown"
        w["speaker_local_track_id"] = local_tid
        w["speaker_local_tag"] = local_tid or "unknown"


__all__ = [
    "apply_turn_consistency_smoothing",
    "bind_audio_turns_to_local_tracks",
    "calibrate_lrasd_word_confidence",
    "evaluate_lrasd_assignment_policy",
    "lrasd_abstention_reason",
]
