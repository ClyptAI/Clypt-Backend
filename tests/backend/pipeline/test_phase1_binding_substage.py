"""Unit tests for phase1 LR-ASD binding sub-stages (policy, hysteresis)."""

import pytest

from backend.pipeline.phase1.lrasd_binding_stages import (
    apply_turn_consistency_smoothing,
    bind_audio_turns_to_local_tracks,
    evaluate_lrasd_assignment_policy,
    lrasd_abstention_reason,
)


def test_bind_audio_turn_respects_assign_vs_unknown_margin_threshold():
    """Near-tied turn-level scores stay unassigned when margin gates are tight."""
    turns = [
        {
            "speaker_id": "SPEAKER_01",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 500,
            "candidates": [
                {"local_track_id": "track_1", "score": 0.82},
                {"local_track_id": "track_2", "score": 0.79},
            ],
        },
        {
            "start_time_ms": 500,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_1", "score": 0.80},
                {"local_track_id": "track_2", "score": 0.81},
            ],
        },
    ]
    bindings = bind_audio_turns_to_local_tracks(
        turns,
        local_candidate_evidence,
        ambiguity_margin=0.02,
    )
    assert bindings[0]["local_track_id"] is None
    assert bindings[0]["ambiguous"] is True


def test_evaluate_lrasd_assignment_policy_abstains_below_margin():
    confident, margin = evaluate_lrasd_assignment_policy(
        best_prob=0.5,
        best_total=0.9,
        best_body=0.5,
        second_total=0.88,
        min_lrasd_prob=0.15,
        min_assignment_margin=0.05,
        min_body_fallback_score=0.62,
        audio_prior_applied=False,
    )
    assert margin is not None
    assert margin < 0.05
    assert confident is False


def test_singleton_hysteresis_suppresses_isolated_one_word_switch():
    """With window=0, neighborhood voting never overrides; singleton pass fixes A-B-A."""
    words = [
        {"speaker_track_id": "A", "speaker_tag": "A", "speaker_local_track_id": "a", "speaker_local_tag": "a"},
        {"speaker_track_id": "B", "speaker_tag": "B", "speaker_local_track_id": "b", "speaker_local_tag": "b"},
        {"speaker_track_id": "A", "speaker_tag": "A", "speaker_local_track_id": "a", "speaker_local_tag": "a"},
    ]
    apply_turn_consistency_smoothing(
        words,
        protected_unknown_key="_never",
        window=0,
        min_neighbor_votes=2,
        suppress_singleton_switches=True,
    )
    assert words[1]["speaker_track_id"] == "A"
    assert words[1]["speaker_local_track_id"] == "a"


def test_lrasd_abstention_reason_maps_low_confidence():
    reason = lrasd_abstention_reason(
        confident_pick=False,
        no_candidates=False,
        audio_prior_abstain=False,
        audio_prior_mismatch=False,
        best_prob=0.5,
        best_body=0.5,
        second_total=0.49,
        best_total=0.51,
        min_lrasd_prob=0.15,
        min_assignment_margin=0.5,
        min_body_fallback_score=0.62,
        audio_prior_applied=False,
    )
    assert reason == "below_min_assignment_margin"
