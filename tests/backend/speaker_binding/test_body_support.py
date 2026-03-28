import pytest
import numpy as np

from backend.speaker_binding.body_support import score_body_continuity_support


def test_stable_body_track_survives_face_flicker():
    result = score_body_continuity_support(
        mean_body_prior=0.83,
        track_quality=0.91,
        prominence=0.88,
        visibility_continuity=0.94,
        face_coverage=0.08,
        face_continuity=0.06,
        hard_reject_ratio=0.0,
        identity_feature=None,
    )

    assert result["candidate_survives"] is True
    assert result["continuity_support_score"] > 0.75
    assert result["tie_break_bonus"] > 0.0


def test_large_prominent_body_track_beats_tiny_face_heavy_fragment():
    stable_body = score_body_continuity_support(
        mean_body_prior=0.79,
        track_quality=0.86,
        prominence=0.92,
        visibility_continuity=0.90,
        face_coverage=0.10,
        face_continuity=0.08,
        hard_reject_ratio=0.0,
        identity_feature=None,
    )
    tiny_fragment = score_body_continuity_support(
        mean_body_prior=0.42,
        track_quality=0.48,
        prominence=0.05,
        visibility_continuity=0.42,
        face_coverage=1.0,
        face_continuity=1.0,
        hard_reject_ratio=0.35,
        identity_feature=None,
    )

    assert stable_body["continuity_support_score"] > tiny_fragment["continuity_support_score"]
    assert stable_body["tie_break_bonus"] > tiny_fragment["tie_break_bonus"]
    assert stable_body["candidate_survives"] is True
    assert tiny_fragment["candidate_survives"] is False


def test_identity_features_and_track_quality_raise_candidate_survival():
    baseline = score_body_continuity_support(
        mean_body_prior=0.54,
        track_quality=0.51,
        prominence=0.32,
        visibility_continuity=0.56,
        face_coverage=0.18,
        face_continuity=0.12,
        hard_reject_ratio=0.0,
        identity_feature=None,
    )
    with_identity = score_body_continuity_support(
        mean_body_prior=0.54,
        track_quality=0.78,
        prominence=0.32,
        visibility_continuity=0.56,
        face_coverage=0.18,
        face_continuity=0.12,
        hard_reject_ratio=0.0,
        identity_feature={"embedding": [0.1, 0.2, 0.3]},
    )

    assert baseline["candidate_survives"] is False
    assert with_identity["candidate_survives"] is True
    assert with_identity["continuity_support_score"] > baseline["continuity_support_score"]
    assert with_identity["tie_break_bonus"] > baseline["tie_break_bonus"]


def test_pose_support_remains_dormant_without_explicit_opt_in():
    result = score_body_continuity_support(
        mean_body_prior=0.71,
        track_quality=0.74,
        prominence=0.60,
        visibility_continuity=0.72,
        face_coverage=0.20,
        face_continuity=0.18,
        hard_reject_ratio=0.0,
        identity_feature=None,
    )

    assert result["pose_support_used"] == pytest.approx(0.0, abs=1e-6)


def test_identity_embedding_numpy_array_counts_as_present():
    result = score_body_continuity_support(
        mean_body_prior=0.54,
        track_quality=0.78,
        prominence=0.32,
        visibility_continuity=0.56,
        face_coverage=0.18,
        face_continuity=0.12,
        hard_reject_ratio=0.0,
        identity_feature={"embedding": np.asarray([0.1, 0.2, 0.3], dtype=np.float32)},
    )

    assert result["candidate_survives"] is True
    assert result["identity_feature_signal"] == pytest.approx(1.0, abs=1e-6)
