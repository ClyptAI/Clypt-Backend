"""Body continuity support for speaker binding.

This first pass intentionally keeps pose or head-orientation cues dormant.
If pose support is added later, it should be wired through explicit opt-in.
"""

from __future__ import annotations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _identity_feature_signal(identity_feature: dict | None) -> float:
    if not isinstance(identity_feature, dict):
        return 0.0
    if identity_feature.get("embedding"):
        return 1.0
    if identity_feature.get("face_observations"):
        return 0.7
    if identity_feature.get("cluster_id") not in (None, ""):
        return 0.5
    return 0.0


def score_body_continuity_support(
    *,
    mean_body_prior: float,
    track_quality: float,
    prominence: float,
    visibility_continuity: float,
    face_coverage: float = 0.0,
    face_continuity: float = 0.0,
    hard_reject_ratio: float = 0.0,
    identity_feature: dict | None = None,
    pose_support: float | None = None,
    enable_pose_support: bool = False,
) -> dict[str, float | bool]:
    """Return continuity support signals for LR-ASD candidate survival."""

    identity_signal = _identity_feature_signal(identity_feature)
    pose_support_used = _clamp01(pose_support or 0.0) if enable_pose_support else 0.0

    continuity_support_score = _clamp01(
        (0.30 * _clamp01(mean_body_prior))
        + (0.28 * _clamp01(track_quality))
        + (0.22 * _clamp01(prominence))
        + (0.16 * _clamp01(visibility_continuity))
        + (0.08 * identity_signal)
        - (0.18 * _clamp01(hard_reject_ratio))
        + (0.04 * pose_support_used)
    )
    candidate_survives = bool(
        continuity_support_score >= 0.55
        or (
            continuity_support_score >= 0.48
            and identity_signal > 0.0
            and _clamp01(track_quality) >= 0.70
        )
    )
    tie_break_bonus = min(
        0.18,
        (0.10 * continuity_support_score)
        + (0.05 * identity_signal)
        + (0.02 * _clamp01(face_coverage))
        + (0.01 * _clamp01(face_continuity))
        + (0.02 * pose_support_used),
    )

    return {
        "continuity_support_score": float(continuity_support_score),
        "candidate_survives": candidate_survives,
        "tie_break_bonus": float(tie_break_bonus),
        "identity_feature_signal": float(identity_signal),
        "pose_support_used": float(pose_support_used),
    }
