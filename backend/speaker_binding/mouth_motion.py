from __future__ import annotations

from collections.abc import Iterable, Mapping


def _score_candidate(candidate: Mapping[str, object]) -> float:
    mouth_motion_score = float(candidate.get("mouth_motion_score", 0.0) or 0.0)
    pose_visibility_score = float(candidate.get("pose_visibility_score", 0.0) or 0.0)
    face_visibility_score = float(candidate.get("face_visibility_score", 0.0) or 0.0)
    mapping_confidence = float(candidate.get("mapping_confidence", 0.0) or 0.0)
    return round(
        (0.45 * mouth_motion_score)
        + (0.2 * pose_visibility_score)
        + (0.2 * face_visibility_score)
        + (0.15 * mapping_confidence),
        3,
    )


def choose_visual_speaking_candidate(
    candidates: Iterable[Mapping[str, object]] | None,
    *,
    winning_margin: float = 0.05,
) -> dict[str, object]:
    ranked_candidates = []
    for candidate in candidates or ():
        visual_identity_id = str(candidate.get("visual_identity_id", "") or "").strip()
        if not visual_identity_id:
            continue
        ranked_candidates.append(
            {
                **dict(candidate),
                "visual_identity_id": visual_identity_id,
                "composite_score": _score_candidate(candidate),
            }
        )

    ranked_candidates.sort(
        key=lambda item: (-float(item["composite_score"]), item["visual_identity_id"])
    )
    if not ranked_candidates:
        return {
            "winner_visual_identity_id": None,
            "unresolved": True,
            "reason": "no_candidates",
            "ranked_candidates": [],
        }

    top_score = float(ranked_candidates[0]["composite_score"])
    second_score = float(ranked_candidates[1]["composite_score"]) if len(ranked_candidates) > 1 else 0.0
    if len(ranked_candidates) > 1 and (top_score - second_score) < float(winning_margin):
        return {
            "winner_visual_identity_id": None,
            "unresolved": True,
            "reason": "ambiguous_visual_signals",
            "ranked_candidates": ranked_candidates,
        }

    return {
        "winner_visual_identity_id": ranked_candidates[0]["visual_identity_id"],
        "unresolved": False,
        "reason": "resolved",
        "ranked_candidates": ranked_candidates,
    }
