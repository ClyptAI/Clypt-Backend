from __future__ import annotations

from collections.abc import Iterable, Mapping


def _clamp_non_negative(value: float) -> float:
    return max(0.0, value)


def _clamp_unit(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalized_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _signal_value(row: Mapping[str, object]) -> float:
    present_values = [
        _clamp_unit(row.get("mouth_open_ratio")),
        _clamp_unit(row.get("mouth_wide_ratio")),
        _clamp_unit(row.get("blendshape_jaw_open")),
        _clamp_unit(row.get("blendshape_mouth_open")),
    ]
    filtered_values = [
        value
        for key, value in zip(
            ("mouth_open_ratio", "mouth_wide_ratio", "blendshape_jaw_open", "blendshape_mouth_open"),
            present_values,
            strict=False,
        )
        if row.get(key) is not None
    ]
    if not filtered_values:
        return 0.0
    return sum(filtered_values) / len(filtered_values)


def _sorted_rows(samples: Iterable[Mapping[str, object]] | None) -> list[dict[str, object]]:
    rows = [dict(sample or {}) for sample in (samples or [])]
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("frame_idx", 0) or 0),
            int(bool(row.get("face_detected", False))),
        ),
    )


def sample_span_frame_indices(frame_indices: Iterable[int] | None, max_samples: int = 6) -> list[int]:
    unique_indices = sorted({int(frame_idx) for frame_idx in (frame_indices or [])})
    if not unique_indices or max_samples <= 0:
        return []
    if len(unique_indices) <= max_samples:
        return unique_indices
    if max_samples == 1:
        return [unique_indices[len(unique_indices) // 2]]

    last_index = len(unique_indices) - 1
    sampled_positions = {
        (sample_idx * last_index) // (max_samples - 1)
        for sample_idx in range(max_samples)
    }
    return [unique_indices[position] for position in sorted(sampled_positions)]


def summarize_mouth_landmark_signal(samples: Iterable[Mapping[str, object]] | None) -> dict[str, float | int]:
    sample_rows = _sorted_rows(samples)
    total_frame_count = len(sample_rows)
    usable_face_rows = [row for row in sample_rows if bool(row.get("face_detected", False))]
    usable_face_frame_count = len(usable_face_rows)

    face_visibility_score = _round_score(usable_face_frame_count / total_frame_count) if total_frame_count else 0.0

    blendshape_support_count = 0
    usable_signal_values: list[float] = []
    for row in usable_face_rows:
        usable_signal_values.append(_signal_value(row))
        if row.get("blendshape_jaw_open") is not None or row.get("blendshape_mouth_open") is not None:
            blendshape_support_count += 1

    if usable_signal_values:
        if len(usable_signal_values) >= 2:
            deltas = [
                abs(curr - prev)
                for prev, curr in zip(usable_signal_values, usable_signal_values[1:], strict=False)
            ]
            mean_delta = sum(deltas) / len(deltas)
            dynamic_range = max(usable_signal_values) - min(usable_signal_values)
        else:
            mean_delta = 0.0
            dynamic_range = 0.0
        mean_signal = sum(usable_signal_values) / len(usable_signal_values)
        mouth_motion_score = _round_score(
            (0.55 * min(1.0, mean_delta * 4.0))
            + (0.25 * min(1.0, dynamic_range * 4.0))
            + (0.20 * mean_signal)
        )
    else:
        mouth_motion_score = 0.0

    blendshape_support_score = _round_score(blendshape_support_count / total_frame_count) if total_frame_count else 0.0

    return {
        "mouth_motion_score": mouth_motion_score,
        "face_visibility_score": face_visibility_score,
        "blendshape_support_score": blendshape_support_score,
        "usable_face_frame_count": usable_face_frame_count,
    }


def summarize_pose_signal(samples: Iterable[Mapping[str, object]] | None) -> dict[str, float | int]:
    sample_rows = _sorted_rows(samples)
    total_frame_count = len(sample_rows)

    usable_pose_rows = [
        row
        for row in sample_rows
        if any(
            _clamp_unit(row.get(key)) > 0.0
            for key in ("upper_body_visibility", "head_visibility", "frontal_support")
        )
    ]
    usable_pose_frame_count = len(usable_pose_rows)

    if usable_pose_rows:
        visibility_scores = []
        for row in usable_pose_rows:
            upper_body_visibility = _clamp_unit(row.get("upper_body_visibility"))
            head_visibility = _clamp_unit(row.get("head_visibility"))
            frontal_support = _clamp_unit(row.get("frontal_support"))
            visibility_scores.append(
                (0.5 * upper_body_visibility)
                + (0.25 * head_visibility)
                + (0.25 * frontal_support)
            )
        pose_visibility_score = _round_score(sum(visibility_scores) / len(visibility_scores))

        stability_terms: list[float] = []
        for prev_row, curr_row in zip(usable_pose_rows, usable_pose_rows[1:], strict=False):
            prev_x = prev_row.get("torso_center_x")
            prev_y = prev_row.get("torso_center_y")
            curr_x = curr_row.get("torso_center_x")
            curr_y = curr_row.get("torso_center_y")
            prev_span = float(prev_row.get("shoulder_span", 0.0) or 0.0)
            curr_span = float(curr_row.get("shoulder_span", 0.0) or 0.0)
            if None in (prev_x, prev_y, curr_x, curr_y) or prev_span <= 1e-6 or curr_span <= 1e-6:
                continue
            norm_span = max((prev_span + curr_span) * 0.5, 1e-6)
            dx = abs(float(curr_x) - float(prev_x)) / norm_span
            dy = abs(float(curr_y) - float(prev_y)) / norm_span
            ds = abs(curr_span - prev_span) / norm_span
            instability = (0.45 * dx) + (0.35 * dy) + (0.20 * ds)
            stability_terms.append(_clamp_non_negative(1.0 - (instability * 1.6)))

        if stability_terms:
            pose_stability_score = _round_score(sum(stability_terms) / len(stability_terms))
        else:
            pose_stability_score = _round_score(min(1.0, pose_visibility_score * 0.6))
    else:
        pose_visibility_score = 0.0
        pose_stability_score = 0.0

    return {
        "pose_visibility_score": pose_visibility_score,
        "pose_stability_score": pose_stability_score,
        "usable_pose_frame_count": usable_pose_frame_count,
    }


def combine_visual_candidate_signals(
    *,
    visual_identity_id: object,
    mouth_summary: Mapping[str, object] | None = None,
    pose_summary: Mapping[str, object] | None = None,
    mapping_summary: Mapping[str, object] | None = None,
    local_track_id: object | None = None,
) -> dict[str, object]:
    mouth_summary = dict(mouth_summary or {})
    pose_summary = dict(pose_summary or {})
    mapping_summary = dict(mapping_summary or {})

    candidate = {
        "visual_identity_id": _normalized_id(visual_identity_id),
        "local_track_id": _normalized_id(local_track_id) if local_track_id is not None else "",
        "mouth_motion_score": _round_score(mouth_summary.get("mouth_motion_score", 0.0)),
        "face_visibility_score": _round_score(mouth_summary.get("face_visibility_score", 0.0)),
        "blendshape_support_score": _round_score(mouth_summary.get("blendshape_support_score", 0.0)),
        "usable_face_frame_count": int(mouth_summary.get("usable_face_frame_count", 0) or 0),
        "pose_visibility_score": _round_score(pose_summary.get("pose_visibility_score", 0.0)),
        "pose_stability_score": _round_score(pose_summary.get("pose_stability_score", 0.0)),
        "usable_pose_frame_count": int(pose_summary.get("usable_pose_frame_count", 0) or 0),
        "mapping_confidence": _round_score(mapping_summary.get("mapping_confidence", 0.0)),
    }

    if not candidate["local_track_id"]:
        candidate["local_track_id"] = _normalized_id(mapping_summary.get("local_track_id"))

    candidate["composite_score"] = _round_score(
        (0.42 * float(candidate["mouth_motion_score"]))
        + (0.16 * float(candidate["pose_visibility_score"]))
        + (0.12 * float(candidate["pose_stability_score"]))
        + (0.15 * float(candidate["face_visibility_score"]))
        + (0.10 * float(candidate["mapping_confidence"]))
        + (0.05 * float(candidate["blendshape_support_score"]))
    )

    return candidate
