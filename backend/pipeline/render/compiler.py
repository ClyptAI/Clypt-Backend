from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from backend.pipeline.contracts import ShotTrackletIndex, TrackletGeometry

from .contracts import CaptionPlan, CaptionPreset, PublishMetadata, RenderPlan, RenderPlanClip, RenderPlanSegment
from .presets import load_caption_presets


_TARGET_ASPECT = 9.0 / 16.0
_MIN_AUTO_FOLLOW_CROP_WIDTH = 360.0
_MIN_AUTO_FOLLOW_CROP_HEIGHT = 640.0


def _coerce_caption_plan(caption_plan: CaptionPlan | dict[str, Any]) -> CaptionPlan:
    if isinstance(caption_plan, CaptionPlan):
        return caption_plan
    return CaptionPlan.model_validate(caption_plan)


def _coerce_publish_metadata(
    publish_metadata: PublishMetadata | dict[str, Any],
) -> PublishMetadata:
    if isinstance(publish_metadata, PublishMetadata):
        return publish_metadata
    return PublishMetadata.model_validate(publish_metadata)


def _field(payload: Any, name: str, default: Any):
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _coerce_tracklet_index(
    shot_tracklet_index: ShotTrackletIndex | dict[str, Any] | None,
) -> ShotTrackletIndex:
    if isinstance(shot_tracklet_index, ShotTrackletIndex):
        return shot_tracklet_index
    if shot_tracklet_index is None:
        return ShotTrackletIndex(tracklets=[])
    return ShotTrackletIndex.model_validate(shot_tracklet_index, from_attributes=True)


def _coerce_tracklet_geometry(
    tracklet_geometry: TrackletGeometry | dict[str, Any] | None,
) -> TrackletGeometry:
    if isinstance(tracklet_geometry, TrackletGeometry):
        return tracklet_geometry
    if tracklet_geometry is None:
        return TrackletGeometry(tracklets=[])
    return TrackletGeometry.model_validate(tracklet_geometry, from_attributes=True)


def _camera_segments(camera_intent_timeline: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(segment) for segment in _field(camera_intent_timeline, "segments", [])]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _intent_default_zone(intent: str | None, fallback_zone: str) -> str:
    del intent
    return fallback_zone


def _resolve_camera_segment(
    *,
    clip_id: str,
    absolute_start_ms: int,
    absolute_end_ms: int,
    camera_intent_timeline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    segments = _camera_segments(camera_intent_timeline)
    if not segments:
        return None
    candidates = [
        segment
        for segment in segments
        if _field(segment, "clip_candidate_id", clip_id) in {clip_id, None}
        and int(_field(segment, "start_ms", 0)) <= absolute_start_ms
        and int(_field(segment, "end_ms", 0)) >= absolute_end_ms
    ]
    if not candidates:
        raise ValueError(
            f"clip {clip_id} has unresolved interval {absolute_start_ms}-{absolute_end_ms} "
            "against camera intent timeline"
        )
    candidates.sort(key=lambda segment: int(_field(segment, "end_ms", 0)) - int(_field(segment, "start_ms", 0)))
    return candidates[0]


def _center_band_collision(
    *,
    tracklet_id: str | None,
    absolute_start_ms: int,
    absolute_end_ms: int,
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
) -> bool:
    if not tracklet_id:
        return False
    points = geometry_by_tracklet.get(tracklet_id, [])
    for point in points:
        timestamp_ms = int(point["timestamp_ms"])
        bbox_xyxy = point["bbox_xyxy"]
        if timestamp_ms < absolute_start_ms or timestamp_ms > absolute_end_ms:
            continue
        _, top, _, bottom = bbox_xyxy
        overlap_top = max(top, 620.0)
        overlap_bottom = min(bottom, 1300.0)
        if overlap_bottom > overlap_top:
            return True
    return False


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _tracklet_pose_score(descriptor: Any) -> tuple[int, float, float, float, str]:
    quality = dict(_field(descriptor, "subject_quality", {}) or {})
    eligible = 1 if _field(descriptor, "auto_follow_eligible", True) is not False else 0
    head_ratio = float(quality.get("head_evidence_ratio") or 0.0)
    upper_ratio = float(quality.get("upper_body_anchor_ratio") or 0.0)
    median_confidence = float(quality.get("median_rfdetr_confidence") or 0.0)
    return (
        eligible,
        head_ratio,
        upper_ratio,
        median_confidence,
        str(_field(descriptor, "tracklet_id", "")),
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _tracklet_meets_auto_crop_guard(points: list[dict[str, Any]]) -> bool:
    sizes = [
        _bbox_inside_9x16_size([float(value) for value in point["bbox_xyxy"][:4]])
        for point in points
    ]
    if not sizes:
        return False
    return (
        _median([width for width, _ in sizes]) >= _MIN_AUTO_FOLLOW_CROP_WIDTH
        and _median([height for _, height in sizes]) >= _MIN_AUTO_FOLLOW_CROP_HEIGHT
    )


def _build_shot_subject_plan(
    *,
    tracklet_descriptors: list[Any],
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    by_shot: dict[str, list[Any]] = {}
    for descriptor in tracklet_descriptors:
        tracklet_id = str(_field(descriptor, "tracklet_id", ""))
        if not tracklet_id or tracklet_id not in geometry_by_tracklet:
            continue
        shot_id = str(_field(descriptor, "shot_id", ""))
        if not shot_id:
            continue
        by_shot.setdefault(shot_id, []).append(descriptor)

    selected: dict[str, str] = {}
    for shot_id, descriptors in by_shot.items():
        eligible = [
            descriptor
            for descriptor in descriptors
            if _field(descriptor, "auto_follow_eligible", True) is not False
            and _tracklet_meets_auto_crop_guard(
                geometry_by_tracklet.get(str(_field(descriptor, "tracklet_id", "")), [])
            )
        ]
        pool = eligible or [
            descriptor
            for descriptor in descriptors
            if _tracklet_meets_auto_crop_guard(
                geometry_by_tracklet.get(str(_field(descriptor, "tracklet_id", "")), [])
            )
        ]
        if not pool:
            continue
        winner = max(pool, key=_tracklet_pose_score)
        selected[shot_id] = str(_field(winner, "tracklet_id", ""))
    return selected


def _dominant_shot_id(
    *,
    absolute_start_ms: int,
    absolute_end_ms: int,
    tracklet_descriptors: list[Any],
) -> str | None:
    by_shot: dict[str, int] = {}
    for descriptor in tracklet_descriptors:
        shot_id = str(_field(descriptor, "shot_id", ""))
        if not shot_id:
            continue
        overlap = _overlap_ms(
            absolute_start_ms,
            absolute_end_ms,
            int(_field(descriptor, "start_ms", 0)),
            int(_field(descriptor, "end_ms", 0)),
        )
        if overlap > 0:
            by_shot[shot_id] = max(by_shot.get(shot_id, 0), overlap)
    if not by_shot:
        return None
    return max(by_shot.items(), key=lambda item: (item[1], item[0]))[0]


def _select_auto_tracklet(
    *,
    clip_id: str,
    absolute_start_ms: int,
    absolute_end_ms: int,
    tracklet_descriptors: list[Any],
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
    shot_subjects: dict[str, str] | None = None,
) -> str | None:
    if shot_subjects:
        shot_id = _dominant_shot_id(
            absolute_start_ms=absolute_start_ms,
            absolute_end_ms=absolute_end_ms,
            tracklet_descriptors=tracklet_descriptors,
        )
        if shot_id:
            tracklet_id = shot_subjects.get(shot_id)
            if tracklet_id and tracklet_id in geometry_by_tracklet:
                return tracklet_id

    candidates: list[tuple[int, int, str]] = []
    for descriptor in tracklet_descriptors:
        if _field(descriptor, "auto_follow_eligible", True) is False:
            continue
        tracklet_id = str(_field(descriptor, "tracklet_id", ""))
        if not tracklet_id or tracklet_id not in geometry_by_tracklet:
            continue
        if not _tracklet_meets_auto_crop_guard(geometry_by_tracklet.get(tracklet_id, [])):
            continue
        overlap = _overlap_ms(
            absolute_start_ms,
            absolute_end_ms,
            int(_field(descriptor, "start_ms", 0)),
            int(_field(descriptor, "end_ms", 0)),
        )
        if overlap <= 0:
            continue
        shot_id = str(_field(descriptor, "shot_id", ""))
        digest = hashlib.sha1(f"{clip_id}:{shot_id}:{tracklet_id}".encode("utf-8")).hexdigest()
        candidates.append((-overlap, int(digest[:8], 16), tracklet_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _geometry_points_for_interval(
    *,
    tracklet_id: str,
    absolute_start_ms: int,
    absolute_end_ms: int,
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    points = geometry_by_tracklet.get(tracklet_id, [])
    if not points:
        return []
    interval_points = [
        point
        for point in points
        if absolute_start_ms <= int(point["timestamp_ms"]) <= absolute_end_ms
    ]
    if interval_points:
        return sorted(interval_points, key=lambda point: int(point["timestamp_ms"]))
    midpoint_ms = (absolute_start_ms + absolute_end_ms) / 2.0
    return [
        min(points, key=lambda point: abs(int(point["timestamp_ms"]) - midpoint_ms))
    ]


def _bbox_inside_9x16_size(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    if width / height >= _TARGET_ASPECT:
        crop_height = height
        crop_width = crop_height * _TARGET_ASPECT
    else:
        crop_width = width
        crop_height = crop_width / _TARGET_ASPECT
    return max(2.0, crop_width), max(2.0, crop_height)


def _point_xy(point: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = point.get(name)
    if not value or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _clamp_float(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(float(value), upper))


def _pose_x_for_points(points: list[dict[str, Any]]) -> list[tuple[float | None, str]]:
    raw: list[float | None] = []
    for point in points:
        head = _point_xy(point, "head_center_xy")
        raw.append(head[0] if head else None)

    resolved: list[tuple[float | None, str]] = []
    for index, value in enumerate(raw):
        if value is not None:
            resolved.append((float(value), "pose"))
            continue

        previous_index = next((idx for idx in range(index - 1, -1, -1) if raw[idx] is not None), None)
        next_index = next((idx for idx in range(index + 1, len(raw)) if raw[idx] is not None), None)
        if previous_index is not None and next_index is not None:
            previous_time = float(points[previous_index]["timestamp_ms"])
            next_time = float(points[next_index]["timestamp_ms"])
            current_time = float(points[index]["timestamp_ms"])
            if next_time > previous_time:
                progress = (current_time - previous_time) / (next_time - previous_time)
                interpolated = float(raw[previous_index]) + (
                    (float(raw[next_index]) - float(raw[previous_index])) * progress
                )
                resolved.append((interpolated, "pose_interpolated"))
                continue
        if previous_index is not None:
            resolved.append((float(raw[previous_index]), "pose_hold"))
            continue
        if next_index is not None:
            resolved.append((float(raw[next_index]), "pose_hold"))
            continue
        resolved.append((None, "bbox_center"))
    return resolved


def _crop_keyframe_inside_bbox(
    *,
    point: dict[str, Any],
    clip_start_ms: int,
    run_id: str,
    run: dict[str, Any],
    anchor_x: float | None,
    anchor_source: str,
) -> dict[str, Any]:
    bbox = [float(value) for value in point["bbox_xyxy"][:4]]
    x1, y1, x2, y2 = bbox
    crop_width, crop_height = _bbox_inside_9x16_size(bbox)
    if anchor_x is None:
        anchor_x = (x1 + x2) / 2.0
    raw_x = float(anchor_x) - (crop_width * 0.5)
    x = _clamp_float(raw_x, x1, x2 - crop_width)
    y = _clamp_float(y1, y1, y2 - crop_height)
    return {
        "run_id": run_id,
        "shot_id": run["shot_id"],
        "tracklet_id": str(run["tracklet_id"]),
        "time_ms": max(0, int(point["timestamp_ms"]) - int(clip_start_ms)),
        "x": round(x, 3),
        "y": round(y, 3),
        "w": round(crop_width, 3),
        "h": round(crop_height, 3),
        "anchor_x": round(float(anchor_x), 3),
        "bbox_xyxy": bbox,
        "anchor_source": anchor_source,
    }


def _points_with_run_boundaries(
    *,
    points: list[dict[str, Any]],
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = sorted((dict(point) for point in points), key=lambda point: int(point["timestamp_ms"]))
    if not ordered:
        return []
    start_ms = int(run["start_ms"])
    if int(ordered[0]["timestamp_ms"]) > start_ms:
        first = dict(ordered[0])
        first["timestamp_ms"] = start_ms
        ordered.insert(0, first)
    return ordered


def _dynamic_crop_keyframes_for_run(
    *,
    points: list[dict[str, Any]],
    clip_start_ms: int,
    run: dict[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    run_points = _points_with_run_boundaries(points=points, run=run)
    anchors = _pose_x_for_points(run_points)
    return [
        _crop_keyframe_inside_bbox(
            point=point,
            clip_start_ms=clip_start_ms,
            run_id=run_id,
            run=run,
            anchor_x=anchor_x,
            anchor_source=anchor_source,
        )
        for point, (anchor_x, anchor_source) in zip(run_points, anchors, strict=True)
    ]


def _merge_render_tracklet_runs(
    *,
    clip_start_ms: int,
    render_segments: list[RenderPlanSegment],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for segment in render_segments:
        if not segment.primary_tracklet_id:
            continue
        absolute_start_ms = clip_start_ms + int(segment.start_ms)
        absolute_end_ms = clip_start_ms + int(segment.end_ms)
        if (
            runs
            and runs[-1]["tracklet_id"] == segment.primary_tracklet_id
            and runs[-1]["shot_id"] == segment.shot_id
            and runs[-1]["end_ms"] >= absolute_start_ms
        ):
            runs[-1]["end_ms"] = max(runs[-1]["end_ms"], absolute_end_ms)
            runs[-1]["relative_end_ms"] = max(runs[-1]["relative_end_ms"], int(segment.end_ms))
            continue
        runs.append(
            {
                "tracklet_id": segment.primary_tracklet_id,
                "shot_id": segment.shot_id,
                "start_ms": absolute_start_ms,
                "end_ms": absolute_end_ms,
                "relative_start_ms": int(segment.start_ms),
                "relative_end_ms": int(segment.end_ms),
            }
        )
    return runs


def _build_crop_plan(
    *,
    clip_start_ms: int,
    render_segments: list[RenderPlanSegment],
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    runs = _merge_render_tracklet_runs(
        clip_start_ms=clip_start_ms,
        render_segments=render_segments,
    )
    if not runs:
        return {}

    run_points: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for run in runs:
        points = _geometry_points_for_interval(
            tracklet_id=str(run["tracklet_id"]),
            absolute_start_ms=int(run["start_ms"]),
            absolute_end_ms=int(run["end_ms"]),
            geometry_by_tracklet=geometry_by_tracklet,
        )
        if not points:
            continue
        run_id = f"run_{len(run_points) + 1:04d}"
        run_points.append((run_id, run, points))

    if not run_points:
        return {}

    keyframes: list[dict[str, Any]] = []
    crop_segments: list[dict[str, Any]] = []
    crop_runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_id, run, points in run_points:
        crop_runs.append(
            {
                "run_id": run_id,
                "shot_id": run["shot_id"],
                "tracklet_id": str(run["tracklet_id"]),
                "start_ms": int(run["relative_start_ms"]),
                "end_ms": int(run["relative_end_ms"]),
            }
        )
        crop_segments.append(
            {
                "run_id": run_id,
                "start_ms": int(run["relative_start_ms"]),
                "end_ms": int(run["relative_end_ms"]),
                "tracklet_id": str(run["tracklet_id"]),
                "shot_id": run["shot_id"],
            }
        )
        keyframes.extend(
            _dynamic_crop_keyframes_for_run(
                points=points,
                clip_start_ms=clip_start_ms,
                run=run,
                run_id=run_id,
            )
        )
    keyframes.sort(key=lambda keyframe: int(keyframe["time_ms"]))
    if any(
        float(keyframe["w"]) < _MIN_AUTO_FOLLOW_CROP_WIDTH
        or float(keyframe["h"]) < _MIN_AUTO_FOLLOW_CROP_HEIGHT
        for keyframe in keyframes
    ):
        warnings.append("crop_source_size_below_auto_follow_guard")

    return {
        "mode": "tracklet_follow_9x16_pose_x_dynamic_inside_person",
        "tracklet_ids": _dedupe([str(run["tracklet_id"]) for _, run, _ in run_points]),
        "runs": crop_runs,
        "segments": crop_segments,
        "keyframes": keyframes,
        "warnings": warnings,
    }


def _render_segment_context(
    *,
    clip_id: str,
    clip_start_ms: int,
    segment: Any,
    previous_zone: str | None,
    previous_intent: str | None,
    camera_intent_timeline: dict[str, Any] | None,
    tracklet_to_shot: dict[str, str],
    tracklet_descriptors: list[Any],
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
    shot_subjects: dict[str, str],
) -> dict[str, Any]:
    review_reasons = []
    if segment.review_needed and segment.review_reason:
        review_reasons.append(segment.review_reason)

    absolute_start_ms = clip_start_ms + segment.start_ms
    absolute_end_ms = clip_start_ms + segment.end_ms
    camera_segment = _resolve_camera_segment(
        clip_id=clip_id,
        absolute_start_ms=absolute_start_ms,
        absolute_end_ms=absolute_end_ms,
        camera_intent_timeline=camera_intent_timeline,
    )
    intent = str(_field(camera_segment, "intent", "")) if camera_segment else None
    primary_tracklet_value = _field(camera_segment, "primary_tracklet_id", None) if camera_segment else None
    secondary_tracklet_value = _field(camera_segment, "secondary_tracklet_id", None) if camera_segment else None
    primary_tracklet_id = str(primary_tracklet_value) if primary_tracklet_value not in {"", None} else None
    secondary_tracklet_id = str(secondary_tracklet_value) if secondary_tracklet_value not in {"", None} else None
    if primary_tracklet_id is None:
        primary_tracklet_id = _select_auto_tracklet(
            clip_id=clip_id,
            absolute_start_ms=absolute_start_ms,
            absolute_end_ms=absolute_end_ms,
            tracklet_descriptors=tracklet_descriptors,
            geometry_by_tracklet=geometry_by_tracklet,
            shot_subjects=shot_subjects,
        )
        if primary_tracklet_id is not None:
            intent = "auto_follow"
    for tracklet_id in (primary_tracklet_id, secondary_tracklet_id):
        if not tracklet_id:
            continue
        if tracklet_to_shot and tracklet_id not in tracklet_to_shot:
            raise ValueError(f"camera intent references unresolved tracklet {tracklet_id}")
        if geometry_by_tracklet and tracklet_id not in geometry_by_tracklet:
            raise ValueError(f"camera intent references unresolved tracklet {tracklet_id}")

    preferred_zone = _intent_default_zone(intent, segment.placement_zone)
    caption_zone = preferred_zone
    fallback_applied = False
    zone_transition_reason = ""
    if preferred_zone == "center_band" and _center_band_collision(
        tracklet_id=primary_tracklet_id,
        absolute_start_ms=absolute_start_ms,
        absolute_end_ms=absolute_end_ms,
        geometry_by_tracklet=geometry_by_tracklet,
    ):
        caption_zone = "lower_safe"
        fallback_applied = True
        zone_transition_reason = "collision_fallback:center_band->lower_safe"
        review_reasons.append(
            f"center_band collision with primary tracklet {primary_tracklet_id}; fell back to lower_safe"
        )
    elif previous_zone is not None and caption_zone != previous_zone:
        if previous_intent and intent and previous_intent != intent:
            zone_transition_reason = f"camera_intent:{previous_intent}->{intent}"
        else:
            zone_transition_reason = f"zone_change:{previous_zone}->{caption_zone}"

    shot_id = tracklet_to_shot.get(primary_tracklet_id or "", None)
    return {
        "caption_zone": caption_zone,
        "layout_mode": intent,
        "shot_id": shot_id,
        "primary_tracklet_id": primary_tracklet_id,
        "secondary_tracklet_id": secondary_tracklet_id,
        "semantic_reason": intent or "",
        "review_reasons": _dedupe(review_reasons),
        "review_needed": bool(review_reasons),
        "fallback_applied": fallback_applied,
        "zone_transition_reason": zone_transition_reason,
        "intent": intent,
    }


def compile_render_plan(
    *,
    run_id: str,
    clip_finalists: list[dict[str, Any]] | None = None,
    caption_plan: CaptionPlan | dict[str, Any],
    publish_metadata: PublishMetadata | dict[str, Any],
    source_context: dict[str, Any] | None = None,
    participation_timeline: dict[str, Any] | None = None,
    camera_intent_timeline: dict[str, Any] | None = None,
    shot_tracklet_index: dict[str, Any] | None = None,
    tracklet_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del clip_finalists, source_context, participation_timeline
    parsed_caption_plan = _coerce_caption_plan(caption_plan)
    parsed_publish_metadata = _coerce_publish_metadata(publish_metadata)
    metadata_by_clip = {clip.clip_id: clip for clip in parsed_publish_metadata.clips}
    parsed_tracklet_index = _coerce_tracklet_index(shot_tracklet_index)
    parsed_tracklet_geometry = _coerce_tracklet_geometry(tracklet_geometry)
    tracklet_to_shot = {
        descriptor.tracklet_id: descriptor.shot_id
        for descriptor in parsed_tracklet_index.tracklets
    }
    tracklet_descriptors = list(parsed_tracklet_index.tracklets)
    geometry_by_tracklet = {
        entry.tracklet_id: [
            {
                "timestamp_ms": point.timestamp_ms,
                "bbox_xyxy": point.bbox_xyxy,
                "head_center_xy": point.head_center_xy,
                "shoulder_center_xy": point.shoulder_center_xy,
                "upper_torso_anchor_xy": point.upper_torso_anchor_xy,
            }
            for point in entry.points
        ]
        for entry in parsed_tracklet_geometry.tracklets
    }
    shot_subjects = _build_shot_subject_plan(
        tracklet_descriptors=tracklet_descriptors,
        geometry_by_tracklet=geometry_by_tracklet,
    )

    clips: list[RenderPlanClip] = []
    for clip in parsed_caption_plan.clips:
        overlays = [
            {
                "kind": "captions_ass",
                "path": str(Path("render") / "captions" / f"{clip.clip_id}.ass"),
            }
        ]
        if clip.clip_id in metadata_by_clip:
            overlays.append(
                {
                    "kind": "thumbnail_text",
                    "text": metadata_by_clip[clip.clip_id].thumbnail_text,
                }
            )

        render_segments: list[RenderPlanSegment] = []
        clip_review_reasons: list[str] = []
        previous_zone: str | None = None
        previous_intent: str | None = None
        for segment in clip.segments:
            segment_context = _render_segment_context(
                clip_id=clip.clip_id,
                clip_start_ms=clip.clip_start_ms,
                segment=segment,
                previous_zone=previous_zone,
                previous_intent=previous_intent,
                camera_intent_timeline=camera_intent_timeline,
                tracklet_to_shot=tracklet_to_shot,
                tracklet_descriptors=tracklet_descriptors,
                geometry_by_tracklet=geometry_by_tracklet,
                shot_subjects=shot_subjects,
            )
            previous_zone = segment_context["caption_zone"]
            previous_intent = segment_context["intent"]
            clip_review_reasons.extend(segment_context["review_reasons"])
            render_segments.append(
                RenderPlanSegment(
                    segment_id=f"{segment.segment_id}_render",
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    caption_segment_ids=[segment.segment_id],
                    caption_preset_id=clip.preset_id,
                    caption_zone=segment_context["caption_zone"],
                    highlight_mode=segment.highlight_mode,
                    shot_id=segment_context["shot_id"],
                    layout_mode=segment_context["layout_mode"],
                    primary_tracklet_id=segment_context["primary_tracklet_id"],
                    secondary_tracklet_id=segment_context["secondary_tracklet_id"],
                    semantic_reason=segment_context["semantic_reason"],
                    review_needed=segment_context["review_needed"],
                    review_reasons=segment_context["review_reasons"],
                    fallback_applied=segment_context["fallback_applied"],
                    zone_transition_reason=segment_context["zone_transition_reason"],
                    overlays=list(overlays),
                )
            )
        clip_review_reasons = _dedupe(clip_review_reasons)
        crop_plan = _build_crop_plan(
            clip_start_ms=clip.clip_start_ms,
            render_segments=render_segments,
            geometry_by_tracklet=geometry_by_tracklet,
        )
        clips.append(
            RenderPlanClip(
                clip_id=clip.clip_id,
                clip_start_ms=clip.clip_start_ms,
                clip_end_ms=clip.clip_end_ms,
                caption_plan_ref="caption_plan.json",
                publish_metadata_ref="publish_metadata.json",
                caption_segment_ids=[segment.segment_id for segment in clip.segments],
                caption_zone=render_segments[0].caption_zone if render_segments else clip.default_zone,
                caption_preset_id=clip.preset_id,
                review_needed=bool(clip_review_reasons),
                review_reasons=clip_review_reasons,
                overlays=list(overlays),
                crop_plan=crop_plan,
                segments=render_segments,
            )
        )

    return RenderPlan(
        run_id=run_id,
        source_context_ref="source_context.json",
        caption_plan_ref="caption_plan.json",
        publish_metadata_ref="publish_metadata.json",
        clips=clips,
    ).model_dump(mode="json")


def _ass_time(value_ms: int) -> str:
    total_centiseconds = max(0, int(round(value_ms / 10.0)))
    hours, rem = divmod(total_centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, centiseconds = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return "&H00FFFFFF"
    red = color[0:2]
    green = color[2:4]
    blue = color[4:6]
    return f"&H00{blue}{green}{red}"


def _alignment(zone: str) -> int:
    if zone == "center_band":
        return 5
    if zone == "split_band":
        return 8
    return 2


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _highlight_text(text: str, active_index: int, preset: CaptionPreset) -> str:
    parts = text.split()
    if active_index < 0 or active_index >= len(parts):
        return _escape_ass_text(text)
    break_before_index = None
    if preset.max_lines >= 2 and len(parts) > 3:
        break_before_index = (len(parts) + 1) // 2

    rendered: list[str] = []
    for idx, part in enumerate(parts):
        if break_before_index is not None and idx == break_before_index:
            rendered.append(r"\N")
        escaped = _escape_ass_text(part)
        if idx == active_index:
            scale = str(int(round(preset.active_scale * 100)))
            rendered.append(
                "{\\c"
                + _ass_color(preset.active_fill_color)
                + "\\fscx"
                + scale
                + "\\fscy"
                + scale
                + "}"
                + escaped
                + "{\\c"
                + _ass_color(preset.fill_color)
                + "\\fscx100\\fscy100}"
            )
        else:
            rendered.append(escaped)
    return " ".join(rendered).replace(r" \N ", r"\N")


def compile_ass_subtitles(
    *,
    run_id: str,
    clip_id: str,
    caption_plan: CaptionPlan | dict[str, Any],
    publish_metadata: PublishMetadata | dict[str, Any],
    render_plan: RenderPlan | dict[str, Any],
) -> str:
    del publish_metadata
    parsed_caption_plan = _coerce_caption_plan(caption_plan)
    parsed_render_plan = render_plan if isinstance(render_plan, RenderPlan) else RenderPlan.model_validate(render_plan)
    clip = next(item for item in parsed_caption_plan.clips if item.clip_id == clip_id)
    render_clip = next((item for item in parsed_render_plan.clips if item.clip_id == clip_id), None)
    preset = load_caption_presets()[clip.preset_id]
    style_name = preset.preset_id
    margin_v = preset.safe_margin_bottom_px
    render_segments_by_caption_id = {
        caption_segment_id: segment
        for segment in (render_clip.segments if render_clip else [])
        for caption_segment_id in segment.caption_segment_ids
    }

    header = "\n".join(
        [
            "[Script Info]",
            f"Title: {run_id} {clip_id}",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            (
                "Style: "
                f"{style_name},{preset.font_family},{preset.font_size_px_1080x1920},"
                f"{_ass_color(preset.fill_color)},{_ass_color(preset.active_fill_color)},"
                f"{_ass_color(preset.stroke_color)},&H64000000,"
                f"{-1 if preset.font_weight >= 700 else 0},0,0,0,100,100,{preset.letter_spacing:.1f},0,1,"
                f"{preset.stroke_width:.1f},1,{_alignment(clip.default_zone)},120,120,{margin_v},1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            f"; caption preset: {style_name}",
        ]
    )

    dialogue_lines: list[str] = []
    for segment in clip.segments:
        start = _ass_time(segment.start_ms)
        end = _ass_time(segment.end_ms)
        render_segment = render_segments_by_caption_id.get(segment.segment_id)
        alignment_override = (
            "{\\an" + str(_alignment(render_segment.caption_zone)) + "}"
            if render_segment and render_segment.caption_zone != clip.default_zone
            else ""
        )
        if segment.highlight_mode == "word_highlight":
            word_count = max(1, len(re.findall(r"\S+", segment.text)))
            timings = segment.active_word_timings[:word_count]
            for idx, timing in enumerate(timings):
                timing_end_ms = (
                    timings[idx + 1].start_ms
                    if idx + 1 < len(timings)
                    else max(timing.end_ms, segment.end_ms)
                )
                dialogue_lines.append(
                    "Dialogue: 1,"
                    f"{_ass_time(timing.start_ms)},{_ass_time(timing_end_ms)},{style_name},,0,0,0,,"
                    f"{alignment_override}{_highlight_text(segment.text, idx, preset)}"
                )
            if not timings:
                dialogue_lines.append(
                    f"Dialogue: 1,{start},{end},{style_name},,0,0,0,,"
                    f"{alignment_override}{_escape_ass_text(segment.text)}"
                )
        else:
            dialogue_lines.append(
                f"Dialogue: 1,{start},{end},{style_name},,0,0,0,,"
                f"{alignment_override}{_escape_ass_text(segment.text)}"
            )

    return header + "\n" + "\n".join(dialogue_lines) + "\n"
