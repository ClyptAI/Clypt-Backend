from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from backend.pipeline.contracts import ShotTrackletIndex, TrackletGeometry

from .contracts import CaptionPlan, CaptionPreset, PublishMetadata, RenderPlan, RenderPlanClip, RenderPlanSegment
from .presets import load_caption_presets


_TARGET_ASPECT = 9.0 / 16.0
_PERSON_BOX_CROP_SCALE = 0.92
_CROP_KEYFRAME_MIN_INTERVAL_MS = 650
_CROP_SMOOTH_ALPHA = 0.32
_CROP_DEADZONE_PX = 4.0
_CROP_EMIT_DELTA_PX = 1.5


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
        ]
        pool = eligible or descriptors
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


def _representative_bbox(
    *,
    tracklet_id: str,
    absolute_start_ms: int,
    absolute_end_ms: int,
    geometry_by_tracklet: dict[str, list[dict[str, Any]]],
) -> list[float] | None:
    points = _geometry_points_for_interval(
        tracklet_id=tracklet_id,
        absolute_start_ms=absolute_start_ms,
        absolute_end_ms=absolute_end_ms,
        geometry_by_tracklet=geometry_by_tracklet,
    )
    if not points:
        return None
    return [float(value) for value in points[len(points) // 2]["bbox_xyxy"][:4]]


def _bbox_inside_9x16_size(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    if width / height >= _TARGET_ASPECT:
        crop_height = height * _PERSON_BOX_CROP_SCALE
        crop_width = crop_height * _TARGET_ASPECT
    else:
        crop_width = width * _PERSON_BOX_CROP_SCALE
        crop_height = crop_width / _TARGET_ASPECT
    return max(2.0, crop_width), max(2.0, crop_height)


def _point_xy(point: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = point.get(name)
    if not value or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _anchor_for_point(point: dict[str, Any]) -> tuple[float, float, str]:
    bbox = [float(value) for value in point["bbox_xyxy"][:4]]
    head = _point_xy(point, "head_center_xy")
    shoulders = _point_xy(point, "shoulder_center_xy")
    torso = _point_xy(point, "upper_torso_anchor_xy")
    if head and shoulders:
        return (
            (head[0] * 0.62) + (shoulders[0] * 0.38),
            (head[1] * 0.56) + (shoulders[1] * 0.44),
            "pose",
        )
    if head and torso:
        return (
            (head[0] * 0.64) + (torso[0] * 0.36),
            (head[1] * 0.58) + (torso[1] * 0.42),
            "pose",
        )
    if head:
        return head[0], head[1], "pose"
    if shoulders:
        return shoulders[0], shoulders[1], "pose"
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y1 + ((y2 - y1) * 0.34), "bbox_upper_third"


def _clamp_float(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(float(value), upper))


def _raw_crop_origin_inside_bbox(
    *,
    point: dict[str, Any],
    crop_width: float,
    crop_height: float,
) -> tuple[float, float, str]:
    bbox = [float(value) for value in point["bbox_xyxy"][:4]]
    x1, y1, x2, y2 = bbox
    anchor_x, anchor_y, anchor_source = _anchor_for_point(point)
    raw_x = anchor_x - (crop_width * 0.5)
    raw_y = anchor_y - (crop_height * 0.36)
    x = _clamp_float(raw_x, x1, x2 - crop_width)
    y = _clamp_float(raw_y, y1, y2 - crop_height)
    return x, y, anchor_source


def _sample_crop_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= 2:
        return points
    sampled = [points[0]]
    last_ms = int(points[0]["timestamp_ms"])
    for point in points[1:-1]:
        timestamp_ms = int(point["timestamp_ms"])
        if timestamp_ms - last_ms >= _CROP_KEYFRAME_MIN_INTERVAL_MS:
            sampled.append(point)
            last_ms = timestamp_ms
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    if len(sampled) == 2 and len(points) > 2:
        midpoint = points[len(points) // 2]
        if midpoint is not sampled[0] and midpoint is not sampled[-1]:
            sampled.insert(1, midpoint)
    return sampled


def _smooth_crop_keyframes(
    *,
    points: list[dict[str, Any]],
    clip_start_ms: int,
    crop_width: float,
    crop_height: float,
) -> list[dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    previous_x: float | None = None
    previous_y: float | None = None
    sampled_points = _sample_crop_points(points)
    for index, point in enumerate(sampled_points):
        raw_x, raw_y, anchor_source = _raw_crop_origin_inside_bbox(
            point=point,
            crop_width=crop_width,
            crop_height=crop_height,
        )
        if previous_x is None or previous_y is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            dx = raw_x - previous_x
            dy = raw_y - previous_y
            smooth_x = previous_x if abs(dx) < _CROP_DEADZONE_PX else previous_x + (dx * _CROP_SMOOTH_ALPHA)
            smooth_y = previous_y if abs(dy) < _CROP_DEADZONE_PX else previous_y + (dy * _CROP_SMOOTH_ALPHA)
        bbox = [float(value) for value in point["bbox_xyxy"][:4]]
        smooth_x = _clamp_float(smooth_x, bbox[0], bbox[2] - crop_width)
        smooth_y = _clamp_float(smooth_y, bbox[1], bbox[3] - crop_height)
        previous_x, previous_y = smooth_x, smooth_y
        keyframe = {
            "time_ms": max(0, int(point["timestamp_ms"]) - int(clip_start_ms)),
            "x": round(smooth_x, 3),
            "y": round(smooth_y, 3),
            "bbox_xyxy": bbox,
            "anchor_source": anchor_source,
        }
        if keyframes and index < len(sampled_points) - 1:
            previous_emitted = keyframes[-1]
            moved = max(
                abs(float(previous_emitted["x"]) - float(keyframe["x"])),
                abs(float(previous_emitted["y"]) - float(keyframe["y"])),
            )
            if moved < _CROP_EMIT_DELTA_PX and previous_emitted.get("anchor_source") == anchor_source:
                continue
        keyframes.append(keyframe)
    return keyframes


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

    run_points: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    size_candidates: list[tuple[float, float]] = []
    for run in runs:
        points = _geometry_points_for_interval(
            tracklet_id=str(run["tracklet_id"]),
            absolute_start_ms=int(run["start_ms"]),
            absolute_end_ms=int(run["end_ms"]),
            geometry_by_tracklet=geometry_by_tracklet,
        )
        if not points:
            continue
        run_points.append((run, points))
        size_candidates.extend(
            _bbox_inside_9x16_size([float(value) for value in point["bbox_xyxy"][:4]])
            for point in points
        )

    if not run_points or not size_candidates:
        return {}

    crop_width = min(width for width, _ in size_candidates)
    crop_height = crop_width / _TARGET_ASPECT
    height_limited = min(height for _, height in size_candidates)
    if crop_height > height_limited:
        crop_height = height_limited
        crop_width = crop_height * _TARGET_ASPECT

    keyframes: list[dict[str, Any]] = []
    crop_segments: list[dict[str, Any]] = []
    for run, points in run_points:
        crop_segments.append(
            {
                "start_ms": int(run["relative_start_ms"]),
                "end_ms": int(run["relative_end_ms"]),
                "tracklet_id": str(run["tracklet_id"]),
                "shot_id": run["shot_id"],
            }
        )
        keyframes.extend(
            _smooth_crop_keyframes(
                points=points,
                clip_start_ms=clip_start_ms,
                crop_width=crop_width,
                crop_height=crop_height,
            )
        )
    keyframes.sort(key=lambda keyframe: int(keyframe["time_ms"]))

    return {
        "mode": "tracklet_follow_9x16_smooth_inside_person",
        "crop_width": int(round(crop_width / 2.0) * 2),
        "crop_height": int(round(crop_height / 2.0) * 2),
        "tracklet_ids": _dedupe([str(run["tracklet_id"]) for run, _ in run_points]),
        "segments": crop_segments,
        "keyframes": keyframes,
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
