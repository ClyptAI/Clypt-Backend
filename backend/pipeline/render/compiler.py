from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from backend.pipeline.contracts import ShotTrackletIndex, TrackletGeometry

from .contracts import CaptionPlan, CaptionPreset, PublishMetadata, RenderPlan, RenderPlanClip, RenderPlanSegment
from .presets import load_caption_presets


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
    if intent == "follow":
        return "center_band"
    if intent == "reaction":
        return "lower_safe"
    if intent == "split":
        return "split_band"
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
    geometry_by_tracklet: dict[str, list[tuple[int, list[float]]]],
) -> bool:
    if not tracklet_id:
        return False
    points = geometry_by_tracklet.get(tracklet_id, [])
    for timestamp_ms, bbox_xyxy in points:
        if timestamp_ms < absolute_start_ms or timestamp_ms > absolute_end_ms:
            continue
        _, top, _, bottom = bbox_xyxy
        overlap_top = max(top, 620.0)
        overlap_bottom = min(bottom, 1300.0)
        if overlap_bottom > overlap_top:
            return True
    return False


def _render_segment_context(
    *,
    clip_id: str,
    clip_start_ms: int,
    segment: Any,
    previous_zone: str | None,
    previous_intent: str | None,
    camera_intent_timeline: dict[str, Any] | None,
    tracklet_to_shot: dict[str, str],
    geometry_by_tracklet: dict[str, list[tuple[int, list[float]]]],
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
    geometry_by_tracklet = {
        entry.tracklet_id: [(point.timestamp_ms, point.bbox_xyxy) for point in entry.points]
        for entry in parsed_tracklet_geometry.tracklets
    }

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
                geometry_by_tracklet=geometry_by_tracklet,
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
    active = _escape_ass_text(parts[active_index])
    parts[active_index] = (
        "{\\c"
        + _ass_color(preset.active_fill_color)
        + "\\fscx"
        + str(int(round(preset.active_scale * 100)))
        + "\\fscy"
        + str(int(round(preset.active_scale * 100)))
        + "}"
        + active
        + "{\\r}"
    )
    return " ".join(_escape_ass_text(part) if idx != active_index else part for idx, part in enumerate(parts))


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
        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{alignment_override}{_escape_ass_text(segment.text)}"
        )
        if segment.highlight_mode == "word_highlight":
            word_count = max(1, len(re.findall(r"\S+", segment.text)))
            for idx, timing in enumerate(segment.active_word_timings[:word_count]):
                dialogue_lines.append(
                    "Dialogue: 1,"
                    f"{_ass_time(timing.start_ms)},{_ass_time(timing.end_ms)},{style_name},,0,0,0,,"
                    f"{alignment_override}{_highlight_text(segment.text, idx, preset)}"
                )

    return header + "\n" + "\n".join(dialogue_lines) + "\n"
