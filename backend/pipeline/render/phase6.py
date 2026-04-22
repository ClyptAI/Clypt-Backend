from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from backend.pipeline.artifacts import V31RunPaths, save_json
from backend.pipeline.contracts import (
    CanonicalTimeline,
    CanonicalTurn,
    ClipCandidate,
    SemanticGraphNode,
    ShotTrackletIndex,
    TranscriptWord,
    TrackletGeometry,
)

from .captions.chunker import build_caption_plan
from .compiler import compile_ass_subtitles, compile_render_plan
from .contracts import CaptionPlan, PublishMetadata, RenderPlan, SourceContext
from .metadata.generator import build_publish_metadata


def _field(payload: Any, name: str, default: Any):
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _coerce_words(words: Any) -> list[TranscriptWord]:
    coerced: list[TranscriptWord] = []
    for word in words or []:
        try:
            coerced.append(TranscriptWord.model_validate(word, from_attributes=True))
        except ValidationError:
            continue
    return coerced


def _coerce_turns(turns: Any) -> list[CanonicalTurn]:
    coerced: list[CanonicalTurn] = []
    for turn in turns or []:
        try:
            coerced.append(CanonicalTurn.model_validate(turn, from_attributes=True))
        except ValidationError:
            continue
    return coerced


def _coerce_source_context(source_context: SourceContext | dict[str, Any] | None, *, source_url: str = "") -> SourceContext:
    defaults = {
        "source_url": source_url,
        "youtube_video_id": "unknown",
        "source_title": "Unknown Source",
        "source_description": "",
        "channel_id": "unknown",
        "channel_title": "Unknown Channel",
        "published_at": "",
        "default_audio_language": "und",
        "category_id": "",
        "tags": [],
        "thumbnails": {},
    }
    if isinstance(source_context, SourceContext):
        return source_context
    if source_context is None:
        return SourceContext(**defaults)
    merged = {**defaults, **source_context}
    if not merged["source_url"]:
        merged["source_url"] = source_url
    return SourceContext.model_validate(merged)


def _coerce_timeline(canonical_timeline: CanonicalTimeline | dict[str, Any]) -> CanonicalTimeline:
    if isinstance(canonical_timeline, CanonicalTimeline):
        return canonical_timeline
    try:
        return CanonicalTimeline.model_validate(canonical_timeline, from_attributes=True)
    except ValidationError:
        # Runtime resume tests often pass lightweight namespaces; keep Phase 6
        # packaging best-effort by salvaging any valid timeline fragments.
        return CanonicalTimeline(
            words=_coerce_words(_field(canonical_timeline, "words", [])),
            turns=_coerce_turns(_field(canonical_timeline, "turns", [])),
            source_video_url=_field(canonical_timeline, "source_video_url", None),
            video_gcs_uri=_field(canonical_timeline, "video_gcs_uri", None),
        )


def _candidate_excerpt(candidate: ClipCandidate, canonical_timeline: CanonicalTimeline) -> str:
    words = [
        word.text
        for word in canonical_timeline.words
        if word.end_ms > candidate.start_ms and word.start_ms < candidate.end_ms
    ]
    return " ".join(words).strip()


def plan_caption_artifacts(
    *,
    run_id: str,
    canonical_timeline: CanonicalTimeline | dict[str, Any],
    candidates: list[ClipCandidate],
    preset_id: str = "karaoke_focus",
) -> CaptionPlan:
    payload = build_caption_plan(
        run_id=run_id,
        canonical_timeline=canonical_timeline,
        finalists=[
            {
                "clip_id": candidate.clip_id,
                "start_ms": candidate.start_ms,
                "end_ms": candidate.end_ms,
                "preset_id": preset_id,
            }
            for candidate in candidates
            if candidate.clip_id is not None
        ],
    )
    return CaptionPlan.model_validate(payload)


def generate_publish_metadata(
    *,
    run_id: str,
    candidates: list[ClipCandidate],
    canonical_timeline: CanonicalTimeline | dict[str, Any],
    nodes: list[SemanticGraphNode],
    source_context: SourceContext | dict[str, Any],
) -> PublishMetadata:
    timeline = _coerce_timeline(canonical_timeline)
    context = _coerce_source_context(source_context)
    finalists = []
    for candidate in candidates:
        if candidate.clip_id is None:
            continue
        finalists.append(
            {
                "clip_id": candidate.clip_id,
                "transcript_excerpt": _candidate_excerpt(candidate, timeline),
                "rationale": candidate.rationale,
                "semantic_node_summaries": [
                    node.summary for node in nodes if node.node_id in set(candidate.node_ids)
                ],
                "external_attribution": candidate.external_attribution_json or {},
            }
        )
    payload = build_publish_metadata(
        run_id=run_id,
        source_context=context,
        finalists=finalists,
    )
    return PublishMetadata.model_validate(payload)


def run_phase_6(
    *,
    paths: V31RunPaths,
    canonical_timeline: CanonicalTimeline | dict[str, Any],
    shot_tracklet_index: ShotTrackletIndex | dict[str, Any],
    tracklet_geometry: TrackletGeometry | dict[str, Any],
    candidates: list[ClipCandidate],
    nodes: list[SemanticGraphNode],
    source_context: SourceContext | dict[str, Any] | None,
    participation_timeline: dict[str, Any] | None = None,
    camera_intent_timeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timeline = _coerce_timeline(canonical_timeline)
    context = _coerce_source_context(source_context, source_url=timeline.source_video_url or "")
    caption_plan = plan_caption_artifacts(
        run_id=paths.run_id,
        canonical_timeline=timeline,
        candidates=candidates,
        preset_id="karaoke_focus",
    )
    publish_metadata = generate_publish_metadata(
        run_id=paths.run_id,
        candidates=candidates,
        canonical_timeline=timeline,
        nodes=nodes,
        source_context=context,
    )
    render_plan = RenderPlan.model_validate(
        compile_render_plan(
            run_id=paths.run_id,
            clip_finalists=[
                candidate.model_dump(mode="json")
                for candidate in candidates
                if candidate.clip_id is not None
            ],
            caption_plan=caption_plan,
            publish_metadata=publish_metadata,
            source_context=context.model_dump(mode="json"),
            participation_timeline=participation_timeline,
            camera_intent_timeline=camera_intent_timeline,
            shot_tracklet_index=(
                shot_tracklet_index.model_dump(mode="json")
                if hasattr(shot_tracklet_index, "model_dump")
                else shot_tracklet_index
            ),
            tracklet_geometry=(
                tracklet_geometry.model_dump(mode="json")
                if hasattr(tracklet_geometry, "model_dump")
                else tracklet_geometry
            ),
        )
    )

    save_json(paths.source_context, context.model_dump(mode="json"))
    save_json(paths.caption_plan, caption_plan.model_dump(mode="json"))
    save_json(paths.publish_metadata, publish_metadata.model_dump(mode="json"))
    save_json(paths.render_plan, render_plan.model_dump(mode="json"))

    written_artifacts = {
        "source_context": str(paths.source_context),
        "caption_plan": str(paths.caption_plan),
        "publish_metadata": str(paths.publish_metadata),
        "render_plan": str(paths.render_plan),
    }
    for clip in caption_plan.clips:
        ass_text = compile_ass_subtitles(
            run_id=paths.run_id,
            clip_id=clip.clip_id,
            caption_plan=caption_plan,
            publish_metadata=publish_metadata,
            render_plan=render_plan,
        )
        ass_path = paths.captions_ass(clip.clip_id)
        ass_path.write_text(ass_text, encoding="utf-8")
        written_artifacts[f"captions_{clip.clip_id}.ass"] = str(ass_path)

    return {
        "source_context": context,
        "caption_plan": caption_plan,
        "publish_metadata": publish_metadata,
        "render_plan": render_plan,
        "artifact_paths": written_artifacts,
    }
