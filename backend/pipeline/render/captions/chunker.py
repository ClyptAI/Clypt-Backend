from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from pydantic import ValidationError

from backend.pipeline.contracts import CanonicalTimeline, CanonicalTurn, TranscriptWord

from ..contracts import (
    ActiveWordTiming,
    CaptionPlan,
    CaptionPlanClip,
    CaptionPreset,
    CaptionSegment,
)
from ..presets import load_caption_presets


_PUNCTUATION_RE = re.compile(r"[.!?]$")


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


def _coerce_preset_registry(
    preset_registry: dict[str, CaptionPreset] | dict[str, dict[str, Any]] | None,
) -> dict[str, CaptionPreset]:
    if preset_registry is None:
        return load_caption_presets()
    coerced: dict[str, CaptionPreset] = {}
    for preset_id, payload in preset_registry.items():
        coerced[preset_id] = (
            payload if isinstance(payload, CaptionPreset) else CaptionPreset.model_validate(payload)
        )
    return coerced


def _chunk_words(words: Iterable[TranscriptWord], *, max_words_per_segment: int) -> list[list[TranscriptWord]]:
    chunks: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    for word in words:
        current.append(word)
        if len(current) >= max_words_per_segment or _PUNCTUATION_RE.search(word.text):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _segment_from_words(
    *,
    clip_id: str,
    segment_index: int,
    clip_start_ms: int,
    clip_end_ms: int,
    words: list[TranscriptWord],
    turn: CanonicalTurn,
    preset: CaptionPreset,
) -> CaptionSegment:
    segment_start_ms = max(clip_start_ms, words[0].start_ms)
    segment_end_ms = min(clip_end_ms, words[-1].end_ms)
    return CaptionSegment(
        segment_id=f"{clip_id}_seg_{segment_index:03d}",
        start_ms=max(0, segment_start_ms - clip_start_ms),
        end_ms=max(0, segment_end_ms - clip_start_ms),
        text=" ".join(word.text for word in words).strip(),
        word_ids=[word.word_id for word in words],
        speaker_ids=[turn.speaker_id],
        turn_ids=[turn.turn_id],
        placement_zone=preset.default_zone,
        highlight_mode=preset.highlight_mode,
        review_needed=False,
        review_reason="",
        fallback_applied=False,
        zone_transition_reason="",
        active_word_timings=[
            ActiveWordTiming(
                word_id=word.word_id,
                start_ms=max(0, max(clip_start_ms, word.start_ms) - clip_start_ms),
                end_ms=max(0, min(clip_end_ms, word.end_ms) - clip_start_ms),
                text=word.text,
            )
            for word in words
        ],
    )


def build_caption_plan(
    *,
    run_id: str,
    canonical_timeline: CanonicalTimeline | dict[str, Any],
    finalists: list[dict[str, Any]],
    preset_registry: dict[str, CaptionPreset] | dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timeline = _coerce_timeline(canonical_timeline)
    presets = _coerce_preset_registry(preset_registry)
    words_by_id = {word.word_id: word for word in timeline.words}

    clips: list[CaptionPlanClip] = []
    for finalist in finalists:
        clip_id = str(finalist["clip_id"])
        clip_start_ms = int(finalist.get("clip_start_ms", finalist.get("start_ms", 0)))
        clip_end_ms = int(finalist.get("clip_end_ms", finalist.get("end_ms", clip_start_ms)))
        preset_id = str(finalist.get("preset_id", "karaoke_focus"))
        preset = presets[preset_id]

        segments: list[CaptionSegment] = []
        segment_index = 1
        for turn in timeline.turns:
            if turn.end_ms <= clip_start_ms or turn.start_ms >= clip_end_ms:
                continue
            missing_word_ids = [word_id for word_id in turn.word_ids if word_id not in words_by_id]
            if missing_word_ids:
                raise ValueError(
                    f"clip {clip_id} turn {turn.turn_id} references missing canonical words: "
                    f"{', '.join(missing_word_ids)}"
                )
            turn_words = [
                words_by_id[word_id]
                for word_id in turn.word_ids
                if word_id in words_by_id
                and words_by_id[word_id].end_ms > clip_start_ms
                and words_by_id[word_id].start_ms < clip_end_ms
            ]
            for chunk in _chunk_words(turn_words, max_words_per_segment=preset.max_words_per_segment):
                if not chunk:
                    continue
                segments.append(
                    _segment_from_words(
                        clip_id=clip_id,
                        segment_index=segment_index,
                        clip_start_ms=clip_start_ms,
                        clip_end_ms=clip_end_ms,
                        words=chunk,
                        turn=turn,
                        preset=preset,
                    )
                )
                segment_index += 1
        if not segments:
            raise ValueError(
                f"clip {clip_id} interval {clip_start_ms}-{clip_end_ms} cannot be resolved "
                "against canonical timeline"
            )

        clips.append(
            CaptionPlanClip(
                clip_id=clip_id,
                clip_start_ms=clip_start_ms,
                clip_end_ms=clip_end_ms,
                preset_id=preset_id,
                default_zone=preset.default_zone,
                segments=segments,
            )
        )

    return CaptionPlan(run_id=run_id, clips=clips).model_dump(mode="json")
