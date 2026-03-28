from __future__ import annotations

from collections.abc import Iterable, Mapping


def _overlap_ms(word: Mapping[str, object], span: Mapping[str, object]) -> int:
    word_start = int(word.get("start_time_ms", 0) or 0)
    word_end = int(word.get("end_time_ms", word_start) or word_start)
    span_start = int(span.get("start_time_ms", 0) or 0)
    span_end = int(span.get("end_time_ms", span_start) or span_start)
    return max(0, min(word_end, span_end) - max(word_start, span_start))


def project_words(
    *,
    words: Iterable[Mapping[str, object]] | None,
    span_assignments: Iterable[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    ordered_spans = sorted(
        [dict(span) for span in (span_assignments or [])],
        key=lambda span: (
            int(span.get("start_time_ms", 0) or 0),
            int(span.get("end_time_ms", 0) or 0),
        ),
    )
    projected_words: list[dict[str, object]] = []
    for word in words or []:
        projected = dict(word)
        overlapping_spans = [
            span for span in ordered_spans
            if _overlap_ms(projected, span) > 0
        ]
        overlapping_spans.sort(
            key=lambda span: (
                -_overlap_ms(projected, span),
                int(span.get("start_time_ms", 0) or 0),
                int(span.get("end_time_ms", 0) or 0),
            ),
        )

        speaker_track_ids: list[str] = []
        offscreen_audio_speaker_ids: list[str] = []
        decision_source = None
        require_hard_disambiguation = False

        if not overlapping_spans:
            projected.setdefault("speaker_track_ids", [projected["speaker_track_id"]] if projected.get("speaker_track_id") else [])
            projected.setdefault("offscreen_audio_speaker_ids", [])
            projected.setdefault("speaker_assignment_source", None)
            projected.setdefault("requires_hard_disambiguation", False)
            projected_words.append(projected)
            continue

        for span in overlapping_spans:
            for identity_id in span.get("assigned_visual_identity_ids", ()) or ():
                identity_id = str(identity_id).strip()
                if identity_id and identity_id not in speaker_track_ids:
                    speaker_track_ids.append(identity_id)
            for speaker_id in span.get("offscreen_audio_speaker_ids", ()) or ():
                speaker_id = str(speaker_id).strip()
                if speaker_id and speaker_id not in offscreen_audio_speaker_ids:
                    offscreen_audio_speaker_ids.append(speaker_id)
            if decision_source is None:
                decision_source = span.get("decision_source")
            require_hard_disambiguation = require_hard_disambiguation or bool(
                span.get("require_hard_disambiguation", False)
            )

        projected["speaker_track_ids"] = speaker_track_ids
        projected["offscreen_audio_speaker_ids"] = offscreen_audio_speaker_ids
        projected["speaker_assignment_source"] = decision_source
        projected["requires_hard_disambiguation"] = require_hard_disambiguation
        if len(speaker_track_ids) == 1 and not offscreen_audio_speaker_ids:
            projected["speaker_track_id"] = speaker_track_ids[0]
            projected["speaker_tag"] = speaker_track_ids[0]
        else:
            projected["speaker_track_id"] = None
            projected["speaker_tag"] = "unknown"
            projected["speaker_local_track_id"] = None
            projected["speaker_local_tag"] = "unknown"
        projected_words.append(projected)

    return projected_words
