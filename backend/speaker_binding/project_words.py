"""Project span-level speaker assignments back onto words."""

from __future__ import annotations


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ordered_unique(values: list[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        normalized = str(value or "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def build_speaker_assignment_spans(
    *,
    active_speakers_local: list[dict] | None,
    local_to_global_track_id: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    local_to_global_track_id = {
        str(local_track_id): str(global_track_id)
        for local_track_id, global_track_id in dict(local_to_global_track_id or {}).items()
        if str(local_track_id) and str(global_track_id)
    }

    local_spans: list[dict] = []
    global_spans: list[dict] = []
    for span in active_speakers_local or []:
        if not isinstance(span, dict):
            continue
        normalized_span = {
            "start_time_ms": _as_int(span.get("start_time_ms"), default=0),
            "end_time_ms": _as_int(span.get("end_time_ms", span.get("start_time_ms")), default=0),
            "audio_speaker_ids": _ordered_unique(span.get("audio_speaker_ids") or []),
            "visible_local_track_ids": _ordered_unique(span.get("visible_local_track_ids") or []),
            "visible_track_ids": _ordered_unique(span.get("visible_track_ids") or []),
            "offscreen_audio_speaker_ids": _ordered_unique(span.get("offscreen_audio_speaker_ids") or []),
            "overlap": bool(span.get("overlap", False)),
            "confidence": span.get("confidence"),
            "decision_source": str(span.get("decision_source") or "unknown"),
        }
        local_spans.append(normalized_span)

        global_visible_track_ids = list(normalized_span["visible_track_ids"])
        if not global_visible_track_ids:
            global_visible_track_ids = _ordered_unique(
                [
                    local_to_global_track_id.get(local_track_id, "")
                    for local_track_id in normalized_span["visible_local_track_ids"]
                ]
            )
        global_spans.append(
            {
                **normalized_span,
                "visible_track_ids": global_visible_track_ids,
            }
        )

    return local_spans, global_spans


def project_span_assignments_to_words(
    *,
    words: list[dict] | None,
    speaker_assignment_spans_local: list[dict] | None,
) -> list[dict]:
    normalized_spans: list[dict] = []
    for span in speaker_assignment_spans_local or []:
        if not isinstance(span, dict):
            continue
        start_time_ms = _as_int(span.get("start_time_ms"), default=0)
        end_time_ms = _as_int(span.get("end_time_ms", span.get("start_time_ms")), default=start_time_ms)
        if end_time_ms < start_time_ms:
            start_time_ms, end_time_ms = end_time_ms, start_time_ms
        if end_time_ms <= start_time_ms:
            continue
        normalized_spans.append(
            {
                **dict(span),
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "audio_speaker_ids": _ordered_unique(span.get("audio_speaker_ids") or []),
                "visible_local_track_ids": _ordered_unique(span.get("visible_local_track_ids") or []),
                "visible_track_ids": _ordered_unique(span.get("visible_track_ids") or []),
                "offscreen_audio_speaker_ids": _ordered_unique(span.get("offscreen_audio_speaker_ids") or []),
                "decision_source": str(span.get("decision_source") or "unknown"),
                "overlap": bool(span.get("overlap", False)),
            }
        )

    word_assignments: list[dict] = []
    for word in words or []:
        word_start_ms = _as_int(word.get("start_time_ms"), default=0)
        word_end_ms = _as_int(word.get("end_time_ms", word.get("start_time_ms")), default=word_start_ms)
        if word_end_ms < word_start_ms:
            word_start_ms, word_end_ms = word_end_ms, word_start_ms

        overlapping_spans: list[tuple[int, dict]] = []
        for span in normalized_spans:
            overlap_ms = min(word_end_ms, int(span["end_time_ms"])) - max(word_start_ms, int(span["start_time_ms"]))
            if overlap_ms <= 0:
                continue
            overlapping_spans.append((int(overlap_ms), span))
        overlapping_spans.sort(
            key=lambda item: (
                -int(item[0]),
                int(item[1]["start_time_ms"]),
                int(item[1]["end_time_ms"]),
            )
        )

        if not overlapping_spans:
            dominant_visible_local_track_id = (
                str(word.get("speaker_local_track_id") or "") or None
            )
            dominant_visible_track_id = str(word.get("speaker_track_id") or "") or None
            assignment = {
                "start_time_ms": int(word_start_ms),
                "end_time_ms": int(word_end_ms),
                "audio_speaker_ids": [],
                "visible_local_track_ids": [dominant_visible_local_track_id] if dominant_visible_local_track_id else [],
                "visible_track_ids": [dominant_visible_track_id] if dominant_visible_track_id else [],
                "offscreen_audio_speaker_ids": [],
                "dominant_visible_local_track_id": dominant_visible_local_track_id,
                "dominant_visible_track_id": dominant_visible_track_id,
                "decision_source": "legacy",
                "overlap": False,
            }
            word_assignments.append(assignment)
            continue

        dominant_span = overlapping_spans[0][1]
        dominant_overlap_ms = int(overlapping_spans[0][0])
        selected_spans = [dominant_span]
        if bool(dominant_span.get("overlap", False)):
            selected_spans = [span for _, span in overlapping_spans]
        else:
            materially_tied_spans = [
                span
                for overlap_ms, span in overlapping_spans[1:]
                if int(overlap_ms) == int(dominant_overlap_ms)
            ]
            if materially_tied_spans:
                selected_spans = [dominant_span, *materially_tied_spans]

        merged_audio_speaker_ids = _ordered_unique(
            [speaker_id for span in selected_spans for speaker_id in span["audio_speaker_ids"]]
        )
        merged_visible_local_track_ids = _ordered_unique(
            [track_id for span in selected_spans for track_id in span["visible_local_track_ids"]]
        )
        merged_visible_track_ids = _ordered_unique(
            [track_id for span in selected_spans for track_id in span["visible_track_ids"]]
        )
        merged_offscreen_audio_speaker_ids = _ordered_unique(
            [speaker_id for span in selected_spans for speaker_id in span["offscreen_audio_speaker_ids"]]
        )

        existing_global_track_id = str(word.get("speaker_track_id") or "") or None
        if not merged_visible_track_ids and len(merged_visible_local_track_ids) == 1 and existing_global_track_id:
            merged_visible_track_ids = [existing_global_track_id]

        if len(merged_visible_local_track_ids) == 1 and len(merged_visible_track_ids) <= 1:
            dominant_visible_local_track_id = merged_visible_local_track_ids[0]
            dominant_visible_track_id = (
                merged_visible_track_ids[0] if merged_visible_track_ids else None
            )
        else:
            dominant_visible_local_track_id = None
            dominant_visible_track_id = None

        word["speaker_local_track_id"] = dominant_visible_local_track_id
        word["speaker_local_tag"] = dominant_visible_local_track_id or "unknown"
        word["speaker_track_id"] = dominant_visible_track_id
        word["speaker_tag"] = dominant_visible_track_id or "unknown"

        word_assignments.append(
            {
                "start_time_ms": int(word_start_ms),
                "end_time_ms": int(word_end_ms),
                "audio_speaker_ids": merged_audio_speaker_ids,
                "visible_local_track_ids": merged_visible_local_track_ids,
                "visible_track_ids": merged_visible_track_ids,
                "offscreen_audio_speaker_ids": merged_offscreen_audio_speaker_ids,
                "dominant_visible_local_track_id": dominant_visible_local_track_id,
                "dominant_visible_track_id": dominant_visible_track_id,
                "decision_source": str(dominant_span["decision_source"]),
                "overlap": bool(
                    dominant_span.get("overlap", False)
                    or len(selected_spans) > 1
                    or len(merged_audio_speaker_ids) > 1
                    or len(merged_visible_local_track_ids) > 1
                    or bool(merged_offscreen_audio_speaker_ids)
                ),
            }
        )

    return word_assignments


__all__ = [
    "build_speaker_assignment_spans",
    "project_span_assignments_to_words",
]
