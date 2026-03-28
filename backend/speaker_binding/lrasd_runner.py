"""Span-job planning for LR-ASD execution."""

from __future__ import annotations

from collections.abc import Callable


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _merge_word_subspans(
    *,
    words: list[dict] | None,
    start_time_ms: int,
    end_time_ms: int,
    merge_gap_ms: int = 250,
) -> list[tuple[int, int, list[int]]]:
    spans: list[tuple[int, int, int]] = []
    for word_index, word in enumerate(words or []):
        word_start_ms = _as_int(word.get("start_time_ms"), default=0)
        word_end_ms = _as_int(word.get("end_time_ms"), default=word_start_ms)
        if word_end_ms < word_start_ms:
            word_start_ms, word_end_ms = word_end_ms, word_start_ms
        clipped_start_ms = max(int(start_time_ms), int(word_start_ms))
        clipped_end_ms = min(int(end_time_ms), int(word_end_ms))
        if clipped_end_ms <= clipped_start_ms:
            continue
        spans.append((int(clipped_start_ms), int(clipped_end_ms), int(word_index)))
    spans.sort()
    if not spans:
        return []

    merged: list[list[int | list[int]]] = [[spans[0][0], spans[0][1], [spans[0][2]]]]
    for span_start_ms, span_end_ms, word_index in spans[1:]:
        previous = merged[-1]
        if span_start_ms <= int(previous[1]) + int(merge_gap_ms):
            previous[1] = max(int(previous[1]), int(span_end_ms))
            previous[2] = [*list(previous[2]), int(word_index)]  # type: ignore[list-item]
        else:
            merged.append([int(span_start_ms), int(span_end_ms), [int(word_index)]])

    return [
        (int(span_start_ms), int(span_end_ms), [int(index) for index in word_indices])  # type: ignore[arg-type]
        for span_start_ms, span_end_ms, word_indices in merged
    ]


def build_lrasd_span_jobs(
    *,
    scheduled_hard_spans: list[dict] | None,
    words: list[dict] | None,
    rank_candidates_fn: Callable[[dict], list[dict]],
    merge_gap_ms: int = 250,
) -> tuple[list[dict], list[dict]]:
    jobs: list[dict] = []
    debug_rows: list[dict] = []

    for span in scheduled_hard_spans or []:
        if not isinstance(span, dict):
            continue
        context_start_time_ms = _as_int(
            span.get("context_start_time_ms", span.get("start_time_ms")),
            default=0,
        )
        context_end_time_ms = _as_int(
            span.get("context_end_time_ms", span.get("end_time_ms")),
            default=context_start_time_ms,
        )
        if context_end_time_ms < context_start_time_ms:
            context_start_time_ms, context_end_time_ms = context_end_time_ms, context_start_time_ms
        if context_end_time_ms <= context_start_time_ms:
            continue

        subspans = _merge_word_subspans(
            words=words,
            start_time_ms=context_start_time_ms,
            end_time_ms=context_end_time_ms,
            merge_gap_ms=merge_gap_ms,
        )
        if not subspans:
            continue

        span_id = str(span.get("span_id", "") or f"span-{len(debug_rows)}")
        span_selected_local_track_ids: set[str] = set()
        generated_job_count = 0
        for subspan_index, (subspan_start_ms, subspan_end_ms, word_indices) in enumerate(subspans):
            ranked_candidates = [
                dict(candidate)
                for candidate in list(
                    rank_candidates_fn(
                        {
                            **dict(span),
                            "start_time_ms": int(subspan_start_ms),
                            "end_time_ms": int(subspan_end_ms),
                        }
                    ) or []
                )
                if isinstance(candidate, dict) and str(candidate.get("local_track_id", "") or "")
            ]
            surviving_candidates = [
                dict(candidate)
                for candidate in ranked_candidates
                if bool(candidate.get("candidate_survives", False))
            ]
            if not surviving_candidates:
                continue
            selected_local_track_ids = [
                str(candidate["local_track_id"])
                for candidate in surviving_candidates
                if str(candidate.get("local_track_id", "") or "")
            ]
            span_selected_local_track_ids.update(selected_local_track_ids)
            jobs.append(
                {
                    "job_id": f"{span_id}:subspan-{subspan_index}",
                    "span_id": span_id,
                    "span_type": str(span.get("span_type", "single") or "single"),
                    "speaker_ids": [
                        str(speaker_id)
                        for speaker_id in list(span.get("speaker_ids") or [])
                        if str(speaker_id or "")
                    ],
                    "source_turn_ids": [
                        str(turn_id)
                        for turn_id in list(span.get("source_turn_ids") or [])
                        if str(turn_id or "")
                    ],
                    "overlap": bool(span.get("overlap", False)),
                    "context_start_time_ms": int(context_start_time_ms),
                    "context_end_time_ms": int(context_end_time_ms),
                    "start_time_ms": int(subspan_start_ms),
                    "end_time_ms": int(subspan_end_ms),
                    "selected_local_track_ids": list(selected_local_track_ids),
                    "candidate_rows": [dict(candidate) for candidate in surviving_candidates],
                    "word_indices": [int(index) for index in word_indices],
                }
            )
            generated_job_count += 1

        if generated_job_count:
            debug_rows.append(
                {
                    "span_id": span_id,
                    "job_count": int(generated_job_count),
                    "selected_local_track_ids": sorted(span_selected_local_track_ids),
                }
            )

    return jobs, debug_rows


__all__ = ["build_lrasd_span_jobs"]
