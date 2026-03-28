"""Span-job planning for LR-ASD execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from .metrics import finalize_lrasd_pipeline_metrics, new_lrasd_pipeline_metrics


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


@dataclass(slots=True)
class _PreparedItem:
    seq: int
    value: object
    prep_wallclock_s: float


class LrasdPrepPipeline:
    """Bounded prep queue for LR-ASD subchunk preparation.

    The pipeline parallelizes preparation work while keeping emitted items in
    submission order so downstream inference remains deterministic.
    """

    def __init__(
        self,
        *,
        prepare_fn: Callable[[dict], object],
        prep_workers: int | None = None,
        queue_size: int | None = None,
        infer_workers: int | None = None,
        metrics: dict | None = None,
    ) -> None:
        self._prepare_fn = prepare_fn
        self.prep_workers = max(1, _as_int(prep_workers, default=_as_int(os.getenv("CLYPT_LRASD_PREP_WORKERS", "4"), default=4)))
        self.queue_size = max(1, _as_int(queue_size, default=_as_int(os.getenv("CLYPT_LRASD_PREP_QUEUE_SIZE", "128"), default=128)))
        self.infer_workers = max(1, _as_int(infer_workers, default=_as_int(os.getenv("CLYPT_LRASD_INFER_WORKERS", "1"), default=1)))
        self.metrics = dict(metrics or new_lrasd_pipeline_metrics())
        self.metrics.setdefault("lrasd_infer_workers", int(self.infer_workers))
        self._executor = ThreadPoolExecutor(max_workers=self.prep_workers)
        self._lock = threading.Lock()
        self._futures: dict[int, Future[_PreparedItem]] = {}
        self._next_submit_seq = 0
        self._next_emit_seq = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _prepare_item(self, seq: int, payload: dict) -> _PreparedItem:
        started_at = time.perf_counter()
        value = self._prepare_fn(payload)
        return _PreparedItem(
            seq=int(seq),
            value=value,
            prep_wallclock_s=float(time.perf_counter() - started_at),
        )

    def submit(self, payload: dict) -> list[object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("LR-ASD prep pipeline has been closed")
            seq = self._next_submit_seq
            self._next_submit_seq += 1
            future = self._executor.submit(self._prepare_item, seq, dict(payload))
            self._futures[seq] = future
            current_depth = len(self._futures)
            self.metrics["lrasd_prep_queue_depth"] = int(current_depth)
            self.metrics["lrasd_prep_queue_depth_max"] = max(
                int(self.metrics.get("lrasd_prep_queue_depth_max", 0) or 0),
                int(current_depth),
            )
            self.metrics["lrasd_prep_jobs_submitted"] = int(
                self.metrics.get("lrasd_prep_jobs_submitted", 0) or 0
            ) + 1
        ready = self._drain_ready(block=False)
        if len(self._futures) >= self.queue_size:
            ready.extend(self._drain_ready(block=True))
        return ready

    def drain(self) -> list[object]:
        return self._drain_ready(block=True)

    def _drain_ready(self, *, block: bool) -> list[object]:
        ready_values: list[object] = []
        while True:
            with self._lock:
                future = self._futures.get(self._next_emit_seq)
                if future is None:
                    break
            if not future.done():
                if not block:
                    break
                future.result()
            prepared_item = future.result()
            with self._lock:
                current = self._futures.get(prepared_item.seq)
                if current is not future:
                    continue
                del self._futures[prepared_item.seq]
                self._next_emit_seq += 1
                remaining_depth = len(self._futures)
                self.metrics["lrasd_prep_queue_depth"] = int(remaining_depth)
                self.metrics["lrasd_prep_queue_depth_max"] = max(
                    int(self.metrics.get("lrasd_prep_queue_depth_max", 0) or 0),
                    int(remaining_depth),
                )
                self.metrics["lrasd_prep_jobs_completed"] = int(
                    self.metrics.get("lrasd_prep_jobs_completed", 0) or 0
                ) + 1
                self.metrics["lrasd_prep_wallclock_s"] = float(
                    self.metrics.get("lrasd_prep_wallclock_s", 0.0) or 0.0
                ) + float(prepared_item.prep_wallclock_s)
                self.metrics["lrasd_spans_processed"] = int(
                    self.metrics.get("lrasd_spans_processed", 0) or 0
                ) + 1
            ready_values.append(prepared_item.value)
        self.metrics = finalize_lrasd_pipeline_metrics(self.metrics)
        return ready_values


__all__ = ["LrasdPrepPipeline", "build_lrasd_span_jobs"]
