"""Pyannote scheduling primitives for speaker binding."""

from __future__ import annotations

import os
from typing import Iterable

from .types import DiarizedSpan, ScheduledSpan


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def scheduler_config_from_env() -> dict[str, int]:
    """Return env-driven tuning knobs for pyannote scheduling."""
    return {
        "same_speaker_gap_ms": max(
            0,
            _as_int(os.getenv("CLYPT_BINDING_SCHEDULER_SAME_SPEAKER_GAP_MS"), default=80),
        ),
        "boundary_pad_ms": max(
            0,
            _as_int(os.getenv("CLYPT_BINDING_SCHEDULER_BOUNDARY_PAD_MS"), default=40),
        ),
    }


def normalize_diarization_turns(turns: Iterable[dict] | None) -> list[DiarizedSpan]:
    normalized_turns: list[DiarizedSpan] = []
    for index, raw_turn in enumerate(turns or []):
        if not isinstance(raw_turn, dict):
            continue
        start_time_ms = _as_int(raw_turn.get("start_time_ms"), default=0)
        end_time_ms = _as_int(raw_turn.get("end_time_ms"), default=start_time_ms)
        if end_time_ms < start_time_ms:
            start_time_ms, end_time_ms = end_time_ms, start_time_ms
        if end_time_ms <= start_time_ms:
            continue
        speaker_id = str(raw_turn.get("speaker_id", "") or "")
        if not speaker_id:
            continue
        overlap = _as_bool(raw_turn.get("overlap"), default=False)
        exclusive_raw = raw_turn.get("exclusive")
        exclusive = (not overlap) if exclusive_raw is None else _as_bool(exclusive_raw, default=not overlap)
        normalized_turn: DiarizedSpan = {
            **dict(raw_turn),
            "turn_id": str(raw_turn.get("turn_id") or f"turn-{len(normalized_turns)}"),
            "speaker_id": speaker_id,
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "exclusive": bool(exclusive),
            "overlap": bool(overlap or not exclusive),
        }
        normalized_turns.append(normalized_turn)
    normalized_turns.sort(
        key=lambda turn: (
            int(turn["start_time_ms"]),
            int(turn["end_time_ms"]),
            str(turn["speaker_id"]),
            str(turn["turn_id"]),
        )
    )
    return normalized_turns


def merge_same_speaker_micro_gaps(
    turns: Iterable[DiarizedSpan] | None,
    *,
    same_speaker_gap_ms: int,
) -> list[DiarizedSpan]:
    merged_turns: list[DiarizedSpan] = []
    for turn in turns or []:
        if not merged_turns:
            merged_turns.append(dict(turn))
            continue
        previous_turn = merged_turns[-1]
        gap_ms = int(turn["start_time_ms"]) - int(previous_turn["end_time_ms"])
        if (
            str(previous_turn.get("speaker_id") or "") == str(turn.get("speaker_id") or "")
            and not bool(previous_turn.get("overlap", False))
            and not bool(turn.get("overlap", False))
            and previous_turn.get("exclusive") is not False
            and turn.get("exclusive") is not False
            and gap_ms >= 0
            and gap_ms <= same_speaker_gap_ms
        ):
            previous_turn["end_time_ms"] = max(
                int(previous_turn["end_time_ms"]),
                int(turn["end_time_ms"]),
            )
            source_turn_ids = list(previous_turn.get("source_turn_ids") or [previous_turn["turn_id"]])
            source_turn_ids.extend(list(turn.get("source_turn_ids") or [turn["turn_id"]]))
            previous_turn["source_turn_ids"] = source_turn_ids
            continue
        merged_turns.append(dict(turn))
    return merged_turns


def schedule_diarized_spans(
    turns: Iterable[dict] | None,
    *,
    same_speaker_gap_ms: int | None = None,
    boundary_pad_ms: int | None = None,
) -> list[ScheduledSpan]:
    config = scheduler_config_from_env()
    if same_speaker_gap_ms is None:
        same_speaker_gap_ms = int(config["same_speaker_gap_ms"])
    if boundary_pad_ms is None:
        boundary_pad_ms = int(config["boundary_pad_ms"])

    normalized_turns = normalize_diarization_turns(turns)
    merged_turns = merge_same_speaker_micro_gaps(
        normalized_turns,
        same_speaker_gap_ms=max(0, int(same_speaker_gap_ms)),
    )
    if not merged_turns:
        return []

    boundaries = {
        boundary
        for turn in merged_turns
        for boundary in (int(turn["start_time_ms"]), int(turn["end_time_ms"]))
    }
    scheduled_spans: list[ScheduledSpan] = []
    ordered_boundaries = sorted(boundaries)
    for start_time_ms, end_time_ms in zip(ordered_boundaries, ordered_boundaries[1:]):
        if end_time_ms <= start_time_ms:
            continue
        active_turns = [
            turn
            for turn in merged_turns
            if int(turn["start_time_ms"]) < end_time_ms and int(turn["end_time_ms"]) > start_time_ms
        ]
        if not active_turns:
            continue

        speaker_ids: list[str] = []
        source_turn_ids: list[str] = []
        for turn in active_turns:
            speaker_id = str(turn.get("speaker_id") or "")
            if speaker_id and speaker_id not in speaker_ids:
                speaker_ids.append(speaker_id)
            for turn_id in list(turn.get("source_turn_ids") or [turn.get("turn_id")]):
                turn_id_str = str(turn_id or "")
                if turn_id_str and turn_id_str not in source_turn_ids:
                    source_turn_ids.append(turn_id_str)

        is_overlap = (
            len(speaker_ids) > 1
            or any(bool(turn.get("overlap", False)) for turn in active_turns)
            or any(turn.get("exclusive") is False for turn in active_turns)
        )
        scheduled_spans.append(
            {
                "span_id": f"scheduled-{len(scheduled_spans)}",
                "span_type": "overlap" if is_overlap else "single",
                "speaker_ids": speaker_ids,
                "exclusive": not is_overlap,
                "overlap": bool(is_overlap),
                "start_time_ms": max(0, int(start_time_ms) - max(0, int(boundary_pad_ms))),
                "end_time_ms": int(end_time_ms) + max(0, int(boundary_pad_ms)),
                "context_start_time_ms": int(start_time_ms),
                "context_end_time_ms": int(end_time_ms),
                "source_turn_ids": source_turn_ids,
            }
        )

    return scheduled_spans


__all__ = [
    "merge_same_speaker_micro_gaps",
    "normalize_diarization_turns",
    "schedule_diarized_spans",
    "scheduler_config_from_env",
]
