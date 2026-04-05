from __future__ import annotations

import hashlib
from typing import Any


def frame_time_ms(frame_idx: int, *, video_fps: float) -> int:
    fps = float(video_fps) if float(video_fps) > 1e-6 else 25.0
    return int(round((float(frame_idx) / fps) * 1000.0))


def shot_index_for_time_ms(time_ms: int, *, shot_segments: list[dict[str, Any]]) -> int:
    timestamp_ms = int(time_ms)
    for index, segment in enumerate(shot_segments):
        start_ms = int(segment.get("start_time_ms", 0))
        end_ms = int(segment.get("end_time_ms", 0))
        if index == len(shot_segments) - 1:
            if start_ms <= timestamp_ms <= end_ms:
                return index
        elif start_ms <= timestamp_ms < end_ms:
            return index
    return 0


def _stable_local_track_id(normalized_track_id: str, *, part_index: int) -> int:
    digest = hashlib.sha256(
        f"{normalized_track_id}\0part={int(part_index)}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return max(1, int(value))


def split_tracks_at_shot_boundaries(
    tracks: list[dict[str, Any]],
    *,
    shot_timeline_ms: list[dict[str, Any]] | None,
    video_fps: float,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    metrics: dict[str, int | bool] = {
        "camera_cut_gating_enabled": False,
        "camera_cut_split_source_tracks": 0,
        "camera_cut_split_emitted_segments": 0,
    }
    timeline = [dict(segment) for segment in (shot_timeline_ms or [])]
    if not tracks or len(timeline) <= 1:
        return [dict(track) for track in tracks], metrics

    metrics["camera_cut_gating_enabled"] = True

    by_track_id: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        track_id = str(track.get("track_id", "")).strip()
        if not track_id:
            continue
        by_track_id.setdefault(track_id, []).append(track)

    output_rows: list[dict[str, Any]] = []
    split_source_tracks = 0
    emitted_segments = 0

    for track_id in sorted(by_track_id):
        rows = sorted(
            by_track_id[track_id],
            key=lambda row: (
                int(row.get("frame_idx", -1)),
                int(row.get("chunk_idx", 0)),
                int(row.get("local_frame_idx", 0)),
            ),
        )
        runs: list[list[dict[str, Any]]] = []
        current_run: list[dict[str, Any]] = []
        previous_shot_index: int | None = None
        for row in rows:
            frame_idx = int(row.get("frame_idx", -1))
            timestamp_ms = frame_time_ms(frame_idx, video_fps=video_fps) if frame_idx >= 0 else 0
            shot_index = shot_index_for_time_ms(timestamp_ms, shot_segments=timeline)
            if not current_run:
                current_run = [row]
                previous_shot_index = shot_index
            elif shot_index == previous_shot_index:
                current_run.append(row)
            else:
                runs.append(current_run)
                current_run = [row]
                previous_shot_index = shot_index
        if current_run:
            runs.append(current_run)

        if len(runs) <= 1:
            output_rows.extend(dict(row) for row in rows)
            continue

        split_source_tracks += 1
        emitted_segments += len(runs)
        for part_index, run in enumerate(runs):
            local_track_id = _stable_local_track_id(track_id, part_index=part_index)
            new_track_id = f"track_{local_track_id}"
            for row in run:
                new_row = dict(row)
                new_row["track_id"] = new_track_id
                new_row["local_track_id"] = local_track_id
                output_rows.append(new_row)

    if split_source_tracks == 0:
        return [dict(track) for track in tracks], metrics

    metrics["camera_cut_split_source_tracks"] = int(split_source_tracks)
    metrics["camera_cut_split_emitted_segments"] = int(emitted_segments)
    output_rows.sort(
        key=lambda row: (
            int(row.get("frame_idx", -1)),
            str(row.get("track_id", "")),
            int(row.get("chunk_idx", 0)),
        )
    )
    return output_rows, metrics


__all__ = [
    "frame_time_ms",
    "shot_index_for_time_ms",
    "split_tracks_at_shot_boundaries",
]
