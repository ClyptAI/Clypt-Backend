"""Shot-boundary continuity gating for raw tracks before ReID / global clustering."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from backend.pipeline.phase1.clustering import _feature_key_for_track, _norm_track_id


def _shot_index_for_time_ms(time_ms: int, shot_segments: list[dict[str, Any]]) -> int:
    """Match ``ClyptWorker._shot_index_for_time_ms`` (last segment inclusive end)."""
    t = int(time_ms)
    for i, seg in enumerate(shot_segments):
        s = int(seg.get("start_time_ms", 0))
        e = int(seg.get("end_time_ms", 0))
        if i == len(shot_segments) - 1:
            if s <= t <= e:
                return i
        elif s <= t < e:
            return i
    return 0


def _frame_time_ms(frame_idx: int, video_fps: float) -> int:
    fps = float(video_fps) if float(video_fps) > 1e-6 else 25.0
    return int(round((float(frame_idx) / fps) * 1000.0))


def _stable_local_track_id(norm_tid: str, part_index: int) -> int:
    """Deterministic positive int for ``(logical_track, split_part_index)``."""
    payload = f"{norm_tid}\0part={int(part_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return max(1, int(value))


def _new_track_id(local_id: int) -> str:
    return f"track_{int(local_id)}"


def split_tracks_at_shot_boundaries(
    tracks: list[dict[str, Any]],
    *,
    shot_timeline_ms: list[dict[str, Any]] | None,
    video_fps: float,
    track_identity_features: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]] | None, dict[str, int | bool]]:
    """Split raw detections when a logical track spans more than one editorial shot.

    Each contiguous run of frames in the same shot receives a new deterministic
    ``track_id`` / ``local_track_id``. Rows are shallow-copied; geometry and
    confidence fields are preserved.

    Returns:
        Updated tracks, updated ``track_identity_features`` (or ``None``), and metrics.
    """
    metrics: dict[str, int | bool] = {
        "camera_cut_gating_enabled": False,
        "camera_cut_split_source_tracks": 0,
        "camera_cut_split_emitted_segments": 0,
    }
    timeline = [dict(s) for s in (shot_timeline_ms or [])]
    if not tracks or len(timeline) <= 1:
        return tracks, track_identity_features, metrics

    metrics["camera_cut_gating_enabled"] = True

    by_tid: dict[str, list[dict[str, Any]]] = {}
    missing_tid_rows: list[dict[str, Any]] = []
    for row in tracks:
        tid_raw = str(row.get("track_id", "")).strip()
        if not tid_raw:
            missing_tid_rows.append(row)
            continue
        tid = _norm_track_id(tid_raw)
        by_tid.setdefault(tid, []).append(row)

    out_rows: list[dict[str, Any]] = []
    split_sources = 0
    emitted_segments = 0

    identity_out: dict[str, dict[str, Any]] | None = (
        dict(track_identity_features) if isinstance(track_identity_features, dict) else None
    )

    for norm_tid in sorted(by_tid.keys()):
        rows = by_tid[norm_tid]
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                int(r.get("frame_idx", -1)),
                int(r.get("chunk_idx", 0)),
                int(r.get("local_frame_idx", 0)),
            ),
        )

        runs: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        last_shot: int | None = None
        for r in rows_sorted:
            fi = int(r.get("frame_idx", -1))
            t_ms = _frame_time_ms(fi, video_fps) if fi >= 0 else 0
            shot_i = _shot_index_for_time_ms(t_ms, timeline)
            if not cur:
                cur = [r]
                last_shot = shot_i
            elif shot_i == last_shot:
                cur.append(r)
            else:
                runs.append(cur)
                cur = [r]
                last_shot = shot_i
        if cur:
            runs.append(cur)

        if len(runs) <= 1:
            for r in rows_sorted:
                out_rows.append(dict(r))
            continue

        split_sources += 1
        emitted_segments += len(runs)

        feat_key = _feature_key_for_track(identity_out or {}, norm_tid) if identity_out else None
        feat_template = copy.deepcopy(identity_out[feat_key]) if feat_key and identity_out else None
        if feat_key and identity_out and feat_key in identity_out:
            del identity_out[feat_key]

        for part_idx, run in enumerate(runs):
            local_id = _stable_local_track_id(norm_tid, part_idx)
            new_tid = _new_track_id(local_id)
            if feat_template is not None and identity_out is not None:
                identity_out[new_tid] = copy.deepcopy(feat_template)
            for r in run:
                nr = dict(r)
                nr["track_id"] = new_tid
                nr["local_track_id"] = int(local_id)
                out_rows.append(nr)

    for r in missing_tid_rows:
        out_rows.append(dict(r))

    if split_sources == 0:
        return tracks, track_identity_features, metrics

    metrics["camera_cut_split_source_tracks"] = int(split_sources)
    metrics["camera_cut_split_emitted_segments"] = int(emitted_segments)

    out_rows.sort(
        key=lambda r: (
            int(r.get("frame_idx", -1)),
            str(r.get("track_id", "")),
            int(r.get("chunk_idx", 0)),
        )
    )
    return out_rows, identity_out, metrics


__all__ = [
    "split_tracks_at_shot_boundaries",
    "_shot_index_for_time_ms",
    "_frame_time_ms",
]
