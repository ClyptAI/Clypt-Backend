from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..contracts import (
    ShotTrackletDescriptor,
    ShotTrackletIndex,
    TrackletGeometry,
    TrackletGeometryEntry,
    TrackletGeometryPoint,
)

from ..contracts import ShotTrackletIndex, TrackletGeometry


def build_tracklet_artifacts(*, phase1_visual: dict) -> tuple[ShotTrackletIndex, TrackletGeometry]:
    """Adapt Phase 1 visual ledgers into V3.1 shot/tracklet artifacts."""
    fps = float((phase1_visual.get("video_metadata") or {}).get("fps") or 30.0)
    shot_changes = list(phase1_visual.get("shot_changes") or [])
    tracks = list(phase1_visual.get("tracks") or [])

    def frame_to_ms(frame_idx: int) -> int:
        return int(round((float(frame_idx) / fps) * 1000.0))

    def find_shot_id(timestamp_ms: int) -> str:
        for idx, shot in enumerate(shot_changes, start=1):
            if int(shot["start_time_ms"]) <= timestamp_ms < int(shot["end_time_ms"]):
                return f"shot_{idx:04d}"
        if shot_changes and timestamp_ms == int(shot_changes[-1]["end_time_ms"]):
            return f"shot_{len(shot_changes):04d}"
        return "shot_0001"

    grouped_points: dict[tuple[str, str], list[TrackletGeometryPoint]] = defaultdict(list)
    for track in tracks:
        frame_idx = int(track["frame_idx"])
        timestamp_ms = frame_to_ms(frame_idx)
        shot_id = find_shot_id(timestamp_ms)
        track_id = str(track["track_id"])
        grouped_points[(shot_id, track_id)].append(
            TrackletGeometryPoint(
                frame_index=frame_idx,
                timestamp_ms=timestamp_ms,
                bbox_xyxy=[
                    float(track["x1"]),
                    float(track["y1"]),
                    float(track["x2"]),
                    float(track["y2"]),
                ],
            )
        )

    descriptors: list[ShotTrackletDescriptor] = []
    geometry_entries: list[TrackletGeometryEntry] = []
    for (shot_id, track_id), points in sorted(grouped_points.items(), key=lambda item: (item[0][0], item[0][1])):
        ordered_points = sorted(points, key=lambda point: point.frame_index)
        tracklet_id = f"{shot_id}:{track_id}"
        descriptors.append(
            ShotTrackletDescriptor(
                tracklet_id=tracklet_id,
                shot_id=shot_id,
                start_ms=ordered_points[0].timestamp_ms,
                end_ms=ordered_points[-1].timestamp_ms,
                representative_thumbnail_uris=[],
            )
        )
        geometry_entries.append(
            TrackletGeometryEntry(
                tracklet_id=tracklet_id,
                shot_id=shot_id,
                points=ordered_points,
            )
        )

    return ShotTrackletIndex(tracklets=descriptors), TrackletGeometry(tracklets=geometry_entries)
