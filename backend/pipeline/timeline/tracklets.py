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
from .payload_utils import payload_to_dict


def build_tracklet_artifacts(*, phase1_visual: Any) -> tuple[ShotTrackletIndex, TrackletGeometry]:
    """Adapt Phase 1 visual ledgers into V3.1 shot/tracklet artifacts."""
    payload = payload_to_dict(phase1_visual)
    fps = float((payload.get("video_metadata") or {}).get("fps") or 30.0)
    shot_changes = list(payload.get("shot_changes") or [])
    tracks = list(payload.get("tracks") or [])

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
    grouped_quality: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_eligibility: dict[tuple[str, str], bool] = {}
    pose_anchors_by_track_frame: dict[tuple[str, int], dict[str, Any]] = {}
    for track in tracks:
        frame_idx = int(track["frame_idx"])
        timestamp_ms = frame_to_ms(frame_idx)
        shot_id = find_shot_id(timestamp_ms)
        track_id = str(track["track_id"])
        key = (shot_id, track_id)
        subject_quality = dict(track.get("subject_quality") or {})
        if subject_quality:
            grouped_quality[key] = subject_quality
            for anchor in subject_quality.get("pose_anchor_points") or []:
                try:
                    pose_anchors_by_track_frame[
                        (track_id, int(anchor.get("frame_idx")))
                    ] = dict(anchor)
                except (TypeError, ValueError):
                    continue
        anchor = pose_anchors_by_track_frame.get((track_id, frame_idx), {})
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
                mask_rle=track.get("mask_rle") if isinstance(track.get("mask_rle"), dict) else None,
                head_center_xy=anchor.get("head_center_xy"),
                shoulder_center_xy=anchor.get("shoulder_center_xy"),
                upper_torso_anchor_xy=anchor.get("upper_torso_anchor_xy"),
            )
        )
        if "auto_follow_eligible" in track:
            grouped_eligibility[key] = bool(track.get("auto_follow_eligible"))

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
                auto_follow_eligible=grouped_eligibility.get((shot_id, track_id), True),
                subject_quality=grouped_quality.get((shot_id, track_id), {}),
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
