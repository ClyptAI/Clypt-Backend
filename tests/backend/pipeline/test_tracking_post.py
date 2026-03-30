"""Tests for shot-boundary track splitting (``tracking_post``)."""

from __future__ import annotations

import copy

from backend.pipeline.phase1.tracking_post import split_tracks_at_shot_boundaries


def _row(
    frame_idx: int,
    track_id: str,
    *,
    confidence: float = 0.9,
    local_track_id: int = 1,
) -> dict:
    return {
        "frame_idx": frame_idx,
        "track_id": track_id,
        "local_track_id": local_track_id,
        "chunk_idx": 0,
        "local_frame_idx": frame_idx,
        "class_id": 0,
        "label": "person",
        "confidence": confidence,
        "x1": 0.1,
        "y1": 0.1,
        "x2": 0.5,
        "y2": 0.5,
        "geometry_type": "aabb",
        "x_center": 0.3,
        "y_center": 0.3,
        "width": 0.4,
        "height": 0.4,
        "source": "test",
        "bbox_norm_xywh": {
            "x_center": 0.3,
            "y_center": 0.3,
            "width": 0.4,
            "height": 0.4,
        },
    }


def test_single_shot_timeline_no_op():
    tracks = [_row(0, "track_1"), _row(1, "track_1")]
    out, feats, m = split_tracks_at_shot_boundaries(
        tracks,
        shot_timeline_ms=[{"start_time_ms": 0, "end_time_ms": 10_000}],
        video_fps=25.0,
        track_identity_features=None,
    )
    assert out is tracks
    assert feats is None
    assert m["camera_cut_gating_enabled"] is False
    assert m["camera_cut_split_source_tracks"] == 0


def test_track_not_crossing_cut_unchanged():
    """All frames fall in shot 0; editorial has a cut later — same track id."""
    # Two shots: [0, 500), [500, 2000] — frames 0–10 → times < 500ms stay in shot 0.
    tracks = [_row(i, "track_7") for i in range(0, 11)]
    timeline = [
        {"start_time_ms": 0, "end_time_ms": 500},
        {"start_time_ms": 500, "end_time_ms": 2000},
    ]
    out, feats, m = split_tracks_at_shot_boundaries(
        tracks,
        shot_timeline_ms=timeline,
        video_fps=25.0,
        track_identity_features=None,
    )
    assert out is tracks  # no split — same list reference
    assert all(str(r["track_id"]) == "track_7" for r in out)
    assert m["camera_cut_split_source_tracks"] == 0


def test_track_crossing_one_cut_splits_into_two_ids():
    fps = 25.0
    # Shot boundary at 500ms: frame 12 → 480ms (shot 0), frame 13 → 520ms (shot 1).
    tracks = [_row(i, "track_3") for i in range(12, 22)]
    timeline = [
        {"start_time_ms": 0, "end_time_ms": 500},
        {"start_time_ms": 500, "end_time_ms": 5000},
    ]
    out, _, m = split_tracks_at_shot_boundaries(
        tracks,
        shot_timeline_ms=timeline,
        video_fps=fps,
        track_identity_features=None,
    )
    assert m["camera_cut_gating_enabled"] is True
    assert m["camera_cut_split_source_tracks"] == 1
    assert m["camera_cut_split_emitted_segments"] == 2
    tids = {str(r["track_id"]) for r in out}
    assert len(tids) == 2
    low = [r for r in out if int(r["frame_idx"]) <= 12]
    high = [r for r in out if int(r["frame_idx"]) >= 13]
    assert len(low) >= 1 and len(high) >= 1
    assert len({r["track_id"] for r in low}) == 1
    assert len({r["track_id"] for r in high}) == 1
    assert low[0]["track_id"] != high[0]["track_id"]
    assert abs(float(low[0]["confidence"]) - 0.9) < 1e-6


def test_deterministic_ids_stable_across_runs():
    fps = 25.0
    tracks = [_row(i, "track_9") for i in range(12, 18)]
    timeline = [
        {"start_time_ms": 0, "end_time_ms": 500},
        {"start_time_ms": 500, "end_time_ms": 5000},
    ]
    a, _, _ = split_tracks_at_shot_boundaries(
        copy.deepcopy(tracks),
        shot_timeline_ms=timeline,
        video_fps=fps,
    )
    b, _, _ = split_tracks_at_shot_boundaries(
        copy.deepcopy(tracks),
        shot_timeline_ms=timeline,
        video_fps=fps,
    )
    assert [str(x["track_id"]) for x in sorted(a, key=lambda r: int(r["frame_idx"]))] == [
        str(x["track_id"]) for x in sorted(b, key=lambda r: int(r["frame_idx"]))
    ]
    assert [int(x["local_track_id"]) for x in sorted(a, key=lambda r: int(r["frame_idx"]))] == [
        int(x["local_track_id"]) for x in sorted(b, key=lambda r: int(r["frame_idx"]))
    ]


def test_identity_features_duplicated_per_segment():
    tracks = [_row(12, "track_2"), _row(13, "track_2")]
    timeline = [
        {"start_time_ms": 0, "end_time_ms": 500},
        {"start_time_ms": 500, "end_time_ms": 5000},
    ]
    feats_in = {"track_2": {"embedding": [0.1, 0.2], "embedding_count": 1}}
    out, feats_out, m = split_tracks_at_shot_boundaries(
        tracks,
        shot_timeline_ms=timeline,
        video_fps=25.0,
        track_identity_features=copy.deepcopy(feats_in),
    )
    assert m["camera_cut_split_source_tracks"] == 1
    assert feats_out is not None
    assert "track_2" not in feats_out
    keys = [k for k in feats_out if feats_out[k].get("embedding")]
    assert len(keys) == 2
    assert all(feats_out[k]["embedding"] == [0.1, 0.2] for k in keys)
