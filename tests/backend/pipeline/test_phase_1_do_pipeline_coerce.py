"""Tests for Phase 1 DO pipeline visual ledger coercion."""

from __future__ import annotations

from pathlib import Path

from backend.pipeline.phase_1_do_pipeline import (
    coerce_phase1_visual_ledger_list_fields,
    enrich_visual_ledger_for_downstream,
    validate_phase_handoff,
)


def test_coerce_phase1_visual_ledger_list_fields_fills_none_and_missing():
    vis = {
        "tracks": [{"frame_idx": 0, "track_id": "a", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9}],
        "object_tracking": None,
        "label_detections": None,
    }
    coerce_phase1_visual_ledger_list_fields(vis)
    assert vis["object_tracking"] == []
    assert vis["label_detections"] == []
    assert vis["face_detections"] == []
    assert vis["person_detections"] == []
    assert vis["shot_changes"] == []
    assert len(vis["tracks"]) == 1


def test_enrich_visual_ledger_coerces_null_visual_lists(tmp_path: Path):
    vpath = tmp_path / "v.mp4"
    vpath.write_bytes(b"not-a-real-video")
    visual = {
        "schema_version": "3.0.0",
        "task_type": "person_tracking",
        "coordinate_space": "absolute_original_frame_xyxy",
        "geometry_type": "aabb",
        "class_taxonomy": {"0": "person"},
        "tracking_metrics": {},
        "tracks": [],
        "face_detections": [],
        "person_detections": [],
        "object_tracking": None,
        "label_detections": None,
        "shot_changes": [],
    }
    audio = {"words": []}
    out = enrich_visual_ledger_for_downstream(visual, audio, str(vpath))
    assert out["object_tracking"] == []
    assert out["label_detections"] == []


def test_validate_phase_handoff_coerces_then_passes():
    visual = {
        "tracks": [{"frame_idx": 0, "track_id": "a", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.9}],
        "shot_changes": [],
        "person_detections": [],
        "face_detections": [],
        "object_tracking": None,
        "label_detections": None,
    }
    audio = {"words": []}
    validate_phase_handoff(visual, audio)
    assert visual["object_tracking"] == []
