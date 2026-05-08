from __future__ import annotations

from backend.phase1_runtime.visual_warmup import (
    DEFAULT_VISUAL_WARMUP_SPEC,
    load_visual_warmup_spec_from_env,
)


def test_visual_warmup_defaults(monkeypatch) -> None:
    monkeypatch.delenv("CLYPT_PHASE1_VISUAL_WARMUP_ASSET_ID", raising=False)
    monkeypatch.delenv("CLYPT_PHASE1_VISUAL_WARMUP_GCS_URI", raising=False)

    spec = load_visual_warmup_spec_from_env()

    assert spec == DEFAULT_VISUAL_WARMUP_SPEC


def test_visual_warmup_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_ASSET_ID", "custom")
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_GCS_URI", "gs://bucket/custom.mp4")
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_CLIP_START_MS", "1000")
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_CLIP_END_MS", "2500")
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_MIN_EMITTED_TRACK_ROWS", "2")
    monkeypatch.setenv("CLYPT_PHASE1_VISUAL_WARMUP_MIN_POSE_VALIDATED_TRACKLETS", "3")

    spec = load_visual_warmup_spec_from_env()

    assert spec.asset_id == "custom"
    assert spec.warmup_video_gcs_uri == "gs://bucket/custom.mp4"
    assert spec.clip_start_ms == 1000
    assert spec.clip_end_ms == 2500
    assert spec.min_emitted_track_rows == 2
    assert spec.min_pose_validated_tracklets == 3
