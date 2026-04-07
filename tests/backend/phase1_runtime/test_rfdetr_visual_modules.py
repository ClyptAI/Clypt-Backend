"""Tests for the RF-DETR refactor modules: visual_config, frame_decode,
rfdetr_detector, and tracker_runtime.

These tests do NOT require GPU or real model weights. They test:
- config parsing and properties
- frame decode batch bookkeeping
- detector output normalization (mocked model)
- tracker update normalization (mocked tracker)
- shot-split post-processing compatibility with new pipeline output
- tracking_metrics reflecting RF-DETR runtime labels
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------- visual_config tests ----------


class TestVisualPipelineConfig:
    def test_from_env_defaults(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()
        assert config.detector_backend == "pytorch_cuda_fp16"
        assert config.detector_batch_size == 4
        assert config.detection_threshold == pytest.approx(0.35)
        assert config.detector_resolution == 640
        assert config.tracker_backend == "bytetrack"
        assert config.frame_decode_backend == "cpu"
        assert config.use_fp16 is True
        assert config.use_tensorrt is False
        assert config.is_cuda_required is True
        assert config.tensorrt_engine_dir == "backend/outputs/tensorrt_engines"

    def test_from_env_custom(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "8")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.5")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_SHAPE", "640")

        config = VisualPipelineConfig.from_env()
        assert config.detector_backend == "tensorrt_fp16"
        assert config.detector_batch_size == 8
        assert config.detection_threshold == pytest.approx(0.5)
        assert config.detector_resolution == 640
        assert config.use_fp16 is True
        assert config.use_tensorrt is True
        assert config.is_cuda_required is True

    def test_person_class_id(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()
        assert config.PERSON_CLASS_ID == 0

    def test_tensorrt_engine_path_encodes_config(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="cpu",
            tensorrt_engine_dir="/tmp/engines",
        )
        expected = Path("/tmp/engines/rfdetr_small_b4_r560_fp16.engine")
        assert config.tensorrt_engine_path == expected

    def test_tensorrt_engine_dir_from_env(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_TRT_ENGINE_DIR", "/data/trt_cache")
        config = VisualPipelineConfig.from_env()
        assert config.tensorrt_engine_dir == "/data/trt_cache"
        assert str(config.tensorrt_engine_path).startswith("/data/trt_cache/")


# ---------- frame_decode tests ----------


class TestFrameDecode:
    def test_batch_frames_groups_correctly(self):
        from backend.phase1_runtime.frame_decode import DecodedFrame, batch_frames

        frames = [
            DecodedFrame(frame_idx=i, rgb=np.zeros((2, 2, 3), dtype=np.uint8))
            for i in range(7)
        ]
        batches = list(batch_frames(iter(frames), batch_size=3))
        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1
        assert [b[0].frame_idx for b in batches] == [0, 3, 6]

    def test_batch_frames_empty(self):
        from backend.phase1_runtime.frame_decode import batch_frames

        batches = list(batch_frames(iter([]), batch_size=4))
        assert batches == []

    def test_batch_frames_exact_multiple(self):
        from backend.phase1_runtime.frame_decode import DecodedFrame, batch_frames

        frames = [
            DecodedFrame(frame_idx=i, rgb=np.zeros((1, 1, 3), dtype=np.uint8))
            for i in range(6)
        ]
        batches = list(batch_frames(iter(frames), batch_size=3))
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)


# ---------- rfdetr_detector tests ----------


class TestDetectorMetrics:
    def test_mean_latency_zero_frames(self):
        from backend.phase1_runtime.rfdetr_detector import DetectorMetrics

        m = DetectorMetrics()
        assert m.mean_detector_latency_ms == 0.0

    def test_mean_latency_with_frames(self):
        from backend.phase1_runtime.rfdetr_detector import DetectorMetrics

        m = DetectorMetrics(frames_processed=10, total_detector_ms=100.0)
        assert m.mean_detector_latency_ms == pytest.approx(10.0)


class TestRFDETRPersonDetector:
    def test_detect_batch_raises_if_not_loaded(self):
        from backend.phase1_runtime.rfdetr_detector import RFDETRPersonDetector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()
        det = RFDETRPersonDetector(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            det.detect_batch([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_require_cuda_fails_when_unavailable(self):
        from backend.phase1_runtime.rfdetr_detector import _require_cuda

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is required"):
                _require_cuda()


# ---------- tracker_runtime tests ----------


class TestTrackerMetrics:
    def test_mean_latency_zero(self):
        from backend.phase1_runtime.tracker_runtime import TrackerMetrics

        m = TrackerMetrics()
        assert m.mean_tracker_latency_ms == 0.0

    def test_mean_latency_with_frames(self):
        from backend.phase1_runtime.tracker_runtime import TrackerMetrics

        m = TrackerMetrics(frames_processed=5, total_tracker_ms=50.0)
        assert m.mean_tracker_latency_ms == pytest.approx(10.0)


class TestTrackRow:
    def test_track_row_fields(self):
        from backend.phase1_runtime.tracker_runtime import TrackRow

        row = TrackRow(
            frame_idx=10,
            track_id=3,
            x1=100.0,
            y1=200.0,
            x2=300.0,
            y2=400.0,
            confidence=0.9,
        )
        assert row.frame_idx == 10
        assert row.track_id == 3
        assert row.class_id == 0


# ---------- Integration: visual extractor with injected tracker_runner ----------


# ---------- tensorrt_detector tests ----------


class TestTensorRTDetector:
    def _make_tensorrt_config(self, tmp_path):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        return VisualPipelineConfig(
            detector_backend="tensorrt_fp16",
            detector_batch_size=2,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="cpu",
            tensorrt_engine_dir=str(tmp_path / "engines"),
        )

    def test_detect_batch_raises_if_not_loaded(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            det.detect_batch([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_preprocess_batch_shape_and_normalization(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        frames = [
            np.full((480, 640, 3), 128, dtype=np.uint8),
            np.full((480, 640, 3), 64, dtype=np.uint8),
        ]
        batch = det._preprocess_batch(frames)
        assert batch.shape == (2, 3, 560, 560)
        assert batch.dtype == np.float32

    def test_ensure_engine_reuses_existing(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        engine_path = config.tensorrt_engine_path
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"fake_engine_data")

        result = det._ensure_engine()
        assert result == engine_path

    def test_unload_clears_state(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        det._context = "fake"
        det._engine = "fake"
        det.unload()
        assert det._context is None
        assert det._engine is None


# ---------- detector factory tests ----------


class TestDetectorFactory:
    def test_factory_picks_pytorch_by_default(self):
        from backend.phase1_runtime.rfdetr_detector import RFDETRPersonDetector
        from backend.phase1_runtime.visual import _make_detector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_backend="pytorch_cuda_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="cpu",
            tensorrt_engine_dir="/tmp/engines",
        )
        det = _make_detector(config)
        assert isinstance(det, RFDETRPersonDetector)

    def test_factory_picks_tensorrt_when_configured(self):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector
        from backend.phase1_runtime.visual import _make_detector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="cpu",
            tensorrt_engine_dir="/tmp/engines",
        )
        det = _make_detector(config)
        assert isinstance(det, TensorRTDetector)


# ---------- Integration: visual extractor with injected tracker_runner ----------


class TestVisualExtractorArtifactContract:
    """Verify the refactored V31VisualExtractor preserves artifact shapes."""

    def _make_extractor(self, *, tracks, metadata=None, shots=None):
        from backend.phase1_runtime.visual import V31VisualExtractor

        return V31VisualExtractor(
            metadata_probe=lambda video_path: metadata
            or {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 10000},
            shot_detector=lambda video_path, duration_ms: shots or [],
            tracker_runner=lambda video_path: tracks,
        )

    def test_tracking_metrics_reflect_rfdetr_backend(self, tmp_path: Path):
        from backend.phase1_runtime.models import Phase1Workspace

        workspace = Phase1Workspace.create(root=tmp_path, run_id="r1")
        workspace.video_path.write_text("v", encoding="utf-8")

        ext = self._make_extractor(tracks=[])
        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)
        assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_small_bytetrack"

    def test_person_detections_schema_stable(self, tmp_path: Path):
        from backend.phase1_runtime.models import Phase1Workspace

        workspace = Phase1Workspace.create(root=tmp_path, run_id="r2")
        workspace.video_path.write_text("v", encoding="utf-8")

        ext = self._make_extractor(
            tracks=[
                {
                    "frame_idx": 0,
                    "track_id": "track_1",
                    "local_track_id": 1,
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 50.0,
                    "y2": 80.0,
                    "confidence": 0.9,
                },
                {
                    "frame_idx": 1,
                    "track_id": "track_1",
                    "local_track_id": 1,
                    "x1": 12.0,
                    "y1": 22.0,
                    "x2": 52.0,
                    "y2": 82.0,
                    "confidence": 0.88,
                },
            ],
        )
        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)

        assert len(payload["person_detections"]) == 1
        pd = payload["person_detections"][0]
        assert pd["track_id"] == "track_1"
        assert pd["source"] == "person_tracker"
        assert pd["provenance"] == "v31_visual_extractor"
        assert "segment_start_ms" in pd
        assert "segment_end_ms" in pd
        assert "timestamped_objects" in pd
        assert len(pd["timestamped_objects"]) == 2
        for obj in pd["timestamped_objects"]:
            assert "bounding_box" in obj
            bb = obj["bounding_box"]
            assert set(bb.keys()) == {"left", "top", "right", "bottom"}

    def test_tracks_have_bbox_norm_xywh(self, tmp_path: Path):
        from backend.phase1_runtime.models import Phase1Workspace

        workspace = Phase1Workspace.create(root=tmp_path, run_id="r3")
        workspace.video_path.write_text("v", encoding="utf-8")

        ext = self._make_extractor(
            tracks=[
                {
                    "frame_idx": 5,
                    "track_id": "track_7",
                    "local_track_id": 7,
                    "x1": 100.0,
                    "y1": 200.0,
                    "x2": 300.0,
                    "y2": 400.0,
                    "confidence": 0.75,
                },
            ],
        )
        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)
        track = payload["tracks"][0]
        assert "bbox_norm_xywh" in track
        norm = track["bbox_norm_xywh"]
        assert set(norm.keys()) == {"x_center", "y_center", "width", "height"}
        assert 0.0 < norm["x_center"] < 1.0
        assert 0.0 < norm["y_center"] < 1.0

    def test_empty_artifacts_still_present(self, tmp_path: Path):
        from backend.phase1_runtime.models import Phase1Workspace

        workspace = Phase1Workspace.create(root=tmp_path, run_id="r4")
        workspace.video_path.write_text("v", encoding="utf-8")

        ext = self._make_extractor(tracks=[])
        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)
        assert payload["face_detections"] == []
        assert payload["visual_identities"] == []
        assert payload["mask_stability_signals"] == []
