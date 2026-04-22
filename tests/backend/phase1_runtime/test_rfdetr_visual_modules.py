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
from types import SimpleNamespace


# ---------- visual_config tests ----------


class TestVisualPipelineConfig:
    def test_from_env_defaults(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()
        assert config.detector_model == "nano"
        assert config.detector_backend == "tensorrt_fp16"
        assert config.detector_batch_size == 16
        assert config.detection_threshold == pytest.approx(0.35)
        assert config.detector_resolution == 640
        assert config.tracker_backend == "bytetrack"
        assert config.frame_decode_backend == "gpu"
        assert config.use_fp16 is True
        assert config.use_tensorrt is True
        assert config.is_cuda_required is True
        assert config.tensorrt_engine_dir == "backend/outputs/tensorrt_engines"

    def test_from_env_custom(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_MODEL", "small")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "8")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.5")
        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_SHAPE", "640")

        config = VisualPipelineConfig.from_env()
        assert config.detector_model == "small"
        assert config.detector_backend == "tensorrt_fp16"
        assert config.detector_batch_size == 8
        assert config.detection_threshold == pytest.approx(0.5)
        assert config.detector_resolution == 640
        assert config.use_fp16 is True
        assert config.use_tensorrt is True
        assert config.is_cuda_required is True

    def test_from_env_rejects_cpu_decode_backend(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_DECODE", "cpu")
        with pytest.raises(ValueError, match="GPU decode is required"):
            VisualPipelineConfig.from_env()

    def test_person_class_id(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()
        assert config.PERSON_CLASS_ID == 0

    def test_tensorrt_engine_path_encodes_config(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            tensorrt_engine_dir="/tmp/engines",
        )
        expected = Path("/tmp/engines/rfdetr_nano_b4_r560_fp16.engine")
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
            DecodedFrame(
                frame_idx=i,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                source_width=2,
                source_height=2,
            )
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
            DecodedFrame(
                frame_idx=i,
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                source_width=1,
                source_height=1,
            )
            for i in range(6)
        ]
        batches = list(batch_frames(iter(frames), batch_size=3))
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)

    def test_decode_video_frames_rejects_non_gpu_backend(self, tmp_path: Path):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")
        with pytest.raises(ValueError, match="only 'gpu' is supported"):
            next(decode_video_frames(video_path=video, decode_backend="cpu"))

    def test_decode_video_frames_uses_nv12_then_rgb24_filter(self, monkeypatch, tmp_path: Path):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")

        monkeypatch.setattr(
            "backend.phase1_runtime.frame_decode._probe_frame_dimensions",
            lambda **_: (2, 2),
        )

        class _DummyStdout:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._served = False

            def read(self, _n: int) -> bytes:
                if self._served:
                    return b""
                self._served = True
                return self._payload

            def close(self) -> None:
                return None

        class _DummyStderr:
            def read(self) -> bytes:
                return b""

            def close(self) -> None:
                return None

        class _DummyProcess:
            def __init__(self, payload: bytes) -> None:
                self.stdout = _DummyStdout(payload)
                self.stderr = _DummyStderr()
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

        captured_cmd: list[str] = []

        def _fake_popen(cmd, stdout=None, stderr=None):  # noqa: ARG001
            captured_cmd.extend(cmd)
            return _DummyProcess(payload=(b"\x00" * (2 * 2 * 3)))

        monkeypatch.setattr("backend.phase1_runtime.frame_decode.subprocess.Popen", _fake_popen)

        frames = list(decode_video_frames(video_path=video, decode_backend="gpu"))
        assert len(frames) == 1
        vf_index = captured_cmd.index("-vf")
        assert captured_cmd[vf_index + 1] == "hwdownload,format=nv12,format=rgb24"

    def test_decode_video_frames_can_resize_on_gpu_and_preserve_source_dimensions(self, monkeypatch, tmp_path: Path):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")

        monkeypatch.setattr(
            "backend.phase1_runtime.frame_decode._probe_frame_dimensions",
            lambda **_: (4, 3),
        )

        class _DummyStdout:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._served = False

            def read(self, _n: int) -> bytes:
                if self._served:
                    return b""
                self._served = True
                return self._payload

            def close(self) -> None:
                return None

        class _DummyStderr:
            def read(self) -> bytes:
                return b""

            def close(self) -> None:
                return None

        class _DummyProcess:
            def __init__(self, payload: bytes) -> None:
                self.stdout = _DummyStdout(payload)
                self.stderr = _DummyStderr()
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

        captured_cmd: list[str] = []

        def _fake_popen(cmd, stdout=None, stderr=None):  # noqa: ARG001
            captured_cmd.extend(cmd)
            return _DummyProcess(payload=(b"\x00" * (2 * 2 * 3)))

        monkeypatch.setattr("backend.phase1_runtime.frame_decode.subprocess.Popen", _fake_popen)

        frames = list(
            decode_video_frames(
                video_path=video,
                decode_backend="gpu",
                target_width=2,
                target_height=2,
            )
        )

        assert len(frames) == 1
        assert frames[0].rgb.shape == (2, 2, 3)
        assert frames[0].source_width == 4
        assert frames[0].source_height == 3
        vf_index = captured_cmd.index("-vf")
        assert captured_cmd[vf_index + 1] == "scale_cuda=2:2,hwdownload,format=nv12,format=rgb24"


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

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
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
            detector_model="nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=2,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            tensorrt_engine_dir=str(tmp_path / "engines"),
        )

    def test_detect_batch_raises_if_not_loaded(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            det.detect_batch([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_preprocess_batch_shape_and_normalization(self, tmp_path, monkeypatch):
        import sys
        import types

        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)

        class FakeTensor:
            def __init__(self, array, *, device="cpu", dtype=None):
                self._array = np.asarray(array, dtype=dtype)
                self.device = device
                self.dtype = self._array.dtype

            @property
            def shape(self):
                return self._array.shape

            def to(self, device=None, dtype=None):
                return FakeTensor(
                    self._array.astype(dtype or self._array.dtype, copy=False),
                    device=device or self.device,
                    dtype=dtype or self._array.dtype,
                )

            def permute(self, *dims):
                return FakeTensor(np.transpose(self._array, dims), device=self.device)

            def contiguous(self):
                return self

            def div_(self, value):
                rhs = value._array if isinstance(value, FakeTensor) else value
                self._array = self._array / rhs
                self.dtype = self._array.dtype
                return self

            def sub_(self, value):
                rhs = value._array if isinstance(value, FakeTensor) else value
                self._array = self._array - rhs
                self.dtype = self._array.dtype
                return self

        class FakeTorch:
            float32 = np.float32
            nn = types.SimpleNamespace(
                functional=types.SimpleNamespace(
                    interpolate=lambda tensor, size, mode, align_corners: FakeTensor(
                        np.zeros((tensor.shape[0], tensor.shape[1], size[0], size[1]), dtype=tensor.dtype),
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                )
            )

            @staticmethod
            def from_numpy(arr):
                return FakeTensor(arr)

        monkeypatch.setitem(sys.modules, "torch", FakeTorch)
        frames = [
            np.full((480, 640, 3), 128, dtype=np.uint8),
            np.full((480, 640, 3), 64, dtype=np.uint8),
        ]
        batch = det._preprocess_batch(frames)
        assert batch.shape == (2, 3, 560, 560)
        assert batch.device == "cuda"
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

    def test_convert_onnx_to_engine_uses_skip_inference_cli(self, tmp_path, monkeypatch):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"onnx")
        engine_path = tmp_path / "engine.engine"
        captured: dict[str, list[str]] = {}

        def _fake_run(cmd, capture_output, text, check):  # noqa: ARG001
            captured["cmd"] = cmd
            engine_path.write_bytes(b"engine")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr("subprocess.run", _fake_run)

        det._convert_onnx_to_engine(onnx_path, engine_path)

        assert engine_path.exists()
        assert "--skipInference" in captured["cmd"]
        assert f"--saveEngine={engine_path}" in captured["cmd"]

    def test_unload_clears_state(self, tmp_path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)
        det._context = "fake"
        det._engine = "fake"
        det.unload()
        assert det._context is None
        assert det._engine is None

    def test_allocate_buffers_converts_trt_dims_to_tuple(self, tmp_path, monkeypatch):
        import sys
        import types

        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)

        class FakeDims:
            def __init__(self, *values):
                self._values = values

            def __iter__(self):
                return iter(self._values)

            def __contains__(self, item):
                return item in self._values

        class FakeEngine:
            num_io_tensors = 2

            def get_tensor_name(self, idx):
                return ["input", "boxes"][idx]

            def get_tensor_shape(self, name):
                if name == "input":
                    return FakeDims(-1, 3, 560, 560)
                return FakeDims(2, 300, 4)

            def get_tensor_dtype(self, name):
                return "float32"

            def get_tensor_mode(self, name):
                return fake_trt.TensorIOMode.INPUT if name == "input" else "output"

        class FakeContext:
            def __init__(self):
                self.input_shapes = {}
                self.tensor_addresses = {}

            def set_input_shape(self, name, shape):
                self.input_shapes[name] = shape

            def get_tensor_shape(self, name):
                return self.input_shapes.get(name, (2, 300, 4))

            def set_tensor_address(self, name, address):
                self.tensor_addresses[name] = address

        class FakeBuffer:
            def __init__(self, shape, dtype, device):
                self.shape = shape
                self.dtype = dtype
                self.device = device

            def data_ptr(self):
                return 12345

        class FakeTorch:
            float32 = "float32"

            @staticmethod
            def from_numpy(_arr):
                return types.SimpleNamespace(dtype=FakeTorch.float32)

            @staticmethod
            def empty(shape, dtype=None, device=None):
                if not isinstance(shape, tuple):
                    raise TypeError(f"expected tuple shape, got {type(shape).__name__}")
                return FakeBuffer(shape=shape, dtype=dtype, device=device)

        fake_trt = types.SimpleNamespace(
            TensorIOMode=types.SimpleNamespace(INPUT="input"),
            nptype=lambda dtype: np.float32,
        )

        monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

        det._engine = FakeEngine()
        det._context = FakeContext()

        det._allocate_buffers(FakeTorch)

        assert det._context.input_shapes["input"] == (2, 3, 560, 560)
        assert det._bindings["input"]["shape"] == (2, 3, 560, 560)
        assert det._bindings["boxes"]["shape"] == (2, 300, 4)

    def test_postprocess_decodes_class_logits_before_thresholding(self, tmp_path, monkeypatch):
        import sys
        import types

        from backend.phase1_runtime.rfdetr_detector import COCO_PERSON_CLASS_ID
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)

        class FakeDetections:
            def __init__(self, *, xyxy, confidence, class_id):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id

        monkeypatch.setitem(
            sys.modules,
            "supervision",
            types.SimpleNamespace(Detections=FakeDetections),
        )

        boxes = np.array(
            [
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                ]
            ],
            dtype=np.float32,
        )
        logits = np.full((1, 2, 91), -20.0, dtype=np.float32)
        logits[0, 0, COCO_PERSON_CLASS_ID] = 8.0
        logits[0, 1, 5] = 9.0

        detections = det._postprocess(
            boxes=boxes,
            scores=logits,
            labels=np.zeros_like(logits, dtype=np.int32),
            orig_sizes=[(560, 560)],
            batch_len=1,
        )

        assert len(detections) == 1
        assert detections[0].xyxy.shape == (1, 4)
        assert np.allclose(detections[0].xyxy[0], [10.0, 20.0, 30.0, 40.0])
        assert detections[0].class_id.tolist() == [COCO_PERSON_CLASS_ID]

    def test_detect_batch_uses_explicit_original_sizes_for_rescaling(self, tmp_path, monkeypatch):
        import sys
        import types

        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_tensorrt_config(tmp_path)
        det = TensorRTDetector(config)

        class _FakeTensor:
            def __init__(self, array):
                self._array = np.asarray(array)

            def to(self, *_args, **_kwargs):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._array

            def copy_(self, other):
                self._array = np.asarray(getattr(other, "_array", other))
                return self

        class _FakeTorch:
            @staticmethod
            def from_numpy(arr):
                return _FakeTensor(arr)

        det._context = types.SimpleNamespace(
            execute_async_v3=lambda *_args, **_kwargs: True,
        )
        det._stream = types.SimpleNamespace(
            cuda_stream=object(),
            synchronize=lambda: None,
        )
        det._bindings = {
            "input": {"is_input": True, "buffer": _FakeTensor(np.zeros((2, 3, 560, 560), dtype=np.float16))},
            "boxes": {"is_input": False, "buffer": _FakeTensor(np.zeros((2, 300, 4), dtype=np.float32))},
            "scores": {"is_input": False, "buffer": _FakeTensor(np.zeros((2, 300, 91), dtype=np.float32))},
        }

        monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
        monkeypatch.setattr(
            det,
            "_preprocess_batch",
            lambda batch: _FakeTensor(np.zeros((len(batch), 3, 560, 560), dtype=np.float16)),
        )

        captured: dict[str, object] = {}

        def _fake_postprocess(boxes, scores, labels, orig_sizes, batch_len):
            captured["orig_sizes"] = orig_sizes
            captured["batch_len"] = batch_len
            return ["ok"] * batch_len

        monkeypatch.setattr(det, "_postprocess", _fake_postprocess)

        out = det.detect_batch(
            [np.zeros((560, 560, 3), dtype=np.uint8)],
            orig_sizes=[(1080, 1920)],
        )

        assert out == ["ok"]
        assert captured["orig_sizes"] == [(1080, 1920)]
        assert captured["batch_len"] == 1


# ---------- detector factory tests ----------


class TestDetectorFactory:
    def test_factory_picks_pytorch_when_configured(self):
        from backend.phase1_runtime.rfdetr_detector import RFDETRPersonDetector
        from backend.phase1_runtime.visual import _make_detector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="nano",
            detector_backend="pytorch_cuda_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            tensorrt_engine_dir="/tmp/engines",
        )
        det = _make_detector(config)
        assert isinstance(det, RFDETRPersonDetector)

    def test_factory_picks_tensorrt_when_configured(self):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector
        from backend.phase1_runtime.visual import _make_detector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=560,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
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
        assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_nano_bytetrack"

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
