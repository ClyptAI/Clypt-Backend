from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class TestVisualPipelineConfig:
    def test_from_env_defaults_modal_l40s_tensorrt(self):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig.from_env()

        assert config.detector_model == "nano"
        assert config.detector_backend == "tensorrt_fp16"
        assert config.detector_batch_size == 16
        assert config.detection_threshold == pytest.approx(0.35)
        assert config.detector_resolution == 640
        assert config.tracker_backend == "bytetrack"
        assert config.frame_decode_backend == "gpu"
        assert config.gpu_decode_backend == "nvdec"
        assert config.use_fp16 is True
        assert config.use_tensorrt is True
        assert config.is_gpu_required is True

    def test_from_env_rejects_cpu_decode_backend(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_DECODE", "cpu")

        with pytest.raises(ValueError, match="GPU decode is required"):
            VisualPipelineConfig.from_env()

    def test_from_env_rejects_rocm_backend(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_BACKEND", "rfdetr_rocm_fp16")

        with pytest.raises(ValueError, match="TensorRT is required"):
            VisualPipelineConfig.from_env()

    def test_from_env_rejects_vaapi_decode_backend(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND", "vaapi")

        with pytest.raises(ValueError, match="'nvdec' or 'cuda'"):
            VisualPipelineConfig.from_env()

    def test_tensorrt_engine_path_uses_artifact_dir(self, tmp_path: Path):
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
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path),
        )

        assert config.tensorrt_engine_path == tmp_path / "rfdetr_nano_b4_r560_fp16.engine"


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

        assert [len(batch) for batch in batches] == [3, 3, 1]
        assert [batch[0].frame_idx for batch in batches] == [0, 3, 6]

    def test_decode_video_frames_rejects_non_gpu_backend(self, tmp_path: Path):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")

        with pytest.raises(ValueError, match="only 'gpu' is supported"):
            next(decode_video_frames(video_path=video, decode_backend="cpu"))

    def test_decode_video_frames_rejects_vaapi_backend(self, tmp_path: Path):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")

        with pytest.raises(ValueError, match="expected nvdec or cuda"):
            next(
                decode_video_frames(
                    video_path=video,
                    decode_backend="gpu",
                    gpu_decode_backend="vaapi",
                )
            )

    def test_decode_video_frames_uses_cuda_hwaccel_and_scale_cuda(
        self, monkeypatch, tmp_path: Path
    ):
        from backend.phase1_runtime.frame_decode import decode_video_frames

        video = tmp_path / "sample.mp4"
        video.write_bytes(b"not-a-real-video")
        monkeypatch.setattr(
            "backend.phase1_runtime.frame_decode._probe_frame_dimensions",
            lambda **_: (4, 3),
        )
        monkeypatch.setattr(
            "backend.phase1_runtime.frame_decode.validate_ffmpeg_nvdec_support",
            lambda: None,
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
            return _DummyProcess(payload=b"\x00" * (2 * 2 * 3))

        monkeypatch.setattr("backend.phase1_runtime.frame_decode.subprocess.Popen", _fake_popen)

        frames = list(
            decode_video_frames(
                video_path=video,
                decode_backend="gpu",
                gpu_decode_backend="nvdec",
                target_width=2,
                target_height=2,
            )
        )

        assert len(frames) == 1
        assert frames[0].rgb.shape == (2, 2, 3)
        assert frames[0].source_width == 4
        assert frames[0].source_height == 3
        assert captured_cmd[captured_cmd.index("-hwaccel") + 1] == "cuda"
        assert captured_cmd[captured_cmd.index("-hwaccel_output_format") + 1] == "cuda"
        assert captured_cmd[captured_cmd.index("-vf") + 1] == (
            "scale_cuda=2:2,hwdownload,format=nv12,format=rgb24"
        )


class TestDetectorMetrics:
    def test_mean_latency(self):
        from backend.phase1_runtime.tensorrt_detector import DetectorMetrics

        assert DetectorMetrics().mean_detector_latency_ms == 0.0
        assert DetectorMetrics(frames_processed=10, total_detector_ms=100.0).mean_detector_latency_ms == 10.0


class TestTensorRTDetector:
    def _make_config(self, tmp_path: Path):
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
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path / "engines"),
        )

    def test_detect_batch_raises_if_not_loaded(self, tmp_path: Path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

        with pytest.raises(RuntimeError, match="not loaded"):
            det.detect_batch([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_require_cuda_fails_without_cuda(self, monkeypatch):
        from backend.phase1_runtime.tensorrt_detector import _require_cuda

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda="12.4"),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with pytest.raises(RuntimeError, match="CUDA GPU is required"):
            _require_cuda()

    def test_ensure_engine_reuses_existing(self, tmp_path: Path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_config(tmp_path)
        det = TensorRTDetector(config)
        config.tensorrt_engine_path.parent.mkdir(parents=True, exist_ok=True)
        config.tensorrt_engine_path.write_bytes(b"fake_engine_data")

        assert det._ensure_engine() == config.tensorrt_engine_path

    def test_ensure_engine_preserves_checkpoint_pe_grid_for_640_export(
        self, tmp_path: Path, monkeypatch
    ):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=2,
            detection_threshold=0.35,
            detector_resolution=640,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path / "engines"),
        )
        captured: dict[str, object] = {}

        class _FakeRFDETRNano:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def export(self, *, output_dir, **kwargs):
                captured["export_kwargs"] = kwargs
                Path(output_dir, "inference_model.onnx").write_bytes(b"onnx")

        fake_rfdetr = SimpleNamespace(RFDETRNano=_FakeRFDETRNano)
        monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)

        det = TensorRTDetector(config)
        monkeypatch.setattr(
            det,
            "_convert_onnx_to_engine",
            lambda _onnx_path, engine_path: engine_path.write_bytes(b"engine"),
        )

        assert det._ensure_engine() == config.tensorrt_engine_path
        assert captured["init_kwargs"] == {
            "resolution": 640,
            "positional_encoding_size": 24,
        }
        assert captured["export_kwargs"]["shape"] == (640, 640)

    def test_convert_onnx_to_engine_uses_skip_inference_cli(self, tmp_path: Path, monkeypatch):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        config = self._make_config(tmp_path)
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
        assert not any(arg.startswith("--minShapes=") for arg in captured["cmd"])
        assert not any(arg.startswith("--optShapes=") for arg in captured["cmd"])
        assert not any(arg.startswith("--maxShapes=") for arg in captured["cmd"])

    def test_postprocess_decodes_class_logits_before_thresholding(self, tmp_path: Path, monkeypatch):
        import types

        from backend.phase1_runtime.tensorrt_detector import COCO_PERSON_CLASS_ID, TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

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

        boxes = np.array([[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]], dtype=np.float32)
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
        assert detections[0].class_id.tolist() == [COCO_PERSON_CLASS_ID]


class TestDetectorFactory:
    def test_factory_rejects_non_tensorrt_backend(self):
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
            gpu_decode_backend="nvdec",
            detector_artifact_dir="/tmp/engines",
        )

        with pytest.raises(ValueError, match="TensorRT visual extraction"):
            _make_detector(config)

    def test_factory_returns_tensorrt_detector(self):
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
            gpu_decode_backend="nvdec",
            detector_artifact_dir="/tmp/engines",
        )

        assert isinstance(_make_detector(config), TensorRTDetector)


class TestVisualExtractorArtifactContract:
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
                }
            ],
        )

        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)

        assert len(payload["person_detections"]) == 1
        assert payload["person_detections"][0]["track_id"] == "track_1"
