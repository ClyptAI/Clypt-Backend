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

        assert config.detector_model == "seg_nano"
        assert config.detector_backend == "tensorrt_fp16"
        assert config.detector_batch_size == 16
        assert config.detection_threshold == pytest.approx(0.85)
        assert config.detector_resolution == 648
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

    def test_from_env_rejects_detection_only_model(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_MODEL", "nano")

        with pytest.raises(ValueError, match="seg_nano"):
            VisualPipelineConfig.from_env()

    def test_from_env_rejects_shape_not_divisible_by_seg_patch_size(self, monkeypatch):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        monkeypatch.setenv("CLYPT_PHASE1_VISUAL_SHAPE", "640")

        with pytest.raises(ValueError, match="patch size 12"):
            VisualPipelineConfig.from_env()

    def test_tensorrt_engine_path_uses_artifact_dir(self, tmp_path: Path):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=552,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path),
        )

        assert config.tensorrt_engine_path == tmp_path / "rfdetr_seg_nano_b4_r552_fp16.engine"


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
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=2,
            detection_threshold=0.35,
            detector_resolution=552,
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

    def test_prepare_execution_stream_waits_for_current_torch_stream(self, tmp_path: Path):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))
        waits: list[object] = []
        det._stream = SimpleNamespace(wait_stream=lambda stream: waits.append(stream))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: "torch-default-stream")
        )

        det._prepare_execution_stream(fake_torch)

        assert waits == ["torch-default-stream"]

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

    def test_ensure_engine_preserves_checkpoint_pe_grid_for_seg_nano_export(
        self, tmp_path: Path, monkeypatch
    ):
        from backend.phase1_runtime.tensorrt_detector import TensorRTDetector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=2,
            detection_threshold=0.35,
            detector_resolution=648,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path / "engines"),
        )
        captured: dict[str, object] = {}

        class _FakeRFDETRSegNano:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def export(self, *, output_dir, **kwargs):
                captured["export_kwargs"] = kwargs
                Path(output_dir, "inference_model.onnx").write_bytes(b"onnx")

        fake_rfdetr = SimpleNamespace(RFDETRSegNano=_FakeRFDETRSegNano)
        monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)

        det = TensorRTDetector(config)
        monkeypatch.setattr(
            det,
            "_convert_onnx_to_engine",
            lambda _onnx_path, engine_path: engine_path.write_bytes(b"engine"),
        )

        assert det._ensure_engine() == config.tensorrt_engine_path
        assert captured["init_kwargs"] == {
            "resolution": 648,
            "positional_encoding_size": 26,
        }
        assert captured["export_kwargs"]["shape"] == (648, 648)

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

    def test_postprocess_decodes_class_logits_and_keeps_lowres_masks(
        self, tmp_path: Path, monkeypatch
    ):
        import types

        from backend.phase1_runtime.tensorrt_detector import COCO_PERSON_CLASS_ID, TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

        class FakeDetections:
            def __init__(self, *, xyxy, confidence, class_id, mask=None):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id
                self.mask = mask

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
            masks=np.ones((1, 2, 560, 560), dtype=np.float32),
            orig_sizes=[(560, 560)],
            batch_len=1,
        )

        assert len(detections) == 1
        assert detections[0].xyxy.shape == (1, 4)
        assert detections[0].class_id.tolist() == [COCO_PERSON_CLASS_ID]
        assert detections[0].mask.shape == (1, 560, 560)

    def test_postprocess_decodes_normalized_cxcywh_boxes_to_source_xyxy(
        self, tmp_path: Path, monkeypatch
    ):
        import types

        from backend.phase1_runtime.tensorrt_detector import COCO_PERSON_CLASS_ID, TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

        class FakeDetections:
            def __init__(self, *, xyxy, confidence, class_id, mask=None):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id
                self.mask = mask

        monkeypatch.setitem(
            sys.modules,
            "supervision",
            types.SimpleNamespace(Detections=FakeDetections),
        )

        boxes = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
        logits = np.full((1, 1, 91), -20.0, dtype=np.float32)
        logits[0, 0, COCO_PERSON_CLASS_ID] = 8.0

        detections = det._postprocess(
            boxes=boxes,
            scores=logits,
            labels=np.zeros_like(logits, dtype=np.int32),
            masks=np.ones((1, 1, 560, 560), dtype=np.float32),
            orig_sizes=[(1080, 1920)],
            batch_len=1,
        )

        assert detections[0].xyxy.tolist() == [[720.0, 270.0, 1200.0, 810.0]]
        assert detections[0].mask.shape == (1, 560, 560)

    def test_postprocess_preserves_overlapping_person_queries(self, tmp_path: Path, monkeypatch):
        import types

        from backend.phase1_runtime.tensorrt_detector import COCO_PERSON_CLASS_ID, TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

        class FakeDetections:
            def __init__(self, *, xyxy, confidence, class_id, mask=None):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id
                self.mask = mask

        monkeypatch.setitem(
            sys.modules,
            "supervision",
            types.SimpleNamespace(Detections=FakeDetections),
        )

        boxes = np.array(
            [
                [
                    [0.5, 0.5, 0.25, 0.5],
                    [0.51, 0.5, 0.25, 0.5],
                    [0.1, 0.5, 0.1, 0.4],
                ]
            ],
            dtype=np.float32,
        )
        detections = det._postprocess(
            boxes=boxes,
            scores=np.array([[0.99, 0.9, 0.8]], dtype=np.float32),
            labels=np.array([[COCO_PERSON_CLASS_ID, COCO_PERSON_CLASS_ID, COCO_PERSON_CLASS_ID]], dtype=np.int32),
            masks=np.ones((1, 3, 560, 560), dtype=np.float32),
            orig_sizes=[(1080, 1920)],
            batch_len=1,
        )

        assert detections[0].xyxy.shape == (3, 4)
        assert detections[0].xyxy.tolist()[0] == [720.0, 270.0, 1200.0, 810.0]
        assert detections[0].mask.shape == (3, 560, 560)

    def test_postprocess_fails_hard_without_mask_output(self, tmp_path: Path, monkeypatch):
        import types

        from backend.phase1_runtime.tensorrt_detector import COCO_PERSON_CLASS_ID, TensorRTDetector

        det = TensorRTDetector(self._make_config(tmp_path))

        class FakeDetections:
            def __init__(self, *, xyxy, confidence, class_id, mask=None):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id
                self.mask = mask

        monkeypatch.setitem(
            sys.modules,
            "supervision",
            types.SimpleNamespace(Detections=FakeDetections),
        )

        with pytest.raises(RuntimeError, match="mask output"):
            det._postprocess(
                boxes=np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32),
                scores=np.array([[0.99]], dtype=np.float32),
                labels=np.array([[COCO_PERSON_CLASS_ID]], dtype=np.int32),
                masks=None,
                orig_sizes=[(1080, 1920)],
                batch_len=1,
            )


class TestDetectorFactory:
    def test_factory_rejects_non_tensorrt_backend(self):
        from backend.phase1_runtime.visual import _make_detector
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="seg_nano",
            detector_backend="pytorch_cuda_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=552,
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
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.35,
            detector_resolution=552,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir="/tmp/engines",
        )

        assert isinstance(_make_detector(config), TensorRTDetector)


class TestRfdetrTrackingPipeline:
    def _make_config(self, tmp_path: Path):
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        return VisualPipelineConfig(
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=16,
            detection_threshold=0.85,
            detector_resolution=648,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path / "engines"),
        )

    def test_resets_tracker_at_shot_cut_and_emits_raw_detections(
        self, monkeypatch, tmp_path: Path
    ):
        from backend.phase1_runtime.frame_decode import DecodedFrame
        from backend.phase1_runtime.tracker_runtime import TrackRow
        from backend.phase1_runtime.visual import _run_rfdetr_tracking_pipeline

        class FakeDetections:
            def __init__(self, offset: float) -> None:
                self.xyxy = np.array([[offset, 20.0, offset + 30.0, 80.0]], dtype=np.float32)
                self.confidence = np.array([0.91], dtype=np.float32)
                self.class_id = np.array([1], dtype=np.int32)
                self.mask = np.ones((1, 1080, 1920), dtype=bool)

            def __len__(self) -> int:
                return len(self.xyxy)

        class FakeDetector:
            def __init__(self, _config) -> None:
                self.metrics = SimpleNamespace(
                    frames_processed=0,
                    mean_detector_latency_ms=0.0,
                    warmup_ms=0.0,
                )

            def load(self) -> None:
                return None

            def detect_batch(self, frames, *, orig_sizes=None):  # noqa: ARG002
                self.metrics.frames_processed += len(frames)
                return [FakeDetections(float(index * 10)) for index, _ in enumerate(frames)]

            def unload(self) -> None:
                return None

        events: list[tuple[str, int | None]] = []

        class FakeTracker:
            def __init__(self, _config) -> None:
                self.metrics = SimpleNamespace(mean_tracker_latency_ms=0.0, resets=0)

            def initialize(self, *, frame_rate: float = 30.0) -> None:  # noqa: ARG002
                events.append(("init", None))

            def reset(self) -> None:
                events.append(("reset", None))
                self.metrics.resets += 1

            def update(self, *, frame_idx: int, detections, mask_refs=None) -> list[TrackRow]:
                events.append(("update", frame_idx))
                mask_refs = list(mask_refs or [])
                return [
                    TrackRow(
                        frame_idx=frame_idx,
                        track_id=7,
                        x1=float(detections.xyxy[0][0]),
                        y1=20.0,
                        x2=float(detections.xyxy[0][2]),
                        y2=80.0,
                        confidence=0.91,
                        class_id=1,
                        mask_ref={**mask_refs[0], "track_id": "track_7"},
                    )
                ]

        def fake_decode_video_frames(**_kwargs):
            for frame_idx in (0, 9, 10, 11):
                yield DecodedFrame(
                    frame_idx=frame_idx,
                    rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                    source_width=1920,
                    source_height=1080,
                )

        def fake_batch_frames(frame_stream, *, batch_size):  # noqa: ARG001
            yield list(frame_stream)

        monkeypatch.setattr("backend.phase1_runtime.visual._make_detector", FakeDetector)
        monkeypatch.setattr(
            "backend.phase1_runtime.frame_decode.decode_video_frames",
            fake_decode_video_frames,
        )
        monkeypatch.setattr("backend.phase1_runtime.frame_decode.batch_frames", fake_batch_frames)
        monkeypatch.setattr(
            "backend.phase1_runtime.tracker_runtime.ByteTrackTrackerRuntime",
            FakeTracker,
        )

        tracks, raw_detections, metrics = _run_rfdetr_tracking_pipeline(
            video_path=tmp_path / "video.mp4",
            config=self._make_config(tmp_path),
            shot_segments=[
                {"start_time_ms": 0, "end_time_ms": 1000},
                {"start_time_ms": 1000, "end_time_ms": 2000},
            ],
            video_fps=10.0,
        )

        assert events == [
            ("init", None),
            ("update", 0),
            ("update", 9),
            ("reset", None),
            ("update", 10),
            ("update", 11),
        ]
        assert [row["frame_idx"] for row in tracks] == [0, 9, 10, 11]
        assert [row["frame_idx"] for row in raw_detections] == [0, 9, 10, 11]
        assert raw_detections[0]["source"] == "rfdetr_raw"
        assert raw_detections[0]["mask_ref"]["encoding"] == "lowres_mask_ref_v1"
        assert tracks[0]["mask_ref"]["encoding"] == "lowres_mask_ref_v1"
        assert metrics["mask_artifact_encoding"] == "npz_compressed_lowres_binary_v1"
        assert metrics["mask_artifact_write_ms"] >= 0.0
        assert metrics["payload_size_bytes"] > 0
        assert metrics["segmentation_enabled"] is True
        assert metrics["mask_rows"] == 4
        assert metrics["mask_encoding"] == "lowres_mask_ref_v1"
        assert metrics["tracker_resets_at_shot_boundaries"] == 1


class TestByteTrackMaskAssociation:
    def test_tracker_passes_box_only_detections_and_recovers_mask_ref(
        self, monkeypatch, tmp_path: Path
    ):
        import types

        from backend.phase1_runtime.tracker_runtime import ByteTrackTrackerRuntime
        from backend.phase1_runtime.visual_config import VisualPipelineConfig

        config = VisualPipelineConfig(
            detector_model="seg_nano",
            detector_backend="tensorrt_fp16",
            detector_batch_size=4,
            detection_threshold=0.85,
            detector_resolution=648,
            tracker_backend="bytetrack",
            tracker_lost_buffer=30,
            tracker_match_threshold=0.8,
            frame_decode_backend="gpu",
            gpu_decode_backend="nvdec",
            detector_artifact_dir=str(tmp_path / "engines"),
        )
        captured_tracker_input: list[object] = []

        class FakeDetections:
            def __init__(
                self,
                *,
                xyxy,
                confidence=None,
                class_id=None,
                tracker_id=None,
                mask=None,
                data=None,
            ):
                self.xyxy = np.asarray(xyxy, dtype=np.float32)
                self.confidence = confidence
                self.class_id = class_id
                self.tracker_id = tracker_id
                self.mask = mask
                self.data = data or {}

            def __len__(self) -> int:
                return len(self.xyxy)

        class FakeByteTrackTracker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def update(self, detections):
                captured_tracker_input.append(detections)
                return FakeDetections(
                    xyxy=np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32),
                    confidence=np.array([0.91], dtype=np.float32),
                    class_id=np.array([1], dtype=np.int32),
                    tracker_id=np.array([42], dtype=np.int32),
                )

        monkeypatch.setitem(
            sys.modules,
            "supervision",
            types.SimpleNamespace(Detections=FakeDetections),
        )
        monkeypatch.setitem(
            sys.modules,
            "trackers",
            types.SimpleNamespace(ByteTrackTracker=FakeByteTrackTracker),
        )

        runtime = ByteTrackTrackerRuntime(config)
        runtime.initialize(frame_rate=30.0)
        rows = runtime.update(
            frame_idx=5,
            mask_refs=[
                {
                    "encoding": "lowres_mask_ref_v1",
                    "artifact_id": "visual_masks_lowres_v1",
                    "mask_index": 0,
                    "frame_idx": 5,
                    "detection_id": "raw_5_0",
                }
            ],
            detections=FakeDetections(
                xyxy=np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32),
                confidence=np.array([0.91], dtype=np.float32),
                class_id=np.array([1], dtype=np.int32),
                mask=np.ones((1, 4, 4), dtype=bool),
            ),
        )

        assert captured_tracker_input[0].mask is None
        assert rows[0].track_id == 42
        assert rows[0].mask_ref == {
            "encoding": "lowres_mask_ref_v1",
            "artifact_id": "visual_masks_lowres_v1",
            "mask_index": 0,
            "frame_idx": 5,
            "detection_id": "raw_5_0",
            "track_id": "track_42",
        }


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

        assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_seg_nano_bytetrack"

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
                    "mask_ref": {
                        "encoding": "lowres_mask_ref_v1",
                        "artifact_id": "visual_masks_lowres_v1",
                        "mask_index": 0,
                        "frame_idx": 0,
                        "detection_id": "raw_0_0",
                    },
                }
            ],
        )

        payload = ext.extract(video_path=workspace.video_path, workspace=workspace)

        assert len(payload["person_detections"]) == 1
        assert payload["person_detections"][0]["track_id"] == "track_1"
        assert payload["tracks"][0]["mask_ref"]["encoding"] == "lowres_mask_ref_v1"
        assert (
            payload["person_detections"][0]["timestamped_objects"][0]["mask_ref"]
            == payload["tracks"][0]["mask_ref"]
        )
