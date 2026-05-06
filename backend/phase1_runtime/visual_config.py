from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True, slots=True)
class VisualPipelineConfig:
    detector_model: str
    detector_backend: str
    detector_batch_size: int
    detection_threshold: float
    detector_resolution: int
    tracker_backend: str
    tracker_lost_buffer: int
    tracker_match_threshold: float
    frame_decode_backend: str
    gpu_decode_backend: str
    detector_artifact_dir: str
    pose_validation_enabled: bool = True
    pose_model_path: str = "yolo11s-pose.pt"
    pose_imgsz: int = 640
    pose_batch_size: int = 16
    pose_max_samples_per_tracklet: int = 24
    pose_min_rfdetr_confidence: float = 0.85
    pose_min_head_evidence_ratio: float = 0.40
    pose_min_upper_body_anchor_ratio: float = 0.25
    pose_keypoint_confidence: float = 0.40
    pose_confidence: float = 0.30
    pose_crop_padding_ratio: float = 0.08
    torch_compile: bool = False

    PERSON_CLASS_ID: int = 0

    def __post_init__(self) -> None:
        if self.detector_model != "seg_nano":
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_MODEL="
                f"{self.detector_model!r}; expected 'seg_nano'. "
                "Detection-only RF-DETR models are not supported on the active Modal path."
            )

    @classmethod
    def from_env(cls) -> VisualPipelineConfig:
        config = cls(
            detector_model=_env("CLYPT_PHASE1_VISUAL_MODEL", "seg_nano"),
            detector_backend=_env("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16"),
            detector_batch_size=int(_env("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "16")),
            detection_threshold=float(_env("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.85")),
            detector_resolution=int(_env("CLYPT_PHASE1_VISUAL_SHAPE", "640")),
            tracker_backend=_env("CLYPT_PHASE1_VISUAL_TRACKER", "bytetrack"),
            tracker_lost_buffer=int(_env("CLYPT_PHASE1_VISUAL_TRACKER_BUFFER", "30")),
            tracker_match_threshold=float(
                _env("CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH", "0.7")
            ),
            frame_decode_backend=_env("CLYPT_PHASE1_VISUAL_DECODE", "gpu"),
            gpu_decode_backend=_env("CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND", "nvdec"),
            detector_artifact_dir=_env(
                "CLYPT_PHASE1_VISUAL_ARTIFACT_DIR",
                "backend/outputs/phase1_visual",
            ),
            pose_validation_enabled=_env("CLYPT_PHASE1_VISUAL_POSE_VALIDATION", "1") == "1",
            pose_model_path=_env("CLYPT_PHASE1_VISUAL_POSE_MODEL_PATH", "yolo11s-pose.pt"),
            pose_imgsz=int(_env("CLYPT_PHASE1_VISUAL_POSE_IMGSZ", "640")),
            pose_batch_size=int(_env("CLYPT_PHASE1_VISUAL_POSE_BATCH_SIZE", "16")),
            pose_max_samples_per_tracklet=int(
                _env("CLYPT_PHASE1_VISUAL_POSE_MAX_SAMPLES_PER_TRACKLET", "24")
            ),
            pose_min_rfdetr_confidence=float(
                _env("CLYPT_PHASE1_VISUAL_POSE_MIN_RFDETR_CONFIDENCE", "0.85")
            ),
            pose_min_head_evidence_ratio=float(
                _env("CLYPT_PHASE1_VISUAL_POSE_MIN_HEAD_EVIDENCE_RATIO", "0.40")
            ),
            pose_min_upper_body_anchor_ratio=float(
                _env("CLYPT_PHASE1_VISUAL_POSE_MIN_UPPER_BODY_ANCHOR_RATIO", "0.25")
            ),
            pose_keypoint_confidence=float(
                _env("CLYPT_PHASE1_VISUAL_POSE_KEYPOINT_CONFIDENCE", "0.40")
            ),
            pose_confidence=float(_env("CLYPT_PHASE1_VISUAL_POSE_CONFIDENCE", "0.30")),
            pose_crop_padding_ratio=float(
                _env("CLYPT_PHASE1_VISUAL_POSE_CROP_PADDING_RATIO", "0.08")
            ),
            torch_compile=_env("CLYPT_PHASE1_VISUAL_TORCH_COMPILE", "0") == "1",
        )
        if config.frame_decode_backend != "gpu":
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_DECODE="
                f"{config.frame_decode_backend!r}; GPU decode is required."
            )
        if config.gpu_decode_backend not in {"nvdec", "cuda"}:
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND="
                f"{config.gpu_decode_backend!r}; expected 'nvdec' or 'cuda' for Modal L40S."
            )
        if not config.use_tensorrt:
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_BACKEND="
                f"{config.detector_backend!r}; TensorRT is required for Modal L40S visual extraction."
            )
        if config.gpu_decode_backend not in {"nvdec", "cuda"}:
            raise ValueError(
                "TensorRT visual extraction requires "
                "CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND=nvdec or cuda."
            )
        return config

    @property
    def use_fp16(self) -> bool:
        return "fp16" in self.detector_backend

    @property
    def use_tensorrt(self) -> bool:
        return self.detector_backend.startswith("tensorrt")

    @property
    def is_gpu_required(self) -> bool:
        return True

    @property
    def tensorrt_engine_dir(self) -> str:
        return self.detector_artifact_dir

    @property
    def tensorrt_engine_path(self) -> Path:
        name = (
            f"rfdetr_{self.detector_model}_b{self.detector_batch_size}"
            f"_r{self.detector_resolution}"
            f"_fp16.engine"
        )
        return Path(self.detector_artifact_dir) / name

    @property
    def pose_engine_path(self) -> Path:
        model_stem = Path(self.pose_model_path).stem.replace(".", "_")
        name = f"{model_stem}_b{self.pose_batch_size}_r{self.pose_imgsz}_fp16.engine"
        return Path(self.detector_artifact_dir) / name


__all__ = ["VisualPipelineConfig"]
