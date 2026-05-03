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
    torch_compile: bool = False

    PERSON_CLASS_ID: int = 0

    @classmethod
    def from_env(cls) -> VisualPipelineConfig:
        config = cls(
            detector_model=_env("CLYPT_PHASE1_VISUAL_MODEL", "nano"),
            detector_backend=_env("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16"),
            detector_batch_size=int(_env("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "16")),
            detection_threshold=float(_env("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.35")),
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
        if config.detector_model not in {"nano", "small"}:
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_MODEL="
                f"{config.detector_model!r}; expected 'nano' or 'small'."
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


__all__ = ["VisualPipelineConfig"]
