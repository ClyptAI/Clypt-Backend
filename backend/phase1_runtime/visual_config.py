from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True, slots=True)
class VisualPipelineConfig:
    detector_backend: str
    detector_batch_size: int
    detection_threshold: float
    detector_resolution: int
    tracker_backend: str
    tracker_lost_buffer: int
    tracker_match_threshold: float
    frame_decode_backend: str
    tensorrt_engine_dir: str

    PERSON_CLASS_ID: int = 0

    @classmethod
    def from_env(cls) -> VisualPipelineConfig:
        return cls(
            detector_backend=_env("CLYPT_PHASE1_VISUAL_BACKEND", "pytorch_cuda_fp16"),
            detector_batch_size=int(_env("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "4")),
            detection_threshold=float(_env("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.35")),
            detector_resolution=int(_env("CLYPT_PHASE1_VISUAL_SHAPE", "640")),
            tracker_backend=_env("CLYPT_PHASE1_VISUAL_TRACKER", "bytetrack"),
            tracker_lost_buffer=int(_env("CLYPT_PHASE1_VISUAL_TRACKER_BUFFER", "30")),
            tracker_match_threshold=float(
                _env("CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH", "0.7")
            ),
            frame_decode_backend=_env("CLYPT_PHASE1_VISUAL_DECODE", "cpu"),
            tensorrt_engine_dir=_env(
                "CLYPT_PHASE1_VISUAL_TRT_ENGINE_DIR",
                "backend/outputs/tensorrt_engines",
            ),
        )

    @property
    def use_fp16(self) -> bool:
        return "fp16" in self.detector_backend

    @property
    def use_tensorrt(self) -> bool:
        return self.detector_backend.startswith("tensorrt")

    @property
    def is_cuda_required(self) -> bool:
        return "cuda" in self.detector_backend or self.use_tensorrt

    @property
    def tensorrt_engine_path(self) -> Path:
        name = (
            f"rfdetr_small_b{self.detector_batch_size}"
            f"_r{self.detector_resolution}"
            f"_fp16.engine"
        )
        return Path(self.tensorrt_engine_dir) / name


__all__ = ["VisualPipelineConfig"]
