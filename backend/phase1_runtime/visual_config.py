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
    decode_queue_depth: int = 2
    decode_buffer_chunk_size: int = 3
    decode_hw_accel_device: str = "cuda:0"
    decode_hw_device_index: int = 0
    benchmark_batch_sizes: tuple[int, ...] = (16, 24, 32)

    PERSON_CLASS_ID: int = 0

    @classmethod
    def from_env(cls) -> VisualPipelineConfig:
        config = cls(
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
            tensorrt_engine_dir=_env(
                "CLYPT_PHASE1_VISUAL_TRT_ENGINE_DIR",
                "backend/outputs/tensorrt_engines",
            ),
            decode_queue_depth=int(_env("CLYPT_PHASE1_VISUAL_DECODE_QUEUE_DEPTH", "2")),
            decode_buffer_chunk_size=int(
                _env("CLYPT_PHASE1_VISUAL_DECODE_BUFFER_CHUNKS", "3")
            ),
            decode_hw_accel_device=_env(
                "CLYPT_PHASE1_VISUAL_DECODE_HWACCEL_DEVICE",
                "cuda:0",
            ),
            decode_hw_device_index=int(
                _env("CLYPT_PHASE1_VISUAL_DECODE_HW_DEVICE_INDEX", "0")
            ),
            benchmark_batch_sizes=tuple(
                int(value.strip())
                for value in _env("CLYPT_PHASE1_VISUAL_BENCHMARK_BATCH_SIZES", "16,24,32").split(",")
                if value.strip()
            ),
        )
        if config.frame_decode_backend != "gpu":
            raise ValueError(
                "Unsupported CLYPT_PHASE1_VISUAL_DECODE="
                f"{config.frame_decode_backend!r}; GPU decode is required."
            )
        if config.decode_queue_depth < 1:
            raise ValueError("CLYPT_PHASE1_VISUAL_DECODE_QUEUE_DEPTH must be >= 1.")
        if config.decode_buffer_chunk_size < 1:
            raise ValueError("CLYPT_PHASE1_VISUAL_DECODE_BUFFER_CHUNKS must be >= 1.")
        if not config.benchmark_batch_sizes:
            raise ValueError("CLYPT_PHASE1_VISUAL_BENCHMARK_BATCH_SIZES must not be empty.")
        return config

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
