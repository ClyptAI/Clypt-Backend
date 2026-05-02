"""RF-DETR person detector for Phase 1 visual extraction.

Responsibilities:
- Load the configured RF-DETR model on ROCm/HIP GPU
- optimize_for_inference with FP16
- Emit per-frame sv.Detections filtered to person class only
- Remain stateless across frames
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

    from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)

# rfdetr uses 1-indexed COCO class IDs (COCO_CLASSES = {1: "person", 2: "bicycle", ...})
COCO_PERSON_CLASS_ID = 1
_RFDETR_NATIVE_POSITIONAL_ENCODING_SIZE = {
    "nano": 24,
    "small": 32,
}


@dataclass(slots=True)
class DetectorMetrics:
    frames_processed: int = 0
    total_detector_ms: float = 0.0
    warmup_ms: float = 0.0

    @property
    def mean_detector_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_detector_ms / self.frames_processed


def _require_gpu() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch ROCm is required for RF-DETR visual extraction."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/HIP GPU is required for RF-DETR visual extraction on AMD. "
            "PyTorch exposes HIP devices through torch.cuda; no compatible GPU was found. "
            "Do not fall back to CPU."
        )
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if not hip_version:
        raise RuntimeError(
            "PyTorch ROCm/HIP build is required for Phase1 AMD RF-DETR visual extraction. "
            "torch.cuda is available, but torch.version.hip is empty; do not use a CUDA wheel."
        )


def _require_cuda() -> None:
    """Legacy private alias for retired TensorRT helpers that still import it."""
    _require_gpu()


class RFDETRPersonDetector:
    """Wraps the configured RF-DETR model for person-only detection with ROCm/FP16."""

    def __init__(self, config: VisualPipelineConfig) -> None:
        self._config = config
        self._model = None
        self._metrics = DetectorMetrics()
        self._device: str = "cpu"

    @property
    def metrics(self) -> DetectorMetrics:
        return self._metrics

    def load(self) -> None:
        """Load the model, move to ROCm/HIP GPU, optimize, and warm up."""
        import torch

        _require_gpu()

        try:
            if self._config.detector_model == "nano":
                from rfdetr import RFDETRNano as RFDETRModel
            else:
                from rfdetr import RFDETRSmall as RFDETRModel
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is required for RF-DETR visual extraction. "
                "Install with: pip install rfdetr"
            ) from exc

        # PyTorch ROCm intentionally exposes HIP devices through the cuda namespace.
        self._device = "cuda"
        logger.info(
            "Loading RFDETR%s on %s (resolution=%d, backend=%s)",
            self._config.detector_model.capitalize(),
            self._device,
            self._config.detector_resolution,
            self._config.detector_backend,
        )

        model_kwargs = {"resolution": self._config.detector_resolution}
        native_pe = _RFDETR_NATIVE_POSITIONAL_ENCODING_SIZE.get(self._config.detector_model)
        if native_pe is not None:
            # RF-DETR >=1.6 auto-recomputes positional_encoding_size when a custom
            # resolution is passed. The public Nano/Small checkpoints keep their
            # native PE grid, then interpolate at inference time; recomputing it
            # makes checkpoint loading fail with a position_embeddings mismatch.
            model_kwargs["positional_encoding_size"] = native_pe

        model = RFDETRModel(**model_kwargs)

        torch.backends.cudnn.benchmark = True

        dtype = torch.float16 if self._config.use_fp16 else torch.float32
        logger.info(
            "Optimizing for inference: batch_size=%d, dtype=%s, compile=%s",
            self._config.detector_batch_size,
            dtype,
            not self._config.use_tensorrt,
        )
        model.optimize_for_inference(
            compile=not self._config.use_tensorrt,
            batch_size=self._config.detector_batch_size,
            dtype=dtype,
        )

        warmup_start = time.perf_counter()
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_batch = [dummy] * self._config.detector_batch_size
        model.predict(dummy_batch, threshold=self._config.detection_threshold)
        self._metrics.warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        logger.info("Warmup complete in %.1f ms", self._metrics.warmup_ms)

        self._model = model

    def detect_batch(
        self,
        frames: list[np.ndarray],
        *,
        orig_sizes: list[tuple[int, int]] | None = None,
    ) -> list[sv.Detections]:
        """Run person detection on a batch of BGR/RGB numpy frames.

        Args:
            frames: list of HWC uint8 numpy arrays (RGB).
            orig_sizes: optional original (height, width) for frames decoded with
                GPU resize. Output boxes are rescaled back to source coordinates.

        Returns:
            list of sv.Detections, one per frame, filtered to person class only.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        import supervision as sv

        batch_size = self._config.detector_batch_size
        all_detections: list[sv.Detections] = []

        for batch_start in range(0, len(frames), batch_size):
            batch = frames[batch_start : batch_start + batch_size]
            real_count = len(batch)

            # torch.compile fixes the batch size — pad the tail batch with
            # copies of the last frame so it matches the compiled batch size.
            if real_count < batch_size:
                pad = [batch[-1]] * (batch_size - real_count)
                batch = batch + pad

            t0 = time.perf_counter()
            raw = self._model.predict(batch, threshold=self._config.detection_threshold)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if not isinstance(raw, list):
                raw = [raw]

            # Only keep detections for the real (non-padded) frames
            for offset, det in enumerate(raw[:real_count]):
                person_mask = det.class_id == COCO_PERSON_CLASS_ID
                xyxy = det.xyxy[person_mask]
                if orig_sizes is not None:
                    source_height, source_width = orig_sizes[batch_start + offset]
                    frame_height, frame_width = batch[offset].shape[:2]
                    if frame_width > 0 and frame_height > 0:
                        scale = np.array(
                            [
                                source_width / frame_width,
                                source_height / frame_height,
                                source_width / frame_width,
                                source_height / frame_height,
                            ],
                            dtype=xyxy.dtype,
                        )
                        xyxy = xyxy * scale
                # Build a fresh Detections with only the per-box fields we need.
                # Using det[mask] fails when det.data contains non-per-detection
                # arrays (e.g. image metadata sized by height/width).
                filtered = sv.Detections(
                    xyxy=xyxy,
                    confidence=det.confidence[person_mask] if det.confidence is not None else None,
                    class_id=det.class_id[person_mask] if det.class_id is not None else None,
                )
                all_detections.append(filtered)

            self._metrics.frames_processed += real_count
            self._metrics.total_detector_ms += elapsed_ms

        return all_detections

    def unload(self) -> None:
        """Release model and free VRAM."""
        if self._model is not None:
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("RF-DETR detector unloaded, VRAM released.")


__all__ = ["DetectorMetrics", "RFDETRPersonDetector", "_require_gpu"]
