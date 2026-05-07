"""TensorRT-based RF-DETR-Seg person detector for Phase 1 visual extraction.

Full pipeline:
1. Export the configured RF-DETR-Seg model to ONNX via rfdetr's model.export()
2. Convert ONNX to TensorRT engine via trtexec (must run on target GPU)
3. Load the .engine and run native TensorRT FP16 inference
4. Post-process raw box/mask outputs into sv.Detections filtered to person class

The engine is cached on disk. If it already exists for the current
resolution/batch_size/precision combo, steps 1-2 are skipped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .masks import MASK_THRESHOLD

if TYPE_CHECKING:
    import supervision as sv

    from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)

# rfdetr uses 1-indexed COCO class IDs (COCO_CLASSES = {1: "person", 2: "bicycle", ...})
COCO_PERSON_CLASS_ID = 1
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RFDETR_CHECKPOINT_PE_GRID = {
    "seg_nano": 26,
}


def _decode_rfdetr_boxes_to_source_xyxy(
    boxes: np.ndarray,
    *,
    source_height: int,
    source_width: int,
) -> np.ndarray:
    """Decode RF-DETR ONNX boxes to source-frame xyxy pixels.

    RF-DETR's exported ONNX head emits normalized ``cx, cy, w, h`` boxes. The
    TensorRT path runs outside RF-DETR's Python predictor, so we need to apply
    the same decode here before handing detections to ByteTrack.
    """
    decoded = np.asarray(boxes, dtype=np.float32).copy()
    if decoded.size == 0:
        return decoded.reshape((-1, 4))
    cx = decoded[:, 0]
    cy = decoded[:, 1]
    width = decoded[:, 2]
    height = decoded[:, 3]
    x1 = (cx - (width / 2.0)) * float(source_width)
    y1 = (cy - (height / 2.0)) * float(source_height)
    x2 = (cx + (width / 2.0)) * float(source_width)
    y2 = (cy + (height / 2.0)) * float(source_height)
    xyxy = np.stack([x1, y1, x2, y2], axis=-1)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(source_width))
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(source_height))
    return xyxy


def _frame_masks_for_index(masks: np.ndarray, *, frame_index: int) -> np.ndarray:
    frame_masks = masks[frame_index] if masks.ndim >= 4 else masks
    frame_masks = np.asarray(frame_masks)
    if frame_masks.ndim == 4 and frame_masks.shape[0] == 1:
        frame_masks = frame_masks[0]
    if frame_masks.ndim == 4 and frame_masks.shape[1] == 1:
        frame_masks = frame_masks[:, 0]
    if frame_masks.ndim == 4 and frame_masks.shape[-1] == 1:
        frame_masks = frame_masks[..., 0]
    if frame_masks.ndim != 3:
        raise RuntimeError(
            "RF-DETR-Seg mask output must resolve to [queries, height, width]; "
            f"got shape {frame_masks.shape!r}"
        )
    return frame_masks


@dataclass(slots=True)
class DetectorMetrics:
    frames_processed: int = 0
    total_detector_ms: float = 0.0
    total_output_copy_ms: float = 0.0
    total_postprocess_ms: float = 0.0
    total_mask_extract_ms: float = 0.0
    warmup_ms: float = 0.0
    mask_rows: int = 0
    mask_output_tensor: str | None = None

    @property
    def mean_detector_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_detector_ms / self.frames_processed

    @property
    def mean_output_copy_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_output_copy_ms / self.frames_processed

    @property
    def mean_postprocess_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_postprocess_ms / self.frames_processed


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA PyTorch is required for Modal L40S TensorRT visual extraction."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Modal L40S TensorRT visual extraction. "
            "Do not fall back to CPU."
        )
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not cuda_version:
        raise RuntimeError(
            "CUDA PyTorch build is required for Modal L40S TensorRT visual extraction; "
            "torch.cuda is available, but torch.version.cuda is empty."
        )


class TensorRTDetector:
    """Runs the configured RF-DETR-Seg person detector via a native TensorRT engine."""

    def __init__(self, config: VisualPipelineConfig) -> None:
        self._config = config
        self._context = None
        self._engine = None
        self._stream = None
        self._bindings: dict[str, dict] = {}
        self._metrics = DetectorMetrics()
        self._torch_mean = None
        self._torch_std = None

    @property
    def metrics(self) -> DetectorMetrics:
        return self._metrics

    # ------------------------------------------------------------------
    # Engine build: ONNX export + trtexec conversion
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> Path:
        """Export ONNX and convert to TensorRT if the engine doesn't exist."""
        engine_path = self._config.tensorrt_engine_path
        if engine_path.exists():
            logger.info("TensorRT engine found at %s, skipping build.", engine_path)
            return engine_path

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_dir = engine_path.parent / "onnx_export"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "inference_model.onnx"

        if not onnx_path.exists():
            logger.info(
                "Exporting RFDETRSegNano to ONNX at %s ...",
                onnx_path,
            )
            try:
                from rfdetr import RFDETRSegNano as RFDETRModel
            except ImportError as exc:
                raise RuntimeError(
                    "rfdetr with RFDETRSegNano is required for TensorRT engine build. "
                    "Install with: pip install 'rfdetr[onnx]>=1.5.1'"
                ) from exc

            checkpoint_pe_grid = _RFDETR_CHECKPOINT_PE_GRID[self._config.detector_model]
            model = RFDETRModel(
                resolution=self._config.detector_resolution,
                positional_encoding_size=checkpoint_pe_grid,
            )
            model.export(
                output_dir=str(onnx_dir),
                batch_size=self._config.detector_batch_size,
                shape=(self._config.detector_resolution, self._config.detector_resolution),
            )
            if not onnx_path.exists():
                candidates = list(onnx_dir.glob("*.onnx"))
                if candidates:
                    onnx_path = candidates[0]
                else:
                    raise RuntimeError(
                        f"ONNX export succeeded but no .onnx file found in {onnx_dir}"
                    )
            logger.info("ONNX export complete: %s", onnx_path)
        else:
            logger.info("ONNX model already exists at %s", onnx_path)

        logger.info("Converting ONNX to TensorRT engine (FP16) ...")
        self._convert_onnx_to_engine(onnx_path, engine_path)
        logger.info("TensorRT engine built: %s", engine_path)
        return engine_path

    def _convert_onnx_to_engine(self, onnx_path: Path, engine_path: Path) -> None:
        """Convert an ONNX model to a TensorRT engine via an explicit CLI build.

        We intentionally bypass rfdetr's helper wrapper so bootstrap and first-run
        engine creation do not also pay for an inference benchmark pass.
        """
        import subprocess

        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--skipInference",
            "--memPoolSize=workspace:4096",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"trtexec failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        if not engine_path.exists():
            raise RuntimeError(
                f"trtexec completed without writing TensorRT engine to {engine_path}"
            )

    # ------------------------------------------------------------------
    # Engine loading and inference
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build/find the engine, load it, allocate buffers, and warm up."""
        import torch

        _require_cuda()

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt Python package is required for TensorRT inference. "
                "Install the NVIDIA TensorRT runtime."
            ) from exc

        engine_path = self._ensure_engine()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        self._allocate_buffers(torch)

        warmup_start = time.perf_counter()
        dummy = np.random.randint(
            0, 255,
            (self._config.detector_batch_size, self._config.detector_resolution,
             self._config.detector_resolution, 3),
            dtype=np.uint8,
        )
        frames = [dummy[i] for i in range(self._config.detector_batch_size)]
        self.detect_batch(frames)
        self._metrics.warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        self._metrics.frames_processed = 0
        self._metrics.total_detector_ms = 0.0
        self._metrics.total_output_copy_ms = 0.0
        self._metrics.total_postprocess_ms = 0.0
        self._metrics.total_mask_extract_ms = 0.0
        self._metrics.mask_rows = 0
        logger.info("TensorRT warmup complete in %.1f ms", self._metrics.warmup_ms)

    def _allocate_buffers(self, torch) -> None:
        """Pre-allocate CUDA device tensors for each engine binding."""
        import tensorrt as trt

        self._bindings = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
            dtype_trt = self._engine.get_tensor_dtype(name)
            mode = self._engine.get_tensor_mode(name)

            if -1 in shape:
                shape = list(shape)
                shape[0] = self._config.detector_batch_size
                shape = tuple(shape)
                self._context.set_input_shape(name, shape)
                shape = tuple(int(dim) for dim in self._context.get_tensor_shape(name))

            np_dtype = trt.nptype(dtype_trt)
            torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
            buf = torch.empty(shape, dtype=torch_dtype, device="cuda")

            is_input = mode == trt.TensorIOMode.INPUT
            self._bindings[name] = {
                "buffer": buf,
                "shape": shape,
                "dtype": np_dtype,
                "is_input": is_input,
            }
            self._context.set_tensor_address(name, buf.data_ptr())

    def _preprocess_batch(self, frames: list[np.ndarray]):
        """Move frames to CUDA, normalize there, and return an NCHW tensor."""
        import torch

        res = self._config.detector_resolution
        batch_np = np.stack(frames, axis=0)
        batch = torch.from_numpy(batch_np).to(device="cuda", dtype=torch.float32)
        batch = batch.permute(0, 3, 1, 2).contiguous()
        if batch.shape[-2:] != (res, res):
            batch = torch.nn.functional.interpolate(
                batch,
                size=(res, res),
                mode="bilinear",
                align_corners=False,
            )

        if self._torch_mean is None or self._torch_std is None:
            self._torch_mean = torch.from_numpy(_IMAGENET_MEAN.reshape(1, 3, 1, 1)).to(
                device="cuda",
                dtype=torch.float32,
            )
            self._torch_std = torch.from_numpy(_IMAGENET_STD.reshape(1, 3, 1, 1)).to(
                device="cuda",
                dtype=torch.float32,
            )

        batch = batch.div_(255.0)
        batch = batch.sub_(self._torch_mean)
        batch = batch.div_(self._torch_std)
        return batch

    def _prepare_execution_stream(self, torch) -> None:
        """Make the TensorRT stream wait for PyTorch preprocessing/copy work."""
        if self._stream is None:
            raise RuntimeError("TensorRT CUDA stream is not initialized.")
        self._stream.wait_stream(torch.cuda.current_stream())

    def _postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray | None,
        orig_sizes: list[tuple[int, int]],
        batch_len: int,
    ) -> list:
        """Convert raw TRT outputs to sv.Detections per frame."""
        import supervision as sv

        if masks is None:
            raise RuntimeError("RF-DETR-Seg TensorRT engine did not expose a usable mask output.")

        detections_list = []
        for i in range(batch_len):
            frame_scores = scores[i] if scores.ndim > 1 else scores
            frame_labels = labels[i] if labels.ndim > 1 else labels
            frame_boxes = boxes[i] if boxes.ndim > 2 else boxes
            frame_masks = _frame_masks_for_index(masks, frame_index=i)

            # RF-DETR ONNX/TensorRT export returns per-query class logits with
            # shape [num_queries, num_classes(+background)], not final scores.
            # Decode them to one score and one label per query before filtering.
            if frame_scores.ndim == 2:
                shifted = frame_scores - np.max(frame_scores, axis=-1, keepdims=True)
                exp_scores = np.exp(shifted)
                probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                class_probs = probs[:, 1:] if probs.shape[-1] > 1 else probs
                frame_scores = np.max(class_probs, axis=-1)
                frame_labels = np.argmax(class_probs, axis=-1).astype(np.int32) + 1

            keep = frame_scores > self._config.detection_threshold
            f_scores = frame_scores[keep]
            f_labels = frame_labels[keep].astype(np.int32)
            f_boxes = frame_boxes[keep]
            f_masks = frame_masks[keep]

            if len(f_boxes) > 0:
                oh, ow = orig_sizes[i]
                f_boxes = _decode_rfdetr_boxes_to_source_xyxy(
                    f_boxes,
                    source_height=int(oh),
                    source_width=int(ow),
                )

            person_mask = f_labels == COCO_PERSON_CLASS_ID
            if person_mask.any():
                person_boxes = f_boxes[person_mask].astype(np.float32)
                person_scores = f_scores[person_mask].astype(np.float32)
                person_class_ids = f_labels[person_mask].astype(np.int32)
                person_masks = f_masks[person_mask]
                mask_t0 = time.perf_counter()
                source_masks = (person_masks > MASK_THRESHOLD).astype(np.uint8, copy=False)
                self._metrics.total_mask_extract_ms += (
                    time.perf_counter() - mask_t0
                ) * 1000.0
            else:
                person_boxes = np.empty((0, 4), dtype=np.float32)
                person_scores = np.empty(0, dtype=np.float32)
                person_class_ids = np.empty(0, dtype=np.int32)
                source_masks = np.empty((0, 0, 0), dtype=np.uint8)
            det = sv.Detections(
                xyxy=person_boxes,
                confidence=person_scores,
                class_id=person_class_ids,
                mask=source_masks,
            )
            self._metrics.mask_rows += int(len(source_masks))
            detections_list.append(det)
        return detections_list

    def detect_batch(
        self,
        frames: list[np.ndarray],
        *,
        orig_sizes: list[tuple[int, int]] | None = None,
    ) -> list:
        """Run person detection on a batch of HWC uint8 RGB numpy frames."""
        if self._context is None:
            raise RuntimeError("TensorRT detector not loaded. Call load() first.")
        if orig_sizes is not None and len(orig_sizes) != len(frames):
            raise ValueError("orig_sizes must match the number of frames")

        import torch

        batch_size = self._config.detector_batch_size
        all_detections = []

        for batch_start in range(0, len(frames), batch_size):
            batch = frames[batch_start: batch_start + batch_size]
            batch_orig_sizes = (
                orig_sizes[batch_start: batch_start + batch_size]
                if orig_sizes is not None
                else [(f.shape[0], f.shape[1]) for f in batch]
            )
            actual_batch_len = len(batch)

            if actual_batch_len < batch_size:
                pad_frame = np.zeros_like(batch[0])
                batch = batch + [pad_frame] * (batch_size - actual_batch_len)

            input_tensor = self._preprocess_batch(batch)

            input_binding = None
            for name, info in self._bindings.items():
                if info["is_input"]:
                    input_binding = info
                    break
            buffer_dtype = getattr(input_binding["buffer"], "dtype", None)
            if buffer_dtype is not None:
                input_tensor = input_tensor.to(dtype=buffer_dtype)
            input_binding["buffer"].copy_(input_tensor)
            self._prepare_execution_stream(torch)

            t0 = time.perf_counter()
            self._context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            output_copy_t0 = time.perf_counter()
            output_arrays = {}
            for name, info in self._bindings.items():
                if not info["is_input"]:
                    output_arrays[name] = info["buffer"].cpu().numpy()
            self._metrics.total_output_copy_ms += (
                time.perf_counter() - output_copy_t0
            ) * 1000.0

            out_names = sorted(output_arrays.keys())
            if len(out_names) >= 3:
                boxes_key = next(
                    (k for k in out_names if "box" in k.lower()), out_names[0]
                )
                remaining = [k for k in out_names if k != boxes_key]
                scores_key = next(
                    (k for k in remaining if "score" in k.lower() or "logit" in k.lower()),
                    remaining[0],
                )
                mask_key = next(
                    (
                        k
                        for k in remaining
                        if "mask" in k.lower() or "seg" in k.lower()
                    ),
                    None,
                )
                if mask_key is None:
                    raise RuntimeError(
                        "RF-DETR-Seg TensorRT engine did not expose a mask output. "
                        f"Output tensors: {out_names}"
                    )
                labels_key = next(
                    (
                        k
                        for k in remaining
                        if k != mask_key and ("label" in k.lower() or "class" in k.lower())
                    ),
                    None,
                )
                raw_boxes = output_arrays[boxes_key]
                raw_scores = output_arrays[scores_key]
                raw_labels = (
                    output_arrays[labels_key]
                    if labels_key is not None
                    else np.zeros_like(raw_scores, dtype=np.int32)
                )
                raw_masks = output_arrays[mask_key]
                self._metrics.mask_output_tensor = str(mask_key)
            else:
                raise RuntimeError(
                    f"Expected boxes, scores, and masks from TRT engine, got {len(out_names)} tensors: {out_names}"
                )

            postprocess_t0 = time.perf_counter()
            dets = self._postprocess(
                raw_boxes,
                raw_scores,
                raw_labels,
                raw_masks,
                batch_orig_sizes,
                actual_batch_len,
            )
            self._metrics.total_postprocess_ms += (
                time.perf_counter() - postprocess_t0
            ) * 1000.0
            all_detections.extend(dets)

            self._metrics.frames_processed += actual_batch_len
            self._metrics.total_detector_ms += elapsed_ms

        return all_detections

    def unload(self) -> None:
        """Release TensorRT engine and free VRAM."""
        self._context = None
        self._engine = None
        self._stream = None
        self._bindings = {}
        self._torch_mean = None
        self._torch_std = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("TensorRT detector unloaded, VRAM released.")


__all__ = ["COCO_PERSON_CLASS_ID", "DetectorMetrics", "TensorRTDetector", "_require_cuda"]
