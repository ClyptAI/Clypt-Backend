"""TensorRT-based RF-DETR Small person detector for Phase 1 visual extraction.

Full pipeline:
1. Export RFDETRSmall to ONNX via rfdetr's model.export()
2. Convert ONNX to TensorRT engine via trtexec (must run on target GPU)
3. Load the .engine and run native TensorRT FP16 inference
4. Post-process raw outputs into sv.Detections filtered to person class

The engine is cached on disk. If it already exists for the current
resolution/batch_size/precision combo, steps 1-2 are skipped.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

    from .visual_config import VisualPipelineConfig

from .rfdetr_detector import COCO_PERSON_CLASS_ID, DetectorMetrics, _require_cuda

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TensorRTDetector:
    """Runs RF-DETR Small person detection via a native TensorRT engine."""

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
            logger.info("Exporting RFDETRSmall to ONNX at %s ...", onnx_path)
            try:
                from rfdetr import RFDETRSmall
            except ImportError as exc:
                raise RuntimeError(
                    "rfdetr is required for TensorRT engine build. "
                    "Install with: pip install 'rfdetr[onnx]'"
                ) from exc

            model = RFDETRSmall(resolution=self._config.detector_resolution)
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
        """Convert an ONNX model to a TensorRT engine using rfdetr's trtexec
        wrapper if available, otherwise fall back to the trtexec CLI."""
        try:
            from argparse import Namespace

            from rfdetr.export.tensorrt import trtexec

            args = Namespace(verbose=True, profile=False, dry_run=False)
            trtexec(str(onnx_path), args)

            built = onnx_path.with_suffix(".engine")
            if built.exists() and built != engine_path:
                built.rename(engine_path)
            elif not engine_path.exists():
                raise RuntimeError(
                    f"rfdetr trtexec completed but engine not found at {engine_path}"
                )
            return
        except ImportError:
            logger.info(
                "rfdetr.export.tensorrt not available, falling back to trtexec CLI."
            )

        import subprocess

        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            f"--optShapes=input:{self._config.detector_batch_size}x3"
            f"x{self._config.detector_resolution}x{self._config.detector_resolution}",
            f"--minShapes=input:1x3"
            f"x{self._config.detector_resolution}x{self._config.detector_resolution}",
            f"--maxShapes=input:{self._config.detector_batch_size}x3"
            f"x{self._config.detector_resolution}x{self._config.detector_resolution}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"trtexec failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
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

    def _postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        orig_sizes: list[tuple[int, int]],
        batch_len: int,
    ) -> list:
        """Convert raw TRT outputs to sv.Detections per frame."""
        import supervision as sv

        detections_list = []
        for i in range(batch_len):
            frame_scores = scores[i] if scores.ndim > 1 else scores
            frame_labels = labels[i] if labels.ndim > 1 else labels
            frame_boxes = boxes[i] if boxes.ndim > 2 else boxes

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

            if len(f_boxes) > 0:
                oh, ow = orig_sizes[i]
                res = self._config.detector_resolution
                f_boxes[:, [0, 2]] *= ow / res
                f_boxes[:, [1, 3]] *= oh / res

            person_mask = f_labels == COCO_PERSON_CLASS_ID
            det = sv.Detections(
                xyxy=f_boxes[person_mask].astype(np.float32) if person_mask.any() else np.empty((0, 4), dtype=np.float32),
                confidence=f_scores[person_mask].astype(np.float32) if person_mask.any() else np.empty(0, dtype=np.float32),
                class_id=f_labels[person_mask] if person_mask.any() else np.empty(0, dtype=np.int32),
            )
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

            t0 = time.perf_counter()
            self._context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            output_arrays = {}
            for name, info in self._bindings.items():
                if not info["is_input"]:
                    output_arrays[name] = info["buffer"].cpu().numpy()

            out_names = sorted(output_arrays.keys())
            if len(out_names) >= 2:
                boxes_key = next(
                    (k for k in out_names if "box" in k.lower()), out_names[0]
                )
                remaining = [k for k in out_names if k != boxes_key]
                scores_key = next(
                    (k for k in remaining if "score" in k.lower() or "logit" in k.lower()),
                    remaining[0],
                )
                labels_key = next(
                    (k for k in remaining if "label" in k.lower() or "class" in k.lower()),
                    remaining[-1] if len(remaining) > 1 else scores_key,
                )
                raw_boxes = output_arrays[boxes_key]
                raw_scores = output_arrays[scores_key]
                raw_labels = output_arrays.get(labels_key, np.zeros_like(raw_scores, dtype=np.int32))
            else:
                raise RuntimeError(
                    f"Expected at least 2 output tensors from TRT engine, got {len(out_names)}: {out_names}"
                )

            dets = self._postprocess(
                raw_boxes, raw_scores, raw_labels, batch_orig_sizes, actual_batch_len
            )
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
        except Exception:
            pass
        logger.info("TensorRT detector unloaded, VRAM released.")


__all__ = ["TensorRTDetector"]
