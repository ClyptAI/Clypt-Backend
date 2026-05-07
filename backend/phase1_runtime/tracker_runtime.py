"""ByteTrack tracker runtime for Phase 1 visual extraction.

Responsibilities:
- Own ByteTrackTracker lifecycle
- Accept per-frame sv.Detections from the detector
- Assign persistent track IDs via tracker.update()
- Emit tracker-neutral track rows before post-processing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

    from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackerMetrics:
    frames_processed: int = 0
    total_tracker_ms: float = 0.0
    total_mask_association_ms: float = 0.0
    total_detections_in: int = 0
    total_tracked_out: int = 0
    resets: int = 0

    @property
    def mean_tracker_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_tracker_ms / self.frames_processed

    @property
    def mean_mask_association_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_mask_association_ms / self.frames_processed


@dataclass(slots=True)
class TrackRow:
    """A single detection row with a tracker-assigned persistent ID."""

    frame_idx: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    mask_ref: dict[str, Any] | None = None


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter), where=union > 0)


class ByteTrackTrackerRuntime:
    """Wraps Roboflow trackers ByteTrackTracker for sequential frame updates."""

    def __init__(self, config: VisualPipelineConfig) -> None:
        self._config = config
        self._tracker = None
        self._metrics = TrackerMetrics()
        self._frame_rate = 30.0

    @property
    def metrics(self) -> TrackerMetrics:
        return self._metrics

    def _create_tracker(self, *, frame_rate: float):
        try:
            from trackers import ByteTrackTracker
        except ImportError as exc:
            raise RuntimeError(
                "trackers is required for ByteTrack tracking. "
                "Install with: pip install trackers"
            ) from exc

        return ByteTrackTracker(
            lost_track_buffer=self._config.tracker_lost_buffer,
            frame_rate=frame_rate,
            # track_activation_threshold: min confidence to activate a new track
            track_activation_threshold=self._config.tracker_match_threshold,
        )

    def initialize(self, *, frame_rate: float = 30.0) -> None:
        self._frame_rate = float(frame_rate or 30.0)
        self._tracker = self._create_tracker(frame_rate=self._frame_rate)
        logger.info(
            "ByteTrackTracker initialized (backend=%s, lost_buffer=%d, fps=%.2f, activation_thresh=%.2f)",
            self._config.tracker_backend,
            self._config.tracker_lost_buffer,
            self._frame_rate,
            self._config.tracker_match_threshold,
        )

    def reset(self) -> None:
        """Drop ByteTrack state at hard camera cuts before processing the next frame."""
        self._tracker = self._create_tracker(frame_rate=self._frame_rate)
        self._metrics.resets += 1

    def update(
        self,
        *,
        frame_idx: int,
        detections: sv.Detections,
        mask_refs: list[dict[str, Any]] | None = None,
    ) -> list[TrackRow]:
        """Feed one frame's detections into the tracker and return track rows.

        Must be called in chronological frame order.
        """
        if self._tracker is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        n_in = len(detections)
        source_masks = getattr(detections, "mask", None)
        if source_masks is None:
            raise RuntimeError("RF-DETR-Seg detections must include masks before ByteTrack.")
        source_mask_refs = list(mask_refs or [])
        if len(source_mask_refs) != len(detections):
            raise RuntimeError(
                "RF-DETR-Seg mask refs must align one-to-one with detections before ByteTrack "
                f"(detections={len(detections)}, mask_refs={len(source_mask_refs)})."
            )

        t0 = time.perf_counter()
        import supervision as sv

        box_only = sv.Detections(
            xyxy=detections.xyxy,
            confidence=getattr(detections, "confidence", None),
            class_id=getattr(detections, "class_id", None),
        )
        tracked = self._tracker.update(box_only)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._metrics.frames_processed += 1
        self._metrics.total_tracker_ms += elapsed_ms
        self._metrics.total_detections_in += n_in

        rows: list[TrackRow] = []
        if tracked.tracker_id is None or len(tracked) == 0:
            return rows

        association_t0 = time.perf_counter()
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue
            xyxy = tracked.xyxy[i]
            ious = _box_iou_one_to_many(np.asarray(xyxy, dtype=np.float32), detections.xyxy)
            if ious.size == 0 or float(np.max(ious)) <= 0.0:
                raise RuntimeError(
                    "RF-DETR-Seg mask association failed: tracked row has no same-frame source mask."
                )
            mask_index = int(np.argmax(ious))
            mask_ref = dict(source_mask_refs[mask_index])
            mask_ref["track_id"] = f"track_{int(tid)}"
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            rows.append(
                TrackRow(
                    frame_idx=frame_idx,
                    track_id=int(tid),
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cid,
                    mask_ref=mask_ref,
                )
            )

        self._metrics.total_mask_association_ms += (
            time.perf_counter() - association_t0
        ) * 1000.0
        self._metrics.total_tracked_out += len(rows)
        return rows


__all__ = ["ByteTrackTrackerRuntime", "TrackRow", "TrackerMetrics"]
