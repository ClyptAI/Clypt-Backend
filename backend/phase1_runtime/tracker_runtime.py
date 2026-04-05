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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

    from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackerMetrics:
    frames_processed: int = 0
    total_tracker_ms: float = 0.0
    total_detections_in: int = 0
    total_tracked_out: int = 0

    @property
    def mean_tracker_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_tracker_ms / self.frames_processed


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


class ByteTrackTrackerRuntime:
    """Wraps Roboflow trackers ByteTrackTracker for sequential frame updates."""

    def __init__(self, config: VisualPipelineConfig) -> None:
        self._config = config
        self._tracker = None
        self._metrics = TrackerMetrics()

    @property
    def metrics(self) -> TrackerMetrics:
        return self._metrics

    def initialize(self) -> None:
        try:
            from trackers import ByteTrackTracker
        except ImportError as exc:
            raise RuntimeError(
                "trackers is required for ByteTrack tracking. "
                "Install with: pip install trackers"
            ) from exc

        self._tracker = ByteTrackTracker()
        logger.info(
            "ByteTrackTracker initialized (backend=%s)",
            self._config.tracker_backend,
        )

    def update(self, *, frame_idx: int, detections: sv.Detections) -> list[TrackRow]:
        """Feed one frame's detections into the tracker and return track rows.

        Must be called in chronological frame order.
        """
        if self._tracker is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        n_in = len(detections)

        t0 = time.perf_counter()
        tracked = self._tracker.update(detections)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._metrics.frames_processed += 1
        self._metrics.total_tracker_ms += elapsed_ms
        self._metrics.total_detections_in += n_in

        rows: list[TrackRow] = []
        if tracked.tracker_id is None or len(tracked) == 0:
            return rows

        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue
            xyxy = tracked.xyxy[i]
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
                )
            )

        self._metrics.total_tracked_out += len(rows)
        return rows

    def reset(self) -> None:
        """Reset tracker state (e.g. between videos)."""
        if self._tracker is not None:
            self._tracker = None
            self.initialize()


__all__ = ["ByteTrackTrackerRuntime", "TrackRow", "TrackerMetrics"]
