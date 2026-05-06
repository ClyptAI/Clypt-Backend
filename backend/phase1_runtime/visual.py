"""V3.1 visual extraction pipeline for the Modal L40S RF-DETR-Seg worker.

This module orchestrates:
1. video metadata probing
2. shot boundary detection
3. frame decoding (NVIDIA NVDEC/CUDA GPU decode)
4. RF-DETR-Seg person boxes+masks (TensorRT FP16)
5. ByteTrack box tracking + mask association
6. post-processing into the canonical artifact schemas

It delegates to dedicated modules for detection, tracking, and decoding.
The old Ultralytics/YOLO and PyTorch ROCm paths are removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from .masks import MASK_RLE_ENCODING, encode_mask_rle
from .tracking_post import frame_time_ms, split_tracks_at_shot_boundaries

logger = logging.getLogger(__name__)


def probe_video_metadata(*, video_path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    fps_raw = str(stream.get("avg_frame_rate") or "0/1")
    if "/" in fps_raw:
        left, right = fps_raw.split("/", 1)
        fps = float(left) / max(1e-6, float(right))
    else:
        fps = float(fps_raw or 0.0)
    duration_ms = int(round(float((payload.get("format") or {}).get("duration") or 0.0) * 1000.0))
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps or 30.0,
        "duration_ms": duration_ms,
    }


def detect_shot_boundaries_ms(*, video_path: Path, duration_ms: int, threshold: float = 0.30) -> list[int]:
    if duration_ms <= 0:
        return []
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-filter_complex",
            f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    boundaries_ms: list[int] = []
    for line in output.splitlines():
        if "pts_time:" not in line:
            continue
        _, _, remainder = line.partition("pts_time:")
        raw_value = remainder.strip().split()[0]
        try:
            timestamp_ms = int(round(float(raw_value) * 1000.0))
        except ValueError:
            continue
        if 0 < timestamp_ms < duration_ms and timestamp_ms not in boundaries_ms:
            boundaries_ms.append(timestamp_ms)
    boundaries_ms.sort()
    return boundaries_ms


# ---------------------------------------------------------------------------
# RF-DETR-Seg + ByteTrack extraction pipeline
# ---------------------------------------------------------------------------

def _make_detector(config):
    """Create the active RF-DETR-Seg detector backend."""
    from .tensorrt_detector import TensorRTDetector

    if not config.use_tensorrt:
        raise ValueError("TensorRT visual extraction is required on the Modal L40S path.")
    return TensorRTDetector(config)


def _shot_cut_frame_indices(
    *,
    shot_segments: list[dict[str, Any]] | None,
    video_fps: float,
) -> list[int]:
    if not shot_segments or len(shot_segments) <= 1:
        return []
    fps = float(video_fps) if float(video_fps) > 1e-6 else 30.0
    cut_frames = {
        max(0, int(round((float(segment.get("start_time_ms") or 0) / 1000.0) * fps)))
        for segment in shot_segments[1:]
    }
    return sorted(frame_idx for frame_idx in cut_frames if frame_idx > 0)


def _serialize_raw_detections(*, frame_idx: int, detections) -> list[dict]:
    rows: list[dict] = []
    if len(detections) == 0:
        return rows
    for index in range(len(detections)):
        xyxy = detections.xyxy[index]
        confidence = (
            float(detections.confidence[index])
            if getattr(detections, "confidence", None) is not None
            else 0.0
        )
        class_id = (
            int(detections.class_id[index])
            if getattr(detections, "class_id", None) is not None
            else 0
        )
        mask_rle = None
        masks = getattr(detections, "mask", None)
        if masks is not None:
            mask_rle = encode_mask_rle(masks[index])
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "local_frame_idx": int(frame_idx),
                "chunk_idx": 0,
                "detection_id": f"raw_{int(frame_idx)}_{index}",
                "class_id": class_id,
                "label": "person" if class_id == 1 else str(class_id),
                "confidence": confidence,
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "source": "rfdetr_raw",
                "geometry_type": "aabb",
                **({"mask_rle": mask_rle} if mask_rle is not None else {}),
            }
        )
    return rows


def _run_rfdetr_tracking_pipeline(
    *,
    video_path: Path,
    config,
    shot_segments: list[dict[str, Any]] | None = None,
    video_fps: float | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Decode frames, run RF-DETR-Seg detection, then ByteTrack tracking.

    Returns (track_rows, raw_detection_rows, runtime_metrics).
    """
    from .frame_decode import batch_frames, decode_video_frames
    from .tracker_runtime import ByteTrackTrackerRuntime

    detector = _make_detector(config)
    tracker = ByteTrackTrackerRuntime(config)

    detector.load()

    _fps = float(video_fps) if video_fps and video_fps > 0 else 30.0
    if video_fps is None:
        try:
            import cv2 as _cv2
            _cap2 = _cv2.VideoCapture(str(video_path))
            _raw_fps = _cap2.get(_cv2.CAP_PROP_FPS)
            _cap2.release()
            if _raw_fps and _raw_fps > 0:
                _fps = float(_raw_fps)
        except ImportError:
            pass

    tracker.initialize(frame_rate=_fps)
    cut_frames = _shot_cut_frame_indices(shot_segments=shot_segments, video_fps=_fps)
    next_cut_index = 0

    all_track_rows: list[dict] = []
    all_raw_detection_rows: list[dict] = []
    pipeline_start = time.perf_counter()

    # Estimate total frames for progress display
    total_frames: int | None = None
    try:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(str(video_path))
        total_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or None
        _cap.release()
    except ImportError:
        pass

    _log_interval = max(1, (total_frames or 500) // 20)  # ~20 progress lines per video
    _frames_seen = 0
    _last_log_frame = -1

    try:
        frame_stream = decode_video_frames(
            video_path=video_path,
            decode_backend=config.frame_decode_backend,
            gpu_decode_backend=config.gpu_decode_backend,
            target_width=config.detector_resolution,
            target_height=config.detector_resolution,
        )
        for frame_batch in batch_frames(frame_stream, batch_size=config.detector_batch_size):
            rgb_arrays = [f.rgb for f in frame_batch]
            frame_indices = [f.frame_idx for f in frame_batch]
            orig_sizes = [(f.source_height, f.source_width) for f in frame_batch]

            detections_list = detector.detect_batch(rgb_arrays, orig_sizes=orig_sizes)

            for frame_idx, detections in zip(frame_indices, detections_list, strict=True):
                all_raw_detection_rows.extend(
                    _serialize_raw_detections(frame_idx=frame_idx, detections=detections)
                )
                while next_cut_index < len(cut_frames) and frame_idx >= cut_frames[next_cut_index]:
                    tracker.reset()
                    next_cut_index += 1
                track_rows = tracker.update(frame_idx=frame_idx, detections=detections)
                for row in track_rows:
                    width = max(0.0, row.x2 - row.x1)
                    height = max(0.0, row.y2 - row.y1)
                    all_track_rows.append(
                        {
                            "frame_idx": row.frame_idx,
                            "local_frame_idx": row.frame_idx,
                            "chunk_idx": 0,
                            "track_id": f"track_{row.track_id}",
                            "local_track_id": row.track_id,
                            "class_id": row.class_id,
                            "label": "person",
                            "confidence": row.confidence,
                            "x1": row.x1,
                            "y1": row.y1,
                            "x2": row.x2,
                            "y2": row.y2,
                            "x_center": row.x1 + (width / 2.0),
                            "y_center": row.y1 + (height / 2.0),
                            "width": width,
                            "height": height,
                            "source": "detector",
                            "geometry_type": "aabb",
                            **({"mask_rle": row.mask_rle} if row.mask_rle is not None else {}),
                        }
                    )
                _frames_seen += 1

            # Progress log every ~5% of video
            if _frames_seen - _last_log_frame >= _log_interval:
                elapsed = time.perf_counter() - pipeline_start
                fps_live = _frames_seen / elapsed if elapsed > 0 else 0.0
                if total_frames:
                    pct = _frames_seen / total_frames * 100
                    eta_s = (total_frames - _frames_seen) / fps_live if fps_live > 0 else 0.0
                    logger.info(
                        "[visual]  RF-DETR-Seg %d/%d frames  (%.0f%%)  %.1f fps  ETA %.0f s",
                        _frames_seen, total_frames, pct, fps_live, eta_s,
                    )
                else:
                    logger.info(
                        "[visual]  RF-DETR-Seg %d frames processed  %.1f fps",
                        _frames_seen, fps_live,
                    )
                _last_log_frame = _frames_seen
    finally:
        detector.unload()

    pipeline_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000.0
    det_metrics = detector.metrics
    trk_metrics = tracker.metrics
    effective_fps = (
        det_metrics.frames_processed / (pipeline_elapsed_ms / 1000.0)
        if pipeline_elapsed_ms > 0
        else 0.0
    )

    runtime_metrics = {
        "detector_backend": config.detector_backend,
        "detector_model": f"rfdetr_{config.detector_model}",
        "tracker_backend": f"bytetrack",
        "inference_backend": config.detector_backend,
        "batch_size": config.detector_batch_size,
        "frame_decode_backend": config.frame_decode_backend,
        "gpu_decode_backend": config.gpu_decode_backend,
        "shape": config.detector_resolution,
        "half_precision": config.use_fp16,
        "tensor_rt_enabled": False,
        "frames_processed": det_metrics.frames_processed,
        "mean_detector_latency_ms": round(det_metrics.mean_detector_latency_ms, 2),
        "mean_tracker_latency_ms": round(trk_metrics.mean_tracker_latency_ms, 2),
        "effective_fps": round(effective_fps, 1),
        "warmup_ms": round(det_metrics.warmup_ms, 1),
        "pipeline_elapsed_ms": round(pipeline_elapsed_ms, 1),
        "tracker_resets_at_shot_boundaries": int(getattr(trk_metrics, "resets", 0)),
        "raw_detection_rows": len(all_raw_detection_rows),
        "segmentation_enabled": True,
        "mask_rows": int(getattr(det_metrics, "mask_rows", 0))
        or sum(1 for row in all_track_rows if row.get("mask_rle")),
        "mask_encoding": MASK_RLE_ENCODING,
        "mask_output_tensor": getattr(det_metrics, "mask_output_tensor", None),
    }
    return all_track_rows, all_raw_detection_rows, runtime_metrics


# ---------------------------------------------------------------------------
# Shot segment builder
# ---------------------------------------------------------------------------

def _build_shot_segments(*, boundaries_ms: list[int], duration_ms: int) -> list[dict]:
    ordered = [value for value in boundaries_ms if 0 < int(value) < duration_ms]
    segments: list[dict] = []
    start_ms = 0
    for boundary_ms in ordered:
        segments.append(
            {
                "start_time_ms": int(start_ms),
                "end_time_ms": int(boundary_ms),
            }
        )
        start_ms = int(boundary_ms)
    segments.append(
        {
            "start_time_ms": int(start_ms),
            "end_time_ms": max(0, int(duration_ms)),
        }
    )
    return segments


# ---------------------------------------------------------------------------
# Track normalization (shared by both live and injected paths)
# ---------------------------------------------------------------------------

def _stable_positive_int_from_string(value: str) -> int:
    digest = hashlib.sha256(
        json.dumps({"value": value}, sort_keys=True).encode("utf-8")
    ).digest()
    return max(1, int.from_bytes(digest[:8], "big") % (2**31 - 1))


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize_track_id(raw_track_id: object, *, local_track_id: object | None) -> tuple[str, int]:
    if local_track_id is not None:
        try:
            parsed_local_track_id = int(local_track_id)
            return f"track_{parsed_local_track_id}", parsed_local_track_id
        except (TypeError, ValueError):
            pass
    track_text = str(raw_track_id).strip()
    if track_text.startswith("track_"):
        suffix = track_text.removeprefix("track_")
        if suffix.isdigit():
            return track_text, int(suffix)
    if track_text.isdigit():
        return f"track_{int(track_text)}", int(track_text)
    derived_local_track_id = _stable_positive_int_from_string(track_text or "track")
    return f"track_{derived_local_track_id}", derived_local_track_id


def _normalize_track_row(*, track: dict, metadata: dict) -> dict:
    frame_width = max(1, int(metadata.get("width") or 1))
    frame_height = max(1, int(metadata.get("height") or 1))
    track_id, local_track_id = _normalize_track_id(
        track.get("track_id"),
        local_track_id=track.get("local_track_id"),
    )
    x1 = _clamp(float(track.get("x1") or 0.0), minimum=0.0, maximum=float(frame_width))
    y1 = _clamp(float(track.get("y1") or 0.0), minimum=0.0, maximum=float(frame_height))
    x2 = _clamp(float(track.get("x2") or x1), minimum=x1, maximum=float(frame_width))
    y2 = _clamp(float(track.get("y2") or y1), minimum=y1, maximum=float(frame_height))
    width = max(0.0, float(track.get("width") or (x2 - x1)))
    height = max(0.0, float(track.get("height") or (y2 - y1)))
    x_center = float(track.get("x_center") or (x1 + (width / 2.0)))
    y_center = float(track.get("y_center") or (y1 + (height / 2.0)))
    bbox_norm_xywh = dict(track.get("bbox_norm_xywh") or {})
    normalized = {
        "frame_idx": int(track.get("frame_idx") or 0),
        "local_frame_idx": int(track.get("local_frame_idx", track.get("frame_idx") or 0)),
        "chunk_idx": int(track.get("chunk_idx") or 0),
        "track_id": track_id,
        "local_track_id": int(local_track_id),
        "class_id": int(track.get("class_id") or 0),
        "label": str(track.get("label") or "person"),
        "confidence": float(track.get("confidence") or 0.0),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "source": str(track.get("source") or "detector"),
        "geometry_type": str(track.get("geometry_type") or "aabb"),
        "bbox_norm_xywh": {
            "x_center": float(bbox_norm_xywh.get("x_center", x_center / frame_width)),
            "y_center": float(bbox_norm_xywh.get("y_center", y_center / frame_height)),
            "width": float(bbox_norm_xywh.get("width", width / frame_width)),
            "height": float(bbox_norm_xywh.get("height", height / frame_height)),
        },
    }
    if isinstance(track.get("mask_rle"), dict):
        normalized["mask_rle"] = dict(track["mask_rle"])
    return normalized


def _normalize_bbox_xyxy(*, x1: float, y1: float, x2: float, y2: float, frame_width: int, frame_height: int) -> dict:
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    return {
        "left": float(x1) / width,
        "top": float(y1) / height,
        "right": float(x2) / width,
        "bottom": float(y2) / height,
    }


# ---------------------------------------------------------------------------
# Person detections builder
# ---------------------------------------------------------------------------

def _build_person_detections(*, tracks: list[dict], metadata: dict) -> list[dict]:
    fps = float(metadata.get("fps") or 30.0)
    frame_width = max(1, int(metadata.get("width") or 1))
    frame_height = max(1, int(metadata.get("height") or 1))
    tracks_by_id: dict[str, list[dict]] = {}
    for track in tracks:
        tracks_by_id.setdefault(str(track["track_id"]), []).append(track)

    person_detections: list[dict] = []
    for index, track_id in enumerate(tracks_by_id):
        ordered_tracks = sorted(tracks_by_id[track_id], key=lambda item: int(item["frame_idx"]))
        timestamped_objects = []
        for track in ordered_tracks:
            timestamp_ms = frame_time_ms(int(track["frame_idx"]), video_fps=fps)
            timestamped_object = {
                "time_ms": timestamp_ms,
                "track_id": track_id,
                "confidence": float(track.get("confidence") or 0.0),
                "bounding_box": _normalize_bbox_xyxy(
                    x1=float(track["x1"]),
                    y1=float(track["y1"]),
                    x2=float(track["x2"]),
                    y2=float(track["y2"]),
                    frame_width=frame_width,
                    frame_height=frame_height,
                ),
                "source": "person_tracker",
                "provenance": "v31_visual_extractor",
            }
            if isinstance(track.get("mask_rle"), dict):
                timestamped_object["mask_rle"] = dict(track["mask_rle"])
            timestamped_objects.append(timestamped_object)
        if not timestamped_objects:
            continue
        person_detections.append(
            {
                "track_id": track_id,
                "confidence": (
                    sum(float(item["confidence"]) for item in timestamped_objects)
                    / max(1, len(timestamped_objects))
                ),
                "segment_start_ms": int(timestamped_objects[0]["time_ms"]),
                "segment_end_ms": int(timestamped_objects[-1]["time_ms"]),
                "person_track_index": index,
                "source": "person_tracker",
                "provenance": "v31_visual_extractor",
                "timestamped_objects": timestamped_objects,
            }
        )
    return person_detections


def _apply_pose_subject_reports(
    *,
    tracks: list[dict],
    reports: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict], dict[str, int]]:
    if not reports:
        return [dict(track) for track in tracks], [], {
            "pose_validated_tracklets": 0,
            "pose_auto_follow_eligible_tracklets": 0,
        }
    enriched_tracks: list[dict] = []
    for track in tracks:
        track_id = str(track.get("track_id"))
        report = reports.get(track_id)
        enriched = dict(track)
        if report is not None:
            enriched["auto_follow_eligible"] = bool(report.get("auto_follow_eligible"))
            enriched["subject_quality"] = dict(report.get("subject_quality") or {})
        enriched_tracks.append(enriched)

    identities: list[dict] = []
    for track_id in sorted(reports):
        report = reports[track_id]
        identities.append(
            {
                "track_id": track_id,
                "auto_follow_eligible": bool(report.get("auto_follow_eligible")),
                "subject_quality": dict(report.get("subject_quality") or {}),
                "source": "pose_subject_validator",
                "provenance": "v31_visual_extractor",
            }
        )
    metrics = {
        "pose_validated_tracklets": len(reports),
        "pose_auto_follow_eligible_tracklets": sum(
            1 for report in reports.values() if bool(report.get("auto_follow_eligible"))
        ),
    }
    return enriched_tracks, identities, metrics


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class V31VisualExtractor:
    """Orchestrates the full visual extraction pipeline.

    In production the default tracker_runner uses RF-DETR-Seg + ByteTrack on GPU.
    For testing, inject a callable tracker_runner that returns raw track dicts.
    """

    def __init__(
        self,
        *,
        metadata_probe=None,
        shot_detector=None,
        tracker_runner=None,
        pose_validator=None,
        visual_config=None,
    ) -> None:
        self._metadata_probe = metadata_probe or probe_video_metadata
        self._shot_detector = shot_detector or detect_shot_boundaries_ms
        self._tracker_runner = tracker_runner
        self._pose_validator = pose_validator
        self._visual_config = visual_config

    def _default_tracker_runner(
        self,
        *,
        video_path: Path,
        shot_changes: list[dict[str, Any]],
        fps: float,
    ) -> list[dict]:
        """Run the live RF-DETR-Seg + ByteTrack pipeline."""
        from .visual_config import VisualPipelineConfig

        config = self._visual_config or VisualPipelineConfig.from_env()
        tracks, self._last_raw_detections, self._last_runtime_metrics = _run_rfdetr_tracking_pipeline(
            video_path=video_path,
            config=config,
            shot_segments=shot_changes,
            video_fps=fps,
        )
        return tracks

    def extract(self, *, video_path: Path, workspace) -> dict:
        from .visual_config import VisualPipelineConfig

        self._last_runtime_metrics: dict[str, Any] = {}
        self._last_raw_detections: list[dict] = []
        config = self._visual_config or VisualPipelineConfig.from_env()

        logger.info("[visual]  probing metadata: %s", video_path.name)
        metadata = dict(self._metadata_probe(video_path=video_path))
        duration_ms = int(metadata.get("duration_ms") or 0)
        fps = float(metadata.get("fps") or 30.0)
        logger.info(
            "[visual]  video: %dx%d  %.2f fps  %.1f s  (%.0f est. frames)",
            metadata.get("width", 0),
            metadata.get("height", 0),
            fps,
            duration_ms / 1000.0,
            duration_ms / 1000.0 * fps,
        )

        logger.info("[visual]  detecting shot boundaries ...")
        t_shots = time.perf_counter()
        boundaries_ms = list(
            self._shot_detector(
                video_path=video_path,
                duration_ms=duration_ms,
            )
        )
        shot_changes = _build_shot_segments(
            boundaries_ms=boundaries_ms,
            duration_ms=duration_ms,
        )
        logger.info(
            "[visual]  %d shot boundaries → %d segments (%.1f s)",
            len(boundaries_ms),
            len(shot_changes),
            time.perf_counter() - t_shots,
        )

        if self._tracker_runner is not None:
            raw_tracks = list(self._tracker_runner(video_path=video_path))
        else:
            raw_tracks = self._default_tracker_runner(
                video_path=video_path,
                shot_changes=shot_changes,
                fps=fps,
            )

        normalized_tracks = [
            _normalize_track_row(track=track, metadata=metadata)
            for track in raw_tracks
        ]
        tracks, split_metrics = split_tracks_at_shot_boundaries(
            normalized_tracks,
            shot_timeline_ms=shot_changes,
            video_fps=fps,
        )
        pose_reports: dict[str, dict[str, Any]] = {}
        pose_validator = self._pose_validator
        if (
            pose_validator is None
            and self._tracker_runner is None
            and config.pose_validation_enabled
            and tracks
        ):
            from .pose_subject_validator import YoloPoseSubjectValidator

            pose_validator = YoloPoseSubjectValidator(config=config)
        if pose_validator is not None and tracks:
            pose_reports = dict(
                pose_validator(
                    video_path=video_path,
                    tracks=tracks,
                    metadata=metadata,
                    config=config,
                )
            )
            tracks, visual_identities, pose_metrics = _apply_pose_subject_reports(
                tracks=tracks,
                reports=pose_reports,
            )
        else:
            visual_identities = []
            pose_metrics = {
                "pose_validated_tracklets": 0,
                "pose_auto_follow_eligible_tracklets": 0,
            }
        person_detections = _build_person_detections(tracks=tracks, metadata=metadata)
        logger.info(
            "[visual]  done — %d raw track rows → %d tracks → %d person segments",
            len(raw_tracks),
            len(tracks),
            len(person_detections),
        )

        tracking_metrics = {
            "tracker_backend": f"rfdetr_{config.detector_model}_bytetrack",
            "input_track_rows": len(raw_tracks),
            "emitted_track_rows": len(tracks),
            "emitted_person_detection_segments": len(person_detections),
            "shot_count": len(shot_changes),
            **split_metrics,
            **pose_metrics,
            **self._last_runtime_metrics,
        }
        return {
            "video_metadata": metadata,
            "shot_changes": shot_changes,
            "tracks": tracks,
            "raw_person_detections": self._last_raw_detections,
            "person_detections": person_detections,
            "face_detections": [],
            "visual_identities": visual_identities,
            "mask_stability_signals": [],
            "tracking_metrics": tracking_metrics,
        }


SimpleVisualExtractor = V31VisualExtractor


__all__ = [
    "SimpleVisualExtractor",
    "V31VisualExtractor",
    "detect_shot_boundaries_ms",
    "probe_video_metadata",
]
