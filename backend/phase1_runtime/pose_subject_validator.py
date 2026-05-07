from __future__ import annotations

import logging
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .frame_decode import decode_video_frames
from .masks import LOWRES_MASK_ARTIFACT_ID, MaskArtifactReader

logger = logging.getLogger(__name__)

HEAD_KEYPOINT_INDICES = (0, 1, 2, 3, 4)
SHOULDER_KEYPOINT_INDICES = (5, 6)
_KEYPOINT_WEIGHTS = {
    0: 2.0,
    1: 2.0,
    2: 2.0,
    3: 1.75,
    4: 1.75,
    5: 1.5,
    6: 1.5,
    7: 1.0,
    8: 1.0,
    9: 0.75,
    10: 0.75,
    11: 1.0,
    12: 1.0,
    13: 0.5,
    14: 0.5,
    15: 0.25,
    16: 0.25,
}


@dataclass(frozen=True, slots=True)
class PoseSampleEvidence:
    frame_idx: int
    has_head_evidence: bool
    has_upper_body_anchor: bool
    head_center_xy: tuple[float, float] | None = None
    shoulder_center_xy: tuple[float, float] | None = None
    upper_torso_anchor_xy: tuple[float, float] | None = None


def evaluate_tracklet_pose_quality(
    *,
    track_id: str,
    rfdetr_confidences: Iterable[float],
    samples: list[PoseSampleEvidence],
    min_rfdetr_confidence: float,
    min_head_evidence_ratio: float,
    min_upper_body_anchor_ratio: float,
    pose_backend: str = "yolo11s_pose_tensorrt",
) -> dict[str, Any]:
    confidences = [float(value) for value in rfdetr_confidences]
    median_confidence = statistics.median(confidences) if confidences else 0.0
    sampled_frames = len(samples)
    head_count = sum(1 for sample in samples if sample.has_head_evidence)
    upper_count = sum(1 for sample in samples if sample.has_upper_body_anchor)
    head_ratio = head_count / sampled_frames if sampled_frames else 0.0
    upper_ratio = upper_count / sampled_frames if sampled_frames else 0.0
    auto_follow_eligible = (
        median_confidence >= float(min_rfdetr_confidence)
        and sampled_frames > 0
        and head_ratio >= float(min_head_evidence_ratio)
        and upper_ratio >= float(min_upper_body_anchor_ratio)
    )
    return {
        "auto_follow_eligible": bool(auto_follow_eligible),
        "subject_quality": {
            "pose_backend": pose_backend,
            "sampled_frames": int(sampled_frames),
            "head_evidence_ratio": float(head_ratio),
            "upper_body_anchor_ratio": float(upper_ratio),
            "median_rfdetr_confidence": float(median_confidence),
            "track_id": str(track_id),
            "pose_anchor_points": [
                {
                    "frame_idx": int(sample.frame_idx),
                    "head_center_xy": list(sample.head_center_xy) if sample.head_center_xy else None,
                    "shoulder_center_xy": (
                        list(sample.shoulder_center_xy) if sample.shoulder_center_xy else None
                    ),
                    "upper_torso_anchor_xy": (
                        list(sample.upper_torso_anchor_xy) if sample.upper_torso_anchor_xy else None
                    ),
                }
                for sample in samples
                if sample.head_center_xy or sample.shoulder_center_xy or sample.upper_torso_anchor_xy
            ],
        },
    }


def _sample_track_rows(rows: list[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row.get("frame_idx", 0)))
    if len(ordered) <= max_samples:
        return ordered
    if max_samples <= 1:
        return [ordered[len(ordered) // 2]]
    indices = np.linspace(0, len(ordered) - 1, num=max_samples)
    return [ordered[int(round(index))] for index in indices]


def _crop_frame(
    frame: np.ndarray,
    track: dict[str, Any],
    *,
    padding_ratio: float,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    height, width = frame.shape[:2]
    x1 = float(track.get("x1") or 0.0)
    y1 = float(track.get("y1") or 0.0)
    x2 = float(track.get("x2") or x1)
    y2 = float(track.get("y2") or y1)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < 2.0 or box_height < 2.0:
        return None
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy(), (left, top)


def _center_from_keypoints(
    instance: np.ndarray,
    indices: tuple[int, ...],
    *,
    keypoint_confidence: float,
    offset_xy: tuple[int, int],
) -> tuple[float, float] | None:
    points = instance[indices, :]
    valid = points[:, 2] >= float(keypoint_confidence)
    if not np.any(valid):
        return None
    coords = points[:, :2][valid]
    if coords.size == 0:
        return None
    left, top = offset_xy
    center = np.mean(coords, axis=0)
    return float(center[0] + left), float(center[1] + top)


def _keypoints_data_from_result(result: Any) -> np.ndarray | None:
    keypoints = getattr(result, "keypoints", None)
    data = getattr(keypoints, "data", None)
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)
    if data.size == 0 or data.ndim != 3 or data.shape[1] < 7 or data.shape[2] < 3:
        return None
    return data


def _pose_evidence_from_instance(
    instance: np.ndarray,
    *,
    keypoint_confidence: float,
    offset_xy: tuple[int, int] = (0, 0),
) -> PoseSampleEvidence:
    head_conf = instance[HEAD_KEYPOINT_INDICES, 2]
    shoulder_conf = instance[SHOULDER_KEYPOINT_INDICES, 2]
    head_center = _center_from_keypoints(
        instance,
        HEAD_KEYPOINT_INDICES,
        keypoint_confidence=keypoint_confidence,
        offset_xy=offset_xy,
    )
    shoulder_center = _center_from_keypoints(
        instance,
        SHOULDER_KEYPOINT_INDICES,
        keypoint_confidence=keypoint_confidence,
        offset_xy=offset_xy,
    )
    upper_torso_anchor = None
    if head_center and shoulder_center:
        upper_torso_anchor = (
            (head_center[0] * 0.35) + (shoulder_center[0] * 0.65),
            (head_center[1] * 0.35) + (shoulder_center[1] * 0.65),
        )
    return PoseSampleEvidence(
        frame_idx=-1,
        has_head_evidence=bool(np.any(head_conf >= float(keypoint_confidence))),
        has_upper_body_anchor=bool(np.any(shoulder_conf >= float(keypoint_confidence))),
        head_center_xy=head_center,
        shoulder_center_xy=shoulder_center,
        upper_torso_anchor_xy=upper_torso_anchor,
    )


def _source_point_inside_mask(
    *,
    point_xy: tuple[float, float],
    mask: np.ndarray,
    mask_bbox_xyxy: list[float],
) -> bool:
    x, y = point_xy
    x1, y1, x2, y2 = [float(value) for value in mask_bbox_xyxy[:4]]
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    u = (float(x) - x1) / width
    v = (float(y) - y1) / height
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return False
    mask_height, mask_width = mask.shape[:2]
    mask_x = int(round(u * max(0, mask_width - 1)))
    mask_y = int(round(v * max(0, mask_height - 1)))
    return bool(mask[mask_y, mask_x])


def _keypoint_weight(index: int) -> float:
    return float(_KEYPOINT_WEIGHTS.get(int(index), 0.5))


def _anchor_for_evidence(evidence: PoseSampleEvidence) -> tuple[float, float] | None:
    return (
        evidence.head_center_xy
        or evidence.upper_torso_anchor_xy
        or evidence.shoulder_center_xy
    )


def _anchor_distance(
    anchor_xy: tuple[float, float] | None,
    previous_anchor_xy: tuple[float, float] | None,
) -> float:
    if anchor_xy is None or previous_anchor_xy is None:
        return float("inf")
    dx = float(anchor_xy[0]) - float(previous_anchor_xy[0])
    dy = float(anchor_xy[1]) - float(previous_anchor_xy[1])
    return float(np.hypot(dx, dy))


def _select_pose_evidence_for_track(
    *,
    result: Any,
    keypoint_confidence: float,
    offset_xy: tuple[int, int],
    track: dict[str, Any],
    mask_reader: MaskArtifactReader | None,
    previous_anchor_xy: tuple[float, float] | None,
) -> PoseSampleEvidence:
    data = _keypoints_data_from_result(result)
    if data is None:
        return PoseSampleEvidence(
            frame_idx=-1,
            has_head_evidence=False,
            has_upper_body_anchor=False,
        )
    if data.shape[0] == 1:
        return _pose_evidence_from_instance(
            data[0],
            keypoint_confidence=keypoint_confidence,
            offset_xy=offset_xy,
        )

    mask_ref = track.get("mask_ref")
    if not isinstance(mask_ref, dict):
        raise RuntimeError(
            "YOLO pose returned multiple instances for a tracked person row without mask_ref. "
            "RF-DETR-Seg mask selection is required for disambiguation."
        )
    if mask_reader is None:
        raise RuntimeError(
            "YOLO pose returned multiple instances but no visual mask reader was available."
        )
    mask = mask_reader.get(mask_ref)
    mask_bbox_xyxy = list(mask_ref.get("bbox_xyxy") or [])
    if len(mask_bbox_xyxy) < 4:
        raise RuntimeError("mask_ref missing bbox_xyxy for pose-instance disambiguation")

    candidates: list[tuple[tuple[float, int, int, int, float, float], PoseSampleEvidence]] = []
    for instance in data:
        evidence = _pose_evidence_from_instance(
            instance,
            keypoint_confidence=keypoint_confidence,
            offset_xy=offset_xy,
        )
        weighted_total = 0.0
        weighted_inside = 0.0
        inside_points = 0
        head_inside = 0
        upper_inside = 0
        confidence_total = 0.0
        for index in range(instance.shape[0]):
            confidence = float(instance[index, 2])
            if confidence < float(keypoint_confidence):
                continue
            weight = _keypoint_weight(index)
            weighted_total += weight
            confidence_total += confidence
            point_xy = (
                float(instance[index, 0]) + float(offset_xy[0]),
                float(instance[index, 1]) + float(offset_xy[1]),
            )
            if not _source_point_inside_mask(
                point_xy=point_xy,
                mask=mask,
                mask_bbox_xyxy=mask_bbox_xyxy,
            ):
                continue
            weighted_inside += weight
            inside_points += 1
            if index in HEAD_KEYPOINT_INDICES:
                head_inside = 1
            if index in SHOULDER_KEYPOINT_INDICES:
                upper_inside = 1
        if weighted_total <= 0.0 or inside_points <= 0:
            continue
        anchor_distance = _anchor_distance(_anchor_for_evidence(evidence), previous_anchor_xy)
        candidates.append(
            (
                (
                    weighted_inside / weighted_total,
                    head_inside,
                    upper_inside,
                    inside_points,
                    -anchor_distance if np.isfinite(anchor_distance) else -1e12,
                    confidence_total,
                ),
                evidence,
            )
        )

    if not candidates:
        return PoseSampleEvidence(
            frame_idx=-1,
            has_head_evidence=False,
            has_upper_body_anchor=False,
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _pose_evidence_from_result(
    result: Any,
    *,
    keypoint_confidence: float,
    offset_xy: tuple[int, int] = (0, 0),
) -> PoseSampleEvidence:
    data = _keypoints_data_from_result(result)
    if data is None:
        return PoseSampleEvidence(
            frame_idx=-1,
            has_head_evidence=False,
            has_upper_body_anchor=False,
        )
    if data.shape[0] != 1:
        raise RuntimeError(
            "_pose_evidence_from_result requires exactly one pose instance; "
            "use track-mask selection before computing pose anchors."
        )
    return _pose_evidence_from_instance(
        data[0],
        keypoint_confidence=keypoint_confidence,
        offset_xy=offset_xy,
    )


class YoloPoseSubjectValidator:
    def __init__(self, *, config: Any) -> None:
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO11s-pose subject validation on Modal L40S."
            ) from exc

        engine_path = Path(self._config.pose_engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        if not engine_path.exists():
            source_model = str(self._config.pose_model_path)
            logger.info("[visual] exporting YOLO pose TensorRT engine from %s", source_model)
            exported = YOLO(source_model).export(
                format="engine",
                half=True,
                imgsz=int(self._config.pose_imgsz),
                batch=int(self._config.pose_batch_size),
                dynamic=True,
                device=0,
                verbose=False,
            )
            exported_path = Path(str(exported))
            if not exported_path.exists():
                raise RuntimeError(f"YOLO pose TensorRT export did not create {exported_path}")
            if exported_path.resolve() != engine_path.resolve():
                shutil.copy2(exported_path, engine_path)
        self._model = YOLO(str(engine_path))
        return self._model

    def __call__(
        self,
        *,
        video_path: Path,
        tracks: list[dict[str, Any]],
        metadata: dict[str, Any],
        config: Any,
    ) -> dict[str, dict[str, Any]]:
        del metadata
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for track in tracks:
            grouped[str(track.get("track_id"))].append(track)

        sampled_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        max_samples = int(config.pose_max_samples_per_tracklet)
        for rows in grouped.values():
            for row in _sample_track_rows(rows, max_samples=max_samples):
                frame_idx = int(row.get("frame_idx", 0))
                sampled_by_frame[frame_idx].append(row)

        sample_evidence: dict[str, list[PoseSampleEvidence]] = defaultdict(list)
        if sampled_by_frame:
            model = self._load_model()
            crop_batch: list[np.ndarray] = []
            crop_meta: list[tuple[str, int, tuple[int, int], dict[str, Any]]] = []
            previous_anchor_by_track: dict[str, tuple[float, float]] = {}
            mask_reader: MaskArtifactReader | None = None
            if any(
                isinstance(track.get("mask_ref"), dict)
                for rows in grouped.values()
                for track in rows
            ):
                artifact_path = Path(video_path).parent / f"{LOWRES_MASK_ARTIFACT_ID}.npz"
                mask_reader = MaskArtifactReader(artifact_path=artifact_path)

            def flush_batch() -> None:
                if not crop_batch:
                    return
                results = model.predict(
                    crop_batch,
                    imgsz=int(config.pose_imgsz),
                    conf=float(config.pose_confidence),
                    verbose=False,
                )
                for result, (track_id, frame_idx, offset_xy, track) in zip(results, crop_meta, strict=True):
                    evidence = _select_pose_evidence_for_track(
                        result=result,
                        keypoint_confidence=float(config.pose_keypoint_confidence),
                        offset_xy=offset_xy,
                        track=track,
                        mask_reader=mask_reader,
                        previous_anchor_xy=previous_anchor_by_track.get(track_id),
                    )
                    anchor_xy = _anchor_for_evidence(evidence)
                    if anchor_xy is not None:
                        previous_anchor_by_track[track_id] = anchor_xy
                    sample_evidence[track_id].append(
                        PoseSampleEvidence(
                            frame_idx=frame_idx,
                            has_head_evidence=evidence.has_head_evidence,
                            has_upper_body_anchor=evidence.has_upper_body_anchor,
                            head_center_xy=evidence.head_center_xy,
                            shoulder_center_xy=evidence.shoulder_center_xy,
                            upper_torso_anchor_xy=evidence.upper_torso_anchor_xy,
                        )
                    )
                crop_batch.clear()
                crop_meta.clear()

            wanted_frames = set(sampled_by_frame)
            for decoded in decode_video_frames(
                video_path=video_path,
                decode_backend=config.frame_decode_backend,
                gpu_decode_backend=config.gpu_decode_backend,
            ):
                if decoded.frame_idx not in wanted_frames:
                    continue
                for track in sampled_by_frame[decoded.frame_idx]:
                    track_id = str(track.get("track_id"))
                    cropped = _crop_frame(
                        decoded.rgb,
                        track,
                        padding_ratio=float(config.pose_crop_padding_ratio),
                    )
                    if cropped is None:
                        sample_evidence[track_id].append(
                            PoseSampleEvidence(
                                frame_idx=decoded.frame_idx,
                                has_head_evidence=False,
                                has_upper_body_anchor=False,
                            )
                        )
                        continue
                    crop, offset_xy = cropped
                    crop_batch.append(crop)
                    crop_meta.append((track_id, decoded.frame_idx, offset_xy, track))
                    if len(crop_batch) >= int(config.pose_batch_size):
                        flush_batch()
                if decoded.frame_idx >= max(wanted_frames):
                    break
            flush_batch()

        reports: dict[str, dict[str, Any]] = {}
        for track_id, rows in grouped.items():
            reports[track_id] = evaluate_tracklet_pose_quality(
                track_id=track_id,
                rfdetr_confidences=[float(row.get("confidence") or 0.0) for row in rows],
                samples=sample_evidence.get(track_id, []),
                min_rfdetr_confidence=float(config.pose_min_rfdetr_confidence),
                min_head_evidence_ratio=float(config.pose_min_head_evidence_ratio),
                min_upper_body_anchor_ratio=float(config.pose_min_upper_body_anchor_ratio),
            )
        return reports


__all__ = [
    "PoseSampleEvidence",
    "YoloPoseSubjectValidator",
    "evaluate_tracklet_pose_quality",
]
