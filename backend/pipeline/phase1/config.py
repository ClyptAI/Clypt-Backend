from __future__ import annotations

from dataclasses import dataclass
import os


def _get_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    return value


def _get_float(name: str, default: float, *, min_value: float | None = None) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    return value


def _get_str(name: str, default: str) -> str:
    raw = str(os.getenv(name, default)).strip()
    return raw or default


@dataclass(frozen=True)
class Phase1Config:
    yolo_weights_path: str
    yolo_imgsz: int
    shot_scene_threshold: float
    cluster_attach_max_gap_frames: int
    cluster_attach_gap_weight: float
    cluster_attach_ambiguity_margin: float
    hist_attach_temp_weight: float
    hist_attach_unassign_cost: float
    overlap_follow_min_confidence: float
    overlap_follow_min_evidence_score: int


def get_phase1_config() -> Phase1Config:
    return Phase1Config(
        yolo_weights_path=_get_str("YOLO_WEIGHTS_PATH", "yolo26m.pt"),
        yolo_imgsz=_get_int("CLYPT_YOLO_IMGSZ", 1080, min_value=320),
        shot_scene_threshold=_get_float("CLYPT_SHOT_SCENE_THRESHOLD", 0.35, min_value=0.0),
        cluster_attach_max_gap_frames=_get_int("CLYPT_CLUSTER_ATTACH_MAX_GAP_FRAMES", 180, min_value=0),
        cluster_attach_gap_weight=_get_float("CLYPT_CLUSTER_ATTACH_GAP_WEIGHT", 0.35, min_value=0.0),
        cluster_attach_ambiguity_margin=_get_float("CLYPT_CLUSTER_ATTACH_AMBIGUITY_MARGIN", 0.05, min_value=0.0),
        hist_attach_temp_weight=_get_float("CLYPT_HIST_ATTACH_TEMP_WEIGHT", 0.35, min_value=0.0),
        hist_attach_unassign_cost=_get_float("CLYPT_HIST_ATTACH_UNASSIGN_COST", 2.5, min_value=0.0),
        overlap_follow_min_confidence=_get_float("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", 0.0, min_value=0.0),
        overlap_follow_min_evidence_score=_get_int("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", 0, min_value=0),
    )

