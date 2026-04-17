from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase1AudioAssets(_StrictModel):
    source_audio: str | None = None
    video_gcs_uri: str | None = None
    audio_gcs_uri: str | None = None
    local_video_path: str | None = None
    local_audio_path: str | None = None

    def __getitem__(self, key: str) -> str | None:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


class DiarizationPayload(_StrictModel):
    turns: list[dict[str, Any]] = Field(default_factory=list)
    words: list[dict[str, Any]] = Field(default_factory=list)


class VisualPayload(_StrictModel):
    video_metadata: dict[str, Any] = Field(default_factory=dict)
    shot_changes: list[dict[str, Any]] = Field(default_factory=list)
    tracks: list[dict[str, Any]] = Field(default_factory=list)
    person_detections: list[dict[str, Any]] = Field(default_factory=list)
    face_detections: list[dict[str, Any]] = Field(default_factory=list)
    visual_identities: list[dict[str, Any]] = Field(default_factory=list)
    mask_stability_signals: list[dict[str, Any]] = Field(default_factory=list)
    tracking_metrics: dict[str, Any] = Field(default_factory=dict)


class EmotionSegmentsPayload(_StrictModel):
    segments: list[dict[str, Any]] = Field(default_factory=list)


class YamnetPayload(_StrictModel):
    events: list[dict[str, Any]] = Field(default_factory=list)


class Phase1SidecarOutputs(_StrictModel):
    phase1_audio: Phase1AudioAssets
    diarization_payload: DiarizationPayload
    phase1_visual: VisualPayload
    emotion2vec_payload: EmotionSegmentsPayload
    yamnet_payload: YamnetPayload


__all__ = [
    "DiarizationPayload",
    "EmotionSegmentsPayload",
    "Phase1AudioAssets",
    "Phase1SidecarOutputs",
    "VisualPayload",
    "YamnetPayload",
]
