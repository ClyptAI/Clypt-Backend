from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.common.domain_enums import JobStatus
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase1JobCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str | None = None
    source_path: str | None = None
    runtime_controls: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "Phase1JobCreatePayload":
        if bool(self.source_url) == bool(self.source_path):
            raise ValueError("Provide exactly one of source_url or source_path")
        return self


class Phase1JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    source_url: str | None = None
    source_path: str | None = None
    runtime_controls: dict[str, Any] | None = None
    status: JobStatus
    retries: int = 0
    claim_token: str | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    current_step: str | None = None
    progress_message: str | None = None
    progress_pct: float | None = None
    log_path: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class Phase1Workspace:
    run_id: str
    root: Path
    video_path: Path
    audio_path: Path
    metadata_dir: Path

    @classmethod
    def create(cls, *, root: Path, run_id: str) -> "Phase1Workspace":
        workspace_root = root / run_id
        metadata_dir = workspace_root / "metadata"
        workspace_root.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            root=workspace_root,
            video_path=workspace_root / "source_video.mp4",
            audio_path=workspace_root / "source_audio.wav",
            metadata_dir=metadata_dir,
        )


@dataclass(frozen=True, slots=True)
class Phase1SidecarOutputs:
    phase1_audio: dict
    diarization_payload: dict
    phase1_visual: dict
    emotion2vec_payload: dict
    yamnet_payload: dict


__all__ = [
    "Phase1JobCreatePayload",
    "Phase1JobRecord",
    "Phase1SidecarOutputs",
    "Phase1Workspace",
]
