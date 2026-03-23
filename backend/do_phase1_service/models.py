from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.pipeline.phase1_contract import GcsUriStr, JobState, Phase1Manifest


class JobCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex}")
    source_url: str
    status: JobState | Literal["queued", "running", "succeeded", "failed"]
    retries: int = 0
    claim_token: str | None = None
    manifest: dict[str, Any] | None = None
    manifest_uri: str | None = None
    failure: dict[str, Any] | None = None
    current_step: str | None = None
    progress_message: str | None = None
    progress_pct: float | None = None
    log_path: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class PersistedPhase1Manifest(Phase1Manifest):
    manifest_uri: GcsUriStr
