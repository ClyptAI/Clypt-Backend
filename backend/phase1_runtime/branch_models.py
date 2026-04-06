from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class BranchKind(str, Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    YAMNET = "yamnet"


class BranchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    run_id: str
    branch: BranchKind
    source_url: str | None = None
    source_path: str | None = None
    runtime_controls: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "BranchRequest":
        if bool(self.source_url) == bool(self.source_path):
            raise ValueError("Provide exactly one of source_url or source_path")
        return self


class BranchResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: BranchKind
    ok: bool = True
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_result_state(self) -> "BranchResultEnvelope":
        if self.ok:
            if self.result is None or self.error is not None:
                raise ValueError("ok=True requires result and forbids error")
        else:
            if self.error is None or self.result is not None:
                raise ValueError("ok=False requires error and forbids result")
        return self


class BranchStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: BranchKind
    state: Literal["queued", "running", "succeeded", "failed"]
    message: str | None = None
    pid: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


__all__ = [
    "BranchKind",
    "BranchRequest",
    "BranchResultEnvelope",
    "BranchStatus",
]
