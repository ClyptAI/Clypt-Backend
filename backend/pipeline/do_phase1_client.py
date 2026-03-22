from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from backend.pipeline.phase1_contract import JobState, Phase1Manifest


class Phase1ClientError(RuntimeError):
    """Base error for DigitalOcean Phase 1 client failures."""


class Phase1JobNotFoundError(Phase1ClientError):
    """Raised when the service reports that a job does not exist."""


class Phase1JobNotReadyError(Phase1ClientError):
    """Raised when a job result is requested before completion."""


class Phase1JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    source_url: str
    status: JobState
    retries: int = 0
    claim_token: str | None = None
    manifest: Phase1Manifest | None = None
    manifest_uri: str | None = None
    failure: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {JobState.SUCCEEDED, JobState.FAILED}


Phase1JobSubmission = Phase1JobRecord


class DOPhase1Client:
    """Async client for the DO Phase 1 job service.

    The service is intentionally long-running: submission only enqueues work,
    job state must be polled, and results are fetched separately once the job
    reaches a succeeded terminal state.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "DOPhase1Client":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def submit_job(self, source_url: str) -> Phase1JobSubmission:
        response = await self._client.post("/jobs", json={"source_url": source_url})
        response.raise_for_status()
        return Phase1JobSubmission.model_validate(response.json())

    async def get_job_status(self, job_id: str) -> Phase1JobRecord:
        response = await self._client.get(f"/jobs/{job_id}")
        if response.status_code == httpx.codes.NOT_FOUND:
            raise Phase1JobNotFoundError(f"phase 1 job {job_id!r} was not found")
        response.raise_for_status()
        return Phase1JobRecord.model_validate(response.json())

    async def get_job(self, job_id: str) -> Phase1JobRecord:
        return await self.get_job_status(job_id)

    async def get_result(self, job_id: str) -> Phase1Manifest:
        response = await self._client.get(f"/jobs/{job_id}/result")
        if response.status_code == httpx.codes.NOT_FOUND:
            raise Phase1JobNotFoundError(f"phase 1 job {job_id!r} was not found")
        if response.status_code == httpx.codes.CONFLICT:
            raise Phase1JobNotReadyError(
                f"phase 1 job {job_id!r} is not ready yet; poll job status until it succeeds"
            )
        response.raise_for_status()
        return Phase1Manifest.model_validate(response.json())


__all__ = [
    "DOPhase1Client",
    "Phase1ClientError",
    "Phase1JobNotFoundError",
    "Phase1JobNotReadyError",
    "Phase1JobRecord",
    "Phase1JobSubmission",
]
