from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Phase1SourceVideo(BaseModel):
    source_url: str


class Phase1TranscriptArtifact(BaseModel):
    uri: str
    words: list[dict]
    speaker_bindings: list[dict]


class Phase1VisualArtifact(BaseModel):
    uri: str
    tracks: list[dict]
    detection_blocks: list[dict]


class Phase1EventsArtifact(BaseModel):
    uri: str
    events: list[dict]


class Phase1Artifacts(BaseModel):
    transcript: Phase1TranscriptArtifact
    visual_tracking: Phase1VisualArtifact
    events: Phase1EventsArtifact | None = None


class Phase1RuntimeMetadata(BaseModel):
    provider: str
    worker_id: str | None = None
    region: str | None = None


class Phase1TimingsMetadata(BaseModel):
    ingest_ms: int | None = None
    processing_ms: int | None = None
    upload_ms: int | None = None


class Phase1QualityMetrics(BaseModel):
    schema_pass_rate: float | None = None
    transcript_coverage: float | None = None
    tracking_confidence: float | None = None


class Phase1RetryMetadata(BaseModel):
    attempts: int | None = None
    max_attempts: int | None = None
    last_error: str | None = None


class Phase1FailureMetadata(BaseModel):
    error_type: str | None = None
    error_message: str | None = None
    failed_step: str | None = None


class Phase1Metadata(BaseModel):
    runtime: Phase1RuntimeMetadata
    timings: Phase1TimingsMetadata
    quality_metrics: Phase1QualityMetrics | None = None
    retry: Phase1RetryMetadata | None = None
    failure: Phase1FailureMetadata | None = None


class Phase1Manifest(BaseModel):
    contract_version: str
    job_id: str
    status: JobState
    source_video: Phase1SourceVideo
    canonical_video_gcs_uri: str
    artifacts: Phase1Artifacts
    metadata: Phase1Metadata
