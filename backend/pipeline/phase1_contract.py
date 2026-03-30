from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _validate_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"expected an http(s) URL, got {value!r}")
    return value


def _validate_gcs_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path or parsed.path == "/":
        raise ValueError(f"expected a gs:// URI, got {value!r}")
    return value


HttpUrlStr = Annotated[str, AfterValidator(_validate_http_url)]
GcsUriStr = Annotated[str, AfterValidator(_validate_gcs_uri)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Confidence01 = Annotated[float, Field(ge=0.0, le=1.0)]
Normalized01 = Annotated[float, Field(ge=0.0, le=1.0)]


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Phase1SourceVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: HttpUrlStr


class Phase1Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    speaker_track_id: str | None
    speaker_tag: str
    speaker_local_track_id: str | None = None
    speaker_local_tag: str | None = None
    calibrated_confidence: Confidence01 | None = None
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("word start_time_ms must be <= end_time_ms")
        return self


class Phase1SpeakerBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    word_count: NonNegativeInt

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("speaker binding start_time_ms must be <= end_time_ms")
        return self


class Phase1BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: Normalized01
    top: Normalized01
    right: Normalized01
    bottom: Normalized01

    @model_validator(mode="after")
    def _check_geometry(self):
        if self.left > self.right:
            raise ValueError("bounding box left must be <= right")
        if self.top > self.bottom:
            raise ValueError("bounding box top must be <= bottom")
        return self


class Phase1TimestampedObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_ms: NonNegativeInt
    track_id: str
    confidence: Confidence01
    bounding_box: Phase1BoundingBox
    source: str | None = None
    provenance: dict[str, object] | str | None = None


class Phase1DetectionSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    confidence: Confidence01
    segment_start_ms: NonNegativeInt
    segment_end_ms: NonNegativeInt
    face_track_index: NonNegativeInt | None = None
    person_track_index: NonNegativeInt | None = None
    object_track_index: NonNegativeInt | None = None
    label_track_index: NonNegativeInt | None = None
    source: str | None = None
    provenance: dict[str, object] | str | None = None
    timestamped_objects: list[Phase1TimestampedObject]

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.segment_start_ms > self.segment_end_ms:
            raise ValueError("segment_start_ms must be <= segment_end_ms")
        return self


class Phase1TrackBBoxNorm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_center: Normalized01
    y_center: Normalized01
    width: Normalized01
    height: Normalized01


class Phase1Track(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: NonNegativeInt
    local_frame_idx: NonNegativeInt
    chunk_idx: NonNegativeInt
    track_id: str
    local_track_id: NonNegativeInt
    class_id: NonNegativeInt
    label: str
    confidence: Confidence01
    x1: float
    y1: float
    x2: float
    y2: float
    x_center: float
    y_center: float
    width: float
    height: float
    source: str
    geometry_type: str
    bbox_norm_xywh: Phase1TrackBBoxNorm


class Phase1ShotChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("shot change start_time_ms must be <= end_time_ms")
        return self


class Phase1VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: PositiveInt
    height: PositiveInt
    fps: Annotated[float, Field(gt=0.0)]
    duration_ms: NonNegativeInt


class Phase1AudioSpeakerTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    exclusive: bool | None = None
    overlap: bool | None = None
    confidence: Confidence01 | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("audio speaker turn start_time_ms must be <= end_time_ms")
        return self


class Phase1AudioSpeakerLocalTrackMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    local_track_id: str
    support_segments: PositiveInt
    support_ms: PositiveInt
    confidence: Confidence01


class Phase1SpeakerCandidateDebugCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_track_id: str
    track_id: str
    blended_score: float
    asd_probability: Confidence01 | None = None
    body_prior: Confidence01 | None = None
    detection_confidence: Confidence01 | None = None


class Phase1SpeakerCandidateDebugEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str | None = None
    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    active_audio_speaker_id: str | None = None
    active_audio_local_track_id: str | None = None
    chosen_track_id: str | None = None
    chosen_local_track_id: str | None = None
    decision_source: Literal["visual", "audio_boosted_visual", "unknown"]
    ambiguous: bool
    top_1_top_2_margin: float | None = None
    calibrated_confidence: Confidence01 | None = None
    abstention_reason: str | None = None
    candidates: list[Phase1SpeakerCandidateDebugCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("speaker candidate debug start_time_ms must be <= end_time_ms")
        return self


class Phase1ActiveSpeakerLocalSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    audio_speaker_ids: list[str] = Field(default_factory=list)
    visible_local_track_ids: list[str] = Field(default_factory=list)
    visible_track_ids: list[str] = Field(default_factory=list)
    offscreen_audio_speaker_ids: list[str] = Field(default_factory=list)
    overlap: bool
    confidence: Confidence01 | None = None
    decision_source: str

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("active speaker span start_time_ms must be <= end_time_ms")
        return self


class Phase1OverlapFollowDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time_ms: NonNegativeInt
    end_time_ms: NonNegativeInt
    camera_target_local_track_id: str | None = None
    camera_target_track_id: str | None = None
    stay_wide: bool
    visible_local_track_ids: list[str] = Field(default_factory=list)
    offscreen_audio_speaker_ids: list[str] = Field(default_factory=list)
    decision_model: str | None = None
    decision_source: str
    confidence: Confidence01 | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("overlap follow decision start_time_ms must be <= end_time_ms")
        return self


class Phase1TranscriptArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: GcsUriStr
    source_audio: HttpUrlStr
    video_gcs_uri: GcsUriStr
    words: list[Phase1Word]
    speaker_bindings: list[Phase1SpeakerBinding]
    audio_speaker_turns: list[Phase1AudioSpeakerTurn] = Field(default_factory=list)
    speaker_bindings_local: list[Phase1SpeakerBinding] = Field(default_factory=list)
    speaker_follow_bindings_local: list[Phase1SpeakerBinding] = Field(default_factory=list)
    audio_speaker_local_track_map: list[Phase1AudioSpeakerLocalTrackMapEntry] = Field(default_factory=list)
    speaker_candidate_debug: list[Phase1SpeakerCandidateDebugEntry] = Field(default_factory=list)
    active_speakers_local: list[Phase1ActiveSpeakerLocalSpan] = Field(default_factory=list)
    overlap_follow_decisions: list[Phase1OverlapFollowDecision] = Field(default_factory=list)


class Phase1VisualArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: GcsUriStr
    source_video: HttpUrlStr
    video_gcs_uri: GcsUriStr
    schema_version: str
    task_type: str
    coordinate_space: str
    geometry_type: str
    class_taxonomy: dict[str, str]
    tracking_metrics: dict[str, object]
    tracks: list[Phase1Track]
    tracks_local: list[Phase1Track] = Field(default_factory=list)
    face_detections: list[Phase1DetectionSegment]
    person_detections: list[Phase1DetectionSegment]
    label_detections: list[Phase1DetectionSegment]
    object_tracking: list[Phase1DetectionSegment]
    shot_changes: list[Phase1ShotChange]
    video_metadata: Phase1VideoMetadata
    mask_stability_signals: dict[str, object] = Field(default_factory=dict)


class Phase1EventsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: GcsUriStr
    events: list[dict[str, object]]


class Phase1Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: Phase1TranscriptArtifact
    visual_tracking: Phase1VisualArtifact
    events: Phase1EventsArtifact | None = None


class Phase1RuntimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    worker_id: str | None = None
    region: str | None = None


class Phase1TimingsMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingest_ms: NonNegativeInt | None = None
    processing_ms: NonNegativeInt | None = None
    upload_ms: NonNegativeInt | None = None


class Phase1QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_pass_rate: Confidence01 | None = None
    transcript_coverage: Confidence01 | None = None
    tracking_confidence: Confidence01 | None = None


class Phase1RetryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: NonNegativeInt | None = None
    max_attempts: NonNegativeInt | None = None
    last_error: str | None = None


class Phase1FailureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str | None = None
    error_message: str | None = None
    failed_step: str | None = None


class Phase1Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: Phase1RuntimeMetadata
    timings: Phase1TimingsMetadata
    quality_metrics: Phase1QualityMetrics | None = None
    benchmark_scorecard: dict[str, object] | None = None
    retry: Phase1RetryMetadata | None = None
    failure: Phase1FailureMetadata | None = None


class Phase1Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["v2"]
    job_id: str
    status: JobState
    source_video: Phase1SourceVideo
    canonical_video_gcs_uri: GcsUriStr
    artifacts: Phase1Artifacts
    metadata: Phase1Metadata
