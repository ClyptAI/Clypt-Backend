from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.pipeline.do_phase1_client import (
    DOPhase1Client,
    Phase1JobNotFoundError,
    Phase1JobNotReadyError,
    Phase1JobRecord,
    Phase1JobSubmission,
)


def _manifest_payload() -> dict:
    return {
        "contract_version": "v3",
        "job_id": "job_123",
        "status": "succeeded",
        "source_video": {"source_url": "https://youtube.com/watch?v=x"},
        "canonical_video_gcs_uri": "gs://bucket/phase_1/video.mp4",
        "artifacts": {
            "transcript": {
                "uri": "gs://bucket/transcript.json",
                "source_audio": "https://youtube.com/watch?v=x",
                "video_gcs_uri": "gs://bucket/phase_1/video.mp4",
                "words": [],
                "speaker_bindings": [],
            },
            "visual_tracking": {
                "uri": "gs://bucket/visual.json",
                "source_video": "https://youtube.com/watch?v=x",
                "video_gcs_uri": "gs://bucket/phase_1/video.mp4",
                "schema_version": "3.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "events": None,
        },
        "metadata": {
            "runtime": {"provider": "digitalocean", "worker_id": "worker-1", "region": None},
            "timings": {"ingest_ms": None, "processing_ms": None, "upload_ms": None},
            "quality_metrics": None,
            "retry": None,
            "failure": None,
        },
    }


def _status_payload() -> dict:
    return {
        "job_id": "job_123",
        "source_url": "https://youtube.com/watch?v=x",
        "status": "running",
        "retries": 1,
        "claim_token": "claim_123",
        "manifest": None,
        "manifest_uri": None,
        "failure": None,
        "created_at": "2026-03-21T10:00:00Z",
        "updated_at": "2026-03-21T10:03:00Z",
        "started_at": "2026-03-21T10:01:00Z",
        "completed_at": None,
    }


def test_submit_job_posts_source_url_and_parses_job_record():
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/jobs"
        assert json.loads(request.content) == {
            "source_url": "https://youtube.com/watch?v=x",
            "runtime_controls": {"speaker_binding_mode": "lrasd", "tracking_mode": "direct"},
        }
        return httpx.Response(
            202,
            json={
                "job_id": "job_123",
                "source_url": "https://youtube.com/watch?v=x",
                "runtime_controls": {"speaker_binding_mode": "lrasd", "tracking_mode": "direct"},
                "status": "queued",
                "retries": 0,
                "claim_token": None,
                "manifest": None,
                "manifest_uri": None,
                "failure": None,
                "created_at": "2026-03-21T10:00:00Z",
                "updated_at": "2026-03-21T10:00:00Z",
                "started_at": None,
                "completed_at": None,
            },
        )

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        job = asyncio.run(
            client.submit_job(
                "https://youtube.com/watch?v=x",
                runtime_controls={"speaker_binding_mode": "lrasd", "tracking_mode": "direct"},
            )
        )
    finally:
        asyncio.run(client.aclose())

    assert isinstance(job, Phase1JobSubmission)
    assert job.job_id == "job_123"
    assert job.status == "queued"
    assert job.manifest is None
    assert len(seen_requests) == 1


def test_get_job_parses_nested_manifest_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/jobs/job_123"
        return httpx.Response(
            200,
            json={
                **_status_payload(),
                "status": "succeeded",
                "manifest": _manifest_payload(),
                "manifest_uri": "gs://bucket/job_123/manifest.json",
                "completed_at": "2026-03-21T10:05:00Z",
            },
        )

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        job = asyncio.run(client.get_job("job_123"))
    finally:
        asyncio.run(client.aclose())

    assert isinstance(job, Phase1JobRecord)
    assert job.status == "succeeded"
    assert job.manifest is not None
    assert job.manifest.contract_version == "v3"
    assert job.manifest.artifacts.visual_tracking.video_metadata.duration_ms == 1000


def test_get_job_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/jobs/missing"
        return httpx.Response(404, json={"detail": "job not found"})

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(Phase1JobNotFoundError):
            asyncio.run(client.get_job("missing"))
    finally:
        asyncio.run(client.aclose())


def test_get_result_raises_until_job_is_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/jobs/job_123/result"
        return httpx.Response(409, json={"detail": "job result not ready"})

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(Phase1JobNotReadyError):
            asyncio.run(client.get_result("job_123"))
    finally:
        asyncio.run(client.aclose())


def test_get_result_returns_manifest_after_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/jobs/job_123/result"
        return httpx.Response(200, json=_manifest_payload())

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        manifest = asyncio.run(client.get_result("job_123"))
    finally:
        asyncio.run(client.aclose())

    assert manifest.contract_version == "v3"
    assert manifest.job_id == "job_123"
    assert manifest.status == "succeeded"


def test_get_result_normalizes_overlap_runtime_keys_before_validate():
    payload = _manifest_payload()
    payload["artifacts"]["transcript"]["overlap_follow_decisions"] = [
        {
            "start_time_ms": 0,
            "end_time_ms": 500,
            "stay_wide": True,
            "decision_source": "deterministic",
            "decision_code": "low_overlap_evidence",
            "evidence_context": {"gated_low_evidence": True},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        manifest = asyncio.run(client.get_result("job_ov"))
    finally:
        asyncio.run(client.aclose())

    assert len(manifest.artifacts.transcript.overlap_follow_decisions) == 1
    assert manifest.artifacts.transcript.overlap_follow_decisions[0].decision_source == "deterministic"


def test_get_result_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/jobs/missing/result"
        return httpx.Response(404, json={"detail": "job not found"})

    client = DOPhase1Client(base_url="https://do.example", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(Phase1JobNotFoundError):
            asyncio.run(client.get_result("missing"))
    finally:
        asyncio.run(client.aclose())
