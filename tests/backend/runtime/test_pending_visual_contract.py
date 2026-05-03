from __future__ import annotations

import pytest

from backend.phase1_runtime.payloads import Phase1SidecarOutputs, VisualPayload
from backend.runtime.phase14_live import V31LivePhase14Runner


def test_visual_payload_join_requires_track_and_shot_keys() -> None:
    with pytest.raises(ValueError, match="shot_changes"):
        V31LivePhase14Runner._visual_payload_from_result(
            {"phase1_visual": {"tracks": []}}
        )
    with pytest.raises(ValueError, match="tracks"):
        V31LivePhase14Runner._visual_payload_from_result(
            {"phase1_visual": {"shot_changes": []}}
        )


def test_visual_payload_join_accepts_wrapped_modal_result() -> None:
    payload = V31LivePhase14Runner._visual_payload_from_result(
        {
            "status": "succeeded",
            "queue_wait_ms": 5.0,
            "phase1_visual": {
                "shot_changes": [],
                "tracks": [{"track_id": "p1"}],
                "future_field": "ignored",
            },
        }
    )

    assert isinstance(payload, VisualPayload)
    assert payload.tracks == [{"track_id": "p1"}]


def test_phase24_worker_fails_before_runner_for_pending_visual_without_client() -> None:
    from tests.backend.runtime.test_phase24_worker_app import (
        _FakeRepository,
        _FakeRunner,
        _build_payload,
    )
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    payload = _build_payload()
    payload["phase1_visual_status"] = "pending"
    payload["visual_future"] = {
        "backend": "modal_rfdetr_l40s",
        "call_id": "fc-visual",
        "source_video_gcs_uri": "gs://bucket/video.mp4",
    }
    phase1_outputs = dict(payload["phase1_outputs"])
    phase1_outputs.pop("phase1_visual")
    payload["phase1_outputs"] = phase1_outputs

    repository = _FakeRepository()
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=1,
    )

    with pytest.raises(ValueError, match="visual_result_client"):
        service.handle_task(
            payload=Phase24TaskPayload(**payload),
            job_id="task-001",
            attempt=1,
        )

    assert runner.calls == []
    assert repository.job_record.status == "failed"


def test_phase24_prepare_merges_visual_future_into_loaded_outputs() -> None:
    from backend.runtime.phase24_worker_app import Phase24TaskPayload

    payload = Phase24TaskPayload(
        run_id="run-1",
        source_url="https://example.com/video",
        phase1_visual_status="pending",
        visual_future={
            "backend": "modal_rfdetr_l40s",
            "call_id": "fc-visual",
            "source_video_gcs_uri": "gs://bucket/video.mp4",
        },
        phase1_outputs=Phase1SidecarOutputs(
            phase1_audio={"video_gcs_uri": "gs://bucket/video.mp4"},
            diarization_payload={"turns": [], "words": []},
            phase1_visual_status="pending",
            phase1_visual=None,
            emotion2vec_payload={"segments": []},
            yamnet_payload={"events": []},
        ),
    )

    assert payload.visual_future is not None
    assert payload.phase1_outputs.phase1_visual is None
