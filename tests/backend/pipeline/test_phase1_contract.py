import pytest
from pydantic import ValidationError

from backend.pipeline.phase1_contract import JobState, Phase1Manifest


def test_manifest_requires_contract_version_and_job_state():
    manifest = Phase1Manifest.model_validate(
        {
            "contract_version": "v1",
            "job_id": "job_123",
            "status": JobState.SUCCEEDED,
            "source_video": {"source_url": "https://youtube.com/watch?v=x"},
            "canonical_video_gcs_uri": "gs://clypt-storage-v2/phase_1/video.mp4",
            "artifacts": {
                "transcript": {
                    "uri": "gs://bucket/transcript.json",
                    "words": [],
                    "speaker_bindings": [],
                },
                "visual_tracking": {
                    "uri": "gs://bucket/tracking.json",
                    "tracks": [],
                    "detection_blocks": [],
                },
            },
            "metadata": {
                "runtime": {"provider": "digitalocean"},
                "timings": {},
            },
        }
    )

    assert manifest.contract_version == "v1"
    assert manifest.status is JobState.SUCCEEDED
    assert manifest.artifacts.transcript.words == []
    assert manifest.artifacts.transcript.speaker_bindings == []
    assert manifest.artifacts.visual_tracking.tracks == []
    assert manifest.artifacts.visual_tracking.detection_blocks == []


def test_manifest_requires_legacy_transcript_fields():
    with pytest.raises(ValidationError):
        Phase1Manifest.model_validate(
            {
                "contract_version": "v1",
                "job_id": "job_123",
                "status": JobState.SUCCEEDED,
                "source_video": {"source_url": "https://youtube.com/watch?v=x"},
                "canonical_video_gcs_uri": "gs://clypt-storage-v2/phase_1/video.mp4",
                "artifacts": {
                    "transcript": {
                        "uri": "gs://bucket/transcript.json",
                        "speaker_bindings": [],
                    },
                    "visual_tracking": {
                        "uri": "gs://bucket/tracking.json",
                        "tracks": [],
                        "detection_blocks": [],
                    },
                },
                "metadata": {
                    "runtime": {"provider": "digitalocean"},
                    "timings": {},
                },
            }
        )
