import pytest
from pydantic import ValidationError

from backend.pipeline.phase1_contract import JobState, Phase1Manifest


def _legacy_manifest_payload() -> dict:
    return {
        "contract_version": "v1",
        "job_id": "job_123",
        "status": JobState.SUCCEEDED,
        "source_video": {"source_url": "https://youtube.com/watch?v=RgOz31Gaibw"},
        "canonical_video_gcs_uri": "gs://clypt-storage-v2/phase_1/video.mp4",
        "artifacts": {
            "transcript": {
                "uri": "gs://bucket/phase_1_audio.json",
                "source_audio": "https://youtube.com/watch?v=RgOz31Gaibw",
                "video_gcs_uri": "gs://clypt-storage-v2/phase_1/video.mp4",
                "words": [
                    {
                        "word": "foundational",
                        "start_time_ms": 1040,
                        "end_time_ms": 1600,
                        "speaker_track_id": "Global_Person_0",
                        "speaker_tag": "Global_Person_0",
                    }
                ],
                "speaker_bindings": [
                    {
                        "track_id": "Global_Person_0",
                        "start_time_ms": 400,
                        "end_time_ms": 2160,
                        "word_count": 3,
                    }
                ],
            },
            "visual_tracking": {
                "uri": "gs://bucket/phase_1_visual.json",
                "source_video": "https://youtube.com/watch?v=RgOz31Gaibw",
                "video_gcs_uri": "gs://clypt-storage-v2/phase_1/video.mp4",
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {
                    "schema_pass_rate": 1.0,
                    "throughput_fps": 35.49084254321749,
                },
                "tracks": [
                    {
                        "frame_idx": 0,
                        "local_frame_idx": 0,
                        "chunk_idx": 0,
                        "track_id": "Global_Person_0",
                        "local_track_id": 1,
                        "class_id": 0,
                        "label": "person",
                        "confidence": 0.6850884556770325,
                        "x1": 976.388671875,
                        "y1": 267.7335510253906,
                        "x2": 1574.221435546875,
                        "y2": 959.4959716796875,
                        "x_center": 1275.3050537109375,
                        "y_center": 613.6147613525391,
                        "width": 597.832763671875,
                        "height": 691.7624206542969,
                        "source": "detector",
                        "geometry_type": "aabb",
                        "bbox_norm_xywh": {
                            "x_center": 0.6642213821411133,
                            "y_center": 0.5681618160671658,
                            "width": 0.3113712310791016,
                            "height": 0.6405207598650897,
                        },
                    }
                ],
                "face_detections": [
                    {
                        "track_id": "Global_Person_0",
                        "face_track_index": 0,
                        "segment_start_ms": 0,
                        "segment_end_ms": 10000,
                        "confidence": 0.6850884556770325,
                        "timestamped_objects": [
                            {
                                "time_ms": 0,
                                "track_id": "Global_Person_0",
                                "confidence": 0.6850884556770325,
                                "bounding_box": {
                                    "left": 0.5645825881958008,
                                    "top": 0.26071185133192276,
                                    "right": 0.7638601760864259,
                                    "bottom": 0.555351400869864,
                                },
                            }
                        ],
                    }
                ],
                "person_detections": [
                    {
                        "track_id": "Global_Person_0",
                        "person_track_index": 0,
                        "segment_start_ms": 0,
                        "segment_end_ms": 10000,
                        "confidence": 0.6850884556770325,
                        "timestamped_objects": [
                            {
                                "time_ms": 0,
                                "track_id": "Global_Person_0",
                                "confidence": 0.6850884556770325,
                                "bounding_box": {
                                    "left": 0.5085357666015625,
                                    "top": 0.24790143613462096,
                                    "right": 0.8199069976806641,
                                    "bottom": 0.8884221959997106,
                                },
                            }
                        ],
                    }
                ],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [
                    {"start_time_ms": 0, "end_time_ms": 10000},
                    {"start_time_ms": 10000, "end_time_ms": 20000},
                ],
                "video_metadata": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 29.97,
                    "duration_ms": 123456,
                },
            },
        },
        "metadata": {
            "runtime": {
                "provider": "digitalocean",
                "worker_id": "worker-1",
            },
            "timings": {
                "ingest_ms": 10,
                "processing_ms": 20,
                "upload_ms": 30,
            },
            "quality_metrics": {
                "schema_pass_rate": 1.0,
                "transcript_coverage": 0.98,
                "tracking_confidence": 0.95,
            },
        },
    }


def test_manifest_accepts_realistic_legacy_payload_shape():
    manifest = Phase1Manifest.model_validate(_legacy_manifest_payload())

    assert manifest.contract_version == "v1"
    assert manifest.canonical_video_gcs_uri == "gs://clypt-storage-v2/phase_1/video.mp4"
    assert manifest.artifacts.transcript.speaker_bindings[0].track_id == "Global_Person_0"
    assert manifest.artifacts.transcript.words[0].word == "foundational"
    assert manifest.artifacts.visual_tracking.tracks[0].track_id == "Global_Person_0"
    assert manifest.artifacts.visual_tracking.video_metadata.duration_ms == 123456


@pytest.mark.parametrize(
    "mutator, expected_match",
    [
        (lambda payload: payload["artifacts"]["transcript"].pop("speaker_bindings"), "speaker_bindings"),
        (lambda payload: payload["artifacts"]["visual_tracking"].pop("tracks"), "tracks"),
        (lambda payload: payload.__setitem__("contract_version", "v2"), "v1"),
        (
            lambda payload: payload.__setitem__("canonical_video_gcs_uri", "https://example.com/video.mp4"),
            "gs://",
        ),
    ],
)
def test_manifest_rejects_invalid_legacy_payload(mutator, expected_match):
    payload = _legacy_manifest_payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=expected_match):
        Phase1Manifest.model_validate(payload)
