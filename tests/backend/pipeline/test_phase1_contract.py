from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.pipeline.phase1_contract import JobState, Phase1Manifest


def _legacy_manifest_payload() -> dict:
    return {
        "contract_version": "v2",
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
                        "speaker_track_id": None,
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
                    "track_identity_features": {
                        "Global_Person_0": {
                            "face_observations": [
                                {
                                    "frame_idx": 0,
                                    "confidence": 0.91,
                                    "source": "face_detector",
                                    "provenance": "scrfd_fullframe",
                                }
                            ]
                        }
                    },
                    "tracking_mode": "direct",
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

    assert manifest.contract_version == "v2"
    assert manifest.canonical_video_gcs_uri == "gs://clypt-storage-v2/phase_1/video.mp4"
    assert manifest.artifacts.transcript.speaker_bindings[0].track_id == "Global_Person_0"
    assert manifest.artifacts.transcript.words[0].speaker_track_id is None
    assert manifest.artifacts.transcript.words[0].word == "foundational"
    assert manifest.artifacts.visual_tracking.tracks[0].track_id == "Global_Person_0"
    assert manifest.artifacts.visual_tracking.video_metadata.duration_ms == 123456
    assert manifest.artifacts.visual_tracking.tracking_metrics["tracking_mode"] == "direct"
    assert (
        manifest.artifacts.visual_tracking.tracking_metrics["track_identity_features"]["Global_Person_0"][
            "face_observations"
        ][0]["provenance"]
        == "scrfd_fullframe"
    )


def test_manifest_accepts_local_clip_experiment_fields():
    payload = deepcopy(_legacy_manifest_payload())
    payload["artifacts"]["transcript"]["words"][0]["speaker_local_track_id"] = "track_1"
    payload["artifacts"]["transcript"]["words"][0]["speaker_local_tag"] = "track_1"
    payload["artifacts"]["transcript"]["audio_speaker_turns"] = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 1500,
            "exclusive": True,
            "overlap": False,
            "confidence": 0.91,
        }
    ]
    payload["artifacts"]["transcript"]["speaker_bindings_local"] = [
        {
            "track_id": "track_1",
            "start_time_ms": 400,
            "end_time_ms": 2160,
            "word_count": 3,
        }
    ]
    payload["artifacts"]["transcript"]["speaker_follow_bindings_local"] = [
        {
            "track_id": "track_1",
            "start_time_ms": 400,
            "end_time_ms": 2160,
            "word_count": 3,
        }
    ]
    payload["artifacts"]["transcript"]["audio_speaker_local_track_map"] = [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "track_1",
            "support_segments": 2,
            "support_ms": 1800,
            "confidence": 0.86,
        }
    ]
    payload["artifacts"]["transcript"]["speaker_candidate_debug"] = [
        {
            "word": "foundational",
            "start_time_ms": 1040,
            "end_time_ms": 1600,
            "active_audio_speaker_id": "SPEAKER_00",
            "active_audio_local_track_id": "track_1",
            "chosen_track_id": "Global_Person_0",
            "chosen_local_track_id": "track_1",
            "decision_source": "audio_boosted_visual",
            "ambiguous": False,
            "top_1_top_2_margin": 0.081,
            "candidates": [
                {
                    "local_track_id": "track_1",
                    "track_id": "Global_Person_0",
                    "blended_score": 0.301,
                    "asd_probability": 0.18,
                    "body_prior": 0.56,
                    "detection_confidence": 0.99,
                },
                {
                    "local_track_id": "track_8",
                    "track_id": "Global_Person_1",
                    "blended_score": 0.22,
                    "asd_probability": 0.16,
                    "body_prior": 0.44,
                    "detection_confidence": 0.88,
                },
            ],
        }
    ]
    payload["artifacts"]["visual_tracking"]["tracks_local"] = [
        {
            **payload["artifacts"]["visual_tracking"]["tracks"][0],
            "track_id": "track_1",
        }
    ]

    manifest = Phase1Manifest.model_validate(payload)

    assert manifest.artifacts.transcript.words[0].speaker_local_track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_turns[0].speaker_id == "SPEAKER_00"
    assert manifest.artifacts.transcript.audio_speaker_turns[0].exclusive is True
    assert manifest.artifacts.transcript.speaker_bindings_local[0].track_id == "track_1"
    assert manifest.artifacts.transcript.speaker_follow_bindings_local[0].track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].speaker_id == "SPEAKER_00"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].local_track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].support_segments == 2
    assert manifest.artifacts.transcript.speaker_candidate_debug[0].active_audio_speaker_id == "SPEAKER_00"
    assert manifest.artifacts.transcript.speaker_candidate_debug[0].decision_source == "audio_boosted_visual"
    assert manifest.artifacts.transcript.speaker_candidate_debug[0].candidates[0].local_track_id == "track_1"
    assert manifest.artifacts.visual_tracking.tracks_local[0].track_id == "track_1"


def test_manifest_accepts_overlap_artifacts():
    payload = deepcopy(_legacy_manifest_payload())
    payload["artifacts"]["transcript"]["word_speaker_assignments"] = [
        {
            "start_time_ms": 1040,
            "end_time_ms": 1600,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["track_1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "dominant_visible_local_track_id": "track_1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "turn_binding",
            "overlap": True,
        }
    ]
    payload["artifacts"]["transcript"]["speaker_assignment_spans_local"] = [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["track_1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": 0.73,
            "decision_source": "turn_binding",
        }
    ]
    payload["artifacts"]["transcript"]["speaker_assignment_spans_global"] = [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["track_1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": 0.73,
            "decision_source": "turn_binding",
        }
    ]
    payload["artifacts"]["transcript"]["active_speakers_local"] = [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["track_1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": 0.73,
            "decision_source": "turn_binding",
        }
    ]
    payload["artifacts"]["transcript"]["overlap_follow_decisions"] = [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "camera_target_local_track_id": "track_1",
            "camera_target_track_id": "Global_Person_0",
            "stay_wide": False,
            "visible_local_track_ids": ["track_1"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "decision_model": "gemini-3-flash-preview",
            "decision_source": "gemini",
            "confidence": 0.81,
        }
    ]

    manifest = Phase1Manifest.model_validate(payload)

    assert manifest.artifacts.transcript.active_speakers_local[0].audio_speaker_ids == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert manifest.artifacts.transcript.word_speaker_assignments[0].offscreen_audio_speaker_ids == [
        "SPEAKER_01"
    ]
    assert manifest.artifacts.transcript.word_speaker_assignments[0].dominant_visible_local_track_id == "track_1"
    assert manifest.artifacts.transcript.speaker_assignment_spans_global[0].visible_track_ids == [
        "Global_Person_0"
    ]
    assert manifest.artifacts.transcript.active_speakers_local[0].offscreen_audio_speaker_ids == [
        "SPEAKER_01"
    ]
    assert manifest.artifacts.transcript.overlap_follow_decisions[0].camera_target_local_track_id == "track_1"
    assert manifest.artifacts.transcript.overlap_follow_decisions[0].decision_model == "gemini-3-flash-preview"


def test_manifest_rejects_unknown_audio_speaker_turn_fields():
    payload = deepcopy(_legacy_manifest_payload())
    payload["artifacts"]["transcript"]["audio_speaker_turns"] = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 1500,
            "exclusive": True,
            "unexpected_field": "not-allowed",
        }
    ]

    with pytest.raises(ValidationError, match="unexpected_field"):
        Phase1Manifest.model_validate(payload)


def test_manifest_rejects_unknown_speaker_candidate_debug_fields():
    payload = deepcopy(_legacy_manifest_payload())
    payload["artifacts"]["transcript"]["speaker_candidate_debug"] = [
        {
            "word": "foundational",
            "start_time_ms": 1040,
            "end_time_ms": 1600,
            "decision_source": "visual",
            "ambiguous": False,
            "candidates": [],
            "unexpected_field": "not-allowed",
        }
    ]

    with pytest.raises(ValidationError, match="unexpected_field"):
        Phase1Manifest.model_validate(payload)


def test_manifest_rejects_unknown_overlap_follow_decision_fields():
    payload = deepcopy(_legacy_manifest_payload())
    payload["artifacts"]["transcript"]["overlap_follow_decisions"] = [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "camera_target_local_track_id": "track_1",
            "camera_target_track_id": "Global_Person_0",
            "stay_wide": False,
            "visible_local_track_ids": ["track_1"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "decision_model": "gemini-3-flash-preview",
            "decision_source": "gemini",
            "confidence": 0.81,
            "unexpected_field": True,
        }
    ]

    with pytest.raises(ValidationError, match="unexpected_field"):
        Phase1Manifest.model_validate(payload)


@pytest.mark.parametrize(
    "mutator, expected_match",
    [
        (lambda payload: payload["artifacts"]["transcript"].pop("speaker_bindings"), "speaker_bindings"),
        (lambda payload: payload["artifacts"]["visual_tracking"].pop("tracks"), "tracks"),
        (lambda payload: payload.__setitem__("contract_version", "v1"), "v2"),
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


@pytest.mark.parametrize(
    "mutator, expected_match",
    [
        (
            lambda payload: payload["artifacts"]["transcript"]["words"][0].__setitem__("start_time_ms", 2000),
            "word start_time_ms",
        ),
        (
            lambda payload: payload["artifacts"]["transcript"]["speaker_bindings"][0].__setitem__("word_count", -1),
            "greater than or equal to 0",
        ),
        (
            lambda payload: payload["artifacts"]["visual_tracking"]["face_detections"][0].__setitem__("confidence", 1.2),
            "less than or equal to 1",
        ),
        (
            lambda payload: payload["artifacts"]["visual_tracking"]["tracks"][0]["bbox_norm_xywh"].__setitem__("width", 1.1),
            "less than or equal to 1",
        ),
        (
            lambda payload: payload["artifacts"]["visual_tracking"]["shot_changes"][0].__setitem__("start_time_ms", 20000),
            "shot change start_time_ms",
        ),
    ],
)
def test_manifest_rejects_semantically_invalid_payload(mutator, expected_match):
    payload = deepcopy(_legacy_manifest_payload())
    mutator(payload)

    with pytest.raises(ValidationError, match=expected_match):
        Phase1Manifest.model_validate(payload)


@pytest.mark.parametrize(
    "mutator, expected_match",
    [
        (
            lambda payload: payload["artifacts"]["visual_tracking"]["face_detections"][0]["timestamped_objects"][0]["bounding_box"].__setitem__("left", 0.9),
            "bounding box left must be <= right",
        ),
        (
            lambda payload: payload["artifacts"]["visual_tracking"]["face_detections"][0]["timestamped_objects"][0]["bounding_box"].__setitem__("top", 0.9),
            "bounding box top must be <= bottom",
        ),
        (
            lambda payload: payload["metadata"]["timings"].__setitem__("ingest_ms", -1),
            "greater than or equal to 0",
        ),
        (
            lambda payload: payload["metadata"]["retry"].__setitem__("attempts", -1),
            "greater than or equal to 0",
        ),
        (
            lambda payload: payload["metadata"]["quality_metrics"].__setitem__("schema_pass_rate", 1.2),
            "less than or equal to 1",
        ),
    ],
)
def test_manifest_rejects_geometric_and_metadata_semantics(mutator, expected_match):
    payload = deepcopy(_legacy_manifest_payload())
    payload["metadata"]["retry"] = {
        "attempts": 1,
        "max_attempts": 3,
        "last_error": None,
    }
    mutator(payload)

    with pytest.raises(ValidationError, match=expected_match):
        Phase1Manifest.model_validate(payload)
