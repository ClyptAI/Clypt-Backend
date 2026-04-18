from __future__ import annotations

from backend.phase1_runtime.payloads import (
    DiarizationPayload,
    EmotionSegmentsPayload,
    Phase1AudioAssets,
    VisualPayload,
    YamnetPayload,
)
from backend.pipeline.timeline.audio_events import build_audio_event_timeline
from backend.pipeline.timeline.emotion_events import build_speech_emotion_timeline
from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs
from backend.pipeline.timeline.timeline_builder import build_canonical_timeline
from backend.pipeline.timeline.tracklets import build_tracklet_artifacts


def test_merge_vibevoice_outputs_normalizes_words_and_turns():
    vibevoice_turns = [
        {"Start": 0.0, "End": 0.8, "Speaker": 0, "Content": "Hello world"},
        {"Start": 1.0, "End": 1.4, "Speaker": 1, "Content": "Again"},
    ]
    word_alignments = [
        {"word_id": "w_000001", "text": "Hello", "start_ms": 0, "end_ms": 400, "speaker_id": "SPEAKER_0"},
        {"word_id": "w_000002", "text": "world", "start_ms": 400, "end_ms": 800, "speaker_id": "SPEAKER_0"},
        {"word_id": "w_000003", "text": "Again", "start_ms": 1000, "end_ms": 1400, "speaker_id": "SPEAKER_1"},
    ]

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )

    assert merged["words"][0]["text"] == "Hello"
    assert merged["words"][0]["speaker_id"] == "SPEAKER_0"
    assert merged["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert merged["turns"][0]["transcript_text"] == "Hello world"
    assert merged["turns"][0]["identification_match"] is None
    assert merged["turns"][1]["speaker_id"] == "SPEAKER_1"


def test_merge_vibevoice_outputs_falls_back_to_token_split_without_alignments():
    vibevoice_turns = [
        {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "Hello world"},
    ]
    merged = merge_vibevoice_outputs(vibevoice_turns=vibevoice_turns, word_alignments=[])
    assert len(merged["words"]) == 2
    assert merged["words"][0]["text"] == "Hello"
    assert merged["turns"][0]["speaker_id"] == "SPEAKER_0"


def test_merge_vibevoice_outputs_prefers_monotonic_text_over_raw_overlap():
    vibevoice_turns = [
        {"Start": 0.0, "End": 2.0, "Speaker": 0, "Content": "I think"},
        {"Start": 2.0, "End": 4.0, "Speaker": 1, "Content": "Severely claustrophobic"},
    ]
    # "yeah" and the overlap of "think" spill across the boundary; they should
    # not be attached to turn 2 because turn text doesn't contain them.
    word_alignments = [
        {"word_id": "w_000001", "text": "I", "start_ms": 0, "end_ms": 400, "speaker_id": "UNKNOWN"},
        {"word_id": "w_000002", "text": "think", "start_ms": 400, "end_ms": 2050, "speaker_id": "UNKNOWN"},
        {"word_id": "w_000003", "text": "yeah", "start_ms": 2050, "end_ms": 2200, "speaker_id": "UNKNOWN"},
        {"word_id": "w_000004", "text": "Severely", "start_ms": 2200, "end_ms": 2900, "speaker_id": "UNKNOWN"},
        {
            "word_id": "w_000005",
            "text": "claustrophobic",
            "start_ms": 2900,
            "end_ms": 3600,
            "speaker_id": "UNKNOWN",
        },
    ]

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )

    assert merged["turns"][0]["word_ids"] == ["w_000001", "w_000002"]
    assert merged["turns"][1]["word_ids"] == ["w_000004", "w_000005"]
    assert merged["words"][3]["speaker_id"] == "SPEAKER_1"
    assert merged["words"][4]["speaker_id"] == "SPEAKER_1"


def test_merge_vibevoice_outputs_handles_repeated_tokens_monotonically():
    vibevoice_turns = [
        {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "go go"},
        {"Start": 1.0, "End": 1.5, "Speaker": 1, "Content": "go"},
    ]
    word_alignments = [
        {"word_id": "w_000001", "text": "go", "start_ms": 0, "end_ms": 300, "speaker_id": "UNKNOWN"},
        {"word_id": "w_000002", "text": "go", "start_ms": 300, "end_ms": 600, "speaker_id": "UNKNOWN"},
        {"word_id": "w_000003", "text": "go", "start_ms": 1100, "end_ms": 1300, "speaker_id": "UNKNOWN"},
    ]

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )

    assert merged["turns"][0]["word_ids"] == ["w_000001", "w_000002"]
    assert merged["turns"][1]["word_ids"] == ["w_000003"]


def test_build_canonical_timeline_builds_word_ids_and_turn_text():
    phase1_audio = {
        "source_audio": "https://example.com/video",
        "video_gcs_uri": "gs://bucket/video.mp4",
    }
    # diarization_payload is the merged {words, turns} dict from extract.py
    diarization_payload = {
        "words": [
            {"word_id": "w_000001", "text": "Hello", "start_ms": 0, "end_ms": 400, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000002", "text": "world", "start_ms": 400, "end_ms": 800, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000003", "text": "Again", "start_ms": 1000, "end_ms": 1400, "speaker_id": "SPEAKER_1"},
        ],
        "turns": [
            {
                "turn_id": "t_000001",
                "speaker_id": "SPEAKER_0",
                "start_ms": 0,
                "end_ms": 800,
                "transcript_text": "Hello world",
                "word_ids": ["w_000001", "w_000002"],
                "identification_match": None,
            },
            {
                "turn_id": "t_000002",
                "speaker_id": "SPEAKER_1",
                "start_ms": 1000,
                "end_ms": 1400,
                "transcript_text": "Again",
                "word_ids": ["w_000003"],
                "identification_match": None,
            },
        ],
    }

    timeline = build_canonical_timeline(
        phase1_audio=phase1_audio,
        diarization_payload=diarization_payload,
    )

    assert timeline.source_video_url == "https://example.com/video"
    assert timeline.video_gcs_uri == "gs://bucket/video.mp4"
    assert len(timeline.words) == 3
    assert timeline.words[0].word_id == "w_000001"
    assert timeline.turns[0].turn_id == "t_000001"
    assert timeline.turns[0].word_ids == ["w_000001", "w_000002"]
    assert timeline.turns[0].transcript_text == "Hello world"
    assert timeline.turns[1].speaker_id == "SPEAKER_1"


def test_build_speech_emotion_timeline_normalizes_emotion2vec_plus_output():
    emotion_payload = {
        "segments": [
            {
                "turn_id": "t_000001",
                "labels": ["happy"],
                "scores": [0.93],
                "per_class_scores": {
                    "happy": 0.93,
                    "neutral": 0.05,
                    "sad": 0.02,
                },
            }
        ]
    }

    timeline = build_speech_emotion_timeline(emotion2vec_payload=emotion_payload)

    assert timeline.events[0].turn_id == "t_000001"
    assert timeline.events[0].primary_emotion_label == "happy"
    assert timeline.events[0].primary_emotion_score == 0.93
    assert timeline.events[0].per_class_scores["neutral"] == 0.05


def test_build_audio_event_timeline_merges_adjacent_same_label_events():
    yamnet_payload = {
        "events": [
            {"event_label": "Laughter", "start_ms": 1000, "end_ms": 1400, "confidence": 0.81},
            {"event_label": "Laughter", "start_ms": 1400, "end_ms": 1800, "confidence": 0.88},
            {"event_label": "Applause", "start_ms": 2000, "end_ms": 2400, "confidence": 0.73},
        ]
    }

    timeline = build_audio_event_timeline(yamnet_payload=yamnet_payload)

    assert len(timeline.events) == 2
    assert timeline.events[0].event_label == "Laughter"
    assert timeline.events[0].start_ms == 1000
    assert timeline.events[0].end_ms == 1800
    assert timeline.events[0].confidence == 0.88


def test_build_tracklet_artifacts_groups_tracks_by_shot():
    phase1_visual = {
        "video_metadata": {"fps": 10.0},
        "shot_changes": [
            {"start_time_ms": 0, "end_time_ms": 1000},
            {"start_time_ms": 1000, "end_time_ms": 2000},
        ],
        "tracks": [
            {
                "frame_idx": 0,
                "track_id": "Global_Person_0",
                "x1": 10.0,
                "y1": 20.0,
                "x2": 50.0,
                "y2": 80.0,
            },
            {
                "frame_idx": 5,
                "track_id": "Global_Person_0",
                "x1": 12.0,
                "y1": 22.0,
                "x2": 52.0,
                "y2": 82.0,
            },
            {
                "frame_idx": 12,
                "track_id": "Global_Person_1",
                "x1": 110.0,
                "y1": 120.0,
                "x2": 150.0,
                "y2": 180.0,
            },
        ],
    }

    index, geometry = build_tracklet_artifacts(phase1_visual=phase1_visual)

    assert len(index.tracklets) == 2
    assert index.tracklets[0].shot_id == "shot_0001"
    assert index.tracklets[0].tracklet_id == "shot_0001:Global_Person_0"
    assert index.tracklets[0].start_ms == 0
    assert index.tracklets[0].end_ms == 500
    assert geometry.tracklets[0].points[1].timestamp_ms == 500
    assert geometry.tracklets[1].shot_id == "shot_0002"


def test_timeline_builders_accept_phase1_payload_models():
    phase1_audio = Phase1AudioAssets(
        source_audio="https://example.com/video",
        video_gcs_uri="gs://bucket/video.mp4",
    )
    diarization_payload = DiarizationPayload(
        words=[
            {"word_id": "w_000001", "text": "Hello", "start_ms": 0, "end_ms": 400, "speaker_id": "SPEAKER_0"},
        ],
        turns=[
            {
                "turn_id": "t_000001",
                "speaker_id": "SPEAKER_0",
                "start_ms": 0,
                "end_ms": 400,
                "transcript_text": "Hello",
                "word_ids": ["w_000001"],
                "identification_match": None,
            }
        ],
    )
    emotion_payload = EmotionSegmentsPayload(
        segments=[
            {
                "turn_id": "t_000001",
                "labels": ["happy"],
                "scores": [0.93],
                "per_class_scores": {"happy": 0.93},
            }
        ]
    )
    yamnet_payload = YamnetPayload(
        events=[
            {"event_label": "Laughter", "start_ms": 1000, "end_ms": 1400, "confidence": 0.81},
        ]
    )
    visual_payload = VisualPayload(
        video_metadata={"fps": 10.0},
        shot_changes=[{"start_time_ms": 0, "end_time_ms": 1000}],
        tracks=[
            {
                "frame_idx": 0,
                "track_id": "Global_Person_0",
                "x1": 10.0,
                "y1": 20.0,
                "x2": 50.0,
                "y2": 80.0,
            }
        ],
    )

    canonical_timeline = build_canonical_timeline(
        phase1_audio=phase1_audio,
        diarization_payload=diarization_payload,
    )
    speech_emotion_timeline = build_speech_emotion_timeline(
        emotion2vec_payload=emotion_payload,
    )
    audio_event_timeline = build_audio_event_timeline(
        yamnet_payload=yamnet_payload,
    )
    shot_tracklet_index, tracklet_geometry = build_tracklet_artifacts(
        phase1_visual=visual_payload,
    )

    assert canonical_timeline.video_gcs_uri == "gs://bucket/video.mp4"
    assert canonical_timeline.turns[0].turn_id == "t_000001"
    assert speech_emotion_timeline.events[0].primary_emotion_label == "happy"
    assert audio_event_timeline.events[0].event_label == "Laughter"
    assert shot_tracklet_index.tracklets[0].tracklet_id == "shot_0001:Global_Person_0"
    assert tracklet_geometry.tracklets[0].points[0].frame_index == 0
