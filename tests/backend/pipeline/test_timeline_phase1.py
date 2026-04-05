from __future__ import annotations

from backend.pipeline.timeline.audio_events import build_audio_event_timeline
from backend.pipeline.timeline.emotion_events import build_speech_emotion_timeline
from backend.pipeline.timeline.pyannote_merge import merge_pyannote_outputs
from backend.pipeline.timeline.timeline_builder import build_canonical_timeline
from backend.pipeline.timeline.tracklets import build_tracklet_artifacts


def test_merge_pyannote_outputs_normalizes_words_turns_and_identification():
    diarize_payload = {
        "wordLevelTranscription": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.4, "end": 0.8, "speaker": "SPEAKER_00"},
        ],
        "turnLevelTranscription": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 0.8,
                "text": "Hello world.",
            }
        ],
        "diarization": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.8},
        ],
    }
    identify_payload = {
        "identification": [
            {
                "diarizationSpeaker": "SPEAKER_00",
                "match": "Sam",
                "start": 0.0,
                "end": 0.8,
            }
        ]
    }

    merged = merge_pyannote_outputs(
        diarize_payload=diarize_payload,
        identify_payload=identify_payload,
    )

    assert merged["words"][0]["text"] == "Hello"
    assert merged["words"][0]["speaker_id"] == "SPEAKER_00"
    assert merged["turns"][0]["speaker_id"] == "SPEAKER_00"
    assert merged["turns"][0]["transcript_text"] == "Hello world."
    assert merged["turns"][0]["identification_match"] == "Sam"


def test_build_canonical_timeline_builds_word_ids_and_turn_text():
    phase1_audio = {
        "source_audio": "https://example.com/video",
        "video_gcs_uri": "gs://bucket/video.mp4",
    }
    pyannote_payload = {
        "wordLevelTranscription": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.4, "end": 0.8, "speaker": "SPEAKER_00"},
            {"word": "Again", "start": 1.0, "end": 1.4, "speaker": "SPEAKER_01"},
        ],
        "diarization": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.8},
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 1.4},
        ],
    }

    timeline = build_canonical_timeline(
        phase1_audio=phase1_audio,
        pyannote_payload=pyannote_payload,
    )

    assert timeline.source_video_url == "https://example.com/video"
    assert timeline.video_gcs_uri == "gs://bucket/video.mp4"
    assert len(timeline.words) == 3
    assert timeline.words[0].word_id == "w_000001"
    assert timeline.turns[0].turn_id == "t_000001"
    assert timeline.turns[0].word_ids == ["w_000001", "w_000002"]
    assert timeline.turns[0].transcript_text == "Hello world"
    assert timeline.turns[1].speaker_id == "SPEAKER_01"


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
