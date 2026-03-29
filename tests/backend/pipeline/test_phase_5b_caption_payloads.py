from __future__ import annotations

import json
from pathlib import Path
import tempfile

from backend.pipeline import phase_5b_caption_payloads as subject


def test_build_caption_chunks_splits_on_sentence_pause_and_speaker_change():
    words = [
        {"word": "This", "start_time_ms": 0, "end_time_ms": 180, "speaker_tag": "speaker_1"},
        {"word": "works.", "start_time_ms": 190, "end_time_ms": 360, "speaker_tag": "speaker_1"},
        {"word": "Next", "start_time_ms": 900, "end_time_ms": 1080, "speaker_tag": "speaker_1"},
        {"word": "part", "start_time_ms": 1090, "end_time_ms": 1260, "speaker_tag": "speaker_1"},
        {"word": "switch", "start_time_ms": 1270, "end_time_ms": 1450, "speaker_tag": "speaker_2"},
        {"word": "speaker", "start_time_ms": 1460, "end_time_ms": 1650, "speaker_tag": "speaker_2"},
    ]

    chunks = subject.build_caption_chunks(
        words,
        0,
        2000,
        max_words=4,
        max_chars=24,
        gap_ms=400,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "This works.",
        "Next part",
        "switch speaker",
    ]
    assert chunks[0]["speaker_tag"] == "speaker_1"
    assert chunks[2]["speaker_tag"] == "speaker_2"


def test_augment_main_writes_captioned_payload_without_mutating_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "remotion_payloads_array.json"
        audio_path = tmp_path / "phase_1_audio.json"
        output_path = tmp_path / "remotion_payloads_array_captioned.json"

        input_payloads = [
            {
                "clip_start_ms": 0,
                "clip_end_ms": 1400,
                "final_score": 88.0,
                "tracking_uris": ["gs://bucket/tracking.json"],
            }
        ]
        audio_payload = {
            "words": [
                {"word": "hello", "start_time_ms": 0, "end_time_ms": 180, "speaker_tag": "speaker_1"},
                {"word": "there", "start_time_ms": 190, "end_time_ms": 360, "speaker_tag": "speaker_1"},
                {"word": "general", "start_time_ms": 700, "end_time_ms": 930, "speaker_tag": "speaker_1"},
                {"word": "kenobi", "start_time_ms": 940, "end_time_ms": 1200, "speaker_tag": "speaker_1"},
            ]
        }

        input_path.write_text(json.dumps(input_payloads), encoding="utf-8")
        audio_path.write_text(json.dumps(audio_payload), encoding="utf-8")

        result = subject.main(
            input_path=input_path,
            audio_path=audio_path,
            output_path=output_path,
        )

        assert result["payload_count"] == 1
        written = json.loads(output_path.read_text(encoding="utf-8"))
        original = json.loads(input_path.read_text(encoding="utf-8"))

        assert "captions" not in original[0]
        assert [chunk["text"] for chunk in written[0]["captions"]] == [
            "hello there",
            "general kenobi",
        ]
        assert written[0]["final_score"] == 88.0
