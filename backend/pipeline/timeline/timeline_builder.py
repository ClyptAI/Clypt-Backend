from __future__ import annotations

from typing import Any

from ..contracts import CanonicalTimeline, CanonicalTurn, TranscriptWord
from .payload_utils import payload_to_dict


def build_canonical_timeline(
    *,
    phase1_audio: Any,
    diarization_payload: Any,
) -> CanonicalTimeline:
    """Build the canonical transcript/timing backbone for V3.1."""
    phase1_audio_dict = payload_to_dict(phase1_audio)
    diarization_payload_dict = payload_to_dict(diarization_payload)

    if "turns" not in diarization_payload_dict or "words" not in diarization_payload_dict:
        raise ValueError("diarization_payload must include canonical turns and words.")
    merged = diarization_payload_dict

    words = [
        TranscriptWord(
            word_id=word["word_id"],
            text=word["text"],
            start_ms=word["start_ms"],
            end_ms=word["end_ms"],
            speaker_id=word.get("speaker_id"),
        )
        for word in merged["words"]
    ]
    turns = [
        CanonicalTurn(
            turn_id=turn["turn_id"],
            speaker_id=turn["speaker_id"],
            start_ms=turn["start_ms"],
            end_ms=turn["end_ms"],
            word_ids=turn.get("word_ids", []),
            transcript_text=turn.get("transcript_text", ""),
            identification_match=turn.get("identification_match"),
        )
        for turn in merged["turns"]
    ]

    return CanonicalTimeline(
        words=words,
        turns=turns,
        source_video_url=phase1_audio_dict.get("source_audio"),
        video_gcs_uri=phase1_audio_dict.get("video_gcs_uri"),
    )
