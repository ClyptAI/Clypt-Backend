from __future__ import annotations

from ..contracts import CanonicalTimeline, CanonicalTurn, TranscriptWord
from .pyannote_merge import merge_pyannote_outputs

from ..contracts import CanonicalTimeline


def build_canonical_timeline(
    *,
    phase1_audio: dict,
    pyannote_payload: dict,
    identify_payload: dict | None = None,
) -> CanonicalTimeline:
    """Build the canonical transcript/timing backbone for V3.1."""
    merged = merge_pyannote_outputs(
        diarize_payload=pyannote_payload,
        identify_payload=identify_payload,
    )

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
        source_video_url=phase1_audio.get("source_audio"),
        video_gcs_uri=phase1_audio.get("video_gcs_uri"),
    )
