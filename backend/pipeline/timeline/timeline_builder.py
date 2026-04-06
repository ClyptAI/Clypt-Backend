from __future__ import annotations

from ..contracts import CanonicalTimeline, CanonicalTurn, TranscriptWord
from .vibevoice_merge import merge_vibevoice_outputs


def build_canonical_timeline(
    *,
    phase1_audio: dict,
    diarization_payload: dict,
) -> CanonicalTimeline:
    """Build the canonical transcript/timing backbone for V3.1."""
    # diarization_payload already contains {words, turns} from extract.py
    # If it arrived directly from the merged output, use it as-is.
    # If it arrived as a raw VibeVoice turns list, merge it (fallback path).
    if "turns" in diarization_payload and "words" in diarization_payload:
        merged = diarization_payload
    else:
        # Fallback: treat the whole payload as raw VibeVoice turns with no word alignment
        raw_turns = diarization_payload.get("vibevoice_turns") or []
        merged = merge_vibevoice_outputs(vibevoice_turns=raw_turns, word_alignments=[])

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
