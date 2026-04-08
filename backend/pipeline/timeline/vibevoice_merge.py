from __future__ import annotations

from typing import Any


def _to_ms(value: Any) -> int:
    """Convert seconds (float) or already-ms (large int) to ms int."""
    if value is None:
        return 0
    f = float(value)
    # VibeVoice outputs seconds; if it's already large assume ms
    return int(round(f * 1000)) if f < 10_000 else int(round(f))


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def merge_vibevoice_outputs(
    *,
    vibevoice_turns: list[dict[str, Any]],
    word_alignments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert VibeVoice parsed output + ctc-forced-aligner words into the canonical
    {words, turns} shape expected by timeline_builder and downstream phases.

    Args:
        vibevoice_turns: List of turn dicts from the VibeVoice vLLM provider run().
            Each has keys: Start (float s), End (float s), Speaker (int), Content (str).
        word_alignments: List of word dicts from ForcedAlignmentProvider.run().
            Each has: word_id, text, start_ms, end_ms, speaker_id.
            May be empty if ctc-forced-aligner is not installed.

    Returns:
        {
            "words": [{"word_id", "text", "start_ms", "end_ms", "speaker_id"}, ...],
            "turns": [{"turn_id", "speaker_id", "start_ms", "end_ms",
                        "transcript_text", "word_ids", "identification_match"}, ...],
        }
    """
    # --- Normalize words -------------------------------------------------
    # If we have forced-aligned words, use them directly (already normalized).
    # If not, synthesize word entries from turn content (no per-word timing).
    words: list[dict[str, Any]] = []

    if word_alignments:
        # Re-index to ensure consistent word_id sequence from w_000001
        for idx, word in enumerate(word_alignments, start=1):
            words.append(
                {
                    "word_id": f"w_{idx:06d}",
                    "text": str(word.get("text") or "").strip(),
                    "start_ms": int(word.get("start_ms") or 0),
                    "end_ms": int(word.get("end_ms") or 0),
                    "speaker_id": str(word.get("speaker_id") or "UNKNOWN"),
                }
            )
    else:
        # Fallback: split turn content into word tokens; no individual timestamps
        word_idx = 1
        for turn in vibevoice_turns:
            turn_start_ms = _to_ms(turn.get("Start") or 0)
            turn_end_ms = _to_ms(turn.get("End") or 0)
            speaker_id = f"SPEAKER_{int(turn.get('Speaker') or 0)}"
            content = str(turn.get("Content") or "").strip()
            tokens = content.split()
            if not tokens:
                continue
            # Distribute timing evenly across tokens
            duration_ms = max(0, turn_end_ms - turn_start_ms)
            per_word_ms = duration_ms // len(tokens) if tokens else 0
            for i, token in enumerate(tokens):
                w_start = turn_start_ms + i * per_word_ms
                w_end = w_start + per_word_ms
                words.append(
                    {
                        "word_id": f"w_{word_idx:06d}",
                        "text": token,
                        "start_ms": w_start,
                        "end_ms": w_end,
                        "speaker_id": speaker_id,
                    }
                )
                word_idx += 1

    # --- Normalize turns --------------------------------------------------
    turns: list[dict[str, Any]] = []
    for idx, raw_turn in enumerate(vibevoice_turns, start=1):
        start_ms = _to_ms(raw_turn.get("Start") or 0)
        end_ms = _to_ms(raw_turn.get("End") or 0)
        # Normalize speaker: VibeVoice uses int Speaker IDs → "SPEAKER_0"
        speaker_int = int(raw_turn.get("Speaker") or 0)
        speaker_id = f"SPEAKER_{speaker_int}"
        transcript_text = str(raw_turn.get("Content") or "").strip()

        # Collect word_ids whose timing overlaps this turn
        overlapping_word_ids = [
            word["word_id"]
            for word in words
            if _overlap_ms(start_ms, end_ms, word["start_ms"], word["end_ms"]) > 0
        ]

        # If the turn has no content, reconstruct from overlapping words
        if not transcript_text and overlapping_word_ids:
            word_map = {w["word_id"]: w["text"] for w in words}
            transcript_text = " ".join(word_map.get(wid, "") for wid in overlapping_word_ids).strip()

        turns.append(
            {
                "turn_id": f"t_{idx:06d}",
                "speaker_id": speaker_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "transcript_text": transcript_text,
                "word_ids": overlapping_word_ids,
                "identification_match": None,  # VibeVoice does not have voiceprint ID
            }
        )

    return {"words": words, "turns": turns}


__all__ = ["merge_vibevoice_outputs"]
