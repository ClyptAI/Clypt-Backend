from __future__ import annotations

import re
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


_TOKEN_NORMALIZE_RE = re.compile(r"[^a-z0-9']+")


def _normalize_token(token: str) -> str:
    return _TOKEN_NORMALIZE_RE.sub("", token.lower())


def _tokenize_for_alignment(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.split():
        norm = _normalize_token(raw)
        if norm:
            out.append(norm)
    return out


def _assign_word_ids_by_transcript(
    *,
    turns: list[dict[str, Any]],
    words: list[dict[str, Any]],
) -> list[list[str]]:
    """
    Assign aligned words to turns by monotonic transcript-token matching first,
    with overlap fallback only for turns that still have no words.
    """
    word_norm = [_normalize_token(str(w.get("text") or "")) for w in words]
    word_id_to_index = {w["word_id"]: idx for idx, w in enumerate(words)}
    assigned_word_indices: set[int] = set()

    turn_word_ids: list[list[str]] = []
    cursor = 0
    for turn in turns:
        transcript_text = str(turn.get("transcript_text") or "").strip()
        start_ms = int(turn.get("start_ms") or 0)
        end_ms = int(turn.get("end_ms") or 0)
        turn_tokens = _tokenize_for_alignment(transcript_text)

        matched_ids: list[str] = []
        if turn_tokens:
            for token in turn_tokens:
                found_index = -1
                for idx in range(cursor, len(words)):
                    if word_norm[idx] == token:
                        found_index = idx
                        break
                if found_index < 0:
                    continue
                wid = words[found_index]["word_id"]
                matched_ids.append(wid)
                assigned_word_indices.add(found_index)
                cursor = found_index + 1

        # Fallback for unmatched turns: use unassigned words that overlap the turn.
        if not matched_ids and end_ms > start_ms:
            for idx, word in enumerate(words):
                if idx in assigned_word_indices:
                    continue
                overlap = _overlap_ms(
                    start_ms,
                    end_ms,
                    int(word.get("start_ms") or 0),
                    int(word.get("end_ms") or 0),
                )
                if overlap > 0:
                    wid = word["word_id"]
                    matched_ids.append(wid)
                    assigned_word_indices.add(idx)

        # Boundary-spill repair: drop leading words that start before turn start
        # when they duplicate the previous turn tail.
        if matched_ids and turn_word_ids:
            prev_ids = turn_word_ids[-1]
            while matched_ids and prev_ids:
                first_curr = words[word_id_to_index[matched_ids[0]]]
                if int(first_curr.get("start_ms") or 0) >= start_ms:
                    break
                prev_last = words[word_id_to_index[prev_ids[-1]]]
                if _normalize_token(str(first_curr.get("text") or "")) != _normalize_token(
                    str(prev_last.get("text") or "")
                ):
                    break
                matched_ids.pop(0)

        turn_word_ids.append(matched_ids)

    return turn_word_ids


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

        turns.append(
            {
                "turn_id": f"t_{idx:06d}",
                "speaker_id": speaker_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "transcript_text": transcript_text,
                "word_ids": [],
                "identification_match": None,  # VibeVoice does not have voiceprint ID
            }
        )

    if turns and words:
        assigned = _assign_word_ids_by_transcript(turns=turns, words=words)
        for turn, word_ids in zip(turns, assigned):
            turn["word_ids"] = word_ids

            # If the turn has no content, reconstruct from its assigned words.
            if not turn["transcript_text"] and word_ids:
                word_map = {w["word_id"]: w["text"] for w in words}
                turn["transcript_text"] = " ".join(word_map.get(wid, "") for wid in word_ids).strip()

            # Keep word speaker IDs consistent with turn ownership.
            for wid in word_ids:
                for word in words:
                    if word["word_id"] == wid:
                        word["speaker_id"] = turn["speaker_id"]
                        break

    return {"words": words, "turns": turns}


__all__ = ["merge_vibevoice_outputs"]
