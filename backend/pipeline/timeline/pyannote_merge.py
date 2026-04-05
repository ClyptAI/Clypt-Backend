from __future__ import annotations

from typing import Any


def _to_ms(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1000)) if float(value) < 10_000 else int(round(float(value)))
    text = str(value).strip()
    if not text:
        return 0
    number = float(text)
    return int(round(number * 1000)) if number < 10_000 else int(round(number))


def _get_first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def merge_pyannote_outputs(*, diarize_payload: dict, identify_payload: dict | None = None) -> dict:
    """Merge pyannote diarization/transcription outputs with optional identify data."""
    raw_words = diarize_payload.get("wordLevelTranscription") or diarize_payload.get("words") or []
    words: list[dict[str, Any]] = []
    for idx, raw_word in enumerate(raw_words, start=1):
        text = _get_first(raw_word, "text", "word", default="").strip()
        if not text:
            continue
        words.append(
            {
                "word_id": f"w_{idx:06d}",
                "text": text,
                "start_ms": _to_ms(_get_first(raw_word, "start_ms", "startTimeMs", "start_time_ms", "start")),
                "end_ms": _to_ms(_get_first(raw_word, "end_ms", "endTimeMs", "end_time_ms", "end")),
                "speaker_id": _get_first(raw_word, "speaker_id", "speaker", "speakerId"),
            }
        )

    turn_text_candidates = diarize_payload.get("turnLevelTranscription") or []
    raw_turns = diarize_payload.get("diarization") or turn_text_candidates or []
    turns: list[dict[str, Any]] = []
    for idx, raw_turn in enumerate(raw_turns, start=1):
        start_ms = _to_ms(_get_first(raw_turn, "start_ms", "startTimeMs", "start_time_ms", "start"))
        end_ms = _to_ms(_get_first(raw_turn, "end_ms", "endTimeMs", "end_time_ms", "end"))
        speaker_id = _get_first(raw_turn, "speaker_id", "speaker", "diarizationSpeaker", "speakerId", default="UNKNOWN")
        transcript_text = str(_get_first(raw_turn, "text", "transcript_text", default="") or "").strip()

        if not transcript_text:
            best_text = ""
            best_overlap = -1
            for text_turn in turn_text_candidates:
                text_start = _to_ms(_get_first(text_turn, "start_ms", "startTimeMs", "start_time_ms", "start"))
                text_end = _to_ms(_get_first(text_turn, "end_ms", "endTimeMs", "end_time_ms", "end"))
                text_speaker = _get_first(text_turn, "speaker_id", "speaker", "speakerId")
                if text_speaker and speaker_id and text_speaker != speaker_id:
                    continue
                overlap = _overlap_ms(start_ms, end_ms, text_start, text_end)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_text = str(_get_first(text_turn, "text", "transcript_text", default="") or "").strip()
            transcript_text = best_text

        turns.append(
            {
                "turn_id": f"t_{idx:06d}",
                "speaker_id": speaker_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "transcript_text": transcript_text,
                "identification_match": None,
            }
        )

    for turn in turns:
        overlapping_word_ids = [
            word["word_id"]
            for word in words
            if _overlap_ms(turn["start_ms"], turn["end_ms"], word["start_ms"], word["end_ms"]) > 0
        ]
        turn["word_ids"] = overlapping_word_ids
        if not turn["transcript_text"]:
            turn["transcript_text"] = " ".join(
                word["text"] for word in words if word["word_id"] in overlapping_word_ids
            ).strip()

    identification_entries = (identify_payload or {}).get("identification") or []
    for turn in turns:
        best_match = None
        best_overlap = -1
        for entry in identification_entries:
            entry_speaker = _get_first(entry, "diarizationSpeaker", "speaker", "speaker_id")
            if entry_speaker and entry_speaker != turn["speaker_id"]:
                continue
            entry_start = _to_ms(_get_first(entry, "start_ms", "startTimeMs", "start_time_ms", "start"))
            entry_end = _to_ms(_get_first(entry, "end_ms", "endTimeMs", "end_time_ms", "end"))
            overlap = _overlap_ms(turn["start_ms"], turn["end_ms"], entry_start, entry_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = entry.get("match")
        turn["identification_match"] = best_match

    return {
        "words": words,
        "turns": turns,
    }
