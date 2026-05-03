from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .payloads import DiarizationPayload, EmotionSegmentsPayload, YamnetPayload


_TAG_TEXT_RE = re.compile(r"^\s*[\[(](?P<label>[^)\]]+)[)\]]\s*$")
_SPEAKER_RE = re.compile(r"^speaker[_ -]?(\d+)$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


class ScribeAdapterError(ValueError):
    """Raised when Scribe output cannot satisfy Phase1 audio contracts."""


@dataclass(slots=True)
class ScribePhase1Payloads:
    diarization_payload: DiarizationPayload
    yamnet_payload: YamnetPayload
    emotion2vec_payload: EmotionSegmentsPayload


def _raw_response(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "raw"):
        raw = raw.raw
    if not isinstance(raw, dict):
        raise ScribeAdapterError(
            f"Scribe adapter expected response object, got {type(raw).__name__}"
        )
    return raw


def _ms(value: Any, *, field_name: str) -> int:
    if value is None:
        raise ScribeAdapterError(f"Scribe item missing {field_name}.")
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError) as exc:
        raise ScribeAdapterError(f"Scribe item has invalid {field_name}: {value!r}") from exc


def _repair_non_positive_word_timing(
    *,
    start_ms: int,
    end_ms: int,
    previous_word_end_ms: int | None,
) -> tuple[int, int, bool]:
    if end_ms > start_ms:
        return start_ms, end_ms, False
    repaired_start_ms = start_ms
    if previous_word_end_ms is not None and repaired_start_ms < previous_word_end_ms:
        repaired_start_ms = previous_word_end_ms
    return repaired_start_ms, repaired_start_ms + 1, True


def _speaker_id(raw_speaker: Any, mapping: dict[str, str]) -> str:
    key = str(raw_speaker or "").strip()
    if not key:
        raise ScribeAdapterError("Scribe word token missing speaker_id.")
    match = _SPEAKER_RE.match(key)
    if match:
        return f"SPEAKER_{int(match.group(1))}"
    if key not in mapping:
        sanitized = _NON_ALNUM_RE.sub("_", key.upper()).strip("_") or "UNKNOWN"
        candidate = f"SPEAKER_{sanitized}"
        existing = set(mapping.values())
        if candidate in existing:
            candidate = f"{candidate}_{len(mapping) + 1}"
        mapping[key] = candidate
    return mapping[key]


def _event_label(item: dict[str, Any]) -> str:
    for key in ("label", "event_label", "text"):
        value = str(item.get(key) or "").strip()
        if not value:
            continue
        match = _TAG_TEXT_RE.match(value)
        return (match.group("label") if match else value).strip()
    item_type = str(item.get("type") or "").strip()
    return item_type.replace("_", " ").strip()


def _join_turn_text(words: list[dict[str, Any]]) -> str:
    text = ""
    for word in words:
        token = str(word.get("text") or "")
        if not token:
            continue
        if not text:
            text = token
        elif re.match(r"^[,.;:!?%)]", token):
            text += token
        elif text.endswith(("(", "[", "$")):
            text += token
        else:
            text += " " + token
    return text.strip()


def _build_turns(words: list[dict[str, Any]], *, turn_gap_ms: int) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        turn_index = len(turns) + 1
        turns.append(
            {
                "turn_id": f"t_{turn_index:06d}",
                "speaker_id": current[0]["speaker_id"],
                "start_ms": current[0]["start_ms"],
                "end_ms": current[-1]["end_ms"],
                "transcript_text": _join_turn_text(current),
                "word_ids": [word["word_id"] for word in current],
                "identification_match": None,
            }
        )
        current.clear()

    for word in words:
        if current:
            speaker_changed = word["speaker_id"] != current[-1]["speaker_id"]
            gap_ms = int(word["start_ms"]) - int(current[-1]["end_ms"])
            if speaker_changed or gap_ms > turn_gap_ms:
                flush()
        current.append(word)
    flush()
    return turns


def adapt_scribe_response(
    raw: Any,
    *,
    turn_gap_ms: int = 1200,
) -> ScribePhase1Payloads:
    response = _raw_response(raw)
    items = response.get("words")
    if not isinstance(items, list):
        raise ScribeAdapterError("Scribe response must include a words list.")

    speaker_mapping: dict[str, str] = {}
    canonical_words: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    previous_word_end_ms: int | None = None

    for item in items:
        if not isinstance(item, dict):
            raise ScribeAdapterError("Every Scribe words item must be an object.")
        item_type = str(item.get("type") or "word").strip() or "word"
        if item_type == "word":
            start_ms = _ms(item.get("start"), field_name="start")
            end_ms = _ms(item.get("end"), field_name="end")
            start_ms, end_ms, timing_repaired = _repair_non_positive_word_timing(
                start_ms=start_ms,
                end_ms=end_ms,
                previous_word_end_ms=previous_word_end_ms,
            )
            word_id = f"w_{len(canonical_words) + 1:06d}"
            word = {
                "word_id": word_id,
                "text": str(item.get("text") or ""),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": _speaker_id(item.get("speaker_id"), speaker_mapping),
                "scribe": dict(item),
            }
            if timing_repaired:
                word["scribe_timing_repaired"] = True
            canonical_words.append(word)
            previous_word_end_ms = end_ms
            continue

        has_timing = item.get("start") is not None and item.get("end") is not None
        if has_timing:
            start_ms = _ms(item.get("start"), field_name="start")
            end_ms = _ms(item.get("end"), field_name="end")
            if end_ms <= start_ms:
                continue
            confidence = item.get("confidence")
            if confidence is None:
                confidence = item.get("score")
            events.append(
                {
                    "event_label": _event_label(item),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "confidence": float(confidence) if confidence is not None else None,
                    "source": "scribe_v2",
                    "scribe": dict(item),
                }
            )

    if not canonical_words:
        raise ScribeAdapterError("Scribe response contained no timed word tokens.")

    diarization_payload = DiarizationPayload.model_validate(
        {
            "words": canonical_words,
            "turns": _build_turns(canonical_words, turn_gap_ms=int(turn_gap_ms)),
        }
    )
    yamnet_payload = YamnetPayload.model_validate({"events": events})
    emotion_payload = EmotionSegmentsPayload.model_validate({"segments": []})
    return ScribePhase1Payloads(
        diarization_payload=diarization_payload,
        yamnet_payload=yamnet_payload,
        emotion2vec_payload=emotion_payload,
    )


__all__ = [
    "ScribeAdapterError",
    "ScribePhase1Payloads",
    "adapt_scribe_response",
]
