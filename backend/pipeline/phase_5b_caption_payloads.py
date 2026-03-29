#!/usr/bin/env python3
"""
Phase 5B: build timed caption chunks for already-selected clips.

This stage is intentionally downstream-only:
- input: an existing Remotion payload array produced after clip selection
- input: Phase 1 audio ledger with word timestamps
- output: a new payload array with optional `captions` fields

It does not rescore clips and it does not modify upstream phase outputs unless
the caller explicitly points the output path at the input path.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_AUDIO_PATH = OUTPUTS_DIR / "phase_1_audio.json"
DEFAULT_INPUT_CANDIDATES = (
    OUTPUTS_DIR / "remotion_payloads_array.json",
    OUTPUTS_DIR / "crowd_remotion_payloads_array.json",
    OUTPUTS_DIR / "remotion_payloads_array_audience.json",
)
DEFAULT_MAX_WORDS = int(os.getenv("CAPTION_MAX_WORDS", "4") or 4)
DEFAULT_MAX_CHARS = int(os.getenv("CAPTION_MAX_CHARS", "28") or 28)
DEFAULT_GAP_MS = int(os.getenv("CAPTION_GAP_MS", "300") or 300)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_5b_captions")


def _resolve_input_path(input_path: str | Path | None) -> Path:
    if input_path:
        return Path(input_path)

    override = str(os.getenv("CAPTION_PAYLOAD_INPUT_PATH", "") or "").strip()
    if override:
        return Path(override)

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No clip payload file found. Checked: "
        + ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    )


def _resolve_audio_path(audio_path: str | Path | None) -> Path:
    if audio_path:
        return Path(audio_path)

    override = str(os.getenv("CAPTION_AUDIO_LEDGER_PATH", "") or "").strip()
    if override:
        return Path(override)

    return DEFAULT_AUDIO_PATH


def _resolve_output_path(output_path: str | Path | None, input_payload_path: Path) -> Path:
    if output_path:
        return Path(output_path)

    override = str(os.getenv("CAPTION_PAYLOAD_OUTPUT_PATH", "") or "").strip()
    if override:
        return Path(override)

    return input_payload_path.with_name(f"{input_payload_path.stem}_captioned.json")


def _load_payload_array(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else [payload]


def _load_words(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    words = payload.get("words", []) if isinstance(payload, dict) else []
    return words if isinstance(words, list) else []


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _word_start_ms(word: dict) -> int:
    return int(word.get("start_time_ms", 0) or 0)


def _word_end_ms(word: dict) -> int:
    start_ms = _word_start_ms(word)
    return int(word.get("end_time_ms", start_ms) or start_ms)


def _overlaps_window(word: dict, start_ms: int, end_ms: int) -> bool:
    word_start = _word_start_ms(word)
    word_end = _word_end_ms(word)
    return word_start < end_ms and word_end > start_ms


def _speaker_tag(word: dict) -> str | None:
    speaker = word.get("speaker_tag") or word.get("speaker_track_id")
    if speaker is None:
        return None
    text = str(speaker).strip()
    return text or None


def _ends_sentence(token: str) -> bool:
    return token.rstrip().endswith((".", "!", "?", "...", "…"))


def _finalize_chunk(tokens: list[dict], clip_start_ms: int, clip_end_ms: int) -> dict | None:
    if not tokens:
        return None

    text = " ".join(str(token["text"]) for token in tokens).strip()
    if not text:
        return None

    start_ms = max(clip_start_ms, int(tokens[0]["start_ms"]))
    end_ms = min(clip_end_ms, int(tokens[-1]["end_ms"]))
    if end_ms <= start_ms:
        end_ms = min(clip_end_ms, start_ms + 1)

    return {
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_tag": tokens[0].get("speaker_tag"),
        "words": [str(token["text"]) for token in tokens],
    }


def build_caption_chunks(
    words: list[dict],
    clip_start_ms: int,
    clip_end_ms: int,
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    gap_ms: int = DEFAULT_GAP_MS,
) -> list[dict]:
    clip_words = [
        {
            "text": _word_text(word),
            "start_ms": max(clip_start_ms, _word_start_ms(word)),
            "end_ms": min(clip_end_ms, _word_end_ms(word)),
            "speaker_tag": _speaker_tag(word),
        }
        for word in words
        if _overlaps_window(word, clip_start_ms, clip_end_ms)
    ]
    clip_words = [word for word in clip_words if word["text"]]

    chunks: list[dict] = []
    current_tokens: list[dict] = []

    def flush() -> None:
        chunk = _finalize_chunk(current_tokens, clip_start_ms, clip_end_ms)
        if chunk is not None:
            chunks.append(chunk)
        current_tokens.clear()

    for token in clip_words:
        if current_tokens:
            prev = current_tokens[-1]
            candidate_text = " ".join(
                [*(str(existing["text"]) for existing in current_tokens), str(token["text"])]
            ).strip()
            gap = int(token["start_ms"]) - int(prev["end_ms"])
            speaker_changed = (
                prev.get("speaker_tag")
                and token.get("speaker_tag")
                and prev.get("speaker_tag") != token.get("speaker_tag")
            )
            should_flush = (
                _ends_sentence(str(prev["text"]))
                or gap > gap_ms
                or speaker_changed
                or len(current_tokens) >= max_words
                or len(candidate_text) > max_chars
            )
            if should_flush:
                flush()

        current_tokens.append(token)

    flush()
    return chunks



def augment_payloads_with_captions(
    payloads: list[dict],
    words: list[dict],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    gap_ms: int = DEFAULT_GAP_MS,
) -> list[dict]:
    augmented: list[dict] = []
    for payload in payloads:
        start_ms = int(payload.get("clip_start_ms", 0) or 0)
        end_ms = int(payload.get("clip_end_ms", start_ms) or start_ms)
        next_payload = dict(payload)
        next_payload["captions"] = build_caption_chunks(
            words,
            start_ms,
            end_ms,
            max_words=max_words,
            max_chars=max_chars,
            gap_ms=gap_ms,
        )
        augmented.append(next_payload)
    return augmented


def main(
    *,
    input_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    input_payload_path = _resolve_input_path(input_path)
    resolved_audio_path = _resolve_audio_path(audio_path)
    resolved_output_path = _resolve_output_path(output_path, input_payload_path)

    if not input_payload_path.exists():
        raise FileNotFoundError(f"Missing clip payload input: {input_payload_path}")
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 audio ledger: {resolved_audio_path}")

    payloads = _load_payload_array(input_payload_path)
    words = _load_words(resolved_audio_path)
    augmented = augment_payloads_with_captions(payloads, words)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")

    caption_count = sum(len(payload.get("captions", [])) for payload in augmented)
    log.info("=" * 60)
    log.info("PHASE 5B - CAPTION PAYLOADS")
    log.info("=" * 60)
    log.info("Input payloads: %d", len(payloads))
    log.info("Transcript words: %d", len(words))
    log.info("Caption chunks: %d", caption_count)
    log.info("Output saved -> %s", resolved_output_path)

    return {
        "input_payload_path": str(input_payload_path),
        "audio_path": str(resolved_audio_path),
        "output_path": str(resolved_output_path),
        "payload_count": len(augmented),
        "caption_count": caption_count,
    }


if __name__ == "__main__":
    main()