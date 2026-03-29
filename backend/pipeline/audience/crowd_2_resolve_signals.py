#!/usr/bin/env python3
"""
Crowd Clip Stage 2: resolve comments to concrete video moments using transcript.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.audience.crowd_transcript import load_transcript_words, transcript_duration_ms
from pipeline.audience.crowd_types import CrowdReference, ResolvedCrowdComment
from pipeline.audience.crowd_utils import OUTPUTS_DIR, extract_video_id, normalize_text, tokenize

INPUT_PATH = OUTPUTS_DIR / "crowd_1_youtube_signals.json"
OUTPUT_PATH = OUTPUTS_DIR / "crowd_2_resolved_signals.json"

TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")
QUOTE_PATTERN = re.compile(r"[\u201c\u201d\"\'`]{1,2}([^\u201c\u201d\"\'`]{8,180})[\u201c\u201d\"\'`]{1,2}")
PHRASE_HINT_PATTERNS = [
    re.compile(r"when (?:he|she|they|senator|someone|the guy)\s+(?:says|said)\s+([^.!?\n]{8,160})", re.IGNORECASE),
    re.compile(r"the part where\s+([^.!?\n]{8,160})", re.IGNORECASE),
    re.compile(r"when they ask\s+([^.!?\n]{8,160})", re.IGNORECASE),
]
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at",
    "for", "from", "with", "that", "this", "it", "you", "i", "me", "my", "we", "our", "they", "their", "he",
    "she", "him", "her", "them", "your", "yours", "when", "what", "why", "how", "who", "just", "really",
    "very", "so", "if", "then", "there", "here", "about", "into", "out", "up", "down", "can", "could",
    "would", "should", "know", "yeah", "like", "watch", "watched", "thing", "things", "stuff", "also",
    "even", "ever", "because", "care", "cares", "problem", "right", "wrong", "proof",
}
GENERIC_TOPIC_TOKENS = {
    "senator", "cotton", "tiktok", "tick", "tok", "ceo", "china", "chinese", "singapore", "hearing",
    "congress", "committee", "question", "questions", "video", "clip", "guy", "people", "person", "party",
    "company", "government", "child", "children", "online", "safety", "forbes", "news",
}
GENERIC_COMMENT_PATTERNS = [
    re.compile(r"\bwho('?s| is) here\b", re.IGNORECASE),
    re.compile(r"\bthis is (wild|crazy|insane|nuts)\b", re.IGNORECASE),
    re.compile(r"\bwhat a clown\b", re.IGNORECASE),
    re.compile(r"\b(he|she|they) is (crazy|insane|stupid|ignorant)\b", re.IGNORECASE),
    re.compile(r"\b(lol|lmao|lmfao|wtf)\b", re.IGNORECASE),
]
NOISY_REPLY_PATTERNS = [
    re.compile(r"^\s*@", re.IGNORECASE),
    re.compile(r"\b(which video|where('?s| is) the proof|you know nothing|sounds like a you problem|man up)\b", re.IGNORECASE),
    re.compile(r"\b(who cares|i respect|not lying|not really|anyone could say|wouldn'?t surprise me)\b", re.IGNORECASE),
]
SPAM_PATTERNS = [
    re.compile(r"\b(scam|claim\s+\$?\d+|visit your site|visit our site|gift card|giveaway winner)\b", re.IGNORECASE),
]
MOMENT_CUE_PATTERNS = [
    re.compile(r"\b(when|part|moment|line|scene|timestamp|time|clip)\b", re.IGNORECASE),
    re.compile(r"\b(say|said|says|hear|heard|swear|cuss|laugh|laughed|cry|cried|started|buried|coffin|tombstone)\b", re.IGNORECASE),
]
MIN_KEYWORD_CHARS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_2")


def _parse_timestamp_to_ms(token: str) -> int | None:
    parts = token.split(":")
    try:
        ints = [int(p) for p in parts]
    except ValueError:
        return None
    if len(ints) == 2:
        minutes, seconds = ints
        return ((minutes * 60) + seconds) * 1000
    if len(ints) == 3:
        hours, minutes, seconds = ints
        return ((hours * 3600) + (minutes * 60) + seconds) * 1000
    return None


def _build_transcript_index(words: list[dict]) -> tuple[list[str], list[int], list[int]]:
    tokens: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for word in words:
        token = normalize_text(str(word.get("word", "")))
        if not token:
            continue
        tokens.append(token)
        starts.append(int(word.get("start_time_ms", 0) or 0))
        ends.append(int(word.get("end_time_ms", 0) or 0))
    return tokens, starts, ends


def _is_generic_reaction_comment(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    if len(normalized.split()) <= 3:
        return True
    return any(pattern.search(text) for pattern in [*GENERIC_COMMENT_PATTERNS, *SPAM_PATTERNS])


def _informative_keywords(comment_text: str) -> list[str]:
    ranked: list[str] = []
    for token in tokenize(comment_text):
        if token in STOPWORDS or token in GENERIC_TOPIC_TOKENS:
            continue
        if len(token) < MIN_KEYWORD_CHARS:
            continue
        if token.isdigit():
            continue
        ranked.append(token)
    return ranked


def _looks_like_noisy_reply(comment_text: str) -> bool:
    return any(pattern.search(comment_text) for pattern in NOISY_REPLY_PATTERNS)


def _has_moment_cue(comment_text: str) -> bool:
    return any(pattern.search(comment_text) for pattern in MOMENT_CUE_PATTERNS)


def _find_phrase_match(phrase: str, transcript_tokens: list[str], starts: list[int], ends: list[int]) -> CrowdReference | None:
    phrase_tokens = [tok for tok in tokenize(phrase) if tok not in STOPWORDS]
    if len(phrase_tokens) < 2:
        phrase_tokens = tokenize(phrase)
    if len(phrase_tokens) < 2:
        return None

    target = " ".join(phrase_tokens)
    best: tuple[float, int, int] | None = None

    min_len = max(2, len(phrase_tokens) - 2)
    max_len = min(len(transcript_tokens), len(phrase_tokens) + 3)
    for start_idx in range(0, len(transcript_tokens)):
        for size in range(min_len, max_len + 1):
            end_idx = start_idx + size
            if end_idx > len(transcript_tokens):
                break
            candidate = " ".join(transcript_tokens[start_idx:end_idx])
            score = SequenceMatcher(None, target, candidate).ratio()
            if score < 0.72:
                continue
            if best is None or score > best[0]:
                best = (score, start_idx, end_idx - 1)

    if best is None:
        return None

    score, start_idx, end_idx = best
    start_ms = starts[start_idx]
    end_ms = ends[end_idx]
    anchor_ms = int(round((start_ms + end_ms) / 2))
    return CrowdReference(
        kind="quote_match",
        anchor_ms=anchor_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=min(0.98, max(0.72, score)),
        matched_text=phrase.strip(),
        note="Matched quoted or hinted phrase against transcript",
    )


def _extract_phrase_candidates(comment_text: str) -> list[str]:
    phrases: list[str] = []
    for match in QUOTE_PATTERN.finditer(comment_text):
        phrase = match.group(1).strip()
        if phrase:
            phrases.append(phrase)
    for pattern in PHRASE_HINT_PATTERNS:
        for match in pattern.finditer(comment_text):
            phrase = match.group(1).strip()
            if phrase:
                phrases.append(phrase)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = normalize_text(phrase)
        if key and key not in seen:
            seen.add(key)
            deduped.append(phrase)
    return deduped


def _resolve_keyword_overlap(
    comment_text: str,
    transcript_tokens: list[str],
    starts: list[int],
    ends: list[int],
    transcript_counts: Counter,
) -> CrowdReference | None:
    if _is_generic_reaction_comment(comment_text):
        return None
    if _looks_like_noisy_reply(comment_text):
        return None

    keywords = _informative_keywords(comment_text)
    counts = Counter(keywords)
    ranked = [tok for tok, _ in counts.most_common(8)]
    if len(ranked) < 2:
        return None

    best: tuple[float, int, int, list[str], int, float] | None = None
    has_moment_cue = _has_moment_cue(comment_text)
    for start_idx in range(0, len(transcript_tokens), 5):
        window_start_ms = starts[start_idx]
        window_end_ms = window_start_ms + 9000
        end_idx = start_idx
        while end_idx < len(ends) and ends[end_idx] <= window_end_ms:
            end_idx += 1
        if end_idx <= start_idx:
            continue
        window_tokens = transcript_tokens[start_idx:end_idx]
        overlap_tokens = sorted(set(window_tokens) & set(ranked))
        overlap = len(overlap_tokens)
        if overlap < 2:
            continue
        overlap_positions = [idx for idx, token in enumerate(window_tokens) if token in overlap_tokens]
        if not overlap_positions:
            continue
        span_tokens = (max(overlap_positions) - min(overlap_positions) + 1)
        concentration = max(0.0, 1.0 - max(0.0, span_tokens - 10) / 26.0)
        if concentration < 0.3:
            continue
        weighted_overlap = sum(1.0 / max(1, transcript_counts.get(token, 1)) for token in overlap_tokens)
        rare_overlap_count = sum(1 for token in overlap_tokens if transcript_counts.get(token, 0) <= 2)
        if overlap < 3 and rare_overlap_count == 0 and not has_moment_cue:
            continue
        if weighted_overlap < 0.16:
            continue
        score = weighted_overlap + (0.12 * rare_overlap_count) + (0.05 * min(3, overlap)) + (0.08 * concentration)
        if best is None or score > best[0]:
            best = (score, start_idx, end_idx - 1, overlap_tokens, rare_overlap_count, concentration)

    if best is None:
        return None

    score, start_idx, end_idx, overlap_tokens, rare_overlap_count, concentration = best
    start_ms = starts[start_idx]
    end_ms = ends[end_idx]
    anchor_ms = int(round((start_ms + end_ms) / 2))
    confidence = min(
        0.64,
        0.18
        + (0.12 * min(3, len(overlap_tokens)))
        + (0.16 * min(2, rare_overlap_count))
        + (0.22 * concentration)
        + (0.42 * min(1.0, score)),
    )
    return CrowdReference(
        kind="keyword_overlap",
        anchor_ms=anchor_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
        matched_text=" ".join(overlap_tokens[: min(4, len(overlap_tokens))]),
        note="Matched concentrated informative comment keywords against transcript window",
    )


def _comment_excitement_score(text: str) -> float:
    upper_tokens = sum(1 for token in text.split() if len(token) >= 3 and token.isupper())
    exclamations = text.count("!")
    questions = text.count("?")
    elongated = len(re.findall(r"(.)\1{2,}", text.lower()))
    score = (upper_tokens * 0.2) + (exclamations * 0.12) + (questions * 0.05) + (elongated * 0.08)
    return max(0.0, min(1.0, score))


def main(youtube_url: str | None = None) -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing Crowd Clip ingest artifact: {INPUT_PATH}")

    signals = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    source_url = youtube_url or str(signals.get("source_url", "") or "")
    video_id = str(signals.get("video_id", "") or "")
    if not video_id:
        video_id = extract_video_id(source_url)

    words, transcript_payload = load_transcript_words(video_id)
    if not words:
        raise RuntimeError(f"No transcript words available for Crowd Clip resolution ({video_id}).")
    transcript_tokens, starts, ends = _build_transcript_index(words)
    transcript_counts = Counter(transcript_tokens)
    duration_ms = transcript_duration_ms(words)

    resolved_comments: list[ResolvedCrowdComment] = []
    total_refs = 0

    for comment in signals.get("comments", []):
        text = str(comment.get("text", "") or "")
        if not text.strip():
            continue

        refs: list[CrowdReference] = []
        for match in TIMESTAMP_PATTERN.finditer(text):
            timestamp_text = match.group(1)
            anchor_ms = _parse_timestamp_to_ms(timestamp_text)
            if anchor_ms is None:
                continue
            refs.append(
                CrowdReference(
                    kind="explicit_timestamp",
                    anchor_ms=anchor_ms,
                    start_ms=max(0, anchor_ms - 1500),
                    end_ms=min(duration_ms or anchor_ms + 1500, anchor_ms + 1500),
                    confidence=1.0,
                    matched_text=timestamp_text,
                    note="Direct timestamp mention in comment",
                )
            )

        for phrase in _extract_phrase_candidates(text):
            match_ref = _find_phrase_match(phrase, transcript_tokens, starts, ends)
            if match_ref:
                refs.append(match_ref)

        if not refs:
            keyword_ref = _resolve_keyword_overlap(text, transcript_tokens, starts, ends, transcript_counts)
            if keyword_ref:
                refs.append(keyword_ref)

        resolved = ResolvedCrowdComment(
            comment_id=str(comment.get("comment_id", "") or ""),
            parent_comment_id=comment.get("parent_comment_id"),
            is_reply=bool(comment.get("is_reply", False)),
            author_name=str(comment.get("author_name", "") or ""),
            like_count=int(comment.get("like_count", 0) or 0),
            reply_count=int(comment.get("reply_count", 0) or 0),
            published_at=str(comment.get("published_at", "") or ""),
            text=text,
            excitement_score=_comment_excitement_score(text),
            references=refs,
        )
        total_refs += len(refs)
        resolved_comments.append(resolved)

    payload = {
        "schema_version": "crowd-clip-resolved-v1",
        "source_url": source_url,
        "video_id": video_id,
        "transcript_source": str(transcript_payload.get("transcript_source", "") or ""),
        "transcript_word_count": len(words),
        "transcript_duration_ms": duration_ms,
        "comment_count": len(resolved_comments),
        "resolved_reference_count": total_refs,
        "comments": [comment.model_dump() for comment in resolved_comments],
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("=" * 60)
    log.info("CROWD CLIP \u2014 STAGE 2 SIGNAL RESOLUTION")
    log.info("=" * 60)
    log.info("Transcript words: %d", len(words))
    log.info("Resolved references: %d", total_refs)
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return payload


if __name__ == "__main__":
    main()
