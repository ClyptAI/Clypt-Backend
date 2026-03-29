#!/usr/bin/env python3
"""
Trend Trim Stage 3: match external trend signals against the local video catalog.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.trends.trend_utils import (
    OUTPUTS_DIR,
    clamp,
    combined_terms,
    cosine_similarity,
    embed_text,
    scaled_cosine,
    utc_now_iso,
)

SIGNALS_PATH = OUTPUTS_DIR / "trend_1_external_signals.json"
CATALOG_PATH = OUTPUTS_DIR / "trend_2_catalog_index.json"
OUTPUT_PATH = OUTPUTS_DIR / "trend_3_resurfacing_candidates.json"

MIN_LEXICAL_SCORE = 0.14
MIN_SEMANTIC_SCORE = 0.34
MIN_FINAL_SCORE = 48.0
MAX_CANDIDATES = 20
MAX_PER_VIDEO = 6
MAX_PER_TREND = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2013 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trend_3")


def _load_json(path: Path, *, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _term_overlap(trend_terms: list[str], catalog_terms: list[str]) -> tuple[float, list[str]]:
    if not trend_terms or not catalog_terms:
        return 0.0, []
    trend_set = set(trend_terms)
    catalog_set = set(catalog_terms)
    shared = sorted(trend_set & catalog_set)
    coverage = len(shared) / max(1, min(len(trend_set), 8))
    return clamp(coverage, 0.0, 1.0), shared[:8]


def _seed_query_score(signal: dict, video: dict) -> tuple[float, list[str]]:
    seed_query = str(signal.get("seed_query", "") or "").strip().lower()
    if not seed_query:
        return 0.0, []
    video_title = str(video.get("video_title", "") or "").strip().lower()
    watchlist_terms = [str(term or "").strip().lower() for term in (video.get("watchlist_terms", []) or [])]
    if seed_query in video_title:
        return 1.0, [seed_query]
    query_terms = [term for term in seed_query.split() if term]
    if not query_terms:
        return 0.0, []
    shared = sorted({term for term in query_terms if term in watchlist_terms})
    if not shared:
        return 0.0, []
    return clamp(len(shared) / len(query_terms), 0.0, 1.0), shared


def _title_overlap(signal: dict, video: dict, clip: dict) -> float:
    signal_title = str(signal.get("title", "") or "").strip().lower()
    if not signal_title:
        return 0.0
    video_title = str(video.get("video_title", "") or "").strip().lower()
    clip_text = " ".join(
        part for part in [
            str(clip.get("combined_transcript", "") or "").strip().lower(),
            str(clip.get("justification", "") or "").strip().lower(),
        ] if part
    )
    if signal_title and signal_title in video_title:
        return 1.0
    if signal_title and signal_title in clip_text:
        return 0.75
    return 0.0


def _semantic_score(signal: dict, clip: dict) -> float:
    query_text = str(signal.get("query_text", "") or signal.get("title", "") or "").strip()
    clip_embedding = clip.get("embedding")
    if not query_text or not isinstance(clip_embedding, list) or not clip_embedding:
        return 0.0
    query_embedding = embed_text(query_text)
    if not query_embedding:
        return 0.0
    similarity = cosine_similarity(query_embedding, [float(value) for value in clip_embedding])
    return scaled_cosine(similarity)


def _candidate(signal: dict, video: dict, clip: dict) -> dict | None:
    signal_terms = combined_terms(
        str(signal.get("title", "") or ""),
        str(signal.get("query_text", "") or ""),
        " ".join(signal.get("keywords", []) or []),
        limit=16,
    )
    clip_terms = list(dict.fromkeys([
        *(clip.get("keyword_terms", []) or []),
        *(video.get("video_keywords", []) or [])[:10],
    ]))
    watchlist_terms = list(dict.fromkeys(video.get("watchlist_terms", []) or []))


    lexical_score, shared_terms = _term_overlap(signal_terms, clip_terms)
    watchlist_overlap, watchlist_shared = _term_overlap(signal_terms, watchlist_terms)
    seed_query_score, seed_query_terms = _seed_query_score(signal, video)
    title_overlap = _title_overlap(signal, video, clip)
    semantic_score = _semantic_score(signal, clip)
    base_clip_score = float(clip.get("base_final_score", 0.0) or 0.0) / 100.0
    signal_strength = float(signal.get("signal_strength", 0.0) or 0.0)

    if (
        lexical_score < MIN_LEXICAL_SCORE
        and watchlist_overlap < MIN_LEXICAL_SCORE
        and seed_query_score <= 0.0
        and semantic_score < MIN_SEMANTIC_SCORE
        and title_overlap <= 0.0
    ):
        return None

    match_core = clamp(
        (0.28 * lexical_score)
        + (0.22 * watchlist_overlap)
        + (0.16 * seed_query_score)
        + (0.18 * semantic_score)
        + (0.08 * title_overlap)
        + (0.08 * base_clip_score),
        0.0,
        1.0,
    )
    final_score = round(
        100.0 * clamp((0.68 * match_core) + (0.22 * signal_strength) + (0.10 * base_clip_score), 0.0, 1.0),
        1,
    )
    if final_score < MIN_FINAL_SCORE:
        return None

    reason_parts = []
    if shared_terms:
        reason_parts.append("shared terms: " + ", ".join(shared_terms[:4]))
    if watchlist_shared:
        reason_parts.append("catalog watchlist hit: " + ", ".join(watchlist_shared[:4]))
    if seed_query_terms:
        reason_parts.append("watchlist query match: " + ", ".join(seed_query_terms[:3]))
    if title_overlap >= 0.75:
        reason_parts.append("direct title/topic hit")
    if semantic_score >= 0.45:
        reason_parts.append(f"semantic match {semantic_score:.2f}")
    if signal_strength >= 0.65:
        reason_parts.append("high external momentum")
    reason = "; ".join(reason_parts) or "matched trend topic to catalog clip"

    return {
        "trend_source": signal.get("source"),
        "trend_title": signal.get("title"),
        "trend_query_text": signal.get("query_text"),
        "trend_signal_strength": round(signal_strength, 3),
        "catalog_id": video.get("catalog_id"),
        "video_id": video.get("video_id"),
        "video_gcs_uri": video.get("video_gcs_uri"),
        "video_title": video.get("video_title"),
        "channel_title": video.get("channel_title"),
        "source_url": video.get("source_url"),
        "clip_index": clip.get("clip_index"),
        "clip_start_ms": clip.get("clip_start_ms"),
        "clip_end_ms": clip.get("clip_end_ms"),
        "base_clip_score": round(base_clip_score * 100.0, 1),
        "trend_match_score": final_score,
        "lexical_score": round(lexical_score, 3),
        "watchlist_overlap_score": round(watchlist_overlap, 3),
        "seed_query_score": round(seed_query_score, 3),
        "semantic_score": round(semantic_score, 3),
        "title_overlap_score": round(title_overlap, 3),
        "overlap_terms": list(dict.fromkeys([*shared_terms, *watchlist_shared, *seed_query_terms])),
        "reason": reason,
        "combined_transcript": clip.get("combined_transcript", ""),
        "justification": clip.get("justification", ""),
        "payload": clip.get("payload", {}),
    }


def main() -> dict:
    log.info("=" * 60)
    log.info("TREND TRIM STAGE 3 \u2013 Resurface Catalog")
    log.info("=" * 60)

    signals_payload = _load_json(SIGNALS_PATH, default={})
    catalog_payload = _load_json(CATALOG_PATH, default={})
    signals = signals_payload.get("signals", []) if isinstance(signals_payload, dict) else []
    videos = catalog_payload.get("videos", []) if isinstance(catalog_payload, dict) else []

    candidates = []
    for signal in signals:
        for video in videos:
            for clip in video.get("clips", []) or []:
                candidate = _candidate(signal, video, clip)
                if candidate:
                    candidates.append(candidate)

    deduped: dict[tuple[str, int, str], dict] = {}
    for candidate in candidates:
        key = (
            str(candidate.get("catalog_id", "") or ""),
            int(candidate.get("clip_index", -1) or -1),
            str(candidate.get("trend_title", "") or "").strip().lower(),
        )
        existing = deduped.get(key)
        if existing is None or float(candidate.get("trend_match_score", 0.0) or 0.0) > float(existing.get("trend_match_score", 0.0) or 0.0):
            deduped[key] = candidate

    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -float(item.get("trend_match_score", 0.0) or 0.0),
            -float(item.get("trend_signal_strength", 0.0) or 0.0),
            -float(item.get("base_clip_score", 0.0) or 0.0),
        ),
    )

    capped: list[dict] = []
    by_video: defaultdict[str, int] = defaultdict(int)
    by_trend: defaultdict[str, int] = defaultdict(int)
    for candidate in ranked:
        catalog_id = str(candidate.get("catalog_id", "") or "")
        trend_title = str(candidate.get("trend_title", "") or "").strip().lower()
        if by_video[catalog_id] >= MAX_PER_VIDEO:
            continue
        if by_trend[trend_title] >= MAX_PER_TREND:
            continue
        capped.append(candidate)
        by_video[catalog_id] += 1
        by_trend[trend_title] += 1
        if len(capped) >= MAX_CANDIDATES:
            break

    for idx, candidate in enumerate(capped, start=1):
        candidate["rank"] = idx

    payload = {
        "generated_at": utc_now_iso(),
        "signal_count": len(signals),
        "catalog_video_count": len(videos),
        "candidate_count": len(capped),
        "candidates": capped,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info("Trend resurfacing candidates: %d", len(capped))
    for candidate in capped[:5]:
        log.info(
            "  #%d %s \u2192 %s %s-%s (score %.1f)",
            candidate["rank"],
            candidate["trend_title"],
            candidate["catalog_id"],
            candidate["clip_start_ms"],
            candidate["clip_end_ms"],
            candidate["trend_match_score"],
        )
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return payload


if __name__ == "__main__":
    main()