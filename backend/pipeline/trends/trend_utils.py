#!/usr/bin/env python3
"""
Shared helpers for Trend Trim.
"""

from __future__ import annotations

import math
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import httpx

try:
    from google import genai
except Exception:  # pragma: no cover - optional runtime dependency behavior
    genai = None

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
VIDEOS_DIR = ROOT / "videos"

PROJECT_ID = "clypt-v3"
EMBEDDING_LOCATION = "us-central1"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 768

YOUTUBE_VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "your", "their", "about", "were", "when",
    "what", "where", "which", "while", "will", "would", "could", "should", "into", "onto", "than", "then",
    "them", "they", "just", "really", "there", "here", "also", "after", "before", "because", "been", "being",
    "video", "videos", "clip", "clips", "watch", "watching", "youtube", "podcast", "episode", "part", "full",
    "react", "reaction", "interview", "show", "channel", "today", "viral", "trend", "trending", "shorts",
    "about", "over", "under", "across", "through", "against", "between", "made", "make", "makes", "making",
    "more", "most", "less", "than", "very", "much", "many", "some", "such", "only", "other", "same", "each",
    "onto", "upon", "once", "ever", "still", "even", "like", "liked", "looks", "look", "looking", "good",
    "great", "best", "worse", "worst", "new", "old", "how", "why", "who", "whom", "whose",
    "didn", "doesn", "isn", "aren", "wasn", "weren", "won", "wouldn", "shouldn", "couldn", "hasn", "hadn",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")
_EMBED_CLIENT = None
_EMBEDDING_DISABLED = False
_EMBED_CACHE: dict[str, list[float] | None] = {}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def truthy_env(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("\u2019", "'").replace("`", "'")
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    lowered = _SPACE_RE.sub(" ", lowered).strip()
    return lowered


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def log_norm(value: float, ceiling: float) -> float:
    if value <= 0:
        return 0.0
    return clamp(math.log1p(value) / math.log1p(max(1.0, ceiling)), 0.0, 1.0)


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return 0.0
    return dot / (norm_left * norm_right)


def scaled_cosine(value: float) -> float:
    return clamp((value - 0.2) / 0.55, 0.0, 1.0)


def parse_approx_traffic(text: str) -> int:
    value = (text or "").strip().replace("+", "").replace(",", "")
    if not value:
        return 0
    multiplier = 1
    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]
    try:
        return int(float(value) * multiplier)
    except ValueError:
        return 0


def keyword_terms(
    text: str,
    *,
    limit: int = 16,
    min_len: int = 4,
    extra_stopwords: Iterable[str] | None = None,
) -> list[str]:
    stopwords = STOPWORDS | {normalize_text(token) for token in (extra_stopwords or []) if token}
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < min_len or token in stopwords or token.isdigit():
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def phrase_terms(text: str, *, limit: int = 8) -> list[str]:
    tokens = [tok for tok in tokenize(text) if len(tok) >= 4 and tok not in STOPWORDS and not tok.isdigit()]
    counts: dict[str, int] = {}
    for width in (2, 3):
        for idx in range(len(tokens) - width + 1):
            phrase = " ".join(tokens[idx:idx + width])
            counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [phrase for phrase, _ in ranked[:limit]]


def combined_terms(*parts: str, limit: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    for part in parts:
        for token in keyword_terms(part, limit=limit * 2):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def youtube_api_key() -> str:
    return str(os.getenv("YOUTUBE_API_KEY", "") or "").strip()


def http_get_json(url: str, *, params: dict | None = None, timeout: float = 20.0) -> dict:
    response = httpx.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def http_get_text(url: str, *, params: dict | None = None, timeout: float = 20.0) -> str:
    response = httpx.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_youtube_video_metadata(video_ids: list[str], api_key: str | None = None) -> dict[str, dict]:
    key = (api_key or youtube_api_key()).strip()
    if not key or not video_ids:
        return {}
    deduped = []
    seen: set[str] = set()
    for video_id in video_ids:
        candidate = (video_id or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    payload: dict[str, dict] = {}
    for idx in range(0, len(deduped), 50):
        chunk = deduped[idx:idx + 50]
        data = http_get_json(
            YOUTUBE_VIDEOS_ENDPOINT,
            params={
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk),
                "key": key,
                "maxResults": len(chunk),
            },
        )
        for item in data.get("items", []) or []:
            payload[str(item.get("id", "") or "")] = item
    return payload


def search_youtube_watchlist(query: str, *, region: str = "US", api_key: str | None = None, max_results: int = 5) -> list[dict]:
    key = (api_key or youtube_api_key()).strip()
    if not key or not query.strip():
        return []
    published_after = (datetime.now(UTC) - timedelta(days=21)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data = http_get_json(
        YOUTUBE_SEARCH_ENDPOINT,
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": published_after,
            "regionCode": region,
            "maxResults": max_results,
            "key": key,
        },
    )
    items = data.get("items", []) or []
    video_ids = [str(((item.get("id") or {}).get("videoId") or "")).strip() for item in items]
    metadata = fetch_youtube_video_metadata(video_ids, api_key=key)
    enriched: list[dict] = []
    for item in items:
        video_id = str(((item.get("id") or {}).get("videoId") or "")).strip()
        if not video_id:
            continue
        meta = metadata.get(video_id, {})
        enriched.append(
            {
                "video_id": video_id,
                "snippet": item.get("snippet", {}),
                "statistics": meta.get("statistics", {}),
                "contentDetails": meta.get("contentDetails", {}),
            }
        )
    return enriched


def _make_embed_client():
    if genai is None:
        raise RuntimeError("google-genai is unavailable")
    return genai.Client(vertexai=True, project=PROJECT_ID, location=EMBEDDING_LOCATION)


def embed_text(text: str) -> list[float] | None:
    global _EMBED_CLIENT, _EMBEDDING_DISABLED
    normalized = normalize_text(text)
    if not normalized:
        return None
    if normalized in _EMBED_CACHE:
        return _EMBED_CACHE[normalized]
    if _EMBEDDING_DISABLED:
        _EMBED_CACHE[normalized] = None
        return None
    try:
        if _EMBED_CLIENT is None:
            _EMBED_CLIENT = _make_embed_client()
        response = _EMBED_CLIENT.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[normalized],
            config=genai.types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        embeddings = getattr(response, "embeddings", None) or []
        values = getattr(embeddings[0], "values", None) if embeddings else None
        result = list(values) if values else None
        _EMBED_CACHE[normalized] = result
        return result
    except Exception:
        _EMBEDDING_DISABLED = True
        _EMBED_CACHE[normalized] = None
        return None


def quote_query(query: str) -> str:
    return quote_plus(query.strip())
