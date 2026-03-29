#!/usr/bin/env python3
"""
Trend Trim Stage 1: ingest external trend signals.

MVP sources:
  - Google Trends public RSS (broad public trend feed)
  - YouTube most-popular videos (official YouTube Data API)
  - YouTube watchlist search (official YouTube Data API, optional)
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.trends.trend_utils import (
    GOOGLE_TRENDS_RSS,
    OUTPUTS_DIR,
    YOUTUBE_VIDEOS_ENDPOINT,
    clamp,
    combined_terms,
    http_get_json,
    http_get_text,
    keyword_terms,
    log_norm,
    parse_approx_traffic,
    search_youtube_watchlist,
    utc_now_iso,
    youtube_api_key,
)

OUTPUT_PATH = OUTPUTS_DIR / "trend_1_external_signals.json"
GOOGLE_TRENDS_NS = {"ht": "https://trends.google.com/trending/rss"}

DEFAULT_REGION = "US"
DEFAULT_GOOGLE_TRENDS_LIMIT = 20
DEFAULT_YOUTUBE_POPULAR_LIMIT = 12
DEFAULT_WATCHLIST_SEARCH_LIMIT = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2013 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trend_1")


def _seed_queries() -> list[str]:
    raw = str(os.getenv("TREND_TRIM_WATCHLIST", "") or "").strip()
    if not raw:
        return []
    queries = []
    seen: set[str] = set()
    for part in raw.split(","):
        query = part.strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def _google_trends_items(region: str, limit: int) -> list[dict]:
    xml_text = http_get_text(GOOGLE_TRENDS_RSS, params={"geo": region})
    root = ET.fromstring(xml_text)
    signals: list[dict] = []
    for item in root.findall("./channel/item")[:limit]:
        title = str(item.findtext("title", default="") or "").strip()
        if not title:
            continue
        approx_traffic = str(item.findtext("ht:approx_traffic", default="", namespaces=GOOGLE_TRENDS_NS) or "").strip()
        traffic_value = parse_approx_traffic(approx_traffic)
        news_context = []
        news_titles: list[str] = []
        for news_item in item.findall("ht:news_item", GOOGLE_TRENDS_NS)[:3]:
            news_title = str(news_item.findtext("ht:news_item_title", default="", namespaces=GOOGLE_TRENDS_NS) or "").strip()
            news_url = str(news_item.findtext("ht:news_item_url", default="", namespaces=GOOGLE_TRENDS_NS) or "").strip()
            news_source = str(news_item.findtext("ht:news_item_source", default="", namespaces=GOOGLE_TRENDS_NS) or "").strip()
            if news_title:
                news_titles.append(news_title)
                news_context.append(
                    {
                        "title": news_title,
                        "url": news_url,
                        "source": news_source,
                    }
                )
        query_text = " | ".join(part for part in [title, *news_titles] if part)
        signals.append(
            {
                "source": "google_trends_rss",
                "signal_id": f"google_trends_rss:{title.lower()}",
                "title": title,
                "query_text": query_text or title,
                "published_at": str(item.findtext("pubDate", default="") or "").strip(),
                "signal_strength": round(log_norm(traffic_value or 1, 5_000_000), 3),
                "approx_traffic": approx_traffic,
                "approx_traffic_value": traffic_value,
                "keywords": combined_terms(title, query_text, limit=12),
                "news_context": news_context,
                "context_terms": combined_terms(*news_titles, limit=10),
            }
        )
    return signals


def _youtube_most_popular(region: str, limit: int, api_key: str) -> list[dict]:
    if not api_key:
        return []
    data = http_get_json(
        YOUTUBE_VIDEOS_ENDPOINT,
        params={
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "regionCode": region,
            "maxResults": limit,
            "key": api_key,
        },
    )
    signals: list[dict] = []
    for item in data.get("items", []) or []:
        snippet = item.get("snippet", {}) or {}
        statistics = item.get("statistics", {}) or {}
        title = str(snippet.get("title", "") or "").strip()
        channel_title = str(snippet.get("channelTitle", "") or "").strip()
        if not title:
            continue
        view_count = int(statistics.get("viewCount", 0) or 0)
        comment_count = int(statistics.get("commentCount", 0) or 0)
        like_count = int(statistics.get("likeCount", 0) or 0)

        strength = clamp(
            (0.55 * log_norm(view_count, 100_000_000))
            + (0.20 * log_norm(like_count, 5_000_000))
            + (0.25 * log_norm(comment_count, 500_000)),
            0.0,
            1.0,
        )
        query_text = " | ".join(part for part in [title, channel_title, snippet.get("description", "")] if part)
        signals.append(
            {
                "source": "youtube_most_popular",
                "signal_id": f"youtube_most_popular:{item.get('id')}",
                "title": title,
                "query_text": query_text,
                "published_at": str(snippet.get("publishedAt", "") or "").strip(),
                "signal_strength": round(strength, 3),
                "video_id": str(item.get("id", "") or "").strip(),
                "channel_title": channel_title,
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count,
                "keywords": combined_terms(title, channel_title, snippet.get("description", ""), limit=12),
                "context_terms": keyword_terms(snippet.get("description", ""), limit=10),
            }
        )
    return signals


def _youtube_watchlist_signals(region: str, queries: list[str], api_key: str, per_query_limit: int) -> list[dict]:
    if not api_key:
        return []
    signals: list[dict] = []
    for query in queries:
        try:
            items = search_youtube_watchlist(query, region=region, api_key=api_key, max_results=per_query_limit)
        except Exception as exc:
            log.warning("YouTube watchlist search failed for %s: %s", query, exc)
            continue
        for item in items:
            snippet = item.get("snippet", {}) or {}
            statistics = item.get("statistics", {}) or {}
            title = str(snippet.get("title", "") or "").strip()
            channel_title = str(snippet.get("channelTitle", "") or "").strip()
            if not title:
                continue
            view_count = int(statistics.get("viewCount", 0) or 0)
            comment_count = int(statistics.get("commentCount", 0) or 0)
            like_count = int(statistics.get("likeCount", 0) or 0)
            strength = clamp(
                (0.50 * log_norm(view_count, 50_000_000))
                + (0.20 * log_norm(like_count, 2_500_000))
                + (0.15 * log_norm(comment_count, 250_000))
                + 0.15,
                0.0,
                1.0,
            )
            query_text = " | ".join(part for part in [query, title, channel_title, snippet.get("description", "")] if part)
            signals.append(
                {
                    "source": "youtube_watchlist_search",
                    "signal_id": f"youtube_watchlist_search:{query.lower()}:{item.get('video_id')}",
                    "seed_query": query,
                    "title": title,
                    "query_text": query_text,
                    "published_at": str(snippet.get("publishedAt", "") or "").strip(),
                    "signal_strength": round(strength, 3),
                    "video_id": str(item.get("video_id", "") or "").strip(),
                    "channel_title": channel_title,
                    "view_count": view_count,
                    "like_count": like_count,
                    "comment_count": comment_count,
                    "keywords": combined_terms(query, title, channel_title, snippet.get("description", ""), limit=12),
                    "context_terms": combined_terms(title, snippet.get("description", ""), limit=10),
                }
            )
    return signals


def main() -> dict:
    log.info("=" * 60)
    log.info("TREND TRIM STAGE 1 \u2013 External Signals")
    log.info("=" * 60)

    region = str(os.getenv("TREND_TRIM_REGION", DEFAULT_REGION) or DEFAULT_REGION).strip().upper()
    google_limit = int(os.getenv("TREND_TRIM_GOOGLE_TRENDS_LIMIT", DEFAULT_GOOGLE_TRENDS_LIMIT) or DEFAULT_GOOGLE_TRENDS_LIMIT)
    popular_limit = int(os.getenv("TREND_TRIM_YOUTUBE_POPULAR_LIMIT", DEFAULT_YOUTUBE_POPULAR_LIMIT) or DEFAULT_YOUTUBE_POPULAR_LIMIT)
    watchlist_limit = int(os.getenv("TREND_TRIM_WATCHLIST_SEARCH_LIMIT", DEFAULT_WATCHLIST_SEARCH_LIMIT) or DEFAULT_WATCHLIST_SEARCH_LIMIT)
    api_key = youtube_api_key()
    watchlist_queries = _seed_queries()

    signals: list[dict] = []
    try:
        signals.extend(_google_trends_items(region, google_limit))
    except Exception as exc:
        log.warning("Google Trends RSS fetch failed: %s", exc)

    if api_key:
        try:
            signals.extend(_youtube_most_popular(region, popular_limit, api_key))
        except Exception as exc:
            log.warning("YouTube mostPopular fetch failed: %s", exc)
        if watchlist_queries:
            signals.extend(_youtube_watchlist_signals(region, watchlist_queries, api_key, watchlist_limit))
    else:
        log.info("YOUTUBE_API_KEY not set \u2013 YouTube trend sources skipped.")

    deduped: dict[str, dict] = {}
    for signal in signals:
        key = str(signal.get("signal_id", "") or "").strip()
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or float(signal.get("signal_strength", 0.0) or 0.0) > float(existing.get("signal_strength", 0.0) or 0.0):
            deduped[key] = signal

    payload = {
        "generated_at": utc_now_iso(),
        "region": region,
        "watchlist_queries": watchlist_queries,
        "signal_count": len(deduped),
        "sources": sorted({signal.get("source", "") for signal in deduped.values()}),
        "signals": sorted(
            deduped.values(),
            key=lambda item: (
                -float(item.get("signal_strength", 0.0) or 0.0),
                str(item.get("title", "") or "").lower(),
            ),
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_source: dict[str, int] = {}
    for signal in payload["signals"]:
        source = str(signal.get("source", "") or "")
        by_source[source] = by_source.get(source, 0) + 1

    log.info("Signals fetched: %d", payload["signal_count"])
    for source, count in sorted(by_source.items()):
        log.info("  %s: %d", source, count)
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return payload


if __name__ == "__main__":
    main()