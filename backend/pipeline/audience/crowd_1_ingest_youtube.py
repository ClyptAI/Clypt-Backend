#!/usr/bin/env python3
"""
Crowd Clip Stage 1: ingest public YouTube metadata, stats, and comments.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import httpx

from pipeline.audience.crowd_types import CrowdComment
from pipeline.audience.crowd_utils import OUTPUTS_DIR, extract_video_id, log_weight

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
OUTPUT_PATH = OUTPUTS_DIR / "crowd_1_youtube_signals.json"

COMMENT_RELEVANCE_PAGES = int(os.getenv("CROWD_CLIP_RELEVANCE_PAGES", "3"))
COMMENT_TIME_PAGES = int(os.getenv("CROWD_CLIP_TIME_PAGES", "2"))
MAX_RESULTS_PER_PAGE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_1")


def _iso_to_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _api_get(client: httpx.Client, path: str, params: dict) -> dict:
    response = client.get(f"{YOUTUBE_API_BASE}/{path}", params=params, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _fetch_video_resource(client: httpx.Client, api_key: str, video_id: str) -> dict:
    payload = _api_get(
        client,
        "videos",
        {
            "part": "snippet,statistics,contentDetails",
            "id": video_id,
            "key": api_key,
        },
    )
    items = payload.get("items", [])
    if not items:
        raise RuntimeError(f"YouTube video not found or unavailable: {video_id}")
    return items[0]


def _flatten_comment_item(item: dict) -> list[CrowdComment]:
    out: list[CrowdComment] = []

    top = ((item.get("snippet") or {}).get("topLevelComment") or {})
    top_snippet = top.get("snippet") or {}
    top_comment_id = str(top.get("id", "") or "")
    if top_comment_id:
        out.append(
            CrowdComment(
                comment_id=top_comment_id,
                parent_comment_id=None,
                is_reply=False,
                author_name=str(top_snippet.get("authorDisplayName", "") or ""),
                like_count=int(top_snippet.get("likeCount", 0) or 0),
                reply_count=int((item.get("snippet") or {}).get("totalReplyCount", 0) or 0),
                published_at=str(top_snippet.get("publishedAt", "") or ""),
                updated_at=str(top_snippet.get("updatedAt", "") or ""),
                text=str(top_snippet.get("textDisplay", "") or top_snippet.get("textOriginal", "") or ""),
            )
        )

    replies = ((item.get("replies") or {}).get("comments") or [])
    for reply in replies:
        snippet = reply.get("snippet") or {}
        comment_id = str(reply.get("id", "") or "")
        if not comment_id:
            continue
        out.append(
            CrowdComment(
                comment_id=comment_id,
                parent_comment_id=top_comment_id or None,
                is_reply=True,
                author_name=str(snippet.get("authorDisplayName", "") or ""),
                like_count=int(snippet.get("likeCount", 0) or 0),
                reply_count=0,
                published_at=str(snippet.get("publishedAt", "") or ""),
                updated_at=str(snippet.get("updatedAt", "") or ""),
                text=str(snippet.get("textDisplay", "") or snippet.get("textOriginal", "") or ""),
            )
        )

    return out


def _fetch_comment_threads(
    client: httpx.Client,
    api_key: str,
    video_id: str,
    *,
    order: str,
    max_pages: int,
) -> tuple[list[CrowdComment], bool]:
    comments: list[CrowdComment] = []
    page_token: str | None = None
    comments_available = True

    for _ in range(max_pages):
        params = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": MAX_RESULTS_PER_PAGE,
            "order": order,
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            payload = _api_get(client, "commentThreads", params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                comments_available = False
                break
            raise

        for item in payload.get("items", []):
            comments.extend(_flatten_comment_item(item))

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return comments, comments_available


def _dedupe_comments(comments: list[CrowdComment]) -> list[CrowdComment]:
    deduped: dict[str, CrowdComment] = {}
    for comment in comments:
        existing = deduped.get(comment.comment_id)
        if existing is None or comment.like_count > existing.like_count:
            deduped[comment.comment_id] = comment
    return list(deduped.values())


def _derive_metrics(video: dict, comments: list[CrowdComment]) -> dict:
    snippet = video.get("snippet") or {}
    stats = video.get("statistics") or {}
    published_at = str(snippet.get("publishedAt", "") or "")
    now = datetime.now(timezone.utc)
    published_dt = _iso_to_datetime(published_at)
    age_hours = max(0.0, (now - published_dt).total_seconds() / 3600.0) if published_dt else 0.0

    views = int(stats.get("viewCount", 0) or 0)
    likes = int(stats.get("likeCount", 0) or 0)
    comment_count = int(stats.get("commentCount", 0) or 0)

    return {
        "age_hours": round(age_hours, 3),
        "like_rate_per_1000_views": round((likes / max(1, views)) * 1000.0, 3),
        "comment_rate_per_1000_views": round((comment_count / max(1, views)) * 1000.0, 3),
        "lifetime_avg_views_per_hour": round(views / max(1e-6, age_hours), 3) if age_hours else None,
        "lifetime_avg_comments_per_hour": round(comment_count / max(1e-6, age_hours), 3) if age_hours else None,
        "observed_comment_sample_size": len(comments),
        "observed_comment_like_mass": int(sum(c.like_count for c in comments)),
        "observed_reply_comment_count": int(sum(1 for c in comments if c.is_reply)),
        "engagement_proxy_score": round(
            log_weight(comment_count, 8.0)
            + log_weight(likes, 5.0)
            + log_weight(len(comments), 6.0),
            3,
        ),
    }


def main(youtube_url: str) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Crowd Clip requires YOUTUBE_API_KEY for public comments/statistics ingestion.")

    video_id = extract_video_id(youtube_url)
    log.info("=" * 60)
    log.info("CROWD CLIP — STAGE 1 INGEST")
    log.info("=" * 60)
    log.info("Video: %s", video_id)

    with httpx.Client() as client:
        video = _fetch_video_resource(client, api_key, video_id)
        comments_relevance, relevance_available = _fetch_comment_threads(
            client, api_key, video_id, order="relevance", max_pages=COMMENT_RELEVANCE_PAGES
        )
        comments_time, time_available = _fetch_comment_threads(
            client, api_key, video_id, order="time", max_pages=COMMENT_TIME_PAGES
        )

    comments = _dedupe_comments([*comments_relevance, *comments_time])
    comments.sort(key=lambda c: (-c.like_count, c.published_at, c.comment_id))

    signal_payload = {
        "schema_version": "crowd-clip-signals-v1",
        "source_url": youtube_url,
        "video_id": video_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "availability": {
            "comments": relevance_available or time_available,
            "basic_stats": True,
            "lifetime_velocity_proxy": True,
            "time_series_velocity": False,
            "retention_curve": False,
        },
        "video": {
            "title": str((video.get("snippet") or {}).get("title", "") or ""),
            "channel_title": str((video.get("snippet") or {}).get("channelTitle", "") or ""),
            "published_at": str((video.get("snippet") or {}).get("publishedAt", "") or ""),
            "duration_iso8601": str((video.get("contentDetails") or {}).get("duration", "") or ""),
            "description": str((video.get("snippet") or {}).get("description", "") or ""),
        },
        "stats": {
            "view_count": int((video.get("statistics") or {}).get("viewCount", 0) or 0),
            "like_count": int((video.get("statistics") or {}).get("likeCount", 0) or 0),
            "comment_count": int((video.get("statistics") or {}).get("commentCount", 0) or 0),
        },
        "derived_metrics": _derive_metrics(video, comments),
        "notes": {
            "retention": "Public YouTube APIs do not expose audience retention for arbitrary public videos.",
            "velocity": "This MVP stores only current totals plus lifetime average proxies. Time-series velocity needs repeated snapshots or creator analytics access.",
        },
        "comments": [comment.model_dump() for comment in comments],
    }

    OUTPUT_PATH.write_text(json.dumps(signal_payload, indent=2), encoding="utf-8")
    log.info("Comments captured: %d", len(comments))
    log.info("Output saved → %s", OUTPUT_PATH)
    return signal_payload


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python backend/pipeline/crowd_1_ingest_youtube.py <youtube-url>")
    main(sys.argv[1])
