from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from .contracts import ExternalSignal

UTC = timezone.utc


def resolve_youtube_video_id(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if "youtu.be" in host:
        candidate = path.strip("/")
        return candidate or None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        if qs.get("v"):
            candidate = qs["v"][0].strip()
            return candidate or None
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[0] in {"shorts", "embed", "live"}:
            return segments[1]
    return None


@dataclass(slots=True)
class YouTubeCommentsClient:
    api_key: str
    base_url: str = "https://www.googleapis.com/youtube/v3"

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = dict(params)
        query["key"] = self.api_key
        url = f"{self.base_url.rstrip('/')}/{endpoint}?{urlencode(query, doseq=True)}"
        with urlopen(url, timeout=30) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and parsed.get("error"):
            message = parsed["error"].get("message") if isinstance(parsed["error"], dict) else str(parsed["error"])
            raise RuntimeError(f"youtube_data_api_error: {message}")
        return parsed

    def fetch_threads(
        self,
        *,
        video_id: str,
        order: str,
        max_pages: int,
    ) -> tuple[list[dict[str, Any]], int]:
        threads: list[dict[str, Any]] = []
        page_token: str | None = None
        total_results = 0
        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "order": order,
                "textFormat": "plainText",
                "maxResults": 100,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._get("commentThreads", params)
            page_info = payload.get("pageInfo") or {}
            total_results = int(page_info.get("totalResults") or total_results or 0)
            items = payload.get("items") or []
            if isinstance(items, list):
                threads.extend(items)
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return threads, max(0, total_results)

    def fetch_replies(self, *, parent_id: str, max_replies: int) -> list[dict[str, Any]]:
        replies: list[dict[str, Any]] = []
        page_token: str | None = None
        while len(replies) < max(0, max_replies):
            params: dict[str, Any] = {
                "part": "snippet",
                "parentId": parent_id,
                "textFormat": "plainText",
                "maxResults": min(100, max(1, max_replies - len(replies))),
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._get("comments", params)
            items = payload.get("items") or []
            if isinstance(items, list):
                replies.extend(items)
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return replies[: max(0, max_replies)]


def _author_id(item: dict[str, Any]) -> str | None:
    snippet = item.get("snippet") or {}
    aid = (snippet.get("authorChannelId") or {}).get("value")
    if isinstance(aid, str) and aid.strip():
        return aid.strip()
    return None


def _text(item: dict[str, Any]) -> str:
    snippet = item.get("snippet") or {}
    text = snippet.get("textDisplay") or snippet.get("textOriginal") or ""
    return str(text).strip()


def _published_at(item: dict[str, Any]) -> str | None:
    snippet = item.get("snippet") or {}
    published = snippet.get("publishedAt")
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    except Exception:
        return None


def _same_author_spam_key(item: dict[str, Any]) -> tuple[str, str] | None:
    author = _author_id(item)
    if not author:
        return None
    text = _text(item)
    if not text:
        return None
    return (author, text.lower())


def collapse_same_author_spam(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = _same_author_spam_key(item)
        if key is None:
            out.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def target_top_threads(*, total_threads: int, min_threads: int, max_threads: int) -> int:
    lower = max(1, int(min_threads))
    upper = max(lower, int(max_threads))
    if total_threads <= 0:
        return lower
    return max(lower, min(upper, int(math.ceil(math.sqrt(total_threads)))))


def to_external_signals_from_threads(
    *,
    thread_items: list[dict[str, Any]],
    include_replies: bool,
) -> list[ExternalSignal]:
    signals: list[ExternalSignal] = []
    for thread in thread_items:
        thread_id = str(thread.get("id") or "")
        snippet = thread.get("snippet") or {}
        top_comment = (snippet.get("topLevelComment") or {})
        top_comment_id = str(top_comment.get("id") or thread_id)
        top_snippet = top_comment.get("snippet") or {}
        top_like_count = int(top_snippet.get("likeCount") or 0)
        reply_count = int(snippet.get("totalReplyCount") or 0)
        signals.append(
            ExternalSignal(
                signal_id=f"comment_top:{top_comment_id}",
                signal_type="comment_top",
                source_platform="youtube",
                source_id=top_comment_id,
                author_id=_author_id(top_comment),
                text=_text(top_comment),
                engagement_score=float(top_like_count + reply_count),
                published_at=_published_at(top_comment),
                metadata={
                    "thread_id": thread_id,
                    "like_count": top_like_count,
                    "reply_count": reply_count,
                },
            )
        )
        if not include_replies:
            continue
        replies = ((thread.get("replies") or {}).get("comments") or [])
        for reply in replies:
            reply_id = str(reply.get("id") or "")
            reply_like_count = int((reply.get("snippet") or {}).get("likeCount") or 0)
            signals.append(
                ExternalSignal(
                    signal_id=f"comment_reply:{reply_id}",
                    signal_type="comment_reply",
                    source_platform="youtube",
                    source_id=reply_id,
                    author_id=_author_id(reply),
                    text=_text(reply),
                    engagement_score=float(reply_like_count),
                    published_at=_published_at(reply),
                    metadata={
                        "thread_id": thread_id,
                        "parent_comment_id": top_comment_id,
                        "like_count": reply_like_count,
                        "parent_reply_count": reply_count,
                    },
                )
            )
    return signals


__all__ = [
    "YouTubeCommentsClient",
    "collapse_same_author_spam",
    "resolve_youtube_video_id",
    "target_top_threads",
    "to_external_signals_from_threads",
]
