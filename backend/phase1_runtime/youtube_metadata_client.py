from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from backend.pipeline.render.contracts import SourceContext
from backend.pipeline.signals.comments_client import resolve_youtube_video_id


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


@dataclass(slots=True)
class YouTubeMetadataClient:
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

    def fetch_source_context(self, *, source_url: str) -> dict[str, Any]:
        video_id = resolve_youtube_video_id(source_url)
        if not video_id:
            raise ValueError(
                "source_url must resolve to a YouTube video ID before Phase 1 metadata ingestion."
            )

        payload = self._get(
            "videos",
            {
                "part": "snippet",
                "id": video_id,
                "maxResults": 1,
            },
        )
        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            raise RuntimeError(
                f"youtube_data_api_error: no video metadata returned for youtube_video_id={video_id!r}"
            )
        snippet = (items[0] or {}).get("snippet") or {}
        tags = snippet.get("tags")
        thumbnails = snippet.get("thumbnails")

        context = SourceContext.model_validate(
            {
                "source_url": source_url,
                "youtube_video_id": video_id,
                "source_title": str(snippet.get("title") or ""),
                "source_description": str(snippet.get("description") or ""),
                "channel_id": str(snippet.get("channelId") or ""),
                "channel_title": str(snippet.get("channelTitle") or ""),
                "published_at": str(snippet.get("publishedAt") or ""),
                "default_audio_language": str(
                    snippet.get("defaultAudioLanguage")
                    or snippet.get("defaultLanguage")
                    or ""
                ),
                "category_id": str(snippet.get("categoryId") or ""),
                "tags": list(tags) if isinstance(tags, list) else [],
                "thumbnails": thumbnails if isinstance(thumbnails, dict) else {},
            }
        )
        return context.model_dump(mode="json")


def build_youtube_metadata_client() -> YouTubeMetadataClient | None:
    api_key = _read_env("CLYPT_YOUTUBE_DATA_API_KEY", "YOUTUBE_API_KEY")
    if not api_key:
        return None
    return YouTubeMetadataClient(
        api_key=api_key,
        base_url=_read_env("CLYPT_YOUTUBE_DATA_API_BASE_URL")
        or "https://www.googleapis.com/youtube/v3",
    )


def fetch_youtube_source_context(*, source_url: str) -> dict[str, Any]:
    client = build_youtube_metadata_client()
    if client is None:
        raise ValueError(
            "CLYPT_YOUTUBE_DATA_API_KEY (or YOUTUBE_API_KEY) is required for Phase 1 source_url metadata ingestion."
        )
    return client.fetch_source_context(source_url=source_url)


__all__ = [
    "YouTubeMetadataClient",
    "build_youtube_metadata_client",
    "fetch_youtube_source_context",
]
