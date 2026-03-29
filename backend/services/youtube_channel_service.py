from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_ISO8601_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<days>\d+)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+)H)?"
    r"(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+)S)?"
    r")?$"
)


class YouTubeChannelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedChannel:
    channel_id: str
    channel_name: str
    channel_url: str
    handle: str
    avatar_url: str
    banner_url: str
    description: str
    category: str
    subscriber_count: int
    subscriber_count_label: str
    total_views: int
    total_views_label: str
    upload_frequency_label: str
    joined_date_label: str


@dataclass(frozen=True)
class ChannelVideo:
    video_id: str
    title: str
    views: int
    views_label: str
    duration_seconds: int
    duration_label: str
    likes: int
    likes_label: str
    thumbnail_url: str
    published_at: str
    description: str


@dataclass(frozen=True)
class ChannelResolveResult:
    channel: ResolvedChannel
    recent_shorts: list[ChannelVideo]
    recent_videos: list[ChannelVideo]


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B".rstrip("0").rstrip(".")
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".rstrip("0").rstrip(".")
    if value >= 1_000:
        return f"{value / 1_000:.1f}K".rstrip("0").rstrip(".")
    return str(value)


def _parse_iso8601_duration(duration: str) -> int:
    match = _ISO8601_DURATION_RE.match(duration or "")
    if not match:
        return 0
    parts = {name: int(value or 0) for name, value in match.groupdict().items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def _format_duration(seconds: int) -> str:
    minutes, secs = divmod(max(0, seconds), 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}" if seconds >= 60 else f"{secs}s"


def _joined_date_label(published_at: str) -> str:
    if not published_at:
        return ""
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return published_at
    return str(dt.year)


def _upload_frequency_label(items: list[dict[str, Any]]) -> str:
    timestamps: list[datetime] = []
    for item in items:
        published_at = str(((item.get("snippet") or {}).get("publishedAt", "") or "")).strip()
        if not published_at:
            continue
        try:
            timestamps.append(datetime.fromisoformat(published_at.replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return ""
    timestamps.sort()
    span_days = max(1.0, (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0)
    per_week = round((len(timestamps) / span_days) * 7.0)
    if per_week <= 0:
        return ""
    if per_week == 1:
        return "~1 video/week"
    return f"~{per_week} videos/week"


def _normalize_handle(handle: str) -> str:
    normalized = handle.strip()
    if not normalized:
        return ""
    return normalized if normalized.startswith("@") else f"@{normalized}"


def _channel_url(handle: str, channel_id: str) -> str:
    return f"https://youtube.com/{handle}" if handle else f"https://youtube.com/channel/{channel_id}"


def _normalize_image_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("//"):
        return f"https:{normalized}"
    if normalized.startswith("http://"):
        return "https://" + normalized[len("http://"):]
    return normalized


def _thumbnail_dimensions(entry: dict[str, Any]) -> tuple[int, int]:
    width = int(entry.get("width", 0) or 0)
    height = int(entry.get("height", 0) or 0)
    if width > 0 and height > 0:
        return width, height
    resolution = str(entry.get("resolution", "") or "").strip().lower()
    if "x" not in resolution:
        return 0, 0
    left, right = resolution.split("x", 1)
    try:
        return int(left or 0), int(right or 0)
    except ValueError:
        return 0, 0


def _thumbnail_list_best_url(thumbnails: list[dict[str, Any]] | None) -> str:
    best_url = ""
    best_score = float("-inf")
    for entry in thumbnails or []:
        if not isinstance(entry, dict):
            continue
        url = _normalize_image_url(str(entry.get("url", "") or ""))
        if not url:
            continue
        width, height = _thumbnail_dimensions(entry)
        area = width * height
        square_bonus = 2_000_000 if width > 0 and height > 0 and abs(width - height) <= max(2, width // 12) else 0
        ident = str(entry.get("id", "") or "").strip().lower()
        uncropped_penalty = -3_000_000 if "uncropped" in ident or "=s0" in url else 0
        score = area + square_bonus + uncropped_penalty
        if score > best_score:
            best_score = score
            best_url = url
    return best_url


def _thumbnail_map_best_url(thumbnails: dict[str, Any] | None) -> str:
    if not isinstance(thumbnails, dict):
        return ""
    preferred_order = ("maxres", "standard", "high", "medium", "default")
    for key in preferred_order:
        candidate = thumbnails.get(key)
        if isinstance(candidate, dict):
            url = _normalize_image_url(str(candidate.get("url", "") or ""))
            if url:
                return url
    flattened = [value for value in thumbnails.values() if isinstance(value, dict)]
    return _thumbnail_list_best_url(flattened)


class YouTubeChannelService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.timeout_s = timeout_s
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_env(cls) -> "YouTubeChannelService":
        return cls(api_key=str(os.getenv("YOUTUBE_API_KEY", "") or "").strip())

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "YouTubeChannelService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def resolve_channel(self, query: str, *, recent_video_limit: int = 12) -> ChannelResolveResult:
        normalized = query.strip()
        if not normalized:
            raise YouTubeChannelError("Channel query must not be empty.")

        if self.api_key:
            try:
                return self._resolve_channel_with_api(normalized, recent_video_limit=recent_video_limit)
            except YouTubeChannelError:
                if not self._supports_fallback_query(normalized):
                    raise

        if self._supports_fallback_query(normalized):
            return self._resolve_channel_with_ytdlp(normalized, recent_video_limit=recent_video_limit)

        raise YouTubeChannelError(
            "Real channel search without a YouTube API key currently supports @handles, /channel/ URLs, and full YouTube channel URLs."
        )

    def get_channel_by_id(self, channel_id: str, *, recent_video_limit: int = 12) -> ChannelResolveResult:
        if self.api_key:
            payload = self._api_get(
                "channels",
                {
                    "part": "snippet,statistics,brandingSettings",
                    "id": channel_id,
                    "maxResults": 1,
                },
            )
            items = payload.get("items", [])
            if not items:
                raise YouTubeChannelError(f"YouTube channel not found: {channel_id}")
            snippet = items[0].get("snippet") or {}
            handle = _normalize_handle(str(snippet.get("customUrl", "") or "").strip())
            query = handle or channel_id or str(snippet.get("title", "") or "").strip()
            return self.resolve_channel(query, recent_video_limit=recent_video_limit)

        return self._resolve_channel_with_ytdlp(
            f"https://www.youtube.com/channel/{channel_id}",
            recent_video_limit=recent_video_limit,
        )

    def _resolve_channel_with_api(self, query: str, *, recent_video_limit: int) -> ChannelResolveResult:
        channel_resource = self._resolve_channel_resource(query)
        channel_id = str(channel_resource.get("id", "") or "")
        if not channel_id:
            raise YouTubeChannelError("Resolved YouTube channel did not include an id.")

        recent_items = self._list_recent_videos(channel_id=channel_id, limit=recent_video_limit)
        recent_videos = self._fetch_video_details(
            [str(item.get("id", {}).get("videoId", "") or "") for item in recent_items]
        )

        shorts: list[ChannelVideo] = []
        standard_videos: list[ChannelVideo] = []
        for item in recent_videos:
            if item.duration_seconds <= 90 and len(shorts) < 6:
                shorts.append(item)
            elif len(standard_videos) < 6:
                standard_videos.append(item)

        snippet = channel_resource.get("snippet") or {}
        statistics = channel_resource.get("statistics") or {}
        branding = channel_resource.get("brandingSettings") or {}
        image_settings = branding.get("image") or {}
        snippet_thumbnails = snippet.get("thumbnails") or {}

        custom_url = str(snippet.get("customUrl", "") or "").strip()
        handle = _normalize_handle(custom_url)
        subscriber_count = int(statistics.get("subscriberCount", 0) or 0)
        total_views = int(statistics.get("viewCount", 0) or 0)
        channel = ResolvedChannel(
            channel_id=channel_id,
            channel_name=str(snippet.get("title", "") or ""),
            channel_url=_channel_url(handle, channel_id),
            handle=handle,
            avatar_url=_thumbnail_map_best_url(snippet_thumbnails),
            banner_url=_normalize_image_url(str(image_settings.get("bannerExternalUrl", "") or "")),
            description=str(snippet.get("description", "") or ""),
            category=str((branding.get("channel") or {}).get("defaultLanguage", "") or ""),
            subscriber_count=subscriber_count,
            subscriber_count_label=_format_count(subscriber_count),
            total_views=total_views,
            total_views_label=_format_count(total_views),
            upload_frequency_label=_upload_frequency_label(recent_items),
            joined_date_label=_joined_date_label(str(snippet.get("publishedAt", "") or "")),
        )
        return ChannelResolveResult(
            channel=channel,
            recent_shorts=shorts,
            recent_videos=standard_videos,
        )

    def _resolve_channel_with_ytdlp(self, query: str, *, recent_video_limit: int) -> ChannelResolveResult:
        from yt_dlp import YoutubeDL

        channel_url = self._normalize_channel_url(query)
        opts = {
            "quiet": True,
            "skip_download": True,
            "playlistend": 1,
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

        channel_id = str(info.get("channel_id", "") or self._extract_channel_id(query) or "")
        handle = _normalize_handle(str(info.get("uploader_id", "") or self._extract_handle(query) or ""))
        resolved_channel_url = str(info.get("uploader_url", "") or _channel_url(handle, channel_id))
        avatar_url = _thumbnail_list_best_url(info.get("thumbnails") or [])

        videos = self._extract_tab_videos(resolved_channel_url, "videos", limit=recent_video_limit)
        shorts = self._extract_tab_videos(resolved_channel_url, "shorts", limit=min(recent_video_limit, 6))

        recent_items = [
            {"snippet": {"publishedAt": item.published_at}}
            for item in [*videos[:6], *shorts[:6]]
            if item.published_at
        ]
        subscriber_count = int(info.get("channel_follower_count", 0) or 0)
        total_views = int(info.get("view_count", 0) or 0)
        channel = ResolvedChannel(
            channel_id=channel_id,
            channel_name=str(info.get("channel") or info.get("uploader") or handle or channel_id),
            channel_url=resolved_channel_url or _channel_url(handle, channel_id),
            handle=handle,
            avatar_url=avatar_url,
            banner_url="",
            description=str(info.get("description", "") or ""),
            category="YouTube",
            subscriber_count=subscriber_count,
            subscriber_count_label=_format_count(subscriber_count),
            total_views=total_views,
            total_views_label=_format_count(total_views),
            upload_frequency_label=_upload_frequency_label(recent_items),
            joined_date_label="",
        )
        return ChannelResolveResult(
            channel=channel,
            recent_shorts=shorts[:6],
            recent_videos=videos[:6],
        )

    def _extract_tab_videos(self, channel_url: str, tab: str, *, limit: int) -> list[ChannelVideo]:
        from yt_dlp import YoutubeDL

        tab_url = channel_url.rstrip("/") + f"/{tab}"
        opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": True,
            "playlistend": max(1, min(limit, 12)),
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(tab_url, download=False)

        videos: list[ChannelVideo] = []
        for entry in info.get("entries") or []:
            video_id = str(entry.get("id", "") or "")
            if not video_id or len(videos) >= limit:
                continue
            duration_seconds = int(float(entry.get("duration", 0) or 0))
            views = int(entry.get("view_count", 0) or 0)
            videos.append(
                ChannelVideo(
                    video_id=video_id,
                    title=str(entry.get("title", "") or ""),
                    views=views,
                    views_label=_format_count(views),
                    duration_seconds=duration_seconds,
                    duration_label=_format_duration(duration_seconds),
                    likes=0,
                    likes_label="0",
                    thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                    published_at=str(entry.get("upload_date", "") or ""),
                    description="",
                )
            )
        return videos

    def _resolve_channel_resource(self, query: str) -> dict[str, Any]:
        explicit_channel_id = self._extract_channel_id(query)
        if explicit_channel_id:
            payload = self._api_get(
                "channels",
                {
                    "part": "snippet,statistics,brandingSettings",
                    "id": explicit_channel_id,
                    "maxResults": 1,
                },
            )
            items = payload.get("items", [])
            if items:
                return items[0]

        handle = self._extract_handle(query)
        if handle:
            payload = self._api_get(
                "channels",
                {
                    "part": "snippet,statistics,brandingSettings",
                    "forHandle": handle.lstrip("@"),
                    "maxResults": 1,
                },
            )
            items = payload.get("items", [])
            if items:
                return items[0]

        payload = self._api_get(
            "search",
            {
                "part": "snippet",
                "type": "channel",
                "q": query,
                "maxResults": 1,
            },
        )
        items = payload.get("items", [])
        if not items:
            raise YouTubeChannelError(f"Could not resolve YouTube channel from query: {query}")
        channel_id = str(((items[0].get("id") or {}).get("channelId", "")) or "")
        if not channel_id:
            raise YouTubeChannelError(f"Search result did not include a channel id for query: {query}")
        resolved = self._api_get(
            "channels",
            {
                "part": "snippet,statistics,brandingSettings",
                "id": channel_id,
                "maxResults": 1,
            },
        )
        channel_items = resolved.get("items", [])
        if not channel_items:
            raise YouTubeChannelError(f"Could not fetch YouTube channel details for: {channel_id}")
        return channel_items[0]

    def _list_recent_videos(self, *, channel_id: str, limit: int) -> list[dict[str, Any]]:
        payload = self._api_get(
            "search",
            {
                "part": "snippet",
                "channelId": channel_id,
                "type": "video",
                "order": "date",
                "maxResults": max(1, min(limit, 25)),
            },
        )
        return [item for item in payload.get("items", []) if str((item.get("id") or {}).get("videoId", "") or "")]

    def _fetch_video_details(self, video_ids: list[str]) -> list[ChannelVideo]:
        ids = [video_id for video_id in video_ids if video_id]
        if not ids:
            return []
        payload = self._api_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(ids),
                "maxResults": len(ids),
            },
        )
        items = payload.get("items", [])
        by_id: dict[str, ChannelVideo] = {}
        for item in items:
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}
            content_details = item.get("contentDetails") or {}
            video_id = str(item.get("id", "") or "")
            duration_seconds = _parse_iso8601_duration(str(content_details.get("duration", "") or ""))
            views = int(statistics.get("viewCount", 0) or 0)
            likes = int(statistics.get("likeCount", 0) or 0)
            by_id[video_id] = ChannelVideo(
                video_id=video_id,
                title=str(snippet.get("title", "") or ""),
                views=views,
                views_label=_format_count(views),
                duration_seconds=duration_seconds,
                duration_label=_format_duration(duration_seconds),
                likes=likes,
                likes_label=_format_count(likes),
                thumbnail_url=_thumbnail_map_best_url(snippet.get("thumbnails") or {}),
                published_at=str(snippet.get("publishedAt", "") or ""),
                description=str(snippet.get("description", "") or ""),
            )
        return [by_id[video_id] for video_id in ids if video_id in by_id]

    def _api_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise YouTubeChannelError("Missing YOUTUBE_API_KEY.")
        response = self._client.get(
            f"{YOUTUBE_API_BASE}/{path}",
            params={**params, "key": self.api_key},
            timeout=self.timeout_s,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise YouTubeChannelError(
                f"YouTube API request failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        return response.json()

    @staticmethod
    def _supports_fallback_query(query: str) -> bool:
        return bool(YouTubeChannelService._extract_handle(query) or YouTubeChannelService._extract_channel_id(query) or "youtube.com/" in query)

    @staticmethod
    def _normalize_channel_url(query: str) -> str:
        handle = YouTubeChannelService._extract_handle(query)
        if handle:
            return f"https://www.youtube.com/{handle}"
        channel_id = YouTubeChannelService._extract_channel_id(query)
        if channel_id:
            return f"https://www.youtube.com/channel/{channel_id}"
        return query

    @staticmethod
    def _extract_handle(query: str) -> str:
        stripped = query.strip()
        if stripped.startswith("@"):
            return stripped
        try:
            parsed = urlparse(stripped)
        except ValueError:
            return ""
        path = parsed.path.strip("/")
        if path.startswith("@"):
            return path.split("/", 1)[0]
        return ""

    @staticmethod
    def _extract_channel_id(query: str) -> str:
        stripped = query.strip()
        if stripped.startswith("UC") and "/" not in stripped and " " not in stripped:
            return stripped
        try:
            parsed = urlparse(stripped)
        except ValueError:
            return ""
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] == "channel":
            return path_parts[1]
        if parsed.netloc.endswith("youtube.com") and parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if video_id:
                return ""
        return ""