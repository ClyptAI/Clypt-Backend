from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse


class _FakeHTTPResponse:
    def __init__(self, *, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_fetch_youtube_source_context_uses_existing_youtube_api_envs(
    monkeypatch,
) -> None:
    from backend.phase1_runtime.youtube_metadata_client import fetch_youtube_source_context

    monkeypatch.setenv("CLYPT_YOUTUBE_DATA_API_KEY", "yt-key")
    monkeypatch.setenv("CLYPT_YOUTUBE_DATA_API_BASE_URL", "https://example.test/youtube/v3")

    captured: dict[str, object] = {}

    def fake_urlopen(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            body={
                "items": [
                    {
                        "snippet": {
                            "title": "Fetched Title",
                            "description": "Fetched Description",
                            "channelId": "channel-ctx",
                            "channelTitle": "Fetched Channel",
                            "publishedAt": "2026-04-22T00:00:00Z",
                            "defaultAudioLanguage": "en",
                            "categoryId": "22",
                            "tags": ["clip", "metadata"],
                            "thumbnails": {"default": {"url": "https://example.test/thumb.jpg"}},
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("backend.phase1_runtime.youtube_metadata_client.urlopen", fake_urlopen)

    context = fetch_youtube_source_context(source_url="https://www.youtube.com/watch?v=source-context")

    parsed = urlparse(captured["url"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.test"
    assert parsed.path == "/youtube/v3/videos"
    assert query["id"] == ["source-context"]
    assert query["key"] == ["yt-key"]
    assert captured["timeout"] == 30
    assert context == {
        "source_url": "https://www.youtube.com/watch?v=source-context",
        "youtube_video_id": "source-context",
        "source_title": "Fetched Title",
        "source_description": "Fetched Description",
        "channel_id": "channel-ctx",
        "channel_title": "Fetched Channel",
        "published_at": "2026-04-22T00:00:00Z",
        "default_audio_language": "en",
        "category_id": "22",
        "tags": ["clip", "metadata"],
        "thumbnails": {"default": {"url": "https://example.test/thumb.jpg"}},
    }
