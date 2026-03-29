from __future__ import annotations

from backend.services.youtube_channel_service import (
    _normalize_image_url,
    _thumbnail_list_best_url,
    _thumbnail_map_best_url,
)


def test_normalize_image_url_promotes_https():
    assert _normalize_image_url("//yt3.googleusercontent.com/avatar.jpg") == "https://yt3.googleusercontent.com/avatar.jpg"
    assert _normalize_image_url("http://yt3.googleusercontent.com/avatar.jpg") == "https://yt3.googleusercontent.com/avatar.jpg"


def test_thumbnail_map_best_url_prefers_highest_named_variant():
    thumbnails = {
        "default": {"url": "https://example.com/default.jpg"},
        "medium": {"url": "https://example.com/medium.jpg"},
        "high": {"url": "https://example.com/high.jpg"},
    }
    assert _thumbnail_map_best_url(thumbnails) == "https://example.com/high.jpg"


def test_thumbnail_list_best_url_prefers_square_cropped_avatar_over_uncropped():
    thumbnails = [
        {
            "url": "https://yt3.googleusercontent.com/avatar_uncropped=s0",
            "id": "avatar_uncropped",
            "preference": 1,
        },
        {
            "url": "https://yt3.googleusercontent.com/avatar_cropped=s900-c-k-c0x00ffffff-no-rj",
            "width": 900,
            "height": 900,
            "id": "0",
            "resolution": "900x900",
        },
    ]
    assert _thumbnail_list_best_url(thumbnails) == (
        "https://yt3.googleusercontent.com/avatar_cropped=s900-c-k-c0x00ffffff-no-rj"
    )
