from __future__ import annotations

from types import ModuleType
import sys

import pytest

from backend.pipeline.signals.comments_client import (
    collapse_same_author_spam,
    target_top_threads,
    to_external_signals_from_threads,
)
from backend.pipeline.signals.trends_client import TrendSpygClient, to_external_signals_from_trends


def test_target_top_threads_uses_dynamic_sqrt_with_fixed_bounds() -> None:
    assert target_top_threads(total_threads=0, min_threads=15, max_threads=40) == 15
    assert target_top_threads(total_threads=16, min_threads=15, max_threads=40) == 15
    assert target_top_threads(total_threads=400, min_threads=15, max_threads=40) == 20
    assert target_top_threads(total_threads=2500, min_threads=15, max_threads=40) == 40


def test_collapse_same_author_spam_only_dedupes_same_author_and_text() -> None:
    items = [
        {"snippet": {"authorChannelId": {"value": "author-1"}, "textDisplay": "same"}},
        {"snippet": {"authorChannelId": {"value": "author-1"}, "textDisplay": "same"}},
        {"snippet": {"authorChannelId": {"value": "author-2"}, "textDisplay": "same"}},
    ]

    collapsed = collapse_same_author_spam(items)

    assert len(collapsed) == 2
    assert collapsed[0]["snippet"]["authorChannelId"]["value"] == "author-1"
    assert collapsed[1]["snippet"]["authorChannelId"]["value"] == "author-2"


def test_to_external_signals_from_threads_includes_top_comment_and_replies() -> None:
    thread_items = [
        {
            "id": "thread-1",
            "snippet": {
                "topLevelComment": {
                    "id": "comment-top-1",
                    "snippet": {
                        "authorChannelId": {"value": "author-top"},
                        "textDisplay": "top comment",
                        "likeCount": 12,
                        "publishedAt": "2026-04-09T08:00:00Z",
                    },
                },
                "totalReplyCount": 3,
            },
            "replies": {
                "comments": [
                    {
                        "id": "reply-1",
                        "snippet": {
                            "authorChannelId": {"value": "author-reply"},
                            "textDisplay": "reply comment",
                            "likeCount": 4,
                            "publishedAt": "2026-04-09T08:01:00Z",
                        },
                    }
                ]
            },
        }
    ]

    signals = to_external_signals_from_threads(thread_items=thread_items, include_replies=True)

    assert [signal.signal_type for signal in signals] == ["comment_top", "comment_reply"]
    assert signals[0].engagement_score == 15.0
    assert signals[0].metadata["thread_id"] == "thread-1"
    assert signals[1].metadata["parent_comment_id"] == "comment-top-1"
    assert signals[1].engagement_score == 4.0


def test_to_external_signals_from_trends_does_not_synthesize_fallback_signal() -> None:
    assert to_external_signals_from_trends(query="game theory", items=[]) == []


def test_trendspyg_client_fails_fast_when_no_supported_method_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("trendspyg")

    class _FakeTrends:
        pass

    module.Trends = _FakeTrends  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trendspyg", module)

    client = TrendSpygClient(max_items=5)

    with pytest.raises(RuntimeError, match="supported related trend method"):
        client.fetch_related(query="example query")
