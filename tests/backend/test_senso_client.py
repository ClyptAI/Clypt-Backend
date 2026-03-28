from __future__ import annotations

import json

import httpx
import pytest

from backend.integrations.senso_client import SensoAPIError, SensoClient


def _build_client(handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="https://apiv2.senso.ai/api/v1", transport=transport)
    return SensoClient(api_key="test-key", http_client=http_client)


def test_create_raw_content_sends_expected_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            202,
            json={
                "id": "content_123",
                "type": "raw",
                "title": "My Title",
                "processing_status": "embed_queued",
                "latest_content_version_id": "version_1",
            },
        )

    client = _build_client(handler)

    record = client.create_raw_content(
        title="My Title",
        summary="Short summary",
        text="Transcript body",
        kb_folder_node_id="root_1",
    )

    assert record.id == "content_123"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://apiv2.senso.ai/api/v1/org/kb/raw"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["json"] == {
        "title": "My Title",
        "summary": "Short summary",
        "text": "Transcript body",
        "kb_folder_node_id": "root_1",
    }


def test_search_context_sends_expected_filters():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "query": "What kind of creator is this?",
                "results": [
                    {
                        "content_id": "content_123",
                        "chunk_text": "TypeScript and React ecosystem content.",
                        "score": 0.91,
                        "title": "Recent uploads",
                    }
                ],
                "total_results": 1,
                "max_results": 7,
            },
        )

    client = _build_client(handler)
    response = client.search(
        query="What kind of creator is this?",
        max_results=7,
        content_ids=["content_123", "content_456"],
        require_scoped_ids=True,
        include_answer=False,
    )

    assert len(response.results) == 1
    assert captured["url"] == "https://apiv2.senso.ai/api/v1/org/search/context"
    assert captured["json"] == {
        "query": "What kind of creator is this?",
        "max_results": 7,
        "content_ids": ["content_123", "content_456"],
        "require_scoped_ids": True,
    }


def test_get_prompt_returns_prompt_details():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "prompt_id": "prompt_1",
                "text": "What are the defining content patterns of this creator?",
                "type": "consideration",
            },
        )

    client = _build_client(handler)
    prompt = client.get_prompt("prompt_1")

    assert prompt.prompt_id == "prompt_1"
    assert prompt.text.startswith("What are the defining")
    assert captured["url"] == "https://apiv2.senso.ai/api/v1/org/prompts/prompt_1"


def test_wait_for_search_results_retries_until_results_exist():
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                200,
                json={"query": "creator", "answer": "No results found for your query.", "results": []},
            )
        return httpx.Response(
            200,
            json={
                "query": "creator",
                "answer": "{\"creator_archetype\":\"Educator\"}",
                "results": [{"content_id": "content_123", "chunk_text": "test", "score": 0.8}],
            },
        )

    client = _build_client(handler)
    response = client.wait_for_search_results(
        query="creator",
        content_ids=["content_123"],
        max_wait_s=0.01,
        poll_interval_s=0.0,
    )

    assert response.results[0].content_id == "content_123"
    assert calls["count"] == 2


def test_request_raises_senso_error_with_server_message():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "You do not have access to this resource"})

    client = _build_client(handler)

    with pytest.raises(SensoAPIError) as excinfo:
        client.get_kb_root()

    assert excinfo.value.status_code == 403
    assert "You do not have access to this resource" in str(excinfo.value)
