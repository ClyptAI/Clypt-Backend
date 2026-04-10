from __future__ import annotations

import pytest

from backend.pipeline.contracts import (
    CanonicalTimeline,
    CanonicalTurn,
    SpeechEmotionTimeline,
    AudioEventTimeline,
)


class _FakeResponse:
    def __init__(self, text: str, parsed=None):
        self.text = text
        self.parsed = parsed


class _FakeTransientError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"{status_code} {message}")
        self.status_code = status_code


class _FakeModelAPI:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []
        self.generate_failures_remaining = 0
        self.embed_failures_remaining = 0
        self.generate_next_response = None

    def generate_content(self, *, model, contents, config):
        if self.generate_failures_remaining > 0:
            self.generate_failures_remaining -= 1
            raise _FakeTransientError(429, "RESOURCE_EXHAUSTED")
        self.generate_calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        if self.generate_next_response is not None:
            response = self.generate_next_response
            self.generate_next_response = None
            return response
        return _FakeResponse(
            '{"ok": true, "contents_seen": 1}',
            parsed={"ok": True, "contents_seen": 1},
        )

    def embed_content(self, *, model, contents, config=None):
        if self.embed_failures_remaining > 0:
            self.embed_failures_remaining -= 1
            raise _FakeTransientError(503, "UNAVAILABLE")
        call_index = len(self.embed_calls)
        self.embed_calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        base_vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        if isinstance(contents, list):
            embedding_values = [
                base_vectors[idx % len(base_vectors)] for idx in range(len(contents))
            ]
        else:
            embedding_values = [base_vectors[call_index % len(base_vectors)]]
        return type(
            "EmbedResult",
            (),
            {
                "embeddings": [
                    type("Embed", (), {"values": values})()
                    for values in embedding_values
                ]
            },
        )()


class _FakeGenAIClient:
    def __init__(self):
        self.models = _FakeModelAPI()


def test_vertex_clients_generate_json_and_embed_texts():
    from backend.providers.config import VertexSettings
    from backend.providers.vertex import VertexEmbeddingClient, VertexGeminiClient

    sdk_client = _FakeGenAIClient()
    settings = VertexSettings(
        project="clypt-v3",
        generation_location="global",
        embedding_location="us-central1",
    )

    gemini_client = VertexGeminiClient(settings=settings, sdk_client=sdk_client)
    response = gemini_client.generate_json(prompt="hello world")

    assert response["ok"] is True
    assert sdk_client.models.generate_calls[0]["model"] == settings.generation_model
    assert sdk_client.models.generate_calls[0]["contents"] == "hello world"

    embedding_client = VertexEmbeddingClient(settings=settings, sdk_client=sdk_client)
    embeddings = embedding_client.embed_texts(["alpha", "beta"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert sdk_client.models.embed_calls[0]["model"] == settings.embedding_model
    assert sorted(call["contents"] for call in sdk_client.models.embed_calls[:2]) == [
        "alpha",
        "beta",
    ]

    media_embeddings = embedding_client.embed_media_uris(
        [
            {"file_uri": "gs://bucket/node_1.mp4", "mime_type": "video/mp4"},
            {"file_uri": "gs://bucket/node_2.mp4", "mime_type": "video/mp4"},
        ]
    )

    assert len(media_embeddings) == 2
    assert all(len(vector) == 3 for vector in media_embeddings)
    assert all(call["model"] == settings.embedding_model for call in sdk_client.models.embed_calls)


def test_vertex_gemini_client_prefers_sdk_parsed_json():
    from backend.providers.config import VertexSettings
    from backend.providers.vertex import VertexGeminiClient

    sdk_client = _FakeGenAIClient()
    sdk_client.models.generate_next_response = _FakeResponse(
        text="not valid json",
        parsed={"ok": True, "from": "parsed"},
    )
    settings = VertexSettings(project="clypt-v3")
    gemini_client = VertexGeminiClient(settings=settings, sdk_client=sdk_client)

    response = gemini_client.generate_json(prompt="use parsed")

    assert response == {"ok": True, "from": "parsed"}


def test_vertex_gemini_client_uses_low_thinking_budget_for_pro_only():
    from backend.providers.config import VertexSettings
    from backend.providers.vertex import VertexGeminiClient

    sdk_client = _FakeGenAIClient()
    settings = VertexSettings(
        project="clypt-v3",
        thinking_budget=128,
    )
    gemini_client = VertexGeminiClient(settings=settings, sdk_client=sdk_client)

    gemini_client.generate_json(prompt="pro call", model="gemini-3.1-pro-preview")
    pro_config = sdk_client.models.generate_calls[-1]["config"]
    thinking_cfg = getattr(pro_config, "thinking_config", None)
    assert thinking_cfg is not None
    assert getattr(thinking_cfg, "thinking_budget", None) == 128

    gemini_client.generate_json(prompt="flash call", model="gemini-3-flash-preview")
    flash_config = sdk_client.models.generate_calls[-1]["config"]
    assert getattr(flash_config, "thinking_config", None) is None


def test_vertex_clients_retry_transient_generate_and_embed_errors(monkeypatch: pytest.MonkeyPatch):
    from backend.providers.config import VertexSettings
    from backend.providers.vertex import VertexEmbeddingClient, VertexGeminiClient

    sdk_client = _FakeGenAIClient()
    sdk_client.models.generate_failures_remaining = 2
    sdk_client.models.embed_failures_remaining = 2

    settings = VertexSettings(
        project="clypt-v3",
        generation_location="global",
        embedding_location="us-central1",
        api_max_retries=4,
        api_initial_backoff_s=0.01,
        api_max_backoff_s=0.02,
        api_backoff_multiplier=2.0,
        api_jitter_ratio=0.0,
    )

    sleeps: list[float] = []
    monkeypatch.setattr("backend.providers.vertex.time.sleep", lambda seconds: sleeps.append(seconds))

    gemini_client = VertexGeminiClient(settings=settings, sdk_client=sdk_client)
    response = gemini_client.generate_json(prompt="retry me")
    assert response["ok"] is True
    assert len(sdk_client.models.generate_calls) == 1

    embedding_client = VertexEmbeddingClient(settings=settings, sdk_client=sdk_client)
    vectors = embedding_client.embed_texts(["alpha"])
    assert vectors == [[0.1, 0.2, 0.3]]
    assert len(sdk_client.models.embed_calls) == 1
    assert len(sleeps) == 4


def test_vertex_gemini_client_fails_hard_when_parsed_payload_missing():
    from backend.providers.config import VertexSettings
    from backend.providers.vertex import VertexGeminiClient

    sdk_client = _FakeGenAIClient()
    sdk_client.models.generate_next_response = _FakeResponse('{"broken": "unterminated}')
    settings = VertexSettings(project="clypt-v3")
    gemini_client = VertexGeminiClient(settings=settings, sdk_client=sdk_client)

    with pytest.raises(ValueError, match="did not return SDK-parsed JSON object"):
        gemini_client.generate_json(prompt="bad json")


def test_semantics_runtime_calls_live_client_and_returns_nodes():
    from backend.pipeline.semantics.runtime import run_merge_and_classify_batches

    timeline = CanonicalTimeline(
        words=[],
        turns=[
            CanonicalTurn(
                turn_id="t_000001",
                speaker_id="S1",
                start_ms=0,
                end_ms=1000,
                word_ids=[],
                transcript_text="hello",
            ),
            CanonicalTurn(
                turn_id="t_000002",
                speaker_id="S1",
                start_ms=1100,
                end_ms=2000,
                word_ids=[],
                transcript_text="world",
            ),
        ],
    )

    calls = []

    class _FakeLLM:
        def generate_json(self, *, prompt, model=None, temperature=0.0, **kwargs):
            calls.append(prompt)
            return {
                "merged_nodes": [
                    {
                        "source_turn_ids": ["t_000001", "t_000002"],
                        "node_type": "claim",
                        "node_flags": [],
                        "summary": "hello world",
                    }
                ]
            }

    nodes, debug = run_merge_and_classify_batches(
        canonical_timeline=timeline,
        speech_emotion_timeline=SpeechEmotionTimeline(events=[]),
        audio_event_timeline=AudioEventTimeline(events=[]),
        llm_client=_FakeLLM(),
        target_turn_count=2,
        halo_turn_count=0,
    )

    assert len(calls) == 1
    assert len(nodes) == 1
    assert nodes[0].transcript_text == "hello world"
    assert debug[0]["batch_id"] == "nb_0001"


def test_candidates_runtime_fails_hard_when_pool_review_is_missing():
    from backend.pipeline.candidates.runtime import run_candidate_pool_review
    from backend.pipeline.contracts import ClipCandidate

    candidates = [
        ClipCandidate(
            clip_id="cand_1",
            node_ids=["node_1"],
            start_ms=0,
            end_ms=1000,
            score=7.0,
            rationale="test",
        )
    ]

    with pytest.raises(ValueError, match="required"):
        run_candidate_pool_review(candidates=candidates, llm_client=None)
