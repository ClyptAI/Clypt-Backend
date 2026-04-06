from __future__ import annotations

import pytest

from backend.pipeline.contracts import (
    CanonicalTimeline,
    CanonicalTurn,
    SpeechEmotionTimeline,
    AudioEventTimeline,
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModelAPI:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        return _FakeResponse('{"ok": true, "contents_seen": 1}')

    def embed_content(self, *, model, contents, config=None):
        self.embed_calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        embedding_count = len(contents) if isinstance(contents, list) else 1
        base_vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        return type(
            "EmbedResult",
            (),
            {
                "embeddings": [
                    type("Embed", (), {"values": base_vectors[idx]})()
                    for idx in range(embedding_count)
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
    assert sdk_client.models.embed_calls[0]["contents"] == ["alpha", "beta"]

    media_embeddings = embedding_client.embed_media_uris(
        [
            {"file_uri": "gs://bucket/node_1.mp4", "mime_type": "video/mp4"},
            {"file_uri": "gs://bucket/node_2.mp4", "mime_type": "video/mp4"},
        ]
    )

    assert media_embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert sdk_client.models.embed_calls[1]["model"] == settings.embedding_model


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
        def generate_json(self, *, prompt, model=None, temperature=0.0):
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
