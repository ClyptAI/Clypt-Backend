from __future__ import annotations

from backend.pipeline.signals.contracts import ExternalSignal, ExternalSignalCluster
from backend.pipeline.signals.llm_runtime import (
    classify_comment_with_llm,
    consolidate_thread_with_llm,
    generate_cluster_prompt_with_llm,
    resolve_cluster_span_with_llm,
)


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_json(self, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        schema = kwargs["response_schema"]
        if schema.get("properties", {}).get("prompt"):
            return {"prompt": "Find the exact moment"}
        if schema.get("properties", {}).get("node_ids"):
            return {"node_ids": ["node-1", "node-2"]}
        if schema.get("properties", {}).get("quality"):
            return {"quality": "high_signal", "reason": "specific and useful"}
        return {"thread_summary": "summary", "moment_hints": ["hint"]}


def test_signal_llm_wrappers_use_expected_schema_and_token_limit() -> None:
    llm_client = _FakeLLMClient()
    cluster = ExternalSignalCluster(
        cluster_id="comment_cluster_001",
        cluster_type="comment",
        summary_text="audience likes the payoff",
        member_signal_ids=["signal-1"],
        cluster_weight=0.0,
        embedding=[1.0, 0.0],
        metadata={},
    )

    assert consolidate_thread_with_llm(
        llm_client=llm_client,
        model="gemini-3-flash",
        thread_payload={"thread": "payload"},
    ).thread_summary == "summary"
    assert classify_comment_with_llm(
        llm_client=llm_client,
        model="gemini-3.1-flash-lite",
        signal=ExternalSignal(
            signal_id="signal-1",
            signal_type="comment_top",
            source_platform="youtube",
            source_id="comment-1",
            text="this is useful",
            engagement_score=1.0,
            metadata={},
        ),
    ).quality == "high_signal"
    assert generate_cluster_prompt_with_llm(
        llm_client=llm_client,
        model="gemini-3-flash",
        cluster=cluster,
    ).prompt == "Find the exact moment"
    assert resolve_cluster_span_with_llm(
        llm_client=llm_client,
        model="gemini-3-flash",
        cluster=cluster,
        neighborhood_payload={"node_ids": ["node-1", "node-2"]},
    ).node_ids == ["node-1", "node-2"]

    assert len(llm_client.calls) == 4
    for call in llm_client.calls:
        assert call["max_output_tokens"] == 32768
        assert call["temperature"] == 0.0
        assert "thinking_level" not in call
        assert "response_schema" in call
