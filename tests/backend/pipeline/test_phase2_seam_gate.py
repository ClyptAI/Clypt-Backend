from __future__ import annotations

import json

from backend.pipeline.contracts import CanonicalTimeline, CanonicalTurn
from backend.pipeline.semantics.responses import BoundarySkipDecision
from backend.pipeline.semantics.runtime import run_merge_classify_and_reconcile


class _BoundaryMustNotRunLLM:
    def __init__(self) -> None:
        self.merge_calls = 0
        self.boundary_calls = 0

    def generate_json(self, *, prompt, **kwargs):
        if "Boundary payload:\n" in prompt:
            self.boundary_calls += 1
            raise AssertionError("boundary LLM call should have been skipped by heuristic")
        if "Neighborhood payload:\n" not in prompt:
            raise AssertionError(f"unexpected prompt: {prompt[:120]}")
        self.merge_calls += 1
        payload = json.loads(prompt.split("Neighborhood payload:\n", 1)[1].strip())
        target_turn_ids = list(payload.get("target_turn_ids") or [])
        first_turn_id = target_turn_ids[0]
        if first_turn_id == "t_000001":
            return {
                "merged_nodes": [
                    {
                        "source_turn_ids": target_turn_ids,
                        "node_type": "claim",
                        "node_flags": [],
                        "summary": "Early setup section.",
                    }
                ]
            }
        return {
            "merged_nodes": [
                {
                    "source_turn_ids": target_turn_ids,
                    "node_type": "reaction_beat",
                    "node_flags": [],
                    "summary": "Later reaction section.",
                }
            ]
        }


def test_run_merge_classify_and_reconcile_skips_clear_seam_llm_call() -> None:
    timeline = CanonicalTimeline(
        turns=[
            CanonicalTurn(
                turn_id="t_000001",
                speaker_id="SPEAKER_0",
                start_ms=0,
                end_ms=800,
                word_ids=[],
                transcript_text="Opening setup.",
            ),
            CanonicalTurn(
                turn_id="t_000002",
                speaker_id="SPEAKER_0",
                start_ms=900,
                end_ms=1500,
                word_ids=[],
                transcript_text="More setup.",
            ),
            CanonicalTurn(
                turn_id="t_000005",
                speaker_id="SPEAKER_1",
                start_ms=9000,
                end_ms=9600,
                word_ids=[],
                transcript_text="Hard cut to later reaction.",
            ),
            CanonicalTurn(
                turn_id="t_000006",
                speaker_id="SPEAKER_1",
                start_ms=9700,
                end_ms=10300,
                word_ids=[],
                transcript_text="Later reaction continues.",
            ),
        ]
    )
    llm_client = _BoundaryMustNotRunLLM()

    nodes, merge_debug, boundary_debug = run_merge_classify_and_reconcile(
        canonical_timeline=timeline,
        speech_emotion_timeline=None,
        audio_event_timeline=None,
        llm_client=llm_client,
        target_batch_count=2,
        max_turns_per_batch=2,
        max_concurrent=2,
    )

    assert len(nodes) == 2
    assert len(merge_debug) == 2
    assert len(boundary_debug) == 1
    assert boundary_debug[0]["heuristic_skip"] is True
    assert boundary_debug[0]["heuristic_reason"] == "large_time_gap"
    assert boundary_debug[0]["diagnostics"]["heuristic_sent_to_llm"] is False
    assert llm_client.merge_calls == 2
    assert llm_client.boundary_calls == 0


def test_run_merge_classify_and_reconcile_uses_separate_merge_and_boundary_concurrency(
    monkeypatch,
) -> None:
    from backend.pipeline.contracts import SemanticGraphNode, SemanticNodeEvidence
    from backend.pipeline.semantics import runtime as semantics_runtime

    timeline = CanonicalTimeline(
        turns=[
            CanonicalTurn(
                turn_id="t_000001",
                speaker_id="SPEAKER_0",
                start_ms=0,
                end_ms=800,
                word_ids=[],
                transcript_text="Turn one.",
            ),
            CanonicalTurn(
                turn_id="t_000002",
                speaker_id="SPEAKER_0",
                start_ms=900,
                end_ms=1500,
                word_ids=[],
                transcript_text="Turn two.",
            ),
            CanonicalTurn(
                turn_id="t_000003",
                speaker_id="SPEAKER_0",
                start_ms=1600,
                end_ms=2200,
                word_ids=[],
                transcript_text="Turn three.",
            ),
            CanonicalTurn(
                turn_id="t_000004",
                speaker_id="SPEAKER_0",
                start_ms=2300,
                end_ms=2900,
                word_ids=[],
                transcript_text="Turn four.",
            ),
        ]
    )

    seen_max_workers: list[int] = []

    class _ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _RecordingExecutor:
        def __init__(self, *, max_workers, **kwargs):
            seen_max_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _ImmediateFuture(fn(*args, **kwargs))

    class _MergeOnlyLLM:
        def generate_json(self, *, prompt, **kwargs):
            payload = json.loads(prompt.split("Neighborhood payload:\n", 1)[1].strip())
            target_turn_ids = list(payload.get("target_turn_ids") or [])
            return {
                "merged_nodes": [
                    {
                        "source_turn_ids": target_turn_ids,
                        "node_type": "claim",
                        "node_flags": [],
                        "summary": f"Summary for {target_turn_ids[0]}",
                    }
                ]
            }

    monkeypatch.setattr(semantics_runtime, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(semantics_runtime, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        semantics_runtime,
        "should_skip_boundary_reconciliation",
        lambda **kwargs: BoundarySkipDecision(
            skip_llm=False,
            reason="ambiguous_default",
            time_gap_ms=0,
            turn_gap=None,
            summary_similarity=0.0,
            transcript_similarity=0.0,
            shared_flag_count=0,
            shared_flags=[],
            overlap_turn_count=0,
            same_node_type=True,
        ),
    )
    monkeypatch.setattr(
        semantics_runtime,
        "run_boundary_reconciliation",
        lambda **kwargs: (
            [
                SemanticGraphNode(
                    node_id="node_boundary",
                    node_type="claim",
                    start_ms=0,
                    end_ms=2900,
                    transcript_text="Combined seam",
                    summary="Combined seam",
                    evidence=SemanticNodeEvidence(),
                    source_turn_ids=["t_000002", "t_000003"],
                )
            ],
            {"prompt": "boundary", "response": {}, "diagnostics": {"latency_ms": 0.0}},
        ),
    )

    run_merge_classify_and_reconcile(
        canonical_timeline=timeline,
        speech_emotion_timeline=None,
        audio_event_timeline=None,
        llm_client=_MergeOnlyLLM(),
        target_batch_count=2,
        max_turns_per_batch=2,
        max_concurrent=8,
        boundary_max_concurrent=10,
    )

    assert seen_max_workers == [8, 10]
