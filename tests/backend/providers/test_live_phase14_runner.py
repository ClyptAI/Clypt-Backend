from __future__ import annotations

import json
import re
from pathlib import Path


def test_live_phase14_runner_executes_provider_backed_phases_2_to_4(tmp_path: Path):
    from backend.phase1_runtime.models import Phase1SidecarOutputs
    from backend.pipeline.config import V31Config
    from backend.repository.models import (
        ClipCandidateRecord,
        Phase24JobRecord,
        PhaseMetricRecord,
        RunRecord,
        SemanticEdgeRecord,
        SemanticNodeRecord,
        TimelineTurnRecord,
    )
    from backend.runtime.phase14_live import V31LivePhase14Runner

    class _FakeRepository:
        def __init__(self) -> None:
            self.timeline_turns: list[TimelineTurnRecord] = []
            self.nodes: list[SemanticNodeRecord] = []
            self.edges: list[SemanticEdgeRecord] = []
            self.candidates: list[ClipCandidateRecord] = []
            self.phase_metrics: list[PhaseMetricRecord] = []
            self.subgraph_provenance = []
            self.external_signals = []
            self.external_signal_clusters = []
            self.prompt_source_links = []
            self.node_signal_links = []
            self.candidate_signal_links = []

        def upsert_run(self, record: RunRecord) -> RunRecord:
            return record

        def get_run(self, run_id: str) -> RunRecord | None:
            return None

        def write_timeline_turns(self, *, run_id: str, turns: list[TimelineTurnRecord]) -> None:
            self.timeline_turns = list(turns)

        def list_timeline_turns(self, *, run_id: str) -> list[TimelineTurnRecord]:
            return list(self.timeline_turns)

        def write_nodes(self, *, run_id: str, nodes: list[SemanticNodeRecord]) -> None:
            self.nodes = list(nodes)

        def list_nodes(self, *, run_id: str) -> list[SemanticNodeRecord]:
            return list(self.nodes)

        def write_edges(self, *, run_id: str, edges: list[SemanticEdgeRecord]) -> None:
            self.edges = list(edges)

        def list_edges(self, *, run_id: str) -> list[SemanticEdgeRecord]:
            return list(self.edges)

        def write_candidates(self, *, run_id: str, candidates: list[ClipCandidateRecord]) -> None:
            self.candidates = list(candidates)

        def list_candidates(self, *, run_id: str) -> list[ClipCandidateRecord]:
            return list(self.candidates)

        def write_phase_metric(self, record: PhaseMetricRecord) -> PhaseMetricRecord:
            self.phase_metrics.append(record)
            return record

        def list_phase_metrics(self, *, run_id: str) -> list[PhaseMetricRecord]:
            return list(self.phase_metrics)

        def upsert_phase24_job(self, record: Phase24JobRecord) -> Phase24JobRecord:
            return record

        def get_phase24_job(self, run_id: str) -> Phase24JobRecord | None:
            return None

        def write_subgraph_provenance(self, *, run_id: str, provenance) -> None:
            self.subgraph_provenance = list(provenance)

        def write_external_signals(self, *, run_id: str, signals) -> None:
            self.external_signals = list(signals)

        def write_external_signal_clusters(self, *, run_id: str, clusters) -> None:
            self.external_signal_clusters = list(clusters)

        def write_prompt_source_links(self, *, run_id: str, links) -> None:
            self.prompt_source_links = list(links)

        def write_node_signal_links(self, *, run_id: str, links) -> None:
            self.node_signal_links = list(links)

        def write_candidate_signal_links(self, *, run_id: str, links) -> None:
            self.candidate_signal_links = list(links)

    class _FakeEmbeddingClient:
        def embed_texts(self, texts, *, task_type=None, model=None):
            return [[float(idx), 0.5, 0.25] for idx, _ in enumerate(texts, start=1)]

        def embed_media_uris(self, media_items, *, model=None):
            return [[float(idx), 0.25, 0.5] for idx, _ in enumerate(media_items, start=1)]

    class _FakeLLMClient:
        def __init__(self):
            self.calls = []

        def generate_json(self, *, prompt, model=None, temperature=0.0, **kwargs):
            self.calls.append(prompt)
            prompt_lc = prompt.lower()
            def _extract_json_payload(marker: str) -> dict:
                if marker not in prompt:
                    return {}
                raw = prompt.split(marker, 1)[1].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {}
            if "merge contiguous target turns" in prompt_lc:
                payload = _extract_json_payload("Neighborhood payload:\n")
                target_turn_ids = list(payload.get("target_turn_ids") or [])
                if not target_turn_ids:
                    target_turn_ids = ["t_000001"]
                return {
                    "merged_nodes": [
                        {
                            "source_turn_ids": target_turn_ids,
                            "node_type": "claim",
                            "node_flags": ["high_resonance_candidate"],
                            "summary": "Merged target unit.",
                        }
                    ]
                }
            if "boundary between two semantic batches" in prompt_lc:
                payload = _extract_json_payload("Boundary payload:\n")
                left = ((payload.get("left_batch_nodes") or [{}])[0]) if payload else {}
                right = ((payload.get("right_batch_nodes") or [{}])[0]) if payload else {}

                def _boundary_node(item: dict, fallback_id: str) -> dict:
                    return {
                        "existing_node_id": item.get("node_id") or item.get("existing_node_id") or fallback_id,
                        "source_turn_ids": list(item.get("source_turn_ids") or []),
                        "node_type": item.get("node_type") or "claim",
                        "node_flags": list(item.get("node_flags") or []),
                        "summary": item.get("summary") or "Boundary node.",
                    }

                return {
                    "resolution": "keep_both",
                    "nodes": [
                        _boundary_node(left, "left_node"),
                        _boundary_node(right, "right_node"),
                    ],
                }
            if (
                "draw only local semantic graph edges" in prompt_lc
                or "identify meaningful semantic edges only between target nodes" in prompt_lc
            ):
                return {"edges": []}
            if (
                "adjudicate only callback_to and topic_recurrence" in prompt_lc
                or "adjudicating potential long-range semantic edges" in prompt_lc
            ):
                return {"edges": []}
            if "designing retrieval queries to find the best short-form clip candidates" in prompt_lc:
                target_count_match = re.search(r"task:\s*generate exactly\s+(\d+)\s+targeted retrieval prompts", prompt_lc)
                target_count = int(target_count_match.group(1)) if target_count_match else 1
                return {
                    "prompts": [
                        f"Find the strongest standalone moment #{idx}."
                        for idx in range(1, target_count + 1)
                    ]
                }
            if (
                "selecting clip candidates from a local semantic subgraph" in prompt_lc
                or "review this local semantic subgraph" in prompt_lc
            ):
                payload = _extract_json_payload("Subgraph payload:\n")
                subgraph_id = payload.get("subgraph_id") or "sg_0001"
                seed_node_id = payload.get("seed_node_id") or "node_seed_001"
                nodes = list(payload.get("nodes") or [])
                first_node = nodes[0] if nodes else {}
                return {
                    "subgraph_id": subgraph_id,
                    "seed_node_id": seed_node_id,
                    "reject_all": False,
                    "reject_reason": "",
                    "candidates": [
                        {
                            "node_ids": [first_node.get("node_id") or seed_node_id],
                            "start_ms": int(first_node.get("start_ms") or 0),
                            "end_ms": int(first_node.get("end_ms") or 1600),
                            "score": 8.2,
                            "rationale": "Strong standalone thought.",
                        }
                    ],
                }
            if (
                "final quality review of a pool of clip candidates" in prompt_lc
                or "review this candidate pool" in prompt_lc
            ):
                payload = _extract_json_payload("Candidate pool:\n")
                candidates = list(payload.get("candidates") or [])
                ranked = []
                dropped = []
                for idx, candidate in enumerate(candidates, start=1):
                    clip_id = candidate.get("clip_id") or f"cand_tmp_{idx:03d}"
                    if idx == 1:
                        ranked.append(
                            {
                                "candidate_temp_id": clip_id,
                                "keep": True,
                                "pool_rank": 1,
                                "score": 8.9,
                                "score_breakdown": {"virality": 8.9, "coherence": 8.9, "engagement": 8.9},
                                "rationale": "Best clip.",
                            }
                        )
                    else:
                        dropped.append(clip_id)
                return {
                    "ranked_candidates": ranked,
                    "dropped_candidate_temp_ids": dropped,
                }
            raise AssertionError(f"unexpected prompt: {prompt[:120]}")

    repository = _FakeRepository()
    log_events: list[dict[str, object]] = []

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=_FakeLLMClient(),
        embedding_client=_FakeEmbeddingClient(),
        repository=repository,
        query_version="graph-v2",
        log_event=lambda **payload: log_events.append(payload),
        node_media_preparer=lambda **kwargs: [
            {
                "node_id": node.node_id,
                "file_uri": f"gs://bucket/{node.node_id}.mp4",
                "mime_type": "video/mp4",
                "local_path": str(tmp_path / f"{node.node_id}.mp4"),
            }
            for node in kwargs["nodes"]
        ],
    )

    summary = runner.run(
        run_id="run_live",
        source_url="https://example.com/video",
        phase1_outputs=Phase1SidecarOutputs(
            phase1_audio={
                "source_audio": "https://example.com/video",
                "video_gcs_uri": "gs://bucket/video.mp4",
                "local_video_path": str(tmp_path / "source_video.mp4"),
            },
            diarization_payload={
                "words": [
                    {"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 400, "speaker_id": "SPEAKER_0"},
                    {"word_id": "w_000002", "text": "world", "start_ms": 600, "end_ms": 1000, "speaker_id": "SPEAKER_0"},
                    {"word_id": "w_000003", "text": "goodbye", "start_ms": 1200, "end_ms": 1600, "speaker_id": "SPEAKER_1"},
                ],
                "turns": [
                    {"turn_id": "t_000001", "speaker_id": "SPEAKER_0", "start_ms": 0, "end_ms": 1000, "transcript_text": "hello world", "word_ids": ["w_000001", "w_000002"], "identification_match": None},
                    {"turn_id": "t_000002", "speaker_id": "SPEAKER_1", "start_ms": 1200, "end_ms": 1600, "transcript_text": "goodbye", "word_ids": ["w_000003"], "identification_match": None},
                ],
            },
            phase1_visual={
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 2000}],
                "tracks": [],
            },
            emotion2vec_payload={
                "segments": [
                    {
                        "turn_id": "t_000001",
                        "labels": ["neutral"],
                        "scores": [0.7],
                        "per_class_scores": {"neutral": 0.7},
                    },
                    {
                        "turn_id": "t_000002",
                        "labels": ["happy"],
                        "scores": [0.8],
                        "per_class_scores": {"happy": 0.8},
                    },
                ]
            },
            yamnet_payload={"events": []},
        ),
    )

    assert summary.run_id == "run_live"
    assert summary.artifact_paths == {}
    assert len(repository.timeline_turns) == 2
    assert len(repository.nodes) >= 1
    assert len(repository.edges) >= 1
    assert len(repository.candidates) == 1
    assert repository.candidates[0].rationale == "Best clip."
    assert [metric.phase_name for metric in repository.phase_metrics] == ["phase2", "phase3", "phase4", "phase24"]
    assert all(metric.query_version == "graph-v2" for metric in repository.phase_metrics)
    assert any(event["event"] == "candidate_summary" for event in log_events)
