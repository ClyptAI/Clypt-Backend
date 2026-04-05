from __future__ import annotations

from pathlib import Path

from backend.pipeline.config import V31Config
from backend.pipeline.contracts import Phase14RunSummary, SemanticGraphNode, SemanticNodeEvidence
from backend.pipeline.orchestrator import V31Phase14Orchestrator, V31Phase14RunInputs
from backend.pipeline.semantics.node_embeddings import embed_semantic_nodes
from backend.pipeline.candidates.query_embeddings import embed_prompt_texts


def _semantic_node(node_id: str, text: str) -> SemanticGraphNode:
    return SemanticGraphNode(
        node_id=node_id,
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        source_turn_ids=[f"turn_{node_id}"],
        word_ids=[],
        transcript_text=text,
        node_flags=[],
        summary=text,
        evidence=SemanticNodeEvidence(),
    )


def test_embed_semantic_nodes_and_prompt_texts_are_deterministic():
    nodes = [
        _semantic_node("node_1", "creators fake authenticity"),
        _semantic_node("node_2", "burnout returns later"),
    ]

    first_pass = embed_semantic_nodes(nodes=nodes)
    second_pass = embed_semantic_nodes(nodes=nodes)

    assert len(first_pass[0].semantic_embedding or []) == len(second_pass[0].semantic_embedding or [])
    assert first_pass[0].semantic_embedding == second_pass[0].semantic_embedding
    assert len(first_pass[0].multimodal_embedding or []) == len(second_pass[0].multimodal_embedding or [])
    assert first_pass[0].multimodal_embedding == second_pass[0].multimodal_embedding

    prompt_embeddings = embed_prompt_texts(prompts=["best hook", "most surprising moment"])
    assert [item["prompt_id"] for item in prompt_embeddings] == ["prompt_001", "prompt_002"]
    assert len(prompt_embeddings[0]["embedding"]) == len(first_pass[0].semantic_embedding or [])


def test_orchestrator_runs_phases_1_to_4_and_writes_artifacts(tmp_path: Path):
    orchestrator = V31Phase14Orchestrator(
        config=V31Config(output_root=tmp_path),
    )

    summary = orchestrator.run(
        run_id="run_demo",
        source_url="https://example.com/video",
        inputs=V31Phase14RunInputs(
            phase1_audio={
                "source_audio": "https://example.com/video",
                "video_gcs_uri": "gs://bucket/video.mp4",
            },
            phase1_visual={
                "video_metadata": {"fps": 10.0},
                "shot_changes": [
                    {"start_time_ms": 0, "end_time_ms": 1200},
                    {"start_time_ms": 1200, "end_time_ms": 2400},
                ],
                "tracks": [
                    {"frame_idx": 0, "track_id": "Global_Person_0", "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 80.0},
                    {"frame_idx": 6, "track_id": "Global_Person_0", "x1": 11.0, "y1": 21.0, "x2": 51.0, "y2": 81.0},
                    {"frame_idx": 13, "track_id": "Global_Person_1", "x1": 100.0, "y1": 120.0, "x2": 140.0, "y2": 180.0},
                ],
            },
            pyannote_payload={
                "wordLevelTranscription": [
                    {"word": "I", "start": 0.0, "end": 0.2, "speaker": "S1"},
                    {"word": "thought", "start": 0.2, "end": 0.4, "speaker": "S1"},
                    {"word": "this", "start": 0.4, "end": 0.55, "speaker": "S1"},
                    {"word": "would", "start": 0.55, "end": 0.7, "speaker": "S1"},
                    {"word": "fail", "start": 0.7, "end": 0.9, "speaker": "S1"},
                    {"word": "yeah", "start": 1.0, "end": 1.1, "speaker": "S2"},
                    {"word": "but", "start": 1.2, "end": 1.35, "speaker": "S1"},
                    {"word": "it", "start": 1.35, "end": 1.45, "speaker": "S1"},
                    {"word": "worked", "start": 1.45, "end": 1.75, "speaker": "S1"},
                    {"word": "wow", "start": 1.9, "end": 2.0, "speaker": "S2"},
                ],
                "diarization": [
                    {"speaker": "S1", "start": 0.0, "end": 0.9},
                    {"speaker": "S2", "start": 1.0, "end": 1.1},
                    {"speaker": "S1", "start": 1.2, "end": 1.75},
                    {"speaker": "S2", "start": 1.9, "end": 2.0},
                ],
            },
            emotion2vec_payload={
                "segments": [
                    {
                        "turn_id": "t_000001",
                        "labels": ["fearful"],
                        "scores": [0.81],
                        "per_class_scores": {"fearful": 0.81, "neutral": 0.19},
                    },
                    {
                        "turn_id": "t_000004",
                        "labels": ["surprised"],
                        "scores": [0.88],
                        "per_class_scores": {"surprised": 0.88, "happy": 0.12},
                    },
                ]
            },
            yamnet_payload={
                "events": [
                    {"event_label": "Laughter", "start_ms": 950, "end_ms": 2050, "confidence": 0.84},
                ]
            },
            phase2_target_turn_count=2,
            phase2_halo_turn_count=1,
            phase2_merge_responses={
                "nb_0001": {
                    "merged_nodes": [
                        {
                            "source_turn_ids": ["t_000001", "t_000002"],
                            "node_type": "setup_payoff",
                            "node_flags": ["backchannel_dense"],
                            "summary": "Expectation of failure with quick backchannel.",
                        }
                    ]
                },
                "nb_0002": {
                    "merged_nodes": [
                        {
                            "source_turn_ids": ["t_000003", "t_000004"],
                            "node_type": "reveal",
                            "node_flags": ["high_resonance_candidate"],
                            "summary": "The reveal lands and gets a reaction.",
                        }
                    ]
                },
            },
            phase2_boundary_responses={
                "nb_0001__nb_0002": {
                    "resolution": "keep_both",
                    "nodes": [
                        {
                            "existing_node_id": "node_t_000001__t_000002",
                            "source_turn_ids": ["t_000001", "t_000002"],
                            "node_type": "setup_payoff",
                            "node_flags": ["backchannel_dense"],
                            "summary": "Expectation of failure with quick backchannel.",
                        },
                        {
                            "existing_node_id": "node_t_000003__t_000004",
                            "source_turn_ids": ["t_000003", "t_000004"],
                            "node_type": "reveal",
                            "node_flags": ["high_resonance_candidate"],
                            "summary": "The reveal lands and gets a reaction.",
                        },
                    ],
                }
            },
            phase3_local_edge_responses=[
                {
                    "batch_id": "edge_batch_01",
                    "target_node_ids": ["node_t_000001__t_000002", "node_t_000003__t_000004"],
                    "context_node_ids": ["node_t_000001__t_000002", "node_t_000003__t_000004"],
                    "edges": [
                        {
                            "source_node_id": "node_t_000003__t_000004",
                            "target_node_id": "node_t_000001__t_000002",
                            "edge_type": "payoff_of",
                            "rationale": "The later node lands the earlier setup.",
                            "confidence": 0.87,
                        },
                        {
                            "source_node_id": "node_t_000003__t_000004",
                            "target_node_id": "node_t_000001__t_000002",
                            "edge_type": "reaction_to",
                            "rationale": "Also functions as the reaction beat.",
                            "confidence": 0.71,
                        },
                    ],
                }
            ],
            phase3_long_range_response={"edges": []},
            phase4_extra_prompt_texts=["best setup payoff moment"],
            phase4_subgraph_responses={
                "sg_0002": {
                    "subgraph_id": "sg_0002",
                    "seed_node_id": "node_t_000001__t_000002",
                    "reject_all": False,
                    "reject_reason": "",
                    "candidates": [
                        {
                            "node_ids": ["node_t_000001__t_000002", "node_t_000003__t_000004"],
                            "start_ms": 0,
                            "end_ms": 2000,
                            "score": 8.1,
                            "rationale": "Complete setup and reveal sequence.",
                        }
                    ],
                },
            },
            phase4_pool_response={
                "ranked_candidates": [
                    {
                        "candidate_temp_id": "sg_0002_cand_01",
                        "keep": True,
                        "pool_rank": 1,
                        "score": 8.6,
                        "score_breakdown": {"overall_clip_quality": 8.6},
                        "rationale": "Best standalone moment in the run.",
                    }
                ],
                "dropped_candidate_temp_ids": [],
            },
        ),
    )

    assert isinstance(summary, Phase14RunSummary)
    assert Path(summary.artifact_paths["canonical_timeline"]).exists()
    assert Path(summary.artifact_paths["semantic_graph_nodes"]).exists()
    assert Path(summary.artifact_paths["semantic_graph_edges"]).exists()
    assert Path(summary.artifact_paths["clip_candidates"]).exists()

    clip_candidates_path = Path(summary.artifact_paths["clip_candidates"])
    clip_payload = clip_candidates_path.read_text(encoding="utf-8")
    assert "sg_0002_cand_01" in clip_payload
    assert "\"pool_rank\": 1" in clip_payload
