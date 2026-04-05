from __future__ import annotations

from pathlib import Path


def test_live_phase14_runner_executes_provider_backed_phases_2_to_4(tmp_path: Path):
    from backend.phase1_runtime.models import Phase1SidecarOutputs
    from backend.pipeline.config import V31Config
    from backend.runtime.phase14_live import V31LivePhase14Runner

    class _FakeEmbeddingClient:
        def embed_texts(self, texts, *, task_type=None, model=None):
            return [[float(idx), 0.5, 0.25] for idx, _ in enumerate(texts, start=1)]

        def embed_media_uris(self, media_items, *, model=None):
            return [[float(idx), 0.25, 0.5] for idx, _ in enumerate(media_items, start=1)]

    class _FakeLLMClient:
        def __init__(self):
            self.calls = []

        def generate_json(self, *, prompt, model=None, temperature=0.0):
            self.calls.append(prompt)
            if "Merge contiguous target turns" in prompt:
                return {
                    "merged_nodes": [
                        {
                            "source_turn_ids": ["t_000001", "t_000002"],
                            "node_type": "claim",
                            "node_flags": ["high_resonance_candidate"],
                            "summary": "One merged unit.",
                        }
                    ]
                }
            if "Draw only local semantic graph edges" in prompt:
                return {"edges": []}
            if "Adjudicate only callback_to and topic_recurrence" in prompt:
                return {"edges": []}
            if "Review this local semantic subgraph" in prompt:
                return {
                    "subgraph_id": "sg_0001",
                    "seed_node_id": "node_t_000001__t_000002",
                    "reject_all": False,
                    "reject_reason": "",
                        "candidates": [
                            {
                                "node_ids": ["node_t_000001__t_000002"],
                                "start_ms": 0,
                                "end_ms": 1000,
                                "score": 8.2,
                                "rationale": "Strong standalone thought.",
                            }
                        ],
                }
            if "Review this candidate pool" in prompt:
                return {
                    "ranked_candidates": [
                        {
                            "candidate_temp_id": "sg_0001_cand_01",
                            "keep": True,
                            "pool_rank": 1,
                            "score": 8.9,
                            "score_breakdown": {"overall_clip_quality": 8.9},
                            "rationale": "Best clip.",
                        }
                    ],
                    "dropped_candidate_temp_ids": [],
                }
            raise AssertionError(f"unexpected prompt: {prompt[:120]}")

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=_FakeLLMClient(),
        embedding_client=_FakeEmbeddingClient(),
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
            pyannote_payload={
                "wordLevelTranscription": [
                    {"word": "hello", "start": 0.0, "end": 0.4, "speaker": "S1"},
                    {"word": "world", "start": 0.6, "end": 1.0, "speaker": "S1"},
                ],
                "diarization": [
                    {"speaker": "S1", "start": 0.0, "end": 0.4},
                    {"speaker": "S1", "start": 0.6, "end": 1.0},
                ],
            },
            identify_payload=None,
            phase1_visual={
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1200}],
                "tracks": [],
            },
            emotion2vec_payload={
                "segments": [
                    {
                        "turn_id": "t_000001",
                        "labels": ["neutral"],
                        "scores": [0.7],
                        "per_class_scores": {"neutral": 0.7},
                    }
                ]
            },
            yamnet_payload={"events": []},
        ),
        phase2_target_turn_count=2,
        phase2_halo_turn_count=0,
        phase3_local_target_node_count=4,
        phase3_local_halo_node_count=0,
    )

    assert Path(summary.artifact_paths["clip_candidates"]).exists()
    payload = Path(summary.artifact_paths["clip_candidates"]).read_text(encoding="utf-8")
    assert "Best clip." in payload
