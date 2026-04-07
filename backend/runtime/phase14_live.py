from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.phase1_runtime.models import Phase1SidecarOutputs
from backend.pipeline.artifacts import V31RunPaths, build_run_paths, save_json
from backend.pipeline.candidates.build_local_subgraphs import build_local_subgraphs
from backend.pipeline.candidates.dedupe_candidates import dedupe_clip_candidates
from backend.pipeline.candidates.prompt_sources import build_meta_prompts
from backend.pipeline.candidates.runtime import (
    embed_prompt_texts_live,
    run_candidate_pool_review,
    run_subgraph_reviews,
)
from backend.pipeline.candidates.seed_retrieval import retrieve_seed_nodes
from backend.pipeline.config import V31Config, get_v31_config
from backend.pipeline.contracts import ClipCandidate, Phase14RunSummary, SemanticGraphEdge, SemanticGraphNode
from backend.pipeline.graph.reconcile_edges import reconcile_semantic_edges
from backend.pipeline.graph.runtime import (
    run_local_semantic_edge_batches,
    run_long_range_edge_adjudication,
)
from backend.pipeline.graph.structural_edges import build_structural_edges
from backend.pipeline.semantics.runtime import (
    embed_semantic_nodes_live,
    prepare_node_media_embeddings,
    run_merge_classify_and_reconcile,
)
from backend.pipeline.timeline.audio_events import build_audio_event_timeline
from backend.pipeline.timeline.emotion_events import build_speech_emotion_timeline
from backend.pipeline.timeline.timeline_builder import build_canonical_timeline
from backend.pipeline.timeline.tracklets import build_tracklet_artifacts


@dataclass(slots=True)
class V31LivePhase14Runner:
    config: V31Config
    llm_client: Any
    embedding_client: Any
    storage_client: Any | None = None
    node_media_preparer: Any | None = None

    @classmethod
    def from_env(
        cls,
        *,
        llm_client: Any,
        embedding_client: Any,
        storage_client: Any | None = None,
    ) -> "V31LivePhase14Runner":
        return cls(
            config=get_v31_config(),
            llm_client=llm_client,
            embedding_client=embedding_client,
            storage_client=storage_client,
        )

    def build_run_paths(self, *, run_id: str) -> V31RunPaths:
        return build_run_paths(output_root=self.config.output_root, run_id=run_id)

    def run(
        self,
        *,
        run_id: str,
        source_url: str,
        phase1_outputs: Phase1SidecarOutputs,
        phase2_target_turn_count: int = 8,
        phase2_halo_turn_count: int = 2,
        phase3_local_target_node_count: int = 8,
        phase3_local_halo_node_count: int = 2,
        phase3_long_range_top_k: int = 3,
        phase4_extra_prompt_texts: list[str] | None = None,
    ) -> Phase14RunSummary:
        paths = self.build_run_paths(run_id=run_id)
        phase1 = self.run_phase_1(paths=paths, phase1_outputs=phase1_outputs)
        phase2 = self.run_phase_2(
            paths=paths,
            phase1_outputs=phase1_outputs,
            canonical_timeline=phase1["canonical_timeline"],
            speech_emotion_timeline=phase1["speech_emotion_timeline"],
            audio_event_timeline=phase1["audio_event_timeline"],
            target_turn_count=phase2_target_turn_count,
            halo_turn_count=phase2_halo_turn_count,
        )
        phase3 = self.run_phase_3(
            paths=paths,
            nodes=phase2["nodes"],
            target_node_count=phase3_local_target_node_count,
            halo_node_count=phase3_local_halo_node_count,
            long_range_top_k=phase3_long_range_top_k,
        )
        self.run_phase_4(
            paths=paths,
            source_url=source_url,
            canonical_timeline=phase1["canonical_timeline"],
            nodes=phase2["nodes"],
            edges=phase3["edges"],
            extra_prompt_texts=phase4_extra_prompt_texts or [],
        )
        return Phase14RunSummary(run_id=run_id, artifact_paths=paths.to_dict())

    def run_phase_1(self, *, paths: V31RunPaths, phase1_outputs: Phase1SidecarOutputs) -> dict[str, Any]:
        canonical_timeline = build_canonical_timeline(
            phase1_audio=phase1_outputs.phase1_audio,
            diarization_payload=phase1_outputs.diarization_payload,
        )
        speech_emotion_timeline = build_speech_emotion_timeline(
            emotion2vec_payload=phase1_outputs.emotion2vec_payload,
        )
        audio_event_timeline = build_audio_event_timeline(
            yamnet_payload=phase1_outputs.yamnet_payload,
        )
        shot_tracklet_index, tracklet_geometry = build_tracklet_artifacts(
            phase1_visual=phase1_outputs.phase1_visual,
        )

        save_json(paths.canonical_timeline, canonical_timeline.model_dump(mode="json"))
        save_json(paths.speech_emotion_timeline, speech_emotion_timeline.model_dump(mode="json"))
        save_json(paths.audio_event_timeline, audio_event_timeline.model_dump(mode="json"))
        save_json(paths.shot_tracklet_index, shot_tracklet_index.model_dump(mode="json"))
        save_json(paths.tracklet_geometry, tracklet_geometry.model_dump(mode="json"))

        return {
            "canonical_timeline": canonical_timeline,
            "speech_emotion_timeline": speech_emotion_timeline,
            "audio_event_timeline": audio_event_timeline,
        }

    def run_phase_2(
        self,
        *,
        paths: V31RunPaths,
        phase1_outputs: Phase1SidecarOutputs,
        canonical_timeline,
        speech_emotion_timeline,
        audio_event_timeline,
        target_turn_count: int,
        halo_turn_count: int,
    ) -> dict[str, Any]:
        # Phase 2A (merge/classify) + Phase 2B (boundary reconciliation) in one pass
        nodes, merge_debug, boundary_debug = run_merge_classify_and_reconcile(
            canonical_timeline=canonical_timeline,
            speech_emotion_timeline=speech_emotion_timeline,
            audio_event_timeline=audio_event_timeline,
            llm_client=self.llm_client,
            target_turn_count=target_turn_count,
            halo_turn_count=halo_turn_count,
        )
        if self.node_media_preparer is not None:
            multimodal_media = self.node_media_preparer(nodes=nodes, paths=paths, phase1_outputs=phase1_outputs)
        else:
            local_video_path = (phase1_outputs.phase1_audio or {}).get("local_video_path")
            if not local_video_path:
                raise ValueError("phase1_outputs.phase1_audio.local_video_path is required for live multimodal node embeddings.")
            if self.storage_client is None:
                raise ValueError("storage_client is required for live multimodal node embeddings.")
            multimodal_media = prepare_node_media_embeddings(
                nodes=nodes,
                source_video_path=Path(local_video_path),
                clips_dir=paths.semantics_dir / "node_media_clips",
                storage_client=self.storage_client,
                object_prefix=f"phase14/{paths.run_id}/node_media",
            )
        embedded_nodes = embed_semantic_nodes_live(
            nodes=nodes,
            embedding_client=self.embedding_client,
            multimodal_media=multimodal_media,
        )
        save_json(paths.semantic_graph_nodes, [node.model_dump(mode="json") for node in embedded_nodes])
        save_json(paths.merge_debug, merge_debug)
        save_json(paths.classification_debug, [node.model_dump(mode="json") for node in embedded_nodes])
        save_json(paths.semantics_dir / "node_media_debug.json", multimodal_media)
        save_json(paths.semantics_dir / "boundary_reconciliation_debug.json", boundary_debug)
        return {"nodes": embedded_nodes}

    def run_phase_3(
        self,
        *,
        paths: V31RunPaths,
        nodes: list[SemanticGraphNode],
        target_node_count: int,
        halo_node_count: int,
        long_range_top_k: int,
    ) -> dict[str, Any]:
        structural_edges = build_structural_edges(nodes=nodes)
        local_edges, local_debug = run_local_semantic_edge_batches(
            nodes=nodes,
            llm_client=self.llm_client,
            target_node_count=target_node_count,
            halo_node_count=halo_node_count,
        )
        long_range_edges, long_range_debug = run_long_range_edge_adjudication(
            nodes=nodes,
            llm_client=self.llm_client,
            top_k=long_range_top_k,
        )
        reconciled_semantic_edges = reconcile_semantic_edges(edges=[*local_edges, *long_range_edges])
        final_edges: list[SemanticGraphEdge] = [*structural_edges, *reconciled_semantic_edges]
        save_json(paths.semantic_graph_edges, [edge.model_dump(mode="json") for edge in final_edges])
        save_json(paths.graph_dir / "local_semantic_edges_debug.json", local_debug)
        save_json(paths.graph_dir / "long_range_edges_debug.json", long_range_debug)
        return {"edges": final_edges}

    def run_phase_4(
        self,
        *,
        paths: V31RunPaths,
        source_url: str,
        canonical_timeline,
        nodes: list[SemanticGraphNode],
        edges: list[SemanticGraphEdge],
        extra_prompt_texts: list[str],
    ) -> None:
        duration_s = 0.0
        if canonical_timeline.turns:
            duration_s = canonical_timeline.turns[-1].end_ms / 1000.0
        prompt_texts = [*build_meta_prompts(video_duration_s=duration_s), *extra_prompt_texts]
        embedded_prompts = embed_prompt_texts_live(
            prompts=prompt_texts,
            embedding_client=self.embedding_client,
        )
        seeds = retrieve_seed_nodes(
            prompts=embedded_prompts,
            nodes=nodes,
            top_k_per_prompt=self.config.phase4_subgraphs.seed_top_k_per_prompt,
        )
        subgraphs = build_local_subgraphs(
            seeds=seeds,
            nodes=nodes,
            edges=edges,
            config=self.config.phase4_subgraphs,
        )
        reviews, subgraph_debug = run_subgraph_reviews(
            subgraphs=subgraphs,
            llm_client=self.llm_client,
        )
        raw_candidates: list[ClipCandidate] = []
        for review in reviews:
            raw_candidates.extend(review.candidates)
        deduped_candidates = dedupe_clip_candidates(candidates=raw_candidates)
        pooled = run_candidate_pool_review(
            candidates=deduped_candidates,
            llm_client=self.llm_client,
        ) if deduped_candidates else None

        final_candidates: list[ClipCandidate] = []
        if pooled:
            candidate_by_id = {
                (candidate.clip_id or f"cand_tmp_{idx:03d}"): candidate
                for idx, candidate in enumerate(deduped_candidates, start=1)
            }
            for decision in pooled.ranked_candidates:
                candidate = candidate_by_id[decision.candidate_temp_id]
                final_candidates.append(
                    candidate.model_copy(
                        update={
                            "pool_rank": decision.pool_rank,
                            "score": decision.score,
                            "score_breakdown": decision.score_breakdown,
                            "rationale": decision.rationale,
                        }
                    )
                )

        save_json(paths.retrieval_prompts_debug, embedded_prompts)
        save_json(paths.seed_nodes_debug, seeds)
        save_json(paths.local_subgraphs_debug, [subgraph.model_dump(mode="json") for subgraph in subgraphs])
        save_json(paths.candidate_dedup_debug, [candidate.model_dump(mode="json") for candidate in deduped_candidates])
        save_json(paths.clip_candidates, [candidate.model_dump(mode="json") for candidate in final_candidates])
        save_json(paths.candidates_dir / "subgraph_review_debug.json", subgraph_debug)
        save_json(
            paths.phase_4_summary,
            {
                "source_url": source_url,
                "prompt_count": len(embedded_prompts),
                "seed_count": len(seeds),
                "subgraph_count": len(subgraphs),
                "raw_candidate_count": len(raw_candidates),
                "deduped_candidate_count": len(deduped_candidates),
                "final_candidate_count": len(final_candidates),
            },
        )


__all__ = ["V31LivePhase14Runner"]
