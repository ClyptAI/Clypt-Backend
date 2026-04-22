from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.phase1_runtime.payloads import (
    DiarizationPayload,
    EmotionSegmentsPayload,
    Phase1AudioAssets,
    VisualPayload,
    YamnetPayload,
)

from .artifacts import V31RunPaths, build_run_paths, save_json
from .config import V31Config, get_v31_config
from .contracts import (
    CanonicalTimeline,
    ClipCandidate,
    ShotTrackletIndex,
    Phase14RunSummary,
    SemanticGraphEdge,
    SemanticGraphNode,
    TrackletGeometry,
)
from .render.phase6 import run_phase_6
from .timeline.audio_events import build_audio_event_timeline
from .timeline.emotion_events import build_speech_emotion_timeline
from .timeline.timeline_builder import build_canonical_timeline
from .timeline.tracklets import build_tracklet_artifacts
from .semantics.turn_neighborhoods import build_turn_neighborhoods
from .semantics.responses import (
    BoundaryReconciliationResponse,
    SemanticsMergeAndClassifyBatchResponse,
)
from .semantics.merge_and_classify import merge_and_classify_neighborhood
from .semantics.boundary_reconciliation import reconcile_boundary_nodes
from .semantics.node_embeddings import embed_semantic_nodes
from .graph.structural_edges import build_structural_edges
from .graph.local_semantic_edges import build_local_semantic_edges
from .graph.long_range_edges import build_long_range_edges, shortlist_long_range_pairs
from .graph.reconcile_edges import reconcile_semantic_edges
from .candidates.prompt_sources import build_meta_prompts
from .candidates.query_embeddings import embed_prompt_texts
from .candidates.seed_retrieval import retrieve_seed_nodes
from .candidates.build_local_subgraphs import build_local_subgraphs
from .candidates.review_subgraphs import review_local_subgraph
from .candidates.dedupe_candidates import dedupe_clip_candidates
from .candidates.review_candidate_pool import review_candidate_pool


@dataclass(slots=True)
class V31Phase14RunInputs:
    phase1_audio: Phase1AudioAssets
    diarization_payload: DiarizationPayload
    phase1_visual: VisualPayload | None = None
    emotion2vec_payload: EmotionSegmentsPayload | None = None
    yamnet_payload: YamnetPayload | None = None
    phase2_target_turn_count: int = 8
    phase2_halo_turn_count: int = 2
    phase2_merge_responses: dict[str, dict] | None = None
    phase2_boundary_responses: dict[str, dict] | None = None
    phase3_local_edge_responses: list[dict] | None = None
    phase3_long_range_top_k: int = 3
    phase3_long_range_response: dict | None = None
    phase4_extra_prompt_texts: list[str] | None = None
    phase4_subgraph_responses: dict[str, dict] | None = None
    phase4_pool_response: dict | None = None
    participation_timeline: dict[str, Any] | None = None
    camera_intent_timeline: dict[str, Any] | None = None
    source_context: dict[str, Any] | None = None


@dataclass(slots=True)
class V31Phase14Orchestrator:
    config: V31Config

    @classmethod
    def from_env(cls) -> "V31Phase14Orchestrator":
        return cls(config=get_v31_config())

    def build_run_paths(self, *, run_id: str) -> V31RunPaths:
        return build_run_paths(output_root=self.config.output_root, run_id=run_id)

    @staticmethod
    def _json_payload(payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        return payload

    def run(self, *, run_id: str, source_url: str, inputs: V31Phase14RunInputs) -> Phase14RunSummary:
        paths = self.build_run_paths(run_id=run_id)
        phase1_outputs = self.run_phase_1(run_id=run_id, source_url=source_url, paths=paths, inputs=inputs)
        phase2_outputs = self.run_phase_2(run_id=run_id, paths=paths, inputs=inputs, phase1_outputs=phase1_outputs)
        phase3_outputs = self.run_phase_3(run_id=run_id, paths=paths, inputs=inputs, phase2_outputs=phase2_outputs)
        phase4_outputs = self.run_phase_4(
            run_id=run_id,
            paths=paths,
            inputs=inputs,
            phase1_outputs=phase1_outputs,
            phase2_outputs=phase2_outputs,
            phase3_outputs=phase3_outputs,
        )
        phase6_outputs = self.run_phase_6(
            run_id=run_id,
            source_url=source_url,
            paths=paths,
            inputs=inputs,
            phase1_outputs=phase1_outputs,
            phase2_outputs=phase2_outputs,
            phase4_outputs=phase4_outputs,
        )
        artifact_paths = paths.to_dict()
        artifact_paths.update(phase6_outputs.get("artifact_paths", {}))
        return Phase14RunSummary(run_id=run_id, artifact_paths=artifact_paths)

    def run_phase_1(self, *, run_id: str, source_url: str, paths: V31RunPaths, inputs: V31Phase14RunInputs) -> dict[str, Any]:
        canonical_timeline = build_canonical_timeline(
            phase1_audio=self._json_payload(inputs.phase1_audio),
            diarization_payload=self._json_payload(inputs.diarization_payload),
        )
        speech_emotion_timeline = build_speech_emotion_timeline(
            emotion2vec_payload=(
                self._json_payload(inputs.emotion2vec_payload)
                if inputs.emotion2vec_payload is not None
                else {}
            )
        )
        audio_event_timeline = build_audio_event_timeline(
            yamnet_payload=(
                self._json_payload(inputs.yamnet_payload)
                if inputs.yamnet_payload is not None
                else {}
            )
        )
        shot_tracklet_index, tracklet_geometry = build_tracklet_artifacts(
            phase1_visual=(
                self._json_payload(inputs.phase1_visual)
                if inputs.phase1_visual is not None
                else {}
            )
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
            "shot_tracklet_index": shot_tracklet_index,
            "tracklet_geometry": tracklet_geometry,
        }

    def run_phase_2(self, *, run_id: str, paths: V31RunPaths, inputs: V31Phase14RunInputs, phase1_outputs: dict[str, Any]) -> dict[str, Any]:
        neighborhoods = build_turn_neighborhoods(
            canonical_timeline=phase1_outputs["canonical_timeline"],
            speech_emotion_timeline=phase1_outputs["speech_emotion_timeline"],
            audio_event_timeline=phase1_outputs["audio_event_timeline"],
            target_turn_count=inputs.phase2_target_turn_count,
            halo_turn_count=inputs.phase2_halo_turn_count,
        )
        save_json(paths.turn_neighborhoods_debug, neighborhoods)

        merge_responses = inputs.phase2_merge_responses or {}
        batch_nodes: list[list[SemanticGraphNode]] = []
        merge_debug: list[dict[str, Any]] = []
        for neighborhood in neighborhoods:
            batch_id = neighborhood["batch_id"]
            response_data = merge_responses.get(batch_id)
            if response_data is None:
                raise ValueError(f"missing phase 2 merge response for batch {batch_id}")
            response = SemanticsMergeAndClassifyBatchResponse.model_validate(response_data)
            nodes = merge_and_classify_neighborhood(
                neighborhood_payload=neighborhood,
                llm_response=response,
            )
            batch_nodes.append(nodes)
            merge_debug.append({"batch_id": batch_id, "response": response.model_dump(mode="json")})

        final_nodes: list[SemanticGraphNode] = []
        if batch_nodes:
            final_nodes.extend(batch_nodes[0])
            boundary_responses = inputs.phase2_boundary_responses or {}
            for idx in range(1, len(batch_nodes)):
                left_batch_id = neighborhoods[idx - 1]["batch_id"]
                right_batch_id = neighborhoods[idx]["batch_id"]
                boundary_key = f"{left_batch_id}__{right_batch_id}"
                boundary_response_data = boundary_responses.get(boundary_key)
                next_nodes = batch_nodes[idx]
                if boundary_response_data and final_nodes and next_nodes:
                    boundary_response = BoundaryReconciliationResponse.model_validate(boundary_response_data)
                    reconciled_boundary_nodes = reconcile_boundary_nodes(
                        left_batch_nodes=[final_nodes[-1]],
                        right_batch_nodes=[next_nodes[0]],
                        llm_response=boundary_response,
                    )
                    final_nodes = [*final_nodes[:-1], *reconciled_boundary_nodes, *next_nodes[1:]]
                else:
                    final_nodes.extend(next_nodes)

        embedded_nodes = embed_semantic_nodes(nodes=final_nodes)
        save_json(paths.semantic_graph_nodes, [node.model_dump(mode="json") for node in embedded_nodes])
        save_json(paths.merge_debug, merge_debug)
        save_json(paths.classification_debug, [node.model_dump(mode="json") for node in embedded_nodes])

        return {
            "nodes": embedded_nodes,
            "turn_neighborhoods": neighborhoods,
        }

    def run_phase_3(self, *, run_id: str, paths: V31RunPaths, inputs: V31Phase14RunInputs, phase2_outputs: dict[str, Any]) -> dict[str, Any]:
        nodes = phase2_outputs["nodes"]
        structural_edges = build_structural_edges(nodes=nodes)
        local_semantic_edges = build_local_semantic_edges(
            nodes=nodes,
            llm_responses=inputs.phase3_local_edge_responses or [],
        )
        long_range_pairs = shortlist_long_range_pairs(
            nodes=nodes,
            top_k=inputs.phase3_long_range_top_k,
        )
        long_range_edges = build_long_range_edges(
            candidate_pairs=long_range_pairs,
            llm_response=inputs.phase3_long_range_response or {"edges": []},
        )
        reconciled_semantic_edges = reconcile_semantic_edges(
            edges=[*local_semantic_edges, *long_range_edges]
        )
        final_edges: list[SemanticGraphEdge] = [*structural_edges, *reconciled_semantic_edges]
        save_json(paths.semantic_graph_edges, [edge.model_dump(mode="json") for edge in final_edges])
        return {
            "edges": final_edges,
            "long_range_pairs": long_range_pairs,
        }

    def run_phase_4(self, *, run_id: str, paths: V31RunPaths, inputs: V31Phase14RunInputs, phase1_outputs: dict[str, Any], phase2_outputs: dict[str, Any], phase3_outputs: dict[str, Any]) -> dict[str, Any]:
        duration_s = 0.0
        turns = phase1_outputs["canonical_timeline"].turns
        if turns:
            duration_s = turns[-1].end_ms / 1000.0
        prompt_texts = [
            *build_meta_prompts(video_duration_s=duration_s),
            *(inputs.phase4_extra_prompt_texts or []),
        ]
        embedded_prompts = embed_prompt_texts(prompts=prompt_texts)
        seeds = retrieve_seed_nodes(
            prompts=embedded_prompts,
            nodes=phase2_outputs["nodes"],
            top_k_per_prompt=self.config.phase4_subgraphs.seed_top_k_per_prompt,
        )
        subgraphs = build_local_subgraphs(
            seeds=seeds,
            nodes=phase2_outputs["nodes"],
            edges=phase3_outputs["edges"],
            config=self.config.phase4_subgraphs,
        )
        save_json(paths.retrieval_prompts_debug, embedded_prompts)
        save_json(paths.seed_nodes_debug, seeds)
        save_json(paths.local_subgraphs_debug, [subgraph.model_dump(mode="json") for subgraph in subgraphs])

        subgraph_responses = inputs.phase4_subgraph_responses or {}
        reviewed_subgraphs = []
        raw_candidates: list[ClipCandidate] = []
        for subgraph in subgraphs:
            response = subgraph_responses.get(subgraph.subgraph_id)
            if response is None:
                raise ValueError(f"missing phase 4 subgraph review response for {subgraph.subgraph_id}")
            reviewed = review_local_subgraph(subgraph=subgraph, llm_response=response)
            reviewed_subgraphs.append(reviewed)
            raw_candidates.extend(reviewed.candidates)

        deduped_candidates = dedupe_clip_candidates(candidates=raw_candidates)
        save_json(paths.candidate_dedup_debug, [candidate.model_dump(mode="json") for candidate in deduped_candidates])

        final_candidates: list[ClipCandidate] = []
        if deduped_candidates:
            pooled = review_candidate_pool(
                candidates=deduped_candidates,
                llm_response=inputs.phase4_pool_response,
            )
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

        save_json(paths.clip_candidates, [candidate.model_dump(mode="json") for candidate in final_candidates])
        save_json(
            paths.phase_4_summary,
            {
                "prompt_count": len(embedded_prompts),
                "seed_count": len(seeds),
                "subgraph_count": len(subgraphs),
                "raw_candidate_count": len(raw_candidates),
                "deduped_candidate_count": len(deduped_candidates),
                "final_candidate_count": len(final_candidates),
            },
        )
        return {"candidates": final_candidates}

    def run_phase_6(
        self,
        *,
        run_id: str,
        source_url: str,
        paths: V31RunPaths,
        inputs: V31Phase14RunInputs,
        phase1_outputs: dict[str, Any],
        phase2_outputs: dict[str, Any],
        phase4_outputs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        phase4_outputs = phase4_outputs or {}
        participation_timeline = inputs.participation_timeline
        if participation_timeline is None:
            participation_timeline = phase4_outputs.get("participation_timeline")
        camera_intent_timeline = inputs.camera_intent_timeline
        if camera_intent_timeline is None:
            camera_intent_timeline = phase4_outputs.get("camera_intent_timeline")
        canonical_timeline = phase1_outputs.get("canonical_timeline")
        shot_tracklet_index = phase1_outputs.get("shot_tracklet_index", ShotTrackletIndex(tracklets=[]))
        tracklet_geometry = phase1_outputs.get("tracklet_geometry", TrackletGeometry(tracklets=[]))
        if not isinstance(canonical_timeline, CanonicalTimeline):
            canonical_timeline = CanonicalTimeline(
                words=list(getattr(canonical_timeline, "words", [])) if canonical_timeline is not None else [],
                turns=list(getattr(canonical_timeline, "turns", [])) if canonical_timeline is not None else [],
                source_video_url=getattr(canonical_timeline, "source_video_url", source_url) if canonical_timeline is not None else source_url,
                video_gcs_uri=getattr(canonical_timeline, "video_gcs_uri", None) if canonical_timeline is not None else None,
            )
        return run_phase_6(
            paths=paths,
            canonical_timeline=canonical_timeline,
            shot_tracklet_index=shot_tracklet_index,
            tracklet_geometry=tracklet_geometry,
            candidates=phase4_outputs.get("candidates", []),
            nodes=phase2_outputs.get("nodes", []),
            source_context=inputs.source_context,
            participation_timeline=participation_timeline,
            camera_intent_timeline=camera_intent_timeline,
        )


__all__ = [
    "V31Phase14Orchestrator",
    "V31Phase14RunInputs",
]
