from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Callable

from backend.phase1_runtime.models import Phase1SidecarOutputs
from backend.pipeline.artifacts import V31RunPaths, build_run_paths, save_json
from backend.pipeline.candidates.build_local_subgraphs import build_local_subgraphs
from backend.pipeline.candidates.dedupe_candidates import dedupe_clip_candidates
from backend.pipeline.candidates.runtime import (
    embed_prompt_texts_live,
    generate_meta_prompts_live,
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
from backend.repository import (
    ClipCandidateRecord,
    Phase14Repository,
    PhaseMetricRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    TimelineTurnRecord,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass(slots=True)
class V31LivePhase14Runner:
    config: V31Config
    llm_client: Any
    embedding_client: Any
    flash_model: str = "gemini-3-flash-preview"
    storage_client: Any | None = None
    node_media_preparer: Any | None = None
    repository: Phase14Repository | None = None
    query_version: str | None = None
    debug_snapshots: bool = False
    log_event: Callable[..., None] | None = None

    @classmethod
    def from_env(
        cls,
        *,
        llm_client: Any,
        embedding_client: Any,
        flash_model: str = "gemini-3-flash-preview",
        storage_client: Any | None = None,
        repository: Phase14Repository | None = None,
        query_version: str | None = None,
        debug_snapshots: bool = False,
        log_event: Callable[..., None] | None = None,
    ) -> "V31LivePhase14Runner":
        return cls(
            config=get_v31_config(),
            llm_client=llm_client,
            embedding_client=embedding_client,
            flash_model=flash_model,
            storage_client=storage_client,
            repository=repository,
            query_version=query_version,
            debug_snapshots=debug_snapshots,
            log_event=log_event,
        )

    def build_run_paths(self, *, run_id: str) -> V31RunPaths:
        return build_run_paths(output_root=self.config.output_root, run_id=run_id)

    def run(
        self,
        *,
        run_id: str,
        source_url: str,
        phase1_outputs: Phase1SidecarOutputs,
        phase3_long_range_top_k: int = 3,
        phase4_extra_prompt_texts: list[str] | None = None,
        job_id: str | None = None,
        attempt: int = 1,
    ) -> Phase14RunSummary:
        paths = self.build_run_paths(run_id=run_id)
        phase1 = self.run_phase_1(paths=paths, phase1_outputs=phase1_outputs)

        run_started_at = datetime.now(UTC)
        run_started = time.perf_counter()
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="phase24",
            event="phase_start",
            attempt=attempt,
            status="start",
        )
        try:
            phase2 = self._execute_phase(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                phase_name="phase2",
                operation=lambda: self.run_phase_2(
                    paths=paths,
                    phase1_outputs=phase1_outputs,
                    canonical_timeline=phase1["canonical_timeline"],
                    speech_emotion_timeline=phase1["speech_emotion_timeline"],
                    audio_event_timeline=phase1["audio_event_timeline"],
                ),
                metric_metadata_builder=lambda payload: {"node_count": len(payload["nodes"])},
            )
            phase3 = self._execute_phase(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                phase_name="phase3",
                operation=lambda: self.run_phase_3(
                    paths=paths,
                    nodes=phase2["nodes"],
                    long_range_top_k=phase3_long_range_top_k,
                ),
                metric_metadata_builder=lambda payload: {"edge_count": len(payload["edges"])},
            )
            phase4 = self._execute_phase(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                phase_name="phase4",
                operation=lambda: self.run_phase_4(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    paths=paths,
                    source_url=source_url,
                    canonical_timeline=phase1["canonical_timeline"],
                    nodes=phase2["nodes"],
                    edges=phase3["edges"],
                    extra_prompt_texts=phase4_extra_prompt_texts or [],
                ),
                metric_metadata_builder=lambda payload: {
                    "seed_count": payload["seed_count"],
                    "subgraph_count": payload["subgraph_count"],
                    "candidate_count": payload["final_candidate_count"],
                },
            )
        except Exception as exc:
            ended_at = datetime.now(UTC)
            total_duration_ms = (time.perf_counter() - run_started) * 1000.0
            self._write_phase_metric(
                run_id=run_id,
                phase_name="phase24",
                status="failed",
                started_at=run_started_at,
                ended_at=ended_at,
                error_payload=self._error_payload(exc),
            )
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="phase24",
                event="run_terminal",
                attempt=attempt,
                status="terminal_failure",
                duration_ms=total_duration_ms,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                final_status="FAILED",
                total_duration_ms=total_duration_ms,
            )
            raise

        ended_at = datetime.now(UTC)
        total_duration_ms = (time.perf_counter() - run_started) * 1000.0
        self._write_phase_metric(
            run_id=run_id,
            phase_name="phase24",
            status="succeeded",
            started_at=run_started_at,
            ended_at=ended_at,
            metadata={
                "node_count": len(phase2["nodes"]),
                "edge_count": len(phase3["edges"]),
                "candidate_count": phase4["final_candidate_count"],
            },
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="phase24",
            event="run_terminal",
            attempt=attempt,
            status="success",
            duration_ms=total_duration_ms,
            final_status="PHASE24_DONE",
            total_duration_ms=total_duration_ms,
        )
        return Phase14RunSummary(
            run_id=run_id,
            artifact_paths=paths.to_dict() if self.debug_snapshots else {},
            metadata={
                "node_count": len(phase2["nodes"]),
                "edge_count": len(phase3["edges"]),
                "candidate_count": phase4["final_candidate_count"],
                "query_version": self.query_version,
            },
        )

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

        if self.repository is not None:
            self.repository.write_timeline_turns(
                run_id=paths.run_id,
                turns=[
                    TimelineTurnRecord(
                        run_id=paths.run_id,
                        turn_id=turn.turn_id,
                        speaker_id=turn.speaker_id,
                        start_ms=turn.start_ms,
                        end_ms=turn.end_ms,
                        word_ids=list(turn.word_ids),
                        transcript_text=turn.transcript_text,
                        identification_match=turn.identification_match,
                    )
                    for turn in canonical_timeline.turns
                ],
            )

        self._save_debug_json(paths.canonical_timeline, canonical_timeline.model_dump(mode="json"))
        self._save_debug_json(paths.speech_emotion_timeline, speech_emotion_timeline.model_dump(mode="json"))
        self._save_debug_json(paths.audio_event_timeline, audio_event_timeline.model_dump(mode="json"))
        self._save_debug_json(paths.shot_tracklet_index, shot_tracklet_index.model_dump(mode="json"))
        self._save_debug_json(paths.tracklet_geometry, tracklet_geometry.model_dump(mode="json"))

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
    ) -> dict[str, Any]:
        nodes, merge_debug, boundary_debug = run_merge_classify_and_reconcile(
            canonical_timeline=canonical_timeline,
            speech_emotion_timeline=speech_emotion_timeline,
            audio_event_timeline=audio_event_timeline,
            llm_client=self.llm_client,
            target_batch_count=self.config.phase2_target_batch_count,
            max_turns_per_batch=self.config.phase2_max_turns_per_batch,
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
        )
        if self.node_media_preparer is not None:
            multimodal_media = self.node_media_preparer(nodes=nodes, paths=paths, phase1_outputs=phase1_outputs)
        else:
            local_video_path = (phase1_outputs.phase1_audio or {}).get("local_video_path")
            if not local_video_path:
                raise ValueError(
                    "phase1_outputs.phase1_audio.local_video_path is required for live multimodal node embeddings."
                )
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
        if self.repository is not None:
            self.repository.write_nodes(
                run_id=paths.run_id,
                nodes=[
                    SemanticNodeRecord(
                        run_id=paths.run_id,
                        node_id=node.node_id,
                        node_type=node.node_type,
                        start_ms=node.start_ms,
                        end_ms=node.end_ms,
                        source_turn_ids=list(node.source_turn_ids),
                        word_ids=list(node.word_ids),
                        transcript_text=node.transcript_text,
                        node_flags=list(node.node_flags),
                        summary=node.summary,
                        evidence=node.evidence.model_dump(mode="json"),
                        semantic_embedding=node.semantic_embedding,
                        multimodal_embedding=node.multimodal_embedding,
                    )
                    for node in embedded_nodes
                ],
            )
        self._save_debug_json(paths.semantic_graph_nodes, [node.model_dump(mode="json") for node in embedded_nodes])
        self._save_debug_json(paths.merge_debug, merge_debug)
        self._save_debug_json(paths.classification_debug, [node.model_dump(mode="json") for node in embedded_nodes])
        self._save_debug_json(paths.semantics_dir / "node_media_debug.json", multimodal_media)
        self._save_debug_json(paths.semantics_dir / "boundary_reconciliation_debug.json", boundary_debug)
        return {"nodes": embedded_nodes}

    def run_phase_3(
        self,
        *,
        paths: V31RunPaths,
        nodes: list[SemanticGraphNode],
        long_range_top_k: int,
    ) -> dict[str, Any]:
        structural_edges = build_structural_edges(nodes=nodes)
        local_edges, local_debug = run_local_semantic_edge_batches(
            nodes=nodes,
            llm_client=self.llm_client,
            target_batch_count=self.config.phase3_target_batch_count,
            max_nodes_per_batch=self.config.phase3_max_nodes_per_batch,
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
        )
        long_range_edges, long_range_debug = run_long_range_edge_adjudication(
            nodes=nodes,
            llm_client=self.llm_client,
            top_k=long_range_top_k,
            model=self.flash_model,
        )
        reconciled_semantic_edges = reconcile_semantic_edges(edges=[*local_edges, *long_range_edges])
        final_edges: list[SemanticGraphEdge] = [*structural_edges, *reconciled_semantic_edges]
        if self.repository is not None:
            self.repository.write_edges(
                run_id=paths.run_id,
                edges=[
                    SemanticEdgeRecord(
                        run_id=paths.run_id,
                        source_node_id=edge.source_node_id,
                        target_node_id=edge.target_node_id,
                        edge_type=edge.edge_type,
                        rationale=edge.rationale,
                        confidence=edge.confidence,
                        support_count=edge.support_count,
                        batch_ids=list(edge.batch_ids),
                    )
                    for edge in final_edges
                ],
            )
        self._save_debug_json(paths.semantic_graph_edges, [edge.model_dump(mode="json") for edge in final_edges])
        self._save_debug_json(paths.graph_dir / "local_semantic_edges_debug.json", local_debug)
        self._save_debug_json(paths.graph_dir / "long_range_edges_debug.json", long_range_debug)
        return {"edges": final_edges}

    def run_phase_4(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        paths: V31RunPaths,
        source_url: str,
        canonical_timeline,
        nodes: list[SemanticGraphNode],
        edges: list[SemanticGraphEdge],
        extra_prompt_texts: list[str],
    ) -> dict[str, Any]:
        duration_s = 0.0
        if canonical_timeline.turns:
            duration_s = canonical_timeline.turns[-1].end_ms / 1000.0
        dynamic_prompts = generate_meta_prompts_live(
            nodes=nodes,
            llm_client=self.llm_client,
            model=self.flash_model,
            duration_s=duration_s,
        )
        prompt_texts = [*dynamic_prompts, *extra_prompt_texts]
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
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
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

        if self.repository is not None:
            self.repository.write_candidates(
                run_id=paths.run_id,
                candidates=[
                    ClipCandidateRecord(
                        run_id=paths.run_id,
                        clip_id=candidate.clip_id,
                        node_ids=list(candidate.node_ids),
                        start_ms=candidate.start_ms,
                        end_ms=candidate.end_ms,
                        score=candidate.score,
                        rationale=candidate.rationale,
                        source_prompt_ids=list(candidate.source_prompt_ids),
                        seed_node_id=candidate.seed_node_id,
                        subgraph_id=candidate.subgraph_id,
                        query_aligned=candidate.query_aligned,
                        pool_rank=candidate.pool_rank,
                        score_breakdown=candidate.score_breakdown,
                    )
                    for candidate in final_candidates
                ],
            )

        self._save_debug_json(
            paths.candidates_dir / "meta_prompts_debug.json",
            {"dynamic_prompts": dynamic_prompts, "extra_prompt_texts": extra_prompt_texts},
        )
        self._save_debug_json(paths.retrieval_prompts_debug, embedded_prompts)
        self._save_debug_json(paths.seed_nodes_debug, seeds)
        self._save_debug_json(paths.local_subgraphs_debug, [subgraph.model_dump(mode="json") for subgraph in subgraphs])
        self._save_debug_json(paths.candidate_dedup_debug, [candidate.model_dump(mode="json") for candidate in deduped_candidates])
        self._save_debug_json(paths.clip_candidates, [candidate.model_dump(mode="json") for candidate in final_candidates])
        self._save_debug_json(paths.candidates_dir / "subgraph_review_debug.json", subgraph_debug)
        self._save_debug_json(
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
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="phase4",
            event="candidate_summary",
            attempt=attempt,
            status="success",
            seed_count=len(seeds),
            subgraph_count=len(subgraphs),
            candidate_count=len(final_candidates),
        )
        return {
            "candidates": final_candidates,
            "seed_count": len(seeds),
            "subgraph_count": len(subgraphs),
            "raw_candidate_count": len(raw_candidates),
            "deduped_candidate_count": len(deduped_candidates),
            "final_candidate_count": len(final_candidates),
        }

    def _execute_phase(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        phase_name: str,
        operation: Callable[[], dict[str, Any]],
        metric_metadata_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase=phase_name,
            event="phase_start",
            attempt=attempt,
            status="start",
        )
        try:
            result = operation()
        except Exception as exc:
            ended_at = datetime.now(UTC)
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._write_phase_metric(
                run_id=run_id,
                phase_name=phase_name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                error_payload=self._error_payload(exc),
            )
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase=phase_name,
                event="phase_error",
                attempt=attempt,
                status="error",
                duration_ms=duration_ms,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        ended_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._write_phase_metric(
            run_id=run_id,
            phase_name=phase_name,
            status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
            metadata=metric_metadata_builder(result) if metric_metadata_builder is not None else None,
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase=phase_name,
            event="phase_success",
            attempt=attempt,
            status="success",
            duration_ms=duration_ms,
        )
        return result

    def _write_phase_metric(
        self,
        *,
        run_id: str,
        phase_name: str,
        status: str,
        started_at: datetime,
        ended_at: datetime,
        error_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.repository is None:
            return
        self.repository.write_phase_metric(
            PhaseMetricRecord(
                run_id=run_id,
                phase_name=phase_name,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=max(0.0, (ended_at - started_at).total_seconds() * 1000.0),
                error_payload=error_payload,
                query_version=self.query_version,
                metadata=metadata or {},
            )
        )

    def _emit_log(
        self,
        *,
        run_id: str,
        job_id: str | None,
        phase: str,
        event: str,
        attempt: int,
        status: str,
        duration_ms: float | None = None,
        **extra: Any,
    ) -> None:
        if self.log_event is None:
            return
        self.log_event(
            run_id=run_id,
            job_id=job_id or run_id,
            phase=phase,
            event=event,
            attempt=attempt,
            query_version=self.query_version,
            duration_ms=duration_ms,
            status=status,
            **extra,
        )

    def _save_debug_json(self, path: Path, payload: Any) -> None:
        if self.debug_snapshots:
            save_json(path, payload)

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, str]:
        return {
            "code": exc.__class__.__name__,
            "message": str(exc)[:2048],
        }


__all__ = ["V31LivePhase14Runner"]
