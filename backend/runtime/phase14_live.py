from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
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
from backend.pipeline.signals.contracts import SignalPipelineOutput, SignalPromptSpec
from backend.pipeline.signals.comments_client import resolve_youtube_video_id
from backend.pipeline.signals.linking import build_node_signal_links
from backend.pipeline.signals.llm_runtime import explain_candidate_attribution_with_llm
from backend.pipeline.signals.runtime import merge_signal_outputs, start_comments_future, start_trends_future
from backend.pipeline.signals.scoring import apply_signal_scoring
from backend.pipeline.timeline.audio_events import build_audio_event_timeline
from backend.pipeline.timeline.emotion_events import build_speech_emotion_timeline
from backend.pipeline.timeline.timeline_builder import build_canonical_timeline
from backend.pipeline.timeline.tracklets import build_tracklet_artifacts
from backend.repository import (
    CandidateSignalLinkRecord,
    ClipCandidateRecord,
    ExternalSignalClusterRecord,
    ExternalSignalRecord,
    Phase14Repository,
    PhaseMetricRecord,
    NodeSignalLinkRecord,
    PromptSourceLinkRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    SubgraphProvenanceRecord,
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

        resume_phase = self._get_resume_phase(run_id=run_id)
        if resume_phase == "phase4":
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
            return self._finish_phase24_success(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                paths=paths,
                started_at=run_started_at,
                started=run_started,
                phase2_nodes=self._load_resume_nodes(run_id=run_id),
                phase3_edges=self._load_resume_edges(run_id=run_id),
                phase4_candidates=self._load_resume_candidates(run_id=run_id),
                resumed_phases=("phase2", "phase3", "phase4"),
            )

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
        signal_executor = ThreadPoolExecutor(max_workers=2)
        comments_future: Any | None = None
        trends_future: Any | None = None
        comments_future_started_at: float | None = None
        trends_future_started_at: float | None = None
        try:
            if self._comments_enabled_effective() and not resolve_youtube_video_id(source_url):
                raise ValueError(
                    "comments signals are enabled but source URL does not resolve a youtube_video_id"
                )

            comments_future, comments_future_started_at = self._start_signal_future(
                executor=signal_executor,
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                signal_name="comments",
                starter=lambda: start_comments_future(
                    executor=signal_executor,
                    cfg=self.config.signals,
                    llm_client=self.llm_client,
                    embedding_client=self.embedding_client,
                    source_url=source_url,
                ),
            )

            phase1 = self.run_phase_1(paths=paths, phase1_outputs=phase1_outputs)

            phase2_nodes: list[SemanticGraphNode]
            phase3_edges: list[SemanticGraphEdge]
            if resume_phase in {"phase2", "phase3"}:
                phase2_nodes = self._load_resume_nodes(run_id=run_id)
                self._emit_phase_skipped_resume(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    phase_name="phase2",
                    reason="phase2_already_succeeded",
                )
            else:
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
                phase2_nodes = phase2["nodes"]

            if comments_future is not None and hasattr(comments_future, "done") and comments_future.done():
                comments_future.result()

            if self._trends_enabled_effective():
                self._emit_log(
                    run_id=run_id,
                    job_id=job_id,
                    phase="signals",
                    event="trend_fetch_start_after_phase2",
                    attempt=attempt,
                    status="start",
                )
            trends_future, trends_future_started_at = self._start_signal_future(
                executor=signal_executor,
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                signal_name="trends",
                starter=lambda: start_trends_future(
                    executor=signal_executor,
                    cfg=self.config.signals,
                    llm_client=self.llm_client,
                    embedding_client=self.embedding_client,
                    nodes=phase2_nodes,
                    source_url=source_url,
                ),
            )

            if resume_phase == "phase3":
                phase3_edges = self._load_resume_edges(run_id=run_id)
                self._emit_phase_skipped_resume(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    phase_name="phase3",
                    reason="phase3_already_succeeded",
                )
            else:
                phase3 = self._execute_phase(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    phase_name="phase3",
                    operation=lambda: self.run_phase_3(
                        paths=paths,
                        nodes=phase2_nodes,
                        long_range_top_k=phase3_long_range_top_k,
                    ),
                    metric_metadata_builder=lambda payload: {"edge_count": len(payload["edges"])},
                )
                phase3_edges = phase3["edges"]

            comments_output, trends_output = self._join_signal_outputs(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                comments_future=comments_future,
                comments_future_started_at=comments_future_started_at,
                trends_future=trends_future,
                trends_future_started_at=trends_future_started_at,
            )
            signal_output = self._merge_signal_outputs(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                comments_output=comments_output,
                trends_output=trends_output,
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
                    nodes=phase2_nodes,
                    edges=phase3_edges,
                    extra_prompt_texts=phase4_extra_prompt_texts or [],
                    signal_output=signal_output,
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
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="signals_failure",
                attempt=attempt,
                status="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                failed_callpoint_id=self._extract_failed_callpoint_id(exc),
            )
            raise
        finally:
            signal_executor.shutdown(wait=False, cancel_futures=True)

        return self._finish_phase24_success(
            run_id=run_id,
            job_id=job_id,
            attempt=attempt,
            paths=paths,
            started_at=run_started_at,
            started=run_started,
            phase2_nodes=phase2_nodes,
            phase3_edges=phase3_edges,
            phase4_result=phase4,
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
        merge_boundary_started = time.perf_counter()
        nodes, merge_debug, boundary_debug = run_merge_classify_and_reconcile(
            canonical_timeline=canonical_timeline,
            speech_emotion_timeline=speech_emotion_timeline,
            audio_event_timeline=audio_event_timeline,
            llm_client=self.llm_client,
            target_batch_count=self.config.phase2_target_batch_count,
            max_turns_per_batch=self.config.phase2_max_turns_per_batch,
            merge_max_output_tokens=self.config.phase2_merge_max_output_tokens,
            boundary_max_output_tokens=self.config.phase2_boundary_max_output_tokens,
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
            merge_thinking_level=self.config.phase2_merge_thinking_level,
            boundary_thinking_level=self.config.phase2_boundary_thinking_level,
        )
        logger.info("[phase2] merge+boundary total done in %.1f s", time.perf_counter() - merge_boundary_started)
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
            media_started = time.perf_counter()
            multimodal_media = prepare_node_media_embeddings(
                nodes=nodes,
                source_video_path=Path(local_video_path),
                clips_dir=paths.semantics_dir / "node_media_clips",
                storage_client=self.storage_client,
                object_prefix=f"phase14/{paths.run_id}/node_media",
            )
            logger.info(
                "[phase2] node clip extraction+upload done in %.1f s (nodes=%d)",
                time.perf_counter() - media_started,
                len(nodes),
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
        local_started = time.perf_counter()
        local_edges, local_debug = run_local_semantic_edge_batches(
            nodes=nodes,
            llm_client=self.llm_client,
            target_batch_count=self.config.phase3_target_batch_count,
            max_nodes_per_batch=self.config.phase3_max_nodes_per_batch,
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
            thinking_level=self.config.phase3_local_thinking_level,
        )
        logger.info("[phase3] local semantic edges done in %.1f s", time.perf_counter() - local_started)
        long_range_started = time.perf_counter()
        long_range_edges, long_range_debug = run_long_range_edge_adjudication(
            nodes=nodes,
            llm_client=self.llm_client,
            top_k=long_range_top_k,
            model=self.flash_model,
            thinking_level=self.config.phase3_long_range_thinking_level,
        )
        logger.info("[phase3] long-range adjudication done in %.1f s", time.perf_counter() - long_range_started)
        reconciled_semantic_edges = reconcile_semantic_edges(edges=[*local_edges, *long_range_edges])
        final_edges: list[SemanticGraphEdge] = [*structural_edges, *reconciled_semantic_edges]
        if self.repository is not None:
            persist_started = time.perf_counter()
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
            logger.info("[phase3] edge persistence done in %.1f s", time.perf_counter() - persist_started)
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
        signal_output: SignalPipelineOutput | None = None,
    ) -> dict[str, Any]:
        duration_s = 0.0
        if canonical_timeline.turns:
            duration_s = canonical_timeline.turns[-1].end_ms / 1000.0
        meta_started = time.perf_counter()
        dynamic_prompts = generate_meta_prompts_live(
            nodes=nodes,
            llm_client=self.llm_client,
            model=self.flash_model,
            duration_s=duration_s,
            thinking_level=self.config.phase4_meta_thinking_level,
        )
        logger.info("[phase4] meta prompt generation done in %.1f s", time.perf_counter() - meta_started)
        signal_output = signal_output or SignalPipelineOutput()
        general_prompt_specs = [
            SignalPromptSpec(
                prompt_id=f"general_prompt_{idx:03d}",
                text=text,
                prompt_source_type="general",
            )
            for idx, text in enumerate([*dynamic_prompts, *extra_prompt_texts], start=1)
        ]
        augmentation_prompt_specs = list(signal_output.prompt_specs)
        all_prompt_specs = [*general_prompt_specs, *augmentation_prompt_specs]

        prompt_source_links = self._build_prompt_source_links(run_id=paths.run_id, prompt_specs=all_prompt_specs)

        embedded_prompts = embed_prompt_texts_live(
            prompts=all_prompt_specs,
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
        subgraph_provenance = self._build_subgraph_provenance(
            run_id=paths.run_id,
            subgraphs=subgraphs,
            prompt_specs=all_prompt_specs,
        )
        subgraph_provenance_by_id = {
            record.subgraph_id: record.model_dump(mode="json")
            for record in subgraph_provenance
        }
        if self.repository is not None and subgraph_provenance:
            self.repository.write_subgraph_provenance(run_id=paths.run_id, provenance=subgraph_provenance)

        subgraph_review_started = time.perf_counter()
        reviews, subgraph_debug = run_subgraph_reviews(
            subgraphs=subgraphs,
            llm_client=self.llm_client,
            model=self.flash_model,
            max_concurrent=self.config.gemini_max_concurrent,
            thinking_level=self.config.phase4_subgraph_thinking_level,
            subgraph_provenance_by_id=subgraph_provenance_by_id,
        )
        logger.info("[phase4] subgraph reviews done in %.1f s", time.perf_counter() - subgraph_review_started)
        raw_candidates: list[ClipCandidate] = []
        for review in reviews:
            raw_candidates.extend(review.candidates)
        deduped_candidates = dedupe_clip_candidates(candidates=raw_candidates)
        pool_review_started = time.perf_counter()
        pooled = run_candidate_pool_review(
            candidates=deduped_candidates,
            llm_client=self.llm_client,
            model=self.flash_model,
            thinking_level=self.config.phase4_pool_thinking_level,
        ) if deduped_candidates else None
        if deduped_candidates:
            logger.info("[phase4] pooled candidate review done in %.1f s", time.perf_counter() - pool_review_started)

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

        prompt_embeddings = {item["prompt_id"]: item["embedding"] for item in embedded_prompts}
        node_signal_links = build_node_signal_links(
            clusters=list(signal_output.clusters),
            prompt_specs=all_prompt_specs,
            prompt_embeddings=prompt_embeddings,
            nodes=nodes,
            edges=edges,
            llm_client=self.llm_client,
            model=self.config.signals.llm.model_5,
            thinking_level=self.config.signals.llm.thinking_5,
            max_hops=self.config.signals.max_hops,
            time_window_ms=self.config.signals.time_window_ms,
            fail_fast=self.config.signals.llm_fail_fast,
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_node_linking_done",
            attempt=attempt,
            status="success",
            link_count=len(node_signal_links),
        )
        signal_scoring = apply_signal_scoring(
            candidates=final_candidates,
            nodes=nodes,
            signals=list(signal_output.external_signals),
            clusters=list(signal_output.clusters),
            node_links=node_signal_links,
            prompt_specs=all_prompt_specs,
            cfg=self.config.signals,
        )
        final_candidates = signal_scoring.candidates
        candidate_signal_links = signal_scoring.candidate_signal_links
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_candidate_attribution_done",
            attempt=attempt,
            status="success",
            candidate_link_count=len(candidate_signal_links),
        )

        for idx, candidate in enumerate(final_candidates):
            if not candidate.external_attribution_json:
                continue
            started = time.perf_counter()
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="signals_llm_call_start",
                attempt=attempt,
                status="start",
                callpoint_id="11",
                model=self.config.signals.llm.model_11,
                thinking_level=self.config.signals.llm.thinking_11,
            )
            try:
                explanation = explain_candidate_attribution_with_llm(
                    llm_client=self.llm_client,
                    model=self.config.signals.llm.model_11,
                    thinking_level=self.config.signals.llm.thinking_11,
                    evidence_payload=candidate.external_attribution_json,
                    fail_fast=self.config.signals.llm_fail_fast,
                )
            except Exception as exc:
                self._emit_log(
                    run_id=run_id,
                    job_id=job_id,
                    phase="signals",
                    event="signals_failure",
                    attempt=attempt,
                    status="error",
                    error_code=exc.__class__.__name__,
                    error_message=str(exc),
                    failed_callpoint_id="11",
                )
                raise
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="signals_llm_call_done",
                attempt=attempt,
                status="success",
                callpoint_id="11",
                model=self.config.signals.llm.model_11,
                thinking_level=self.config.signals.llm.thinking_11,
                latency_ms=duration_ms,
            )
            enriched = dict(candidate.external_attribution_json)
            enriched["explanation"] = explanation
            final_candidates[idx] = candidate.model_copy(
                update={"external_attribution_json": enriched}
            )

        if self.repository is not None:
            self.repository.write_external_signals(
                run_id=paths.run_id,
                signals=[
                    ExternalSignalRecord(
                        run_id=paths.run_id,
                        signal_id=signal.signal_id,
                        signal_type=signal.signal_type,
                        source_platform=signal.source_platform,
                        source_id=signal.source_id,
                        author_id=signal.author_id,
                        text=signal.text,
                        engagement_score=signal.engagement_score,
                        published_at=signal.published_at,
                        metadata=signal.metadata,
                    )
                    for signal in signal_output.external_signals
                ],
            )
            self.repository.write_external_signal_clusters(
                run_id=paths.run_id,
                clusters=[
                    ExternalSignalClusterRecord(
                        run_id=paths.run_id,
                        cluster_id=cluster.cluster_id,
                        cluster_type=cluster.cluster_type,
                        summary_text=cluster.summary_text,
                        member_signal_ids=list(cluster.member_signal_ids),
                        cluster_weight=cluster.cluster_weight,
                        embedding=list(cluster.embedding),
                        metadata=cluster.metadata,
                    )
                    for cluster in signal_output.clusters
                ],
            )
            if prompt_source_links:
                self.repository.write_prompt_source_links(
                    run_id=paths.run_id,
                    links=prompt_source_links,
                )
            self.repository.write_node_signal_links(
                run_id=paths.run_id,
                links=[
                    NodeSignalLinkRecord(
                        run_id=paths.run_id,
                        node_id=link.node_id,
                        cluster_id=link.cluster_id,
                        link_type=link.link_type,
                        hop_distance=link.hop_distance,
                        time_offset_ms=link.time_offset_ms,
                        similarity=link.similarity,
                        link_score=link.link_score,
                        evidence=link.evidence,
                    )
                    for link in node_signal_links
                ],
            )
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
                        external_signal_score=candidate.external_signal_score,
                        agreement_bonus=candidate.agreement_bonus,
                        external_attribution_json=candidate.external_attribution_json,
                    )
                    for candidate in final_candidates
                ],
            )
            self.repository.write_candidate_signal_links(
                run_id=paths.run_id,
                links=[
                    CandidateSignalLinkRecord(
                        run_id=paths.run_id,
                        clip_id=link.clip_id,
                        cluster_id=link.cluster_id,
                        cluster_type=link.cluster_type,
                        aggregated_link_score=link.aggregated_link_score,
                        coverage_ms=link.coverage_ms,
                        direct_node_count=link.direct_node_count,
                        inferred_node_count=link.inferred_node_count,
                        agreement_flags=list(link.agreement_flags),
                        bonus_applied=link.bonus_applied,
                        evidence=link.evidence,
                    )
                    for link in candidate_signal_links
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
        self._save_debug_json(paths.candidates_dir / "prompt_source_links_debug.json", [link.model_dump(mode="json") for link in prompt_source_links])
        self._save_debug_json(
            paths.candidates_dir / "subgraph_provenance_debug.json",
            [record.model_dump(mode="json") for record in subgraph_provenance],
        )
        self._save_debug_json(
            paths.candidates_dir / "signal_output_debug.json",
            signal_output.model_dump(mode="json"),
        )
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
                "comment_cluster_count": len(signal_output.clusters) if signal_output else 0,
                "comment_prompt_count": len([spec for spec in augmentation_prompt_specs if spec.prompt_source_type == "comment"]),
                "trend_prompt_count": len([spec for spec in augmentation_prompt_specs if spec.prompt_source_type == "trend"]),
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

    def _start_signal_future(
        self,
        *,
        executor: ThreadPoolExecutor,
        run_id: str,
        job_id: str | None,
        attempt: int,
        signal_name: str,
        starter: Callable[[], Any | None],
    ) -> tuple[Any | None, float | None]:
        if signal_name == "comments" and not self._comments_enabled_effective():
            return None, None
        if signal_name == "trends" and not self._trends_enabled_effective():
            return None, None
        started_at = time.perf_counter()
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_future_start",
            attempt=attempt,
            status="start",
            signal_name=signal_name,
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_fetch_start",
            attempt=attempt,
            status="start",
            signal_name=signal_name,
        )
        future = starter()
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_future_submitted",
            attempt=attempt,
            status="submitted",
            signal_name=signal_name,
        )
        return future, started_at

    def _comments_enabled_effective(self) -> bool:
        return bool(self.config.signals.enable_comment_signals) and int(self.config.signals.comment_top_threads_max) > 0

    def _trends_enabled_effective(self) -> bool:
        return bool(self.config.signals.enable_trend_signals) and int(self.config.signals.trend_max_items) > 0

    def _join_signal_future(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        signal_name: str,
        future: Any | None,
        started_at: float | None,
    ) -> SignalPipelineOutput:
        if future is None:
            return SignalPipelineOutput()
        join_started = time.perf_counter()
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_future_join_start",
            attempt=attempt,
            status="start",
            signal_name=signal_name,
        )
        try:
            output = future.result()
        except Exception as exc:
            duration_ms = (time.perf_counter() - (started_at or join_started)) * 1000.0
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="signal_future_join_error",
                attempt=attempt,
                status="error",
                signal_name=signal_name,
                duration_ms=duration_ms,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="signals_failure",
                attempt=attempt,
                status="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                signal_name=signal_name,
                failed_callpoint_id=self._extract_failed_callpoint_id(exc),
            )
            raise
        duration_ms = (time.perf_counter() - (started_at or join_started)) * 1000.0
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_future_join_success",
            attempt=attempt,
            status="success",
            signal_name=signal_name,
            duration_ms=duration_ms,
            prompt_count=len(output.prompt_specs),
            cluster_count=len(output.clusters),
            signal_count=len(output.external_signals),
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_fetch_done",
            attempt=attempt,
            status="success",
            signal_name=signal_name,
            duration_ms=duration_ms,
            prompt_count=len(output.prompt_specs),
            cluster_count=len(output.clusters),
            signal_count=len(output.external_signals),
        )
        return output

    def _join_signal_outputs(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        comments_future: Any | None,
        comments_future_started_at: float | None,
        trends_future: Any | None,
        trends_future_started_at: float | None,
    ) -> tuple[SignalPipelineOutput, SignalPipelineOutput]:
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_hard_join_wait_start",
            attempt=attempt,
            status="start",
            comment_enabled=comments_future is not None,
            trend_enabled=trends_future is not None,
        )
        try:
            comments_output = self._join_signal_future(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                signal_name="comments",
                future=comments_future,
                started_at=comments_future_started_at,
            )
        except Exception:
            if trends_future is not None and hasattr(trends_future, "cancel"):
                trends_future.cancel()
            raise
        trends_output = self._join_signal_future(
            run_id=run_id,
            job_id=job_id,
            attempt=attempt,
            signal_name="trends",
            future=trends_future,
            started_at=trends_future_started_at,
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signals_hard_join_wait_done",
            attempt=attempt,
            status="success",
            comment_prompt_count=len(comments_output.prompt_specs),
            trend_prompt_count=len(trends_output.prompt_specs),
            comment_cluster_count=len(comments_output.clusters),
            trend_cluster_count=len(trends_output.clusters),
        )
        return comments_output, trends_output

    def _merge_signal_outputs(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        comments_output: SignalPipelineOutput,
        trends_output: SignalPipelineOutput,
    ) -> SignalPipelineOutput:
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_augmentation_merge_start",
            attempt=attempt,
            status="start",
            comment_prompt_count=len(comments_output.prompt_specs),
            trend_prompt_count=len(trends_output.prompt_specs),
        )
        merged = merge_signal_outputs(comments=comments_output, trends=trends_output)
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_augmentation_merge_success",
            attempt=attempt,
            status="success",
            prompt_count=len(merged.prompt_specs),
            cluster_count=len(merged.clusters),
            signal_count=len(merged.external_signals),
        )
        if comments_output.metadata:
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="comments_threads_count",
                attempt=attempt,
                status="success",
                threads_total=int(comments_output.metadata.get("threads_total") or 0),
                threads_selected=int(comments_output.metadata.get("threads_selected") or 0),
            )
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="comments_replies_count",
                attempt=attempt,
                status="success",
                replies_total=int(comments_output.metadata.get("replies_total") or 0),
            )
        if trends_output.metadata:
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="trend_items_count",
                attempt=attempt,
                status="success",
                trend_items_count=int(trends_output.metadata.get("trend_items_count") or 0),
            )
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event="trend_retained_count",
                attempt=attempt,
                status="success",
                trend_retained_count=int(trends_output.metadata.get("trend_retained_count") or 0),
            )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_clusters_built",
            attempt=attempt,
            status="success",
            cluster_count=len(merged.clusters),
            prompt_count=len(merged.prompt_specs),
            signal_count=len(merged.external_signals),
        )
        return merged

    def _extract_failed_callpoint_id(self, exc: BaseException) -> str | None:
        pattern = re.compile(r"callpoint(?:_id)?[=:\s]+(?P<id>[0-9]+)")
        current: BaseException | None = exc
        while current is not None:
            callpoint = getattr(current, "callpoint_id", None)
            if callpoint is not None:
                return str(callpoint)
            match = pattern.search(str(current))
            if match:
                return match.group("id")
            current = current.__cause__
        return None

    def _build_prompt_source_links(
        self,
        *,
        run_id: str,
        prompt_specs: list[SignalPromptSpec],
    ) -> list[PromptSourceLinkRecord]:
        links: list[PromptSourceLinkRecord] = []
        for idx, prompt in enumerate(prompt_specs, start=1):
            metadata = {
                "prompt_text": prompt.text,
                "prompt_index": idx,
            }
            links.append(
                PromptSourceLinkRecord(
                    run_id=run_id,
                    prompt_id=prompt.prompt_id,
                    prompt_source_type=prompt.prompt_source_type,
                    source_cluster_id=prompt.source_cluster_id,
                    source_cluster_type=prompt.source_cluster_type,
                    metadata=metadata,
                )
            )
        return links

    def _build_subgraph_provenance(
        self,
        *,
        run_id: str,
        subgraphs: list[Any],
        prompt_specs: list[SignalPromptSpec],
    ) -> list[SubgraphProvenanceRecord]:
        prompt_by_id = {prompt.prompt_id: prompt for prompt in prompt_specs}
        provenance_records: list[SubgraphProvenanceRecord] = []
        for subgraph in subgraphs:
            seed_prompt_ids = list(getattr(subgraph, "source_prompt_ids", []) or [])
            source_types: list[str] = []
            source_cluster_ids: list[str] = []
            source_type_counts: dict[str, int] = {}
            for prompt_id in seed_prompt_ids:
                prompt = prompt_by_id.get(prompt_id)
                if prompt is None:
                    continue
                if prompt.prompt_source_type not in source_types:
                    source_types.append(prompt.prompt_source_type)
                source_type_counts[prompt.prompt_source_type] = source_type_counts.get(prompt.prompt_source_type, 0) + 1
                if prompt.source_cluster_id and prompt.source_cluster_id not in source_cluster_ids:
                    source_cluster_ids.append(prompt.source_cluster_id)
            if not source_types:
                source_types = ["general"]
                source_type_counts = {"general": len(seed_prompt_ids)}
            provenance_records.append(
                SubgraphProvenanceRecord(
                    run_id=run_id,
                    subgraph_id=str(getattr(subgraph, "subgraph_id")),
                    seed_source_set=source_types,
                    seed_prompt_ids=seed_prompt_ids,
                    source_cluster_ids=source_cluster_ids,
                    support_summary={
                        "seed_prompt_count": len(seed_prompt_ids),
                        "source_type_counts": source_type_counts,
                        "node_count": len(getattr(subgraph, "nodes", []) or []),
                    },
                    canonical_selected=True,
                    dedupe_overlap_ratio=None,
                    selection_reason="retained_after_dedupe",
                    metadata={
                        "seed_node_id": getattr(subgraph, "seed_node_id", None),
                        "node_count": len(getattr(subgraph, "nodes", []) or []),
                    },
                )
            )
        return provenance_records

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

    def _get_resume_phase(self, *, run_id: str) -> str | None:
        if self.repository is None:
            return None
        list_phase_metrics = getattr(self.repository, "list_phase_metrics", None)
        if not callable(list_phase_metrics):
            return None
        metrics = list_phase_metrics(run_id=run_id)
        succeeded_phases = {
            metric.phase_name
            for metric in metrics
            if getattr(metric, "status", None) == "succeeded"
        }
        for phase_name in ("phase4", "phase3", "phase2"):
            if phase_name in succeeded_phases:
                return phase_name
        return None

    def _load_repository_records(self, *, run_id: str, method_name: str) -> list[Any]:
        if self.repository is None:
            return []
        method = getattr(self.repository, method_name, None)
        if not callable(method):
            return []
        return list(method(run_id=run_id))

    def _load_resume_nodes(self, *, run_id: str) -> list[SemanticGraphNode]:
        raw_nodes = self._load_repository_records(run_id=run_id, method_name="list_nodes")
        if not raw_nodes:
            raise RuntimeError(
                "Resume requested but no persisted nodes were found for this run."
            )
        nodes: list[SemanticGraphNode] = []
        for item in raw_nodes:
            if isinstance(item, SemanticGraphNode):
                nodes.append(item)
                continue
            payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            payload.pop("run_id", None)
            nodes.append(SemanticGraphNode(**payload))
        return nodes

    def _load_resume_edges(self, *, run_id: str) -> list[SemanticGraphEdge]:
        raw_edges = self._load_repository_records(run_id=run_id, method_name="list_edges")
        if not raw_edges:
            raise RuntimeError(
                "Resume requested but no persisted edges were found for this run."
            )
        edges: list[SemanticGraphEdge] = []
        for item in raw_edges:
            if isinstance(item, SemanticGraphEdge):
                edges.append(item)
                continue
            payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            payload.pop("run_id", None)
            edges.append(SemanticGraphEdge(**payload))
        return edges

    def _load_resume_candidates(self, *, run_id: str) -> list[ClipCandidate]:
        raw_candidates = self._load_repository_records(run_id=run_id, method_name="list_candidates")
        candidates: list[ClipCandidate] = []
        for item in raw_candidates:
            if isinstance(item, ClipCandidate):
                candidates.append(item)
                continue
            payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            payload.pop("run_id", None)
            candidates.append(ClipCandidate(**payload))
        return candidates

    def _emit_phase_skipped_resume(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        phase_name: str,
        reason: str,
    ) -> None:
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase=phase_name,
            event="phase_skipped_resume",
            attempt=attempt,
            status="skipped",
            reason=reason,
        )

    def _finish_phase24_success(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        paths: V31RunPaths,
        started_at: datetime,
        started: float,
        phase2_nodes: list[Any],
        phase3_edges: list[Any],
        phase4_candidates: list[Any] | None = None,
        phase4_result: dict[str, Any] | None = None,
        resumed_phases: tuple[str, ...] = (),
    ) -> Phase14RunSummary:
        if resumed_phases:
            for phase_name in resumed_phases:
                self._emit_phase_skipped_resume(
                    run_id=run_id,
                    job_id=job_id,
                    attempt=attempt,
                    phase_name=phase_name,
                    reason=f"{phase_name}_already_succeeded",
                )
        ended_at = datetime.now(UTC)
        total_duration_ms = (time.perf_counter() - started) * 1000.0
        candidate_count = (
            int(phase4_result["final_candidate_count"])
            if phase4_result is not None
            else len(phase4_candidates or [])
        )
        self._write_phase_metric(
            run_id=run_id,
            phase_name="phase24",
            status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
            metadata={
                "node_count": len(phase2_nodes),
                "edge_count": len(phase3_edges),
                "candidate_count": candidate_count,
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
                "node_count": len(phase2_nodes),
                "edge_count": len(phase3_edges),
                "candidate_count": candidate_count,
                "query_version": self.query_version,
            },
        )

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
