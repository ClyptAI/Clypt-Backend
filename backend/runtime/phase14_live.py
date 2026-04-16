from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    run_candidate_pool_review_with_debug,
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
    apply_node_embeddings,
    embed_multimodal_media_live,
    embed_text_semantic_nodes_live,
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
    PhaseSubstepRecord,
    NodeSignalLinkRecord,
    PromptSourceLinkRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    SubgraphProvenanceRecord,
    TimelineTurnRecord,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc
_PHASE4_PROMPT_SOURCE_PRIORITY = {
    "general": 0,
    "comment": 1,
    "trend": 2,
}
_PHASE4_POOL_CANDIDATES_PER_CALL = 12


@dataclass(slots=True)
class V31LivePhase14Runner:
    config: V31Config
    llm_client: Any
    embedding_client: Any
    flash_model: str = "Qwen/Qwen3.5-27B"
    storage_client: Any | None = None
    node_media_preparer: Any | None = None
    repository: Phase14Repository | None = None
    query_version: str | None = None
    debug_snapshots: bool = False
    log_event: Callable[..., None] | None = None
    _prefetched_phase3_local_future: Any | None = field(default=None, init=False, repr=False)
    _prefetched_phase3_local_executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _prefetched_phase3_local_node_ids: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        llm_client: Any,
        embedding_client: Any,
        flash_model: str = "Qwen/Qwen3.5-27B",
        storage_client: Any | None = None,
        node_media_preparer: Any | None = None,
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
            node_media_preparer=node_media_preparer,
            repository=repository,
            query_version=query_version,
            debug_snapshots=debug_snapshots,
            log_event=log_event,
        )

    def build_run_paths(self, *, run_id: str) -> V31RunPaths:
        return build_run_paths(output_root=self.config.output_root, run_id=run_id)

    def _apply_phase4_prompt_budget(
        self,
        *,
        general_prompt_specs: list[SignalPromptSpec],
        augmentation_prompt_specs: list[SignalPromptSpec],
    ) -> tuple[list[SignalPromptSpec], dict[str, Any]]:
        all_prompt_specs = [*general_prompt_specs, *augmentation_prompt_specs]
        max_total_prompts = max(1, int(self.config.phase4_budget.max_total_prompts))
        selected_prompt_specs = list(all_prompt_specs)
        if len(all_prompt_specs) > max_total_prompts:
            ranked = sorted(
                enumerate(all_prompt_specs),
                key=lambda item: (
                    _PHASE4_PROMPT_SOURCE_PRIORITY.get(item[1].prompt_source_type, 99),
                    item[0],
                ),
            )
            kept_indices = {idx for idx, _ in ranked[:max_total_prompts]}
            selected_prompt_specs = [
                prompt_spec
                for idx, prompt_spec in enumerate(all_prompt_specs)
                if idx in kept_indices
            ]

        dropped_prompt_specs = [
            prompt_spec
            for prompt_spec in all_prompt_specs
            if prompt_spec.prompt_id not in {item.prompt_id for item in selected_prompt_specs}
        ]
        return selected_prompt_specs, {
            "max_total_prompts": max_total_prompts,
            "input_prompt_count": len(all_prompt_specs),
            "selected_prompt_count": len(selected_prompt_specs),
            "dropped_prompt_count": len(dropped_prompt_specs),
            "selected_prompt_ids": [item.prompt_id for item in selected_prompt_specs],
            "dropped_prompt_ids": [item.prompt_id for item in dropped_prompt_specs],
            "general_prompt_count": len(general_prompt_specs),
            "augmentation_prompt_count": len(augmentation_prompt_specs),
            "selected_prompt_source_counts": {
                "general": sum(1 for item in selected_prompt_specs if item.prompt_source_type == "general"),
                "comment": sum(1 for item in selected_prompt_specs if item.prompt_source_type == "comment"),
                "trend": sum(1 for item in selected_prompt_specs if item.prompt_source_type == "trend"),
            },
        }

    def _apply_phase4_subgraph_budget(
        self,
        *,
        subgraphs: list[Any],
        seeds: list[dict[str, Any]],
    ) -> tuple[list[Any], dict[str, Any]]:
        max_subgraphs = max(1, int(self.config.phase4_budget.max_subgraphs_per_run))
        selected_subgraphs = list(subgraphs)
        seed_by_node_id = {
            str(seed.get("node_id")): dict(seed)
            for seed in seeds
            if seed.get("node_id") is not None
        }
        if len(subgraphs) > max_subgraphs:
            ranked = sorted(
                subgraphs,
                key=lambda subgraph: (
                    -float(seed_by_node_id.get(subgraph.seed_node_id, {}).get("retrieval_score") or 0.0),
                    -len(list(subgraph.source_prompt_ids or [])),
                    -len(list(subgraph.nodes or [])),
                    int(subgraph.start_ms),
                    str(subgraph.subgraph_id),
                ),
            )
            selected_subgraphs = ranked[:max_subgraphs]

        dropped_subgraphs = [
            subgraph
            for subgraph in subgraphs
            if subgraph.subgraph_id not in {item.subgraph_id for item in selected_subgraphs}
        ]
        return selected_subgraphs, {
            "max_subgraphs_per_run": max_subgraphs,
            "input_subgraph_count": len(subgraphs),
            "selected_subgraph_count": len(selected_subgraphs),
            "dropped_subgraph_count": len(dropped_subgraphs),
            "selected_subgraph_ids": [item.subgraph_id for item in selected_subgraphs],
            "dropped_subgraph_ids": [item.subgraph_id for item in dropped_subgraphs],
        }

    def _apply_phase4_pool_candidate_budget(
        self,
        *,
        candidates: list[ClipCandidate],
    ) -> tuple[list[ClipCandidate], dict[str, Any]]:
        max_review_calls = max(0, int(self.config.phase4_budget.max_final_review_calls))
        max_pool_candidates = max_review_calls * _PHASE4_POOL_CANDIDATES_PER_CALL
        if max_pool_candidates <= 0:
            return [], {
                "max_final_review_calls": max_review_calls,
                "max_pool_candidates": 0,
                "input_candidate_count": len(candidates),
                "budget_selected_candidate_count": 0,
                "budget_dropped_candidate_count": len(candidates),
                "budget_selected_candidate_ids": [],
                "budget_dropped_candidate_ids": [
                    candidate.clip_id or f"cand_tmp_{idx:03d}"
                    for idx, candidate in enumerate(candidates, start=1)
                ],
            }

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.score),
                -(1 if bool(candidate.query_aligned) else 0),
                -len(list(candidate.source_prompt_ids or [])),
                int(candidate.start_ms),
                str(candidate.clip_id or ""),
            ),
        )
        selected_candidates = ranked[:max_pool_candidates]
        dropped_candidates = ranked[max_pool_candidates:]
        return selected_candidates, {
            "max_final_review_calls": max_review_calls,
            "max_pool_candidates": max_pool_candidates,
            "input_candidate_count": len(candidates),
            "budget_selected_candidate_count": len(selected_candidates),
            "budget_dropped_candidate_count": len(dropped_candidates),
            "budget_selected_candidate_ids": [
                candidate.clip_id or f"cand_tmp_{idx:03d}"
                for idx, candidate in enumerate(selected_candidates, start=1)
            ],
            "budget_dropped_candidate_ids": [
                candidate.clip_id or f"cand_tmp_{idx:03d}"
                for idx, candidate in enumerate(dropped_candidates, start=1)
            ],
        }

    def run(
        self,
        *,
        run_id: str,
        source_url: str,
        phase1_outputs: Phase1SidecarOutputs,
        phase3_long_range_top_k: int | None = None,
        phase4_extra_prompt_texts: list[str] | None = None,
        job_id: str | None = None,
        attempt: int = 1,
    ) -> Phase14RunSummary:
        paths = self.build_run_paths(run_id=run_id)
        effective_phase3_long_range_top_k = (
            phase3_long_range_top_k
            if phase3_long_range_top_k is not None
            else self.config.phase3_long_range_top_k
        )

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
        signal_event_logger = self._build_signal_event_logger(run_id=run_id, job_id=job_id, attempt=attempt)
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
                    signal_event_logger=signal_event_logger,
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
                            raw_nodes_ready_callback=self._start_prefetched_phase3_local_lane,
                        ),
                        metric_metadata_builder=lambda payload: {"node_count": len(payload["nodes"])},
                    )
                except Exception:
                    self._discard_prefetched_phase3_local_lane()
                    self._cancel_signal_future(
                        run_id=run_id,
                        job_id=job_id,
                        attempt=attempt,
                        signal_name="comments",
                        future=comments_future,
                    )
                    raise
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
                    signal_event_logger=signal_event_logger,
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
                try:
                    phase3 = self._execute_phase(
                        run_id=run_id,
                        job_id=job_id,
                        attempt=attempt,
                        phase_name="phase3",
                        operation=lambda: self.run_phase_3(
                            paths=paths,
                            nodes=phase2_nodes,
                            long_range_top_k=effective_phase3_long_range_top_k,
                        ),
                        metric_metadata_builder=lambda payload: {"edge_count": len(payload["edges"])},
                    )
                except Exception:
                    self._discard_prefetched_phase3_local_lane()
                    self._cancel_signal_future(
                        run_id=run_id,
                        job_id=job_id,
                        attempt=attempt,
                        signal_name="comments",
                        future=comments_future,
                    )
                    self._cancel_signal_future(
                        run_id=run_id,
                        job_id=job_id,
                        attempt=attempt,
                        signal_name="trends",
                        future=trends_future,
                    )
                    raise
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
            self._discard_prefetched_phase3_local_lane()
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
        raw_nodes_ready_callback: Callable[[list[SemanticGraphNode]], None] | None = None,
    ) -> dict[str, Any]:
        phase2_substeps: list[PhaseSubstepRecord] = []
        merge_boundary_started_at = datetime.now(UTC)
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
            max_concurrent=self.config.phase2_max_concurrent,
            boundary_max_concurrent=self.config.phase2_boundary_max_concurrent,
        )
        merge_boundary_duration_ms = (time.perf_counter() - merge_boundary_started) * 1000.0
        merge_boundary_ended_at = datetime.now(UTC)
        logger.info("[phase2] merge+boundary total done in %.1f s", merge_boundary_duration_ms / 1000.0)
        phase2_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase2",
                step_name="merge_boundary_total",
                step_key="merge_boundary_total",
                status="succeeded",
                started_at=merge_boundary_started_at,
                ended_at=merge_boundary_ended_at,
                duration_ms=merge_boundary_duration_ms,
                metadata={
                    "merge_batch_count": len(merge_debug),
                    "boundary_seam_count": len(boundary_debug),
                    "node_count": len(nodes),
                },
            )
        )
        for item in merge_debug:
            diagnostics = dict(item.get("diagnostics") or {})
            phase2_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase2",
                    step_name="merge_batch",
                    step_key=str(item.get("batch_id") or f"merge_batch_{len(phase2_substeps):04d}"),
                    status="succeeded",
                    duration_ms=float(diagnostics.get("latency_ms") or 0.0),
                    metadata=diagnostics,
                )
            )
        for item in boundary_debug:
            diagnostics = dict(item.get("diagnostics") or {})
            seam_key = f"{item.get('left_batch_id', 'left')}__{item.get('right_batch_id', 'right')}"
            phase2_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase2",
                    step_name="boundary_seam",
                    step_key=seam_key,
                    status="succeeded",
                    duration_ms=float(diagnostics.get("latency_ms") or 0.0),
                    metadata={
                        **diagnostics,
                        "left_batch_id": item.get("left_batch_id"),
                        "right_batch_id": item.get("right_batch_id"),
                    },
                )
            )
        if raw_nodes_ready_callback is not None:
            raw_nodes_ready_callback(nodes)
        media_started_at = datetime.now(UTC)
        media_started = time.perf_counter()
        semantic_started_at = datetime.now(UTC)
        semantic_started = time.perf_counter()

        def _prepare_multimodal_media() -> list[dict[str, str]]:
            if self.node_media_preparer is not None:
                return self.node_media_preparer(nodes=nodes, paths=paths, phase1_outputs=phase1_outputs)
            local_video_path = (phase1_outputs.phase1_audio or {}).get("local_video_path")
            if not local_video_path:
                raise ValueError(
                    "phase1_outputs.phase1_audio.local_video_path is required for live multimodal node embeddings."
                )
            if self.storage_client is None:
                raise ValueError("storage_client is required for live multimodal node embeddings.")
            return prepare_node_media_embeddings(
                nodes=nodes,
                source_video_path=Path(local_video_path),
                clips_dir=paths.semantics_dir / "node_media_clips",
                storage_client=self.storage_client,
                object_prefix=f"phase14/{paths.run_id}/node_media",
            )

        with ThreadPoolExecutor(max_workers=2) as phase2_pool:
            semantic_future = phase2_pool.submit(
                embed_text_semantic_nodes_live,
                nodes=nodes,
                embedding_client=self.embedding_client,
            )
            media_future = phase2_pool.submit(_prepare_multimodal_media)
            multimodal_media = media_future.result()
            semantic_embeddings, semantic_diagnostics = semantic_future.result()

        logger.info(
            "[phase2] node clip extraction+upload done in %.1f s (nodes=%d)",
            time.perf_counter() - media_started,
            len(nodes),
        )
        media_duration_ms = (time.perf_counter() - media_started) * 1000.0
        media_ended_at = datetime.now(UTC)
        phase2_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase2",
                step_name="media_prep",
                step_key="node_media",
                status="succeeded",
                started_at=media_started_at,
                ended_at=media_ended_at,
                duration_ms=media_duration_ms,
                metadata={
                    "node_count": len(nodes),
                    "prepared_media_count": len(multimodal_media),
                    "media_backend": "custom_preparer" if self.node_media_preparer is not None else "local_storage_upload",
                },
            )
        )
        phase2_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase2",
                step_name="semantic_embedding",
                step_key="semantic_embedding",
                status="succeeded",
                started_at=semantic_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=float(semantic_diagnostics.get("semantic_duration_ms") or 0.0),
                metadata=semantic_diagnostics,
            )
        )
        multimodal_started_at = datetime.now(UTC)
        multimodal_embeddings, multimodal_diagnostics = embed_multimodal_media_live(
            multimodal_media=multimodal_media,
            embedding_client=self.embedding_client,
        )
        phase2_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase2",
                step_name="multimodal_embedding",
                step_key="multimodal_embedding",
                status="succeeded",
                started_at=multimodal_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=float(multimodal_diagnostics.get("multimodal_duration_ms") or 0.0),
                metadata=multimodal_diagnostics,
            )
        )
        embedded_nodes = apply_node_embeddings(
            nodes=nodes,
            semantic_embeddings=semantic_embeddings,
            multimodal_embeddings=multimodal_embeddings,
        )
        if self.repository is not None:
            persist_started_at = datetime.now(UTC)
            persist_started = time.perf_counter()
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
            phase2_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase2",
                    step_name="persist_nodes",
                    step_key="semantic_nodes",
                    status="succeeded",
                    started_at=persist_started_at,
                    ended_at=datetime.now(UTC),
                    duration_ms=(time.perf_counter() - persist_started) * 1000.0,
                    metadata={"node_count": len(embedded_nodes)},
                )
            )
        self._write_phase_substeps(run_id=paths.run_id, substeps=phase2_substeps)
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
        phase3_substeps: list[PhaseSubstepRecord] = []
        structural_edges = build_structural_edges(nodes=nodes)
        long_range_started_at = datetime.now(UTC)
        prefetched_local = self._consume_prefetched_phase3_local_lane(nodes=nodes)
        local_started_at = datetime.now(UTC)

        try:
            with ThreadPoolExecutor(max_workers=1 if prefetched_local is not None else 2) as phase3_pool:
                if prefetched_local is None:
                    local_future = phase3_pool.submit(self._run_phase_3_local_lane, nodes=nodes)
                else:
                    local_future = prefetched_local
                long_range_future = phase3_pool.submit(
                    self._run_phase_3_long_range_lane,
                    nodes=nodes,
                    long_range_top_k=long_range_top_k,
                )
                local_edges, local_debug, local_duration_ms = local_future.result()
                long_range_edges, long_range_debug, long_range_duration_ms = long_range_future.result()
        finally:
            if self._prefetched_phase3_local_executor is not None:
                self._prefetched_phase3_local_executor.shutdown(wait=False, cancel_futures=True)
                self._prefetched_phase3_local_executor = None

        logger.info("[phase3] local semantic edges done in %.1f s", local_duration_ms / 1000.0)
        phase3_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase3",
                step_name="local_edges_total",
                step_key="local_edges_total",
                status="succeeded",
                started_at=local_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=local_duration_ms,
                metadata={
                    "batch_count": len(local_debug),
                    "edge_count": len(local_edges),
                },
            )
        )
        for item in local_debug:
            diagnostics = dict(item.get("diagnostics") or {})
            phase3_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase3",
                    step_name="local_edge_batch",
                    step_key=str(item.get("batch_id") or f"local_edge_batch_{len(phase3_substeps):04d}"),
                    status="succeeded",
                    duration_ms=float(diagnostics.get("latency_ms") or 0.0),
                    metadata={
                        **diagnostics,
                        "sanitized_edge_count": int(item.get("sanitized_edge_count") or 0),
                        "dropped_edge_count": int(item.get("dropped_edge_count") or 0),
                    },
                )
            )

        logger.info("[phase3] long-range adjudication done in %.1f s", long_range_duration_ms / 1000.0)
        phase3_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase3",
                step_name="long_range_adjudication",
                step_key="long_range_adjudication",
                status="succeeded",
                started_at=long_range_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=long_range_duration_ms,
                metadata={
                    **dict(long_range_debug.get("diagnostics") or {}),
                    "shard_count": len(list(long_range_debug.get("shards") or [])),
                },
            )
        )
        for shard_debug in list(long_range_debug.get("shards") or []):
            diagnostics = dict(shard_debug.get("diagnostics") or {})
            phase3_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase3",
                    step_name="long_range_shard",
                    step_key=str(shard_debug.get("shard_id") or f"long_range_shard_{len(phase3_substeps):04d}"),
                    status="succeeded",
                    duration_ms=float(diagnostics.get("latency_ms") or 0.0),
                    metadata=diagnostics,
                )
            )
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
            persist_duration_ms = (time.perf_counter() - persist_started) * 1000.0
            logger.info("[phase3] edge persistence done in %.1f s", persist_duration_ms / 1000.0)
            phase3_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase3",
                    step_name="persist_edges",
                    step_key="semantic_edges",
                    status="succeeded",
                    duration_ms=persist_duration_ms,
                    metadata={"edge_count": len(final_edges)},
                )
            )
        self._write_phase_substeps(run_id=paths.run_id, substeps=phase3_substeps)
        self._save_debug_json(paths.semantic_graph_edges, [edge.model_dump(mode="json") for edge in final_edges])
        self._save_debug_json(paths.graph_dir / "local_semantic_edges_debug.json", local_debug)
        self._save_debug_json(paths.graph_dir / "long_range_edges_debug.json", long_range_debug)
        return {"edges": final_edges}

    def _run_phase_3_local_lane(
        self,
        *,
        nodes: list[SemanticGraphNode],
    ) -> tuple[list[SemanticGraphEdge], list[dict[str, Any]], float]:
        started = time.perf_counter()
        local_edges, local_debug = run_local_semantic_edge_batches(
            nodes=nodes,
            llm_client=self.llm_client,
            target_batch_count=self.config.phase3_target_batch_count,
            max_nodes_per_batch=self.config.phase3_max_nodes_per_batch,
            model=self.flash_model,
            max_concurrent=self.config.phase3_local_max_concurrent,
            max_output_tokens=self.config.phase3_local_max_output_tokens,
        )
        return local_edges, local_debug, (time.perf_counter() - started) * 1000.0

    def _run_phase_3_long_range_lane(
        self,
        *,
        nodes: list[SemanticGraphNode],
        long_range_top_k: int,
    ) -> tuple[list[SemanticGraphEdge], dict[str, Any], float]:
        started = time.perf_counter()
        long_range_edges, long_range_debug = run_long_range_edge_adjudication(
            nodes=nodes,
            llm_client=self.llm_client,
            top_k=long_range_top_k,
            model=self.flash_model,
            max_output_tokens=self.config.phase3_long_range_max_output_tokens,
            pairs_per_shard=self.config.phase3_long_range_pairs_per_shard,
            max_concurrent=self.config.phase3_long_range_max_concurrent,
        )
        return long_range_edges, long_range_debug, (time.perf_counter() - started) * 1000.0

    def _start_prefetched_phase3_local_lane(self, nodes: list[SemanticGraphNode]) -> None:
        if self._prefetched_phase3_local_future is not None:
            return
        node_ids = tuple(node.node_id for node in nodes)
        executor = ThreadPoolExecutor(max_workers=1)
        self._prefetched_phase3_local_executor = executor
        self._prefetched_phase3_local_node_ids = node_ids
        self._prefetched_phase3_local_future = executor.submit(self._run_phase_3_local_lane, nodes=nodes)

    def _consume_prefetched_phase3_local_lane(self, *, nodes: list[SemanticGraphNode]) -> Any | None:
        if self._prefetched_phase3_local_future is None:
            return None
        node_ids = tuple(node.node_id for node in nodes)
        if self._prefetched_phase3_local_node_ids != node_ids:
            self._discard_prefetched_phase3_local_lane()
            return None
        future = self._prefetched_phase3_local_future
        self._prefetched_phase3_local_future = None
        self._prefetched_phase3_local_node_ids = None
        return future

    def _discard_prefetched_phase3_local_lane(self) -> None:
        future = self._prefetched_phase3_local_future
        executor = self._prefetched_phase3_local_executor
        self._prefetched_phase3_local_future = None
        self._prefetched_phase3_local_executor = None
        self._prefetched_phase3_local_node_ids = None
        if future is not None and hasattr(future, "cancel"):
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

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
        phase4_substeps: list[PhaseSubstepRecord] = []
        duration_s = 0.0
        if canonical_timeline.turns:
            duration_s = canonical_timeline.turns[-1].end_ms / 1000.0
        meta_started_at = datetime.now(UTC)
        meta_started = time.perf_counter()
        dynamic_prompts, meta_debug = generate_meta_prompts_live(
            nodes=nodes,
            llm_client=self.llm_client,
            model=self.flash_model,
            duration_s=duration_s,
            max_output_tokens=self.config.phase4_meta_max_output_tokens,
            return_debug=True,
        )
        meta_duration_ms = (time.perf_counter() - meta_started) * 1000.0
        logger.info("[phase4] meta prompt generation done in %.1f s", meta_duration_ms / 1000.0)
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="meta_prompt_generation",
                step_key="meta_prompt_generation",
                status="succeeded",
                started_at=meta_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=meta_duration_ms,
                metadata=meta_debug,
            )
        )
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
        all_prompt_specs, prompt_budget_debug = self._apply_phase4_prompt_budget(
            general_prompt_specs=general_prompt_specs,
            augmentation_prompt_specs=augmentation_prompt_specs,
        )

        prompt_source_links = self._build_prompt_source_links(run_id=paths.run_id, prompt_specs=all_prompt_specs)

        prompt_embedding_started_at = datetime.now(UTC)
        embedded_prompts, prompt_embedding_debug = embed_prompt_texts_live(
            prompts=all_prompt_specs,
            embedding_client=self.embedding_client,
            return_debug=True,
        )
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="prompt_embedding",
                step_key="prompt_embedding",
                status="succeeded",
                started_at=prompt_embedding_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=float(prompt_embedding_debug.get("latency_ms") or 0.0),
                metadata={**prompt_embedding_debug, **prompt_budget_debug},
            )
        )
        seed_started_at = datetime.now(UTC)
        seed_started = time.perf_counter()
        seeds = retrieve_seed_nodes(
            prompts=embedded_prompts,
            nodes=nodes,
            top_k_per_prompt=self.config.phase4_subgraphs.seed_top_k_per_prompt,
        )
        seed_duration_ms = (time.perf_counter() - seed_started) * 1000.0
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="seed_retrieval",
                step_key="seed_retrieval",
                status="succeeded",
                started_at=seed_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=seed_duration_ms,
                metadata={
                    "prompt_count": len(embedded_prompts),
                    "node_count": len(nodes),
                    "seed_count": len(seeds),
                    "top_k_per_prompt": self.config.phase4_subgraphs.seed_top_k_per_prompt,
                },
            )
        )
        subgraph_build_started_at = datetime.now(UTC)
        subgraph_build_started = time.perf_counter()
        subgraphs = build_local_subgraphs(
            seeds=seeds,
            nodes=nodes,
            edges=edges,
            config=self.config.phase4_subgraphs,
        )
        subgraphs, subgraph_budget_debug = self._apply_phase4_subgraph_budget(
            subgraphs=subgraphs,
            seeds=seeds,
        )
        subgraph_build_duration_ms = (time.perf_counter() - subgraph_build_started) * 1000.0
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="subgraph_build",
                step_key="subgraph_build",
                status="succeeded",
                started_at=subgraph_build_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=subgraph_build_duration_ms,
                metadata={
                    **subgraph_budget_debug,
                    "seed_count": len(seeds),
                    "subgraph_count": len(subgraphs),
                    "edge_count": len(edges),
                },
            )
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
            provenance_started_at = datetime.now(UTC)
            provenance_started = time.perf_counter()
            self.repository.write_subgraph_provenance(run_id=paths.run_id, provenance=subgraph_provenance)
            phase4_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase4",
                    step_name="persist_subgraph_provenance",
                    step_key="subgraph_provenance",
                    status="succeeded",
                    started_at=provenance_started_at,
                    ended_at=datetime.now(UTC),
                    duration_ms=(time.perf_counter() - provenance_started) * 1000.0,
                    metadata={"subgraph_count": len(subgraph_provenance)},
                )
            )

        subgraph_review_started_at = datetime.now(UTC)
        subgraph_review_started = time.perf_counter()
        reviews, subgraph_debug = run_subgraph_reviews(
            subgraphs=subgraphs,
            llm_client=self.llm_client,
            model=self.flash_model,
            max_concurrent=self.config.phase4_subgraph_max_concurrent,
            subgraph_provenance_by_id=subgraph_provenance_by_id,
            max_output_tokens=self.config.phase4_subgraph_max_output_tokens,
        )
        subgraph_review_duration_ms = (time.perf_counter() - subgraph_review_started) * 1000.0
        logger.info("[phase4] subgraph reviews done in %.1f s", subgraph_review_duration_ms / 1000.0)
        subgraph_review_diagnostics = [
            dict(item.get("diagnostics") or {})
            for item in subgraph_debug
            if isinstance(item, dict)
        ]
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="subgraph_review_total",
                step_key="subgraph_review_total",
                status="succeeded",
                started_at=subgraph_review_started_at,
                ended_at=datetime.now(UTC),
                duration_ms=subgraph_review_duration_ms,
                metadata={"subgraph_count": len(subgraph_debug)},
            )
        )
        for item in subgraph_debug:
            if not isinstance(item, dict):
                continue
            diagnostics = dict(item.get("diagnostics") or {})
            phase4_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="phase4",
                    step_name="subgraph_review",
                    step_key=str(item.get("subgraph_id") or f"subgraph_review_{len(phase4_substeps):04d}"),
                    status="succeeded",
                    duration_ms=float(diagnostics.get("latency_ms") or 0.0),
                    metadata={
                        **diagnostics,
                        "seed_node_id": item.get("seed_node_id"),
                    },
                )
            )

        def _diag_mean(key: str) -> float:
            values = [float(diag.get(key) or 0.0) for diag in subgraph_review_diagnostics]
            return sum(values) / len(values) if values else 0.0

        def _diag_p95(key: str) -> float:
            values = sorted(float(diag.get(key) or 0.0) for diag in subgraph_review_diagnostics)
            if not values:
                return 0.0
            rank = max(0, int((len(values) - 1) * 0.95))
            return values[rank]

        def _diag_max(key: str) -> float:
            values = [float(diag.get(key) or 0.0) for diag in subgraph_review_diagnostics]
            return max(values) if values else 0.0

        invalid_structured_output_count = sum(
            1 for diag in subgraph_review_diagnostics if bool(diag.get("invalid_structured_output"))
        )
        reject_all_count = sum(
            1 for diag in subgraph_review_diagnostics if bool(diag.get("reject_all"))
        )
        total_reviewed_candidates = sum(
            int(diag.get("candidate_count") or 0) for diag in subgraph_review_diagnostics
        )
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="phase4",
            event="subgraph_review_diagnostics",
            attempt=attempt,
            status="success",
            reviewed_subgraph_count=len(subgraph_review_diagnostics),
            rejected_subgraph_count=reject_all_count,
            invalid_structured_output_count=invalid_structured_output_count,
            reviewed_candidate_count=total_reviewed_candidates,
            avg_latency_ms=_diag_mean("latency_ms"),
            p95_latency_ms=_diag_p95("latency_ms"),
            max_latency_ms=_diag_max("latency_ms"),
            avg_node_count=_diag_mean("node_count"),
            max_node_count=_diag_max("node_count"),
            avg_span_ms=_diag_mean("span_ms"),
            max_span_ms=_diag_max("span_ms"),
            avg_prompt_chars=_diag_mean("prompt_chars"),
            p95_prompt_chars=_diag_p95("prompt_chars"),
            avg_prompt_token_estimate=_diag_mean("prompt_token_estimate"),
            max_prompt_token_estimate=_diag_max("prompt_token_estimate"),
        )
        raw_candidates: list[ClipCandidate] = []
        for review in reviews:
            raw_candidates.extend(review.candidates)
        deduped_candidates = dedupe_clip_candidates(candidates=raw_candidates)
        pool_review_candidates, pool_budget_debug = self._apply_phase4_pool_candidate_budget(
            candidates=deduped_candidates,
        )
        pool_review_started = time.perf_counter()
        pool_review_debug: dict[str, Any] = {
            "candidate_count": len(deduped_candidates),
            "kept_candidate_count": 0,
            "dropped_candidate_count": 0,
            "max_pool_rank": 0,
            "latency_ms": 0.0,
            "prompt_chars": 0,
            "prompt_token_estimate": 0,
            "payload_chars": 0,
            "response_chars": 0,
        }
        pooled = None
        if pool_review_candidates:
            pooled, pool_review_debug = run_candidate_pool_review_with_debug(
                candidates=pool_review_candidates,
                llm_client=self.llm_client,
                model=self.flash_model,
                max_output_tokens=self.config.phase4_pool_max_output_tokens,
            )
        if pool_review_candidates:
            logger.info("[phase4] pooled candidate review done in %.1f s", time.perf_counter() - pool_review_started)
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="phase4",
            event="pooled_review_diagnostics",
            attempt=attempt,
            status="success" if pool_review_candidates else "skipped",
            **pool_review_debug,
            **pool_budget_debug,
        )
        phase4_substeps.append(
            self._make_phase_substep_record(
                run_id=paths.run_id,
                phase_name="phase4",
                step_name="pooled_review",
                step_key="pooled_review",
                status="succeeded" if pool_review_candidates else "skipped",
                duration_ms=float(pool_review_debug.get("latency_ms") or 0.0),
                metadata={**pool_review_debug, **pool_budget_debug},
            )
        )

        final_candidates: list[ClipCandidate] = []
        if pooled:
            candidate_by_id = {
                (candidate.clip_id or f"cand_tmp_{idx:03d}"): candidate
                for idx, candidate in enumerate(pool_review_candidates, start=1)
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
            max_hops=self.config.signals.max_hops,
            time_window_ms=self.config.signals.time_window_ms,
            fail_fast=self.config.signals.llm_fail_fast,
            signal_event_logger=self._build_signal_event_logger(run_id=run_id, job_id=job_id, attempt=attempt),
            max_concurrent=self.config.signals.max_concurrent,
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
            attribution_started_at = datetime.now(UTC)
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
            )
            try:
                explanation = explain_candidate_attribution_with_llm(
                    llm_client=self.llm_client,
                    model=self.config.signals.llm.model_11,
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
                latency_ms=duration_ms,
            )
            phase4_substeps.append(
                self._make_phase_substep_record(
                    run_id=paths.run_id,
                    phase_name="signals",
                    step_name="candidate_attribution_explanation",
                    step_key=str(candidate.clip_id or f"candidate_{idx:03d}"),
                    status="succeeded",
                    started_at=attribution_started_at,
                    ended_at=datetime.now(UTC),
                    duration_ms=duration_ms,
                    metadata={
                        "callpoint_id": "11",
                        "candidate_index": idx,
                        "clip_id": candidate.clip_id,
                    },
                )
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
        self._write_phase_substeps(run_id=paths.run_id, substeps=phase4_substeps)

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
            paths.candidates_dir / "pool_review_diagnostics.json",
            pool_review_debug,
        )
        self._save_debug_json(
            paths.candidates_dir / "subgraph_review_diagnostics.json",
            {
                "run_id": run_id,
                "reviewed_subgraph_count": len(subgraph_review_diagnostics),
                "rejected_subgraph_count": reject_all_count,
                "invalid_structured_output_count": invalid_structured_output_count,
                "reviewed_candidate_count": total_reviewed_candidates,
                "latency_ms": {
                    "avg": _diag_mean("latency_ms"),
                    "p95": _diag_p95("latency_ms"),
                    "max": _diag_max("latency_ms"),
                },
                "node_count": {
                    "avg": _diag_mean("node_count"),
                    "max": _diag_max("node_count"),
                },
                "span_ms": {
                    "avg": _diag_mean("span_ms"),
                    "max": _diag_max("span_ms"),
                },
                "prompt_chars": {
                    "avg": _diag_mean("prompt_chars"),
                    "p95": _diag_p95("prompt_chars"),
                },
                "prompt_token_estimate": {
                    "avg": _diag_mean("prompt_token_estimate"),
                    "max": _diag_max("prompt_token_estimate"),
                },
                "per_subgraph": [
                    {
                        "subgraph_id": str(item.get("subgraph_id") or ""),
                        "seed_node_id": str(item.get("seed_node_id") or ""),
                        **dict(item.get("diagnostics") or {}),
                    }
                    for item in subgraph_debug
                    if isinstance(item, dict)
                ],
            },
        )
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
                "subgraph_reviewed_candidate_count": total_reviewed_candidates,
                "subgraph_rejected_count": reject_all_count,
                "subgraph_invalid_structured_output_count": invalid_structured_output_count,
                "subgraph_review_avg_latency_ms": _diag_mean("latency_ms"),
                "subgraph_review_p95_latency_ms": _diag_p95("latency_ms"),
                "subgraph_review_max_latency_ms": _diag_max("latency_ms"),
                "subgraph_review_avg_node_count": _diag_mean("node_count"),
                "subgraph_review_max_node_count": _diag_max("node_count"),
                "subgraph_review_avg_span_ms": _diag_mean("span_ms"),
                "subgraph_review_max_span_ms": _diag_max("span_ms"),
                "subgraph_review_avg_prompt_chars": _diag_mean("prompt_chars"),
                "subgraph_review_p95_prompt_chars": _diag_p95("prompt_chars"),
                "subgraph_review_avg_prompt_token_estimate": _diag_mean("prompt_token_estimate"),
                "subgraph_review_max_prompt_token_estimate": _diag_max("prompt_token_estimate"),
                "pooled_review_candidate_count": int(pool_review_debug.get("candidate_count") or 0),
                "pooled_review_kept_candidate_count": int(pool_review_debug.get("kept_candidate_count") or 0),
                "pooled_review_dropped_candidate_count": int(pool_review_debug.get("dropped_candidate_count") or 0),
                "pooled_review_latency_ms": float(pool_review_debug.get("latency_ms") or 0.0),
                "pooled_review_prompt_chars": int(pool_review_debug.get("prompt_chars") or 0),
                "pooled_review_prompt_token_estimate": int(pool_review_debug.get("prompt_token_estimate") or 0),
                "pooled_review_payload_chars": int(pool_review_debug.get("payload_chars") or 0),
                "pooled_review_response_chars": int(pool_review_debug.get("response_chars") or 0),
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
            self._cancel_signal_future(
                run_id=run_id,
                job_id=job_id,
                attempt=attempt,
                signal_name="trends",
                future=trends_future,
            )
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

    def _build_signal_event_logger(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
    ) -> Callable[..., None]:
        def _logger(**payload: Any) -> None:
            event = str(payload.pop("event", "signal_event"))
            status = str(payload.pop("status", "info"))
            self._emit_log(
                run_id=run_id,
                job_id=job_id,
                phase="signals",
                event=event,
                attempt=attempt,
                status=status,
                **payload,
            )

        return _logger

    def _cancel_signal_future(
        self,
        *,
        run_id: str,
        job_id: str | None,
        attempt: int,
        signal_name: str,
        future: Any | None,
    ) -> None:
        if future is None:
            return
        cancelled = False
        cancel_error: Exception | None = None
        if hasattr(future, "cancel"):
            try:
                cancelled = bool(future.cancel())
            except Exception as exc:  # pragma: no cover - defensive
                cancel_error = exc
        self._emit_log(
            run_id=run_id,
            job_id=job_id,
            phase="signals",
            event="signal_future_cancel_requested",
            attempt=attempt,
            status="cancel_requested",
            signal_name=signal_name,
            cancelled=cancelled,
            cancel_error=str(cancel_error) if cancel_error is not None else None,
        )

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

    def _make_phase_substep_record(
        self,
        *,
        run_id: str,
        phase_name: str,
        step_name: str,
        step_key: str,
        status: str,
        duration_ms: float | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        error_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PhaseSubstepRecord:
        record_started_at = started_at or datetime.now(UTC)
        if ended_at is not None:
            record_ended_at = ended_at
        elif duration_ms is not None:
            record_ended_at = record_started_at + timedelta(milliseconds=max(0.0, duration_ms))
        else:
            record_ended_at = record_started_at
        return PhaseSubstepRecord(
            run_id=run_id,
            phase_name=phase_name,
            step_name=step_name,
            step_key=str(step_key)[:256],
            status=status,
            started_at=record_started_at,
            ended_at=record_ended_at,
            duration_ms=duration_ms,
            error_payload=error_payload,
            query_version=self.query_version,
            metadata=metadata or {},
        )

    def _write_phase_substeps(
        self,
        *,
        run_id: str,
        substeps: list[PhaseSubstepRecord],
    ) -> None:
        if not substeps or self.repository is None:
            return
        write_phase_substeps = getattr(self.repository, "write_phase_substeps", None)
        if not callable(write_phase_substeps):
            return
        write_phase_substeps(run_id=run_id, substeps=substeps)

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
