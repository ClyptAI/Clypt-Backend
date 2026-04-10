from __future__ import annotations

from .cluster import cluster_signals
from .comments_client import (
    YouTubeCommentsClient,
    collapse_same_author_spam,
    resolve_youtube_video_id,
    target_top_threads,
    to_external_signals_from_threads,
)
from .contracts import (
    CandidateSignalLink,
    ClusterType,
    ExternalSignal,
    ExternalSignalCluster,
    LinkType,
    NodeSignalLink,
    PromptSourceType,
    SignalPipelineOutput,
    SignalPromptSpec,
    SignalType,
    SourcePlatform,
    SubgraphProvenance,
)
from .linking import build_node_signal_links
from .llm_runtime import (
    adjudicate_trend_relevance_with_llm,
    classify_comment_with_llm,
    consolidate_thread_with_llm,
    explain_candidate_attribution_with_llm,
    generate_cluster_prompt_with_llm,
    resolve_cluster_span_with_llm,
    synthesize_trend_queries_with_llm,
)
from .runtime import (
    collect_signal_outputs,
    merge_signal_outputs,
    start_comments_future,
    start_trends_future,
    wait_enabled_signal_futures,
)
from .scoring import SignalScoringResult, apply_signal_scoring
from .trends_client import TrendSpygClient, to_external_signals_from_trends

__all__ = [
    "CandidateSignalLink",
    "ClusterType",
    "ExternalSignal",
    "ExternalSignalCluster",
    "LinkType",
    "NodeSignalLink",
    "PromptSourceType",
    "SignalPipelineOutput",
    "SignalPromptSpec",
    "SignalScoringResult",
    "SignalType",
    "SourcePlatform",
    "SubgraphProvenance",
    "TrendSpygClient",
    "YouTubeCommentsClient",
    "adjudicate_trend_relevance_with_llm",
    "apply_signal_scoring",
    "build_node_signal_links",
    "classify_comment_with_llm",
    "collapse_same_author_spam",
    "collect_signal_outputs",
    "consolidate_thread_with_llm",
    "explain_candidate_attribution_with_llm",
    "generate_cluster_prompt_with_llm",
    "merge_signal_outputs",
    "resolve_cluster_span_with_llm",
    "resolve_youtube_video_id",
    "start_comments_future",
    "start_trends_future",
    "synthesize_trend_queries_with_llm",
    "target_top_threads",
    "to_external_signals_from_threads",
    "to_external_signals_from_trends",
    "wait_enabled_signal_futures",
    "cluster_signals",
]
