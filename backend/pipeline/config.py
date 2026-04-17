from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


_REMOVED_GLOBAL_CONCURRENCY_ENV = "CLYPT_GEMINI_MAX_CONCURRENT"
_RENAMED_CONCURRENCY_ENVS = {
    # old name -> new name (renamed for suffix/stage consistency)
    "CLYPT_PHASE2_MAX_CONCURRENT": "CLYPT_PHASE2_MERGE_MAX_CONCURRENT",
    "CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY": "CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT",
}
_REMOVED_THINKING_ENVS = (
    "CLYPT_PHASE2_MERGE_THINKING_LEVEL",
    "CLYPT_PHASE2_BOUNDARY_THINKING_LEVEL",
    "CLYPT_PHASE3_LOCAL_THINKING_LEVEL",
    "CLYPT_PHASE3_LONG_RANGE_THINKING_LEVEL",
    "CLYPT_PHASE4_META_THINKING_LEVEL",
    "CLYPT_PHASE4_SUBGRAPH_THINKING_LEVEL",
    "CLYPT_PHASE4_POOL_THINKING_LEVEL",
    "CLYPT_SIGNAL_LLM_THINKING_1",
    "CLYPT_SIGNAL_LLM_THINKING_2",
    "CLYPT_SIGNAL_LLM_THINKING_3",
    "CLYPT_SIGNAL_LLM_THINKING_5",
    "CLYPT_SIGNAL_LLM_THINKING_9",
    "CLYPT_SIGNAL_LLM_THINKING_10",
    "CLYPT_SIGNAL_LLM_THINKING_11",
)


def _raise_if_removed_global_concurrency_env_present() -> None:
    if os.getenv(_REMOVED_GLOBAL_CONCURRENCY_ENV):
        raise ValueError(
            f"{_REMOVED_GLOBAL_CONCURRENCY_ENV} has been removed; "
            "set explicit per-phase concurrency env vars instead."
        )
    for old, new in _RENAMED_CONCURRENCY_ENVS.items():
        if os.getenv(old) is not None:
            raise ValueError(
                f"{old} has been renamed to {new}; update your env file."
            )


def _raise_if_removed_thinking_env_present() -> None:
    for name in _REMOVED_THINKING_ENVS:
        if os.getenv(name) is not None:
            raise ValueError(
                f"{name} has been removed because the Phase 2-4 local Qwen path "
                "always runs with thinking disabled."
            )


@dataclass(slots=True)
class Phase4BudgetConfig:
    max_total_prompts: int = 12
    max_subgraphs_per_run: int = 24
    max_final_review_calls: int = 1


@dataclass(slots=True)
class Phase4SubgraphConfig:
    max_duration_s: int = 45
    max_node_count: int = 10
    max_hop_depth: int = 2
    max_branching_factor: int = 3
    min_expansion_score: float = 3.0
    seed_top_k_per_prompt: int = 5
    subgraph_overlap_dedupe_threshold: float = field(
        default_factory=lambda: float(os.getenv("CLYPT_PHASE4_SUBGRAPH_OVERLAP_DEDUPE_THRESHOLD") or "0.70")
    )
    candidate_node_overlap_threshold: float = 0.70
    candidate_span_iou_threshold: float = 0.70


@dataclass(slots=True)
class SignalLLMCallConfig:
    model_1: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_1") or "Qwen/Qwen3.6-35B-A3B")
    model_2: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_2") or "Qwen/Qwen3.6-35B-A3B")
    model_3: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_3") or "Qwen/Qwen3.6-35B-A3B")
    model_5: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_5") or "Qwen/Qwen3.6-35B-A3B")
    model_9: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_9") or "Qwen/Qwen3.6-35B-A3B")
    model_10: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_10") or "Qwen/Qwen3.6-35B-A3B")
    model_11: str = field(default_factory=lambda: os.getenv("CLYPT_SIGNAL_LLM_MODEL_11") or "Qwen/Qwen3.6-35B-A3B")


@dataclass(slots=True)
class SignalConfig:
    mode: str = field(default_factory=lambda: (os.getenv("CLYPT_SIGNAL_MODE") or "augment").strip().lower())
    fail_fast: bool = field(default_factory=lambda: (os.getenv("CLYPT_SIGNAL_FAIL_FAST") or "1") == "1")
    llm_fail_fast: bool = field(default_factory=lambda: (os.getenv("CLYPT_SIGNAL_LLM_FAIL_FAST") or "1") == "1")
    enable_comment_signals: bool = field(default_factory=lambda: (os.getenv("CLYPT_ENABLE_COMMENT_SIGNALS") or "0") == "1")
    enable_trend_signals: bool = field(default_factory=lambda: (os.getenv("CLYPT_ENABLE_TREND_SIGNALS") or "0") == "1")
    youtube_api_key: str | None = field(default_factory=lambda: (os.getenv("CLYPT_YOUTUBE_DATA_API_KEY") or os.getenv("YOUTUBE_API_KEY")))
    youtube_base_url: str = field(default_factory=lambda: os.getenv("CLYPT_YOUTUBE_DATA_API_BASE_URL") or "https://www.googleapis.com/youtube/v3")
    max_comment_pages: int = field(default_factory=lambda: int(os.getenv("CLYPT_COMMENT_MAX_PAGES") or "5"))
    comment_order: str = field(default_factory=lambda: (os.getenv("CLYPT_COMMENT_ORDER") or "relevance").strip().lower())
    comment_top_threads_min: int = field(default_factory=lambda: int(os.getenv("CLYPT_COMMENT_TOP_THREADS_MIN") or "15"))
    comment_top_threads_max: int = field(default_factory=lambda: int(os.getenv("CLYPT_COMMENT_TOP_THREADS_MAX") or "40"))
    comment_max_replies_per_thread: int = field(default_factory=lambda: int(os.getenv("CLYPT_COMMENT_MAX_REPLIES_PER_THREAD") or "0"))
    comment_cluster_sim_threshold: float = field(default_factory=lambda: float(os.getenv("CLYPT_COMMENT_CLUSTER_SIM_THRESHOLD") or "0.82"))
    trend_max_items: int = field(default_factory=lambda: int(os.getenv("CLYPT_TREND_MAX_ITEMS") or "40"))
    trend_relevance_threshold: float = field(default_factory=lambda: float(os.getenv("CLYPT_TREND_RELEVANCE_THRESHOLD") or "0.6"))
    max_hops: int = field(default_factory=lambda: int(os.getenv("CLYPT_SIGNAL_MAX_HOPS") or "2"))
    time_window_ms: int = field(default_factory=lambda: int(os.getenv("CLYPT_SIGNAL_TIME_WINDOW_MS") or "30000"))
    epsilon: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_EPSILON") or "1e-6"))
    cluster_cap: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_CLUSTER_CAP") or "0.12"))
    total_cap: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_TOTAL_CAP") or "0.20"))
    agreement_cap: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_AGREEMENT_CAP") or "0.10"))
    meaningful_min_cluster_contrib: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_MEANINGFUL_MIN_CLUSTER_CONTRIB") or "0.04"))
    meaningful_min_source_coverage: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_MEANINGFUL_MIN_SOURCE_COVERAGE") or "0.15"))
    agreement_bonus_tier1: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_AGREEMENT_BONUS_TIER1") or "0.04"))
    agreement_bonus_tier2: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_AGREEMENT_BONUS_TIER2") or "0.07"))
    engagement_top_like_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_ENGAGEMENT_TOP_LIKE_WEIGHT") or "0.65"))
    engagement_top_reply_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_ENGAGEMENT_TOP_REPLY_WEIGHT") or "0.35"))
    engagement_reply_like_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_ENGAGEMENT_REPLY_LIKE_WEIGHT") or "0.85"))
    engagement_reply_parent_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_ENGAGEMENT_REPLY_PARENT_WEIGHT") or "0.15"))
    cluster_mean_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_CLUSTER_MEAN_WEIGHT") or "0.45"))
    cluster_max_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_CLUSTER_MAX_WEIGHT") or "0.25"))
    cluster_freq_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_CLUSTER_FREQ_WEIGHT") or "0.30"))
    cluster_freq_ref: int = field(default_factory=lambda: int(os.getenv("CLYPT_SIGNAL_CLUSTER_FREQ_REF") or "30"))
    hop_decay_1: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_HOP_DECAY_1") or "0.75"))
    hop_decay_2: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_HOP_DECAY_2") or "0.55"))
    coverage_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_COVERAGE_WEIGHT") or "0.30"))
    direct_ratio_weight: float = field(default_factory=lambda: float(os.getenv("CLYPT_SIGNAL_DIRECT_RATIO_WEIGHT") or "0.15"))
    max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_SIGNAL_MAX_CONCURRENT") or "8")
    )
    llm: SignalLLMCallConfig = field(default_factory=SignalLLMCallConfig)

    def __post_init__(self) -> None:
        _raise_if_removed_global_concurrency_env_present()
        _raise_if_removed_thinking_env_present()
        if self.comment_order != "relevance":
            raise ValueError("CLYPT_COMMENT_ORDER currently supports only 'relevance'")


@dataclass(slots=True)
class V31Config:
    output_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_V31_OUTPUT_ROOT", "backend/outputs/v3_1")
        )
    )
    phase4_budget: Phase4BudgetConfig = field(default_factory=Phase4BudgetConfig)
    phase4_subgraphs: Phase4SubgraphConfig = field(default_factory=Phase4SubgraphConfig)
    phase2_target_batch_count: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE2_TARGET_BATCH_COUNT") or "5")
    )
    phase2_max_turns_per_batch: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE2_MAX_TURNS_PER_BATCH") or "25")
    )
    phase2_merge_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE2_MERGE_MAX_CONCURRENT") or "16")
    )
    phase2_boundary_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT") or "16")
    )
    phase2_merge_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS") or "8192"
        )
    )
    phase2_boundary_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS") or "8192"
        )
    )
    phase3_target_batch_count: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_TARGET_BATCH_COUNT") or "3")
    )
    phase3_max_nodes_per_batch: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_MAX_NODES_PER_BATCH") or "24")
    )
    phase3_local_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE3_LOCAL_MAX_OUTPUT_TOKENS") or "4096"
        )
    )
    phase3_long_range_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE3_LONG_RANGE_MAX_OUTPUT_TOKENS") or "4096"
        )
    )
    phase3_long_range_top_k: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_LONG_RANGE_TOP_K") or "2")
    )
    phase3_long_range_pairs_per_shard: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD") or "24")
    )
    phase3_long_range_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT") or "24")
    )
    phase3_local_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_LOCAL_MAX_CONCURRENT") or "24")
    )
    phase4_meta_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS") or "2048"
        )
    )
    phase4_subgraph_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE4_SUBGRAPH_MAX_OUTPUT_TOKENS") or "4096"
        )
    )
    phase4_pool_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS") or "2048"
        )
    )
    phase4_subgraph_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT") or "16")
    )
    signals: SignalConfig = field(default_factory=SignalConfig)

    def __post_init__(self) -> None:
        _raise_if_removed_global_concurrency_env_present()
        _raise_if_removed_thinking_env_present()


def get_v31_config() -> V31Config:
    return V31Config()


__all__ = [
    "Phase4BudgetConfig",
    "Phase4SubgraphConfig",
    "SignalConfig",
    "SignalLLMCallConfig",
    "V31Config",
    "get_v31_config",
]
