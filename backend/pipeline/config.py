from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


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
    subgraph_overlap_dedupe_threshold: float = 0.70
    candidate_node_overlap_threshold: float = 0.70
    candidate_span_iou_threshold: float = 0.70


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
    phase3_target_batch_count: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_TARGET_BATCH_COUNT") or "5")
    )
    phase3_max_nodes_per_batch: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_PHASE3_MAX_NODES_PER_BATCH") or "15")
    )
    gemini_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("CLYPT_GEMINI_MAX_CONCURRENT") or "5")
    )


def get_v31_config() -> V31Config:
    return V31Config()


__all__ = [
    "Phase4BudgetConfig",
    "Phase4SubgraphConfig",
    "V31Config",
    "get_v31_config",
]
