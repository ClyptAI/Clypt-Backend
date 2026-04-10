from __future__ import annotations

import pytest

from backend.repository.models import SubgraphProvenanceRecord


def test_subgraph_provenance_requires_seed_source_set() -> None:
    with pytest.raises(ValueError, match="seed_source_set must be non-empty"):
        SubgraphProvenanceRecord(
            run_id="run_001",
            subgraph_id="sg_001",
            seed_source_set=[],
            seed_prompt_ids=["general_prompt_001"],
            source_cluster_ids=[],
            support_summary={},
            canonical_selected=True,
            metadata={},
        )


def test_subgraph_provenance_accepts_multi_source_payload() -> None:
    record = SubgraphProvenanceRecord(
        run_id="run_001",
        subgraph_id="sg_001",
        seed_source_set=["general", "comment", "trend"],
        seed_prompt_ids=["general_prompt_001", "comment_prompt_001", "trend_prompt_001"],
        source_cluster_ids=["comment_cluster_001", "trend_cluster_001"],
        support_summary={"source_type_counts": {"general": 1, "comment": 1, "trend": 1}},
        canonical_selected=True,
        dedupe_overlap_ratio=0.74,
        selection_reason="retained_after_dedupe",
        metadata={},
    )

    assert record.seed_source_set == ["general", "comment", "trend"]
    assert record.support_summary["source_type_counts"]["comment"] == 1
