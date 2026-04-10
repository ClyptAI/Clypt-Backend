from __future__ import annotations

import pytest

from backend.repository.models import PromptSourceLinkRecord


def test_prompt_source_links_general_requires_no_cluster_reference() -> None:
    with pytest.raises(ValueError, match="general prompt sources must not reference a source cluster"):
        PromptSourceLinkRecord(
            run_id="run_001",
            prompt_id="general_prompt_001",
            prompt_source_type="general",
            source_cluster_id="comment_cluster_001",
            source_cluster_type="comment",
            metadata={},
        )


def test_prompt_source_links_comment_requires_matching_cluster_type() -> None:
    with pytest.raises(ValueError, match="prompt_source_type must match source_cluster_type"):
        PromptSourceLinkRecord(
            run_id="run_001",
            prompt_id="comment_prompt_001",
            prompt_source_type="comment",
            source_cluster_id="trend_cluster_001",
            source_cluster_type="trend",
            metadata={},
        )
