from __future__ import annotations

from typing import Any

from ..models import Phase1Workspace


def run_visual_branch(
    *,
    workspace: Phase1Workspace,
    visual_extractor: Any,
) -> dict[str, Any]:
    phase1_visual = visual_extractor.extract(
        video_path=workspace.video_path,
        workspace=workspace,
    )
    return {"phase1_visual": phase1_visual}


__all__ = ["run_visual_branch"]
