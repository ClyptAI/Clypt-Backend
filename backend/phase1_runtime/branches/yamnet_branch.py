from __future__ import annotations

from typing import Any

from ..models import Phase1Workspace


def run_yamnet_branch(
    *,
    workspace: Phase1Workspace,
    yamnet_provider: Any,
) -> dict[str, Any]:
    yamnet_payload = yamnet_provider.run(audio_path=workspace.audio_path)
    return {"yamnet_payload": yamnet_payload}


__all__ = ["run_yamnet_branch"]
