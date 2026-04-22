"""Reserved Modal entrypoint for future Phase 6 render/export work."""

from __future__ import annotations

import modal

app = modal.App("clypt-render-video")


@app.function(gpu="L40S", min_containers=1, timeout=60 * 60)
def render_video(*_args, **_kwargs):
    raise NotImplementedError("Phase 6 render/export wiring lands later.")
