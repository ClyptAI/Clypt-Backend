"""RTX 6000 Ada–side Phase 1 audio host package.

This package is imported **only** on the RTX audio box. It houses:

* ``audio_chain`` — the pure-Python VibeVoice → NFA → emotion2vec+ → YAMNet
  pipeline (previously ``_run_audio_chain`` in ``backend.phase1_runtime.extract``).
* ``node_media_prep`` — the ffmpeg/NVENC node-clip extractor (Phase 2).
* ``app`` — the FastAPI application wiring the two endpoints together.

The H200 orchestrator does **not** import this package; it interacts with the
box purely through the HTTP clients in ``backend.providers.audio_host_client``
and ``backend.providers.node_media_prep_client``.
"""

from __future__ import annotations

__all__: list[str] = []
