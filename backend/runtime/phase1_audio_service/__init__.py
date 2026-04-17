"""RTX 6000 Ada–side VibeVoice ASR + node-media-prep host package.

This package is imported **only** on the RTX box. It houses:

* ``app`` — the FastAPI application exposing ``/tasks/vibevoice-asr`` and
  ``/tasks/node-media-prep``.
* ``deps`` — singleton VibeVoice vLLM provider, GCS client, scratch root,
  and bearer-token plumbing.
* ``node_media_prep`` — the ffmpeg/NVENC node-clip extractor (Phase 2).

NFA/emotion2vec+/YAMNet are not here; they run in-process on the H200.
See docs/ERROR_LOG.md 2026-04-17.

The H200 orchestrator does **not** import this package; it interacts with the
box purely through the HTTP clients in
``backend.providers.audio_host_client`` (renamed to ``RemoteVibeVoiceAsrClient``)
and ``backend.providers.node_media_prep_client``.
"""

from __future__ import annotations

__all__: list[str] = []
