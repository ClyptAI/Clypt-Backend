from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_remote_host_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-populate required remote service envs for backend tests."""

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key")
    monkeypatch.setenv("CLYPT_PHASE1_AUDIO_BACKEND", "elevenlabs_scribe_v2")
    monkeypatch.setenv(
        "CLYPT_PHASE24_NODE_MEDIA_PREP_URL", "http://test-node-media-prep:9100"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN", "test-node-media-prep-token"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE1_VISUAL_SERVICE_URL", "http://test-phase1-visual:9200"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN", "test-phase1-visual-token"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE24_DISPATCH_URL", "http://test-phase26-dispatch:9300"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE24_DISPATCH_AUTH_TOKEN", "test-phase26-dispatch-token"
    )
