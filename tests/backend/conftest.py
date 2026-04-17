from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_remote_host_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-populate the required RTX 6000 Ada remote-host envs for backend tests.

    VibeVoice ASR and Phase 2 node-media prep run on the RTX 6000 Ada host;
    NFA/emotion/YAMNet run in-process on the H200. ``load_provider_settings``
    fails fast when the VibeVoice ASR service URL or token is missing, which
    would break any test that exercises the loader. Setting fake defaults
    here keeps the fail-fast semantic while unblocking tests that don't
    care about those fields. Individual tests that want to exercise the
    failure paths can ``monkeypatch.delenv`` these vars.
    """

    monkeypatch.setenv(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL", "http://test-vibevoice-asr:9100"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN",
        "test-vibevoice-asr-token",
    )
    monkeypatch.setenv(
        "CLYPT_PHASE24_NODE_MEDIA_PREP_URL", "http://test-vibevoice-asr:9100"
    )
    monkeypatch.setenv(
        "CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN", "test-node-media-prep-token"
    )
