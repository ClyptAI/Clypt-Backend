from __future__ import annotations

from types import SimpleNamespace

import pytest


def _configured_spanner() -> SimpleNamespace:
    """SimpleNamespace mirroring SpannerSettings with is_configured=True."""
    return SimpleNamespace(
        project="test-project",
        instance="clypt-phase14",
        database="clypt_phase14",
        is_configured=True,
    )


def test_build_phase14_repository_bootstraps_schema(monkeypatch) -> None:
    from backend.phase1_runtime import factory as module

    bootstrap_calls: list[str] = []

    class _FakeRepository:
        def bootstrap_schema(self) -> None:
            bootstrap_calls.append("bootstrapped")

    monkeypatch.setattr(
        module.SpannerPhase14Repository,
        "from_settings",
        lambda settings: _FakeRepository(),
    )

    repository = module._build_phase14_repository(
        settings=SimpleNamespace(spanner=_configured_spanner())
    )

    assert repository is not None
    assert bootstrap_calls == ["bootstrapped"]


def test_build_phase14_repository_returns_none_when_not_configured() -> None:
    """Unconfigured Spanner is the only sanctioned path to None — opt-out only."""
    from backend.phase1_runtime import factory as module

    unconfigured = SimpleNamespace(
        project="",
        instance="clypt-phase14",
        database="clypt_phase14",
        is_configured=False,
    )

    repository = module._build_phase14_repository(
        settings=SimpleNamespace(spanner=unconfigured)
    )

    assert repository is None


def test_build_phase14_repository_propagates_init_failures(monkeypatch) -> None:
    """Rotated creds / DDL drift must fail fast, not return None."""
    from backend.phase1_runtime import factory as module

    class _BoomError(RuntimeError):
        pass

    def _raise(*, settings):  # noqa: ARG001
        raise _BoomError("simulated Spanner init failure")

    monkeypatch.setattr(
        module.SpannerPhase14Repository,
        "from_settings",
        _raise,
    )

    with pytest.raises(_BoomError, match="simulated Spanner init failure"):
        module._build_phase14_repository(
            settings=SimpleNamespace(spanner=_configured_spanner())
        )
