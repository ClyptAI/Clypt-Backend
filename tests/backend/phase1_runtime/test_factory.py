from __future__ import annotations

from types import SimpleNamespace


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

    repository = module._build_phase14_repository(settings=SimpleNamespace(spanner=SimpleNamespace()))

    assert repository is not None
    assert bootstrap_calls == ["bootstrapped"]
