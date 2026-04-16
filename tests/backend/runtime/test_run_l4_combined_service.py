from __future__ import annotations


def test_run_l4_combined_service_bootstraps_vibevoice_before_uvicorn(monkeypatch):
    from backend.runtime import run_l4_combined_service

    captured: dict[str, object] = {}
    events: list[str] = []
    sentinel_app = object()
    sentinel_process = object()

    def fake_create_app():
        events.append("create_app")
        return sentinel_app

    def fake_launch():
        events.append("launch_vibevoice")
        return sentinel_process

    def fake_stop(process):
        events.append("stop_vibevoice")
        captured["process"] = process

    def fake_run(app, *, host, port, reload):
        events.append("uvicorn_run")
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    monkeypatch.setattr(run_l4_combined_service, "create_app", fake_create_app, raising=False)
    monkeypatch.setattr(run_l4_combined_service, "launch_vibevoice_server", fake_launch, raising=False)
    monkeypatch.setattr(run_l4_combined_service, "stop_vibevoice_server", fake_stop, raising=False)
    monkeypatch.setattr(run_l4_combined_service.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_l4_combined_service.py",
            "--host",
            "127.0.0.1",
            "--port",
            "9099",
        ],
    )

    exit_code = run_l4_combined_service.main()

    assert exit_code == 0
    assert events == ["launch_vibevoice", "create_app", "uvicorn_run", "stop_vibevoice"]
    assert captured == {
        "app": sentinel_app,
        "host": "127.0.0.1",
        "port": 9099,
        "reload": False,
        "process": sentinel_process,
    }
