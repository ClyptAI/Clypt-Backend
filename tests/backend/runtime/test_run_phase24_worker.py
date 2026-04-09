from __future__ import annotations


def test_run_phase24_worker_uses_explicit_host_and_port(monkeypatch):
    from backend.runtime import run_phase24_worker

    captured: dict[str, object] = {}
    sentinel_app = object()

    def fake_create_app():
        return sentinel_app

    def fake_run(app, *, host, port, reload):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    monkeypatch.setattr(run_phase24_worker, "create_app", fake_create_app)
    monkeypatch.setattr(run_phase24_worker.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_phase24_worker.py",
            "--host",
            "127.0.0.1",
            "--port",
            "9091",
        ],
    )

    exit_code = run_phase24_worker.main()

    assert exit_code == 0
    assert captured == {
        "app": sentinel_app,
        "host": "127.0.0.1",
        "port": 9091,
        "reload": False,
    }
