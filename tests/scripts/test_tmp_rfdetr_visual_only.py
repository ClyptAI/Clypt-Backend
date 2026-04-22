from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_tmp_visual_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "tmp_rfdetr_visual_only.py"
    spec = importlib.util.spec_from_file_location("tmp_rfdetr_visual_only", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_run_visual_once_uses_shot_changes_payload(monkeypatch, tmp_path: Path):
    module = _load_tmp_visual_script()

    fake_workspace = SimpleNamespace(video_path=tmp_path / "workspace-video.mp4")

    class FakeExtractor:
        def __init__(self, *, visual_config):
            self.visual_config = visual_config

        def extract(self, *, video_path, workspace):
            assert video_path == (tmp_path / "source.mp4")
            assert workspace is fake_workspace
            return {
                "tracks": [{"track_id": "track_1"}, {"track_id": "track_2"}],
                "person_detections": [{"track_id": "track_1"}],
                "shot_changes": [
                    {"start_time_ms": 0, "end_time_ms": 1000},
                    {"start_time_ms": 1000, "end_time_ms": 2000},
                ],
                "tracking_metrics": {
                    "shot_count": 2,
                    "tracker_backend": "rfdetr_nano_bytetrack",
                },
            }

    monkeypatch.setattr(module.Phase1Workspace, "create", lambda root, run_id: fake_workspace)
    monkeypatch.setattr(module, "V31VisualExtractor", FakeExtractor)

    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    config = module.VisualPipelineConfig.from_env()

    summary = module._run_visual_once(
        video_path=video_path,
        work_root=tmp_path / "work",
        run_id="tmp_run",
        config=config,
    )

    assert summary["track_rows"] == 2
    assert summary["person_segments"] == 1
    assert summary["shot_count"] == 2
    assert summary["tracking_metrics"]["tracker_backend"] == "rfdetr_nano_bytetrack"
