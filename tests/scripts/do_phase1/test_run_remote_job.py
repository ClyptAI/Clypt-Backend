import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "do_phase1"
    / "run_remote_job.py"
)


@pytest.fixture()
def subject_module():
    spec = importlib.util.spec_from_file_location("run_remote_job", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_youtube_video_id_supports_common_url_forms(subject_module):
    assert subject_module.extract_youtube_video_id("https://www.youtube.com/watch?v=2jW9lmlfiKQ") == "2jW9lmlfiKQ"
    assert subject_module.extract_youtube_video_id("https://youtu.be/2jW9lmlfiKQ?t=12") == "2jW9lmlfiKQ"
    assert subject_module.extract_youtube_video_id("https://www.youtube.com/shorts/2jW9lmlfiKQ") == "2jW9lmlfiKQ"


def test_extract_youtube_video_id_rejects_non_youtube_urls(subject_module):
    with pytest.raises(ValueError):
        subject_module.extract_youtube_video_id("https://example.com/watch?v=2jW9lmlfiKQ")


def test_relay_artifact_paths_are_derived_from_video_id(subject_module):
    video_id = "2jW9lmlfiKQ"
    assert subject_module.relay_filename(video_id) == "2jW9lmlfiKQ.mp4"
    assert subject_module.remote_relay_path(video_id) == "/opt/clypt-phase1/relay/2jW9lmlfiKQ.mp4"
    assert subject_module.remote_relay_source_url(video_id) == "http://127.0.0.1:8091/2jW9lmlfiKQ.mp4"
    assert subject_module.remote_job_log_path("job_abc") == "/var/lib/clypt/do_phase1_service/workdir/logs/job_abc.log"
