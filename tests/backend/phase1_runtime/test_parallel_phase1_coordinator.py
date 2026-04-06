from __future__ import annotations

from pathlib import Path

import pytest

from backend.phase1_runtime.branch_io import read_branch_request
from backend.phase1_runtime.branch_models import BranchKind, BranchResultEnvelope
from backend.phase1_runtime.models import Phase1Workspace


def _build_workspace(tmp_path: Path) -> Phase1Workspace:
    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")
    return workspace


class _FakeProcess:
    def __init__(self, *, branch: str, exit_codes: list[int | None], pid: int) -> None:
        self.branch = branch
        self._exit_codes = list(exit_codes)
        self.pid = pid
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        if self._exit_codes:
            return self._exit_codes.pop(0)
        return 0

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0


@pytest.fixture
def fake_branch_runner(monkeypatch):
    launched: list[dict] = []
    processes: dict[str, _FakeProcess] = {}
    behavior: dict[str, dict] = {
        "visual": {
            "exit_codes": [None, 0],
            "result": {
                "phase1_visual": {
                    "video_metadata": {"fps": 10.0},
                    "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                    "tracks": [{"track_id": "track_001"}],
                }
            },
        },
        "audio": {
            "exit_codes": [None, 0],
            "result": {
                "diarization_payload": {
                    "turns": [
                        {
                            "turn_id": "t_000001",
                            "speaker_id": "SPEAKER_0",
                            "start_ms": 0,
                            "end_ms": 300,
                            "transcript_text": "hello",
                            "word_ids": ["w_000001"],
                            "identification_match": None,
                        }
                    ],
                    "words": [
                        {
                            "word_id": "w_000001",
                            "text": "hello",
                            "start_ms": 0,
                            "end_ms": 300,
                            "speaker_id": "SPEAKER_0",
                        }
                    ],
                },
                "emotion2vec_payload": {
                    "segments": [
                        {
                            "turn_id": "t_000001",
                            "labels": ["neutral"],
                            "scores": [0.88],
                            "per_class_scores": {"neutral": 0.88},
                        }
                    ]
                },
            },
        },
        "yamnet": {
            "exit_codes": [None, 0],
            "result": {"yamnet_payload": {"events": []}},
        },
    }

    def configure(branch: str, **kwargs) -> None:
        behavior.setdefault(branch, {}).update(kwargs)

    def fake_popen(args, *, env=None, stdout=None, stderr=None, text=None):
        request_path = Path(args[-1])
        request = read_branch_request(request_path)
        branch = request.branch.value
        config = behavior[branch]
        launched.append(
            {
                "branch": branch,
                "args": list(args),
                "env": dict(env or {}),
                "request": request,
                "stdout_name": getattr(stdout, "name", None),
                "stdout": stdout,
            }
        )
        if config.get("popen_error") is not None:
            raise config["popen_error"]
        if config.get("write_raw_result") is not None:
            request_path.parent.joinpath("result.json").write_text(
                str(config["write_raw_result"]),
                encoding="utf-8",
            )
        elif config.get("skip_result_write") is not True and "result" in config:
            envelope = BranchResultEnvelope(
                branch=request.branch,
                ok=config.get("ok", True),
                result=config["result"] if config.get("ok", True) else None,
                error=config.get("error"),
            )
            request_path.parent.joinpath("result.json").write_text(
                envelope.model_dump_json(indent=2),
                encoding="utf-8",
            )
        process = _FakeProcess(
            branch=branch,
            exit_codes=config.get("exit_codes", [0]),
            pid=1000 + len(launched),
        )
        processes[branch] = process
        return process

    monkeypatch.setattr("backend.phase1_runtime.coordinator.subprocess.Popen", fake_popen)
    return {
        "launched": launched,
        "processes": processes,
        "configure": configure,
    }


def test_coordinator_launches_visual_audio_and_yamnet_branches(
    tmp_path: Path,
    fake_branch_runner,
    monkeypatch,
) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    workspace = _build_workspace(tmp_path)

    outputs = run_parallel_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        workspace=workspace,
        branch_timeout_s=30.0,
        poll_interval_s=0.0,
    )

    launched = fake_branch_runner["launched"]
    assert [item["branch"] for item in launched] == ["visual", "audio", "yamnet"]
    assert all(item["request"].run_id == "run_001" for item in launched)
    assert launched[2]["request"].branch is BranchKind.YAMNET
    assert launched[2]["env"].get("CUDA_VISIBLE_DEVICES") in (None, "")
    assert "CUDA_HOME" not in launched[2]["env"]
    assert "NVIDIA_VISIBLE_DEVICES" not in launched[2]["env"]
    assert "NVIDIA_DRIVER_CAPABILITIES" not in launched[2]["env"]
    assert outputs.phase1_visual["tracks"] == [{"track_id": "track_001"}]
    assert outputs.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert outputs.emotion2vec_payload["segments"][0]["labels"] == ["neutral"]
    assert outputs.yamnet_payload["events"] == []
    assert outputs.phase1_audio == {
        "source_audio": "https://youtube.com/watch?v=demo",
        "video_gcs_uri": "gs://bucket/source.mp4",
        "local_video_path": str(workspace.video_path),
        "local_audio_path": str(workspace.audio_path),
    }


def test_coordinator_kills_siblings_when_one_branch_fails(tmp_path: Path, fake_branch_runner) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    fake_branch_runner["configure"]("visual", exit_codes=[None, None, None, None])
    fake_branch_runner["configure"](
        "audio",
        exit_codes=[1],
        ok=False,
        error={"error_type": "RuntimeError", "error_message": "audio failed"},
    )
    fake_branch_runner["configure"]("yamnet", exit_codes=[None, None, None, None])

    workspace = _build_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="audio failed"):
        run_parallel_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            branch_timeout_s=30.0,
            poll_interval_s=0.0,
        )

    processes = fake_branch_runner["processes"]
    assert processes["visual"].terminate_called is True
    assert processes["yamnet"].terminate_called is True


def test_coordinator_times_out_branch_and_kills_siblings(tmp_path: Path, fake_branch_runner, monkeypatch) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    fake_branch_runner["configure"]("audio", exit_codes=[None, None, None, None])
    fake_branch_runner["configure"]("visual", exit_codes=[0])
    fake_branch_runner["configure"]("yamnet", exit_codes=[None, None, None, None])

    workspace = _build_workspace(tmp_path)

    ticks = iter([0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0])
    monkeypatch.setattr("backend.phase1_runtime.coordinator.time.monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="audio"):
        run_parallel_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            branch_timeout_s=2.0,
            poll_interval_s=0.0,
        )

    processes = fake_branch_runner["processes"]
    assert processes["visual"].terminate_called is False
    assert processes["yamnet"].terminate_called is True


def test_coordinator_treats_missing_result_json_as_hard_failure(tmp_path: Path, fake_branch_runner) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    fake_branch_runner["configure"]("visual", exit_codes=[0], skip_result_write=True)
    fake_branch_runner["configure"]("audio", exit_codes=[None, None, None, None])
    fake_branch_runner["configure"]("yamnet", exit_codes=[None, None, None, None])

    workspace = _build_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="visual"):
        run_parallel_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            branch_timeout_s=30.0,
            poll_interval_s=0.0,
        )

    processes = fake_branch_runner["processes"]
    assert processes["audio"].terminate_called is True
    assert processes["yamnet"].terminate_called is True


def test_coordinator_treats_malformed_result_json_as_hard_failure(tmp_path: Path, fake_branch_runner) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    fake_branch_runner["configure"]("visual", exit_codes=[0], write_raw_result="{not-json")
    fake_branch_runner["configure"]("audio", exit_codes=[None, None, None, None])
    fake_branch_runner["configure"]("yamnet", exit_codes=[None, None, None, None])

    workspace = _build_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="visual"):
        run_parallel_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            branch_timeout_s=30.0,
            poll_interval_s=0.0,
        )

    processes = fake_branch_runner["processes"]
    assert processes["audio"].terminate_called is True
    assert processes["yamnet"].terminate_called is True


def test_coordinator_cleans_up_already_started_branches_when_launch_fails(
    tmp_path: Path,
    fake_branch_runner,
) -> None:
    from backend.phase1_runtime.extract import run_parallel_phase1_sidecars

    fake_branch_runner["configure"]("visual", exit_codes=[None, None, None, None])
    fake_branch_runner["configure"]("audio", popen_error=RuntimeError("spawn failed"))

    workspace = _build_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="spawn failed"):
        run_parallel_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            branch_timeout_s=30.0,
            poll_interval_s=0.0,
        )

    launched = fake_branch_runner["launched"]
    processes = fake_branch_runner["processes"]
    assert [item["branch"] for item in launched] == ["visual", "audio"]
    assert processes["visual"].terminate_called is True
    assert launched[0]["stdout"].closed is True


def test_run_phase1_sidecars_preserves_legacy_provider_injection(tmp_path: Path) -> None:
    from backend.phase1_runtime.extract import run_phase1_sidecars

    call_order: list[str] = []

    class _FakeVibeVoice:
        def run(self, *, audio_path: Path, context_info=None):
            call_order.append(f"vibevoice:{audio_path.name}")
            return [
                {"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"},
            ]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            call_order.append(f"forced_aligner:{audio_path.name}:{len(turns)}")
            return [
                {"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 300, "speaker_id": "SPEAKER_0"},
            ]

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            call_order.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotionProvider:
        def run(self, *, audio_path: Path, turns: list[dict]):
            call_order.append(f"emotion:{audio_path.name}:{len(turns)}")
            return {
                "segments": [
                    {
                        "turn_id": turns[0]["turn_id"],
                        "labels": ["neutral"],
                        "scores": [0.88],
                        "per_class_scores": {"neutral": 0.88},
                    }
                ]
            }

    class _FakeYamnetProvider:
        def run(self, *, audio_path: Path):
            call_order.append(f"yamnet:{audio_path.name}")
            return {"events": []}

    workspace = _build_workspace(tmp_path)

    outputs = run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        workspace=workspace,
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisualExtractor(),
        emotion_provider=_FakeEmotionProvider(),
        yamnet_provider=_FakeYamnetProvider(),
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert outputs.diarization_payload["words"][0]["text"] == "hello"
    assert outputs.phase1_visual["video_metadata"]["fps"] == 10.0
    assert outputs.emotion2vec_payload["segments"][0]["labels"] == ["neutral"]
    assert outputs.yamnet_payload["events"] == []
    assert call_order == [
        "visual:source_video.mp4",
        "vibevoice:source_audio.wav",
        "forced_aligner:source_audio.wav:1",
        "emotion:source_audio.wav:1",
        "yamnet:source_audio.wav",
    ]
