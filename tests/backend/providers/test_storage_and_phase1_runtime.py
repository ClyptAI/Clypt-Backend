from __future__ import annotations

from pathlib import Path

import pytest


def test_gcs_storage_client_uploads_with_expected_blob_name(tmp_path: Path):
    from backend.providers.config import StorageSettings
    from backend.providers.storage import GCSStorageClient

    uploaded: dict[str, str] = {}

    class _FakeBlob:
        def __init__(self, name: str):
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            uploaded["name"] = self.name
            uploaded["filename"] = filename

    class _FakeBucket:
        def blob(self, name: str):
            return _FakeBlob(name)

    class _FakeStorageClient:
        def bucket(self, bucket_name: str):
            uploaded["bucket"] = bucket_name
            return _FakeBucket()

    local_file = tmp_path / "video.mp4"
    local_file.write_text("video", encoding="utf-8")

    client = GCSStorageClient(
        settings=StorageSettings(gcs_bucket="clypt-bucket"),
        storage_client=_FakeStorageClient(),
    )
    uri = client.upload_file(local_path=local_file, object_name="runs/run_1/source_video.mp4")

    assert uri == "gs://clypt-bucket/runs/run_1/source_video.mp4"
    assert uploaded == {
        "bucket": "clypt-bucket",
        "name": "runs/run_1/source_video.mp4",
        "filename": str(local_file),
    }


def test_run_phase1_sidecars_runs_vllm_and_audio_chain(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    call_order: list[str] = []
    stage_events: list[dict[str, object]] = []

    class _FakeVibeVoice:
        supports_concurrent_visual = True

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

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    outputs = run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        workspace=workspace,
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisualExtractor(),
        emotion_provider=_FakeEmotionProvider(),
        yamnet_provider=_FakeYamnetProvider(),
        stage_event_logger=lambda **event: stage_events.append(event),
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert outputs.diarization_payload["words"][0]["text"] == "hello"
    assert outputs.phase1_visual["video_metadata"]["fps"] == 10.0
    assert outputs.emotion2vec_payload["segments"][0]["labels"] == ["neutral"]
    assert outputs.yamnet_payload["events"] == []
    assert set(call_order) == {
        "visual:source_video.mp4",
        "vibevoice:source_audio.wav",
        "forced_aligner:source_audio.wav:1",
        "emotion:source_audio.wav:1",
        "yamnet:source_audio.wav",
    }
    # The audio chain is strictly ordered and starts only after ASR.
    i_vv = call_order.index("vibevoice:source_audio.wav")
    i_fa = call_order.index("forced_aligner:source_audio.wav:1")
    i_em = call_order.index("emotion:source_audio.wav:1")
    i_yn = call_order.index("yamnet:source_audio.wav")
    assert i_vv < i_fa < i_em < i_yn
    assert {event["stage_name"] for event in stage_events} == {
        "vibevoice_asr",
        "forced_alignment",
        "emotion2vec",
        "yamnet",
        "visual_extraction",
    }
    assert all(event["status"] == "succeeded" for event in stage_events)


def test_run_phase1_sidecars_fails_when_forced_alignment_returns_zero_words(
    tmp_path: Path,
    monkeypatch,
):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    monkeypatch.delenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT", raising=False)

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            return [{"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return []

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotionProvider:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return {"segments": []}

    class _FakeYamnetProvider:
        def run(self, *, audio_path: Path):
            return {"events": []}

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    with pytest.raises(RuntimeError, match="forced-alignment produced 0 words"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            workspace=workspace,
            vibevoice_provider=_FakeVibeVoice(),
            forced_aligner=_FakeForcedAligner(),
            visual_extractor=_FakeVisualExtractor(),
            emotion_provider=_FakeEmotionProvider(),
            yamnet_provider=_FakeYamnetProvider(),
        )


def test_run_phase1_sidecars_can_bypass_forced_alignment_zero_words_with_env_override(
    tmp_path: Path,
    monkeypatch,
):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    monkeypatch.setenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT", "0")

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            return [{"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return []

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotionProvider:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return {"segments": []}

    class _FakeYamnetProvider:
        def run(self, *, audio_path: Path):
            return {"events": []}

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

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
    # When forced alignment is bypassed, we still synthesize fallback words from ASR turns.
    assert outputs.diarization_payload["words"]
    assert outputs.diarization_payload["words"][0]["text"] == "hello"
