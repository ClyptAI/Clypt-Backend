from __future__ import annotations

from pathlib import Path


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


def test_run_phase1_sidecars_overlaps_pyannote_with_local_worker_tasks(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    call_order: list[str] = []

    class _FakePyannote:
        def run_diarize(self, *, media_url: str):
            call_order.append(f"pyannote:{media_url}")
            return {
                "wordLevelTranscription": [
                    {"word": "hello", "start": 0.0, "end": 0.3, "speaker": "S1"}
                ],
                "diarization": [
                    {"speaker": "S1", "start": 0.0, "end": 0.3}
                ],
            }

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
        pyannote_client=_FakePyannote(),
        visual_extractor=_FakeVisualExtractor(),
        emotion_provider=_FakeEmotionProvider(),
        yamnet_provider=_FakeYamnetProvider(),
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.pyannote_payload["diarization"][0]["speaker"] == "S1"
    assert outputs.phase1_visual["video_metadata"]["fps"] == 10.0
    assert outputs.emotion2vec_payload["segments"][0]["labels"] == ["neutral"]
    assert outputs.yamnet_payload["events"] == []
    assert call_order == [
        "pyannote:gs://bucket/source.mp4",
        "visual:source_video.mp4",
        "emotion:source_audio.wav:1",
        "yamnet:source_audio.wav",
    ]
