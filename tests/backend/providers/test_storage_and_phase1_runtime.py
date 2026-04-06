from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


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


def test_build_default_phase1_job_runner_uses_parallel_sidecar_runtime_and_cpu_yamnet(
    tmp_path: Path,
    monkeypatch,
):
    from backend.phase1_runtime import factory
    from backend.phase1_runtime.models import Phase1SidecarOutputs

    parallel_calls: list[dict[str, object]] = []
    yamnet_devices: list[str] = []

    def fake_parallel_phase1_sidecars(
        *,
        source_url: str,
        video_gcs_uri: str,
        workspace,
        branch_timeout_s: float,
        poll_interval_s: float,
    ):
        parallel_calls.append(
            {
                "source_url": source_url,
                "video_gcs_uri": video_gcs_uri,
                "workspace": workspace,
                "branch_timeout_s": branch_timeout_s,
                "poll_interval_s": poll_interval_s,
            }
        )
        return Phase1SidecarOutputs(
            phase1_audio={
                "source_audio": source_url,
                "video_gcs_uri": video_gcs_uri,
                "local_video_path": str(workspace.video_path),
                "local_audio_path": str(workspace.audio_path),
            },
            diarization_payload={
                "turns": [{"turn_id": "t_000001", "speaker_id": "SPEAKER_0"}],
                "words": [{"word_id": "w_000001", "text": "hello"}],
            },
            phase1_visual={"tracks": [{"track_id": 1}]},
            emotion2vec_payload={"segments": [{"turn_id": "t_000001"}]},
            yamnet_payload={"events": []},
        )

    monkeypatch.setattr(factory, "run_parallel_phase1_sidecars", fake_parallel_phase1_sidecars)
    monkeypatch.setattr(
        factory,
        "load_provider_settings",
        lambda: SimpleNamespace(
            vibevoice=SimpleNamespace(
                backend="native",
                native_venv_python="",
                model_id="model",
                flash_attention=True,
                liger_kernel=True,
                hotwords_context="",
                system_prompt="",
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                attn_implementation="flash_attention_2",
                subprocess_timeout_s=7200,
            ),
            vertex=SimpleNamespace(project="clypt-v3"),
            storage=SimpleNamespace(gcs_bucket="bucket-a"),
            phase1_runtime=SimpleNamespace(
                working_root=tmp_path,
                keep_workdir=False,
                branch_timeout_s=45.0,
                branch_poll_interval_s=0.3,
                phase1_parallel_enabled=True,
                phase1_parallel_gpu_branch_limit=2,
                yamnet_branch_device="cpu",
            ),
        ),
    )
    monkeypatch.setattr(factory, "VibeVoiceASRProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(factory, "ForcedAlignmentProvider", lambda: object())
    monkeypatch.setattr(factory, "VertexEmbeddingClient", lambda settings: object())
    monkeypatch.setattr(factory, "VertexGeminiClient", lambda settings: object())
    monkeypatch.setattr(factory, "GCSStorageClient", lambda settings: object())
    monkeypatch.setattr(
        factory,
        "V31LivePhase14Runner",
        SimpleNamespace(from_env=lambda **kwargs: object()),
    )
    monkeypatch.setattr(factory, "VisualPipelineConfig", SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(factory, "SimpleVisualExtractor", lambda visual_config: object())
    monkeypatch.setattr(factory, "Emotion2VecPlusProvider", lambda: object())

    def fake_yamnet_provider(*, device: str, **kwargs):
        yamnet_devices.append(device)
        return SimpleNamespace(device=device, **kwargs)

    monkeypatch.setattr(factory, "YAMNetProvider", fake_yamnet_provider)

    runner = factory.build_default_phase1_job_runner()

    assert runner.run_phase1_sidecars.__name__ == "run_parallel_phase1_sidecars"

    sentinel_workspace = SimpleNamespace(
        video_path=tmp_path / "source_video.mp4",
        audio_path=tmp_path / "source_audio.wav",
    )
    result = runner.run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        workspace=sentinel_workspace,
    )

    assert result.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert result.phase1_visual["tracks"] == [{"track_id": 1}]
    assert result.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert result.emotion2vec_payload["segments"][0]["turn_id"] == "t_000001"
    assert result.yamnet_payload["events"] == []
    assert parallel_calls == [
        {
            "source_url": "https://youtube.com/watch?v=demo",
            "video_gcs_uri": "gs://bucket/source.mp4",
            "workspace": sentinel_workspace,
            "branch_timeout_s": 45.0,
            "poll_interval_s": 0.3,
        }
    ]
    assert yamnet_devices == ["cpu"]
