from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest


def test_emotion2vec_plus_provider_clips_turns_and_normalizes_segments(tmp_path: Path):
    from backend.providers.emotion2vec import Emotion2VecPlusProvider

    clipped_ranges: list[tuple[int, int]] = []

    def fake_clipper(*, audio_path: Path, start_ms: int, end_ms: int) -> Path:
        clipped_ranges.append((start_ms, end_ms))
        clip_path = tmp_path / f"{start_ms}_{end_ms}.wav"
        clip_path.write_text("clip", encoding="utf-8")
        return clip_path

    class _FakeModel:
        def generate(self, *, input, granularity="utterance"):
            return [
                {
                    "labels": ["neutral"],
                    "scores": [0.91],
                    "per_class_scores": {"neutral": 0.91, "happy": 0.09},
                }
            ]

    provider = Emotion2VecPlusProvider(model=_FakeModel(), clipper=fake_clipper)
    audio_path = tmp_path / "source.wav"
    audio_path.write_text("audio", encoding="utf-8")

    payload = provider.run(
        audio_path=audio_path,
        turns=[
            {"turn_id": "t_000001", "start_ms": 0, "end_ms": 1200},
            {"turn_id": "t_000002", "start_ms": 1500, "end_ms": 2100},
        ],
    )

    assert clipped_ranges == [(0, 1200), (1500, 2100)]
    assert payload["segments"][0]["turn_id"] == "t_000001"
    assert payload["segments"][0]["labels"] == ["neutral"]
    assert payload["segments"][1]["scores"] == [0.91]
    assert provider.last_run_metrics["turn_count"] == 2
    assert provider.last_run_metrics["clip_extract_ms"] >= 0.0
    assert provider.last_run_metrics["infer_ms"] >= 0.0


def test_emotion2vec_default_model_uses_hf_hub(monkeypatch):
    import backend.providers.emotion2vec as emotion2vec

    captured_kwargs = {}

    class _FakeAutoModel:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=_FakeAutoModel))
    monkeypatch.delenv("FUNASR_MODEL_SOURCE", raising=False)
    monkeypatch.delenv("EMOTION2VEC_MODEL_ID", raising=False)

    model = emotion2vec._build_default_model()

    assert isinstance(model, _FakeAutoModel)
    assert captured_kwargs["model"] == "iic/emotion2vec_plus_large"
    assert captured_kwargs["hub"] == "hf"
    assert captured_kwargs["disable_update"] is True


def test_emotion2vec_default_model_ignores_non_hf_source(monkeypatch):
    import backend.providers.emotion2vec as emotion2vec

    captured_kwargs = {}

    class _FakeAutoModel:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=_FakeAutoModel))
    monkeypatch.setenv("FUNASR_MODEL_SOURCE", "modelscope")
    monkeypatch.setenv("EMOTION2VEC_MODEL_ID", "iic/emotion2vec_plus_large")

    model = emotion2vec._build_default_model()

    assert isinstance(model, _FakeAutoModel)
    assert captured_kwargs["hub"] == "hf"


def test_emotion2vec_plus_provider_logs_true_top_label_for_unsorted_scores(tmp_path: Path, caplog):
    from backend.providers.emotion2vec import Emotion2VecPlusProvider

    def fake_clipper(*, audio_path: Path, start_ms: int, end_ms: int) -> Path:
        clip_path = tmp_path / f"{start_ms}_{end_ms}.wav"
        clip_path.write_text("clip", encoding="utf-8")
        return clip_path

    class _FakeModel:
        def generate(self, *, input, granularity="utterance"):
            return [
                {
                    "labels": ["angry", "neutral"],
                    "scores": [0.0, 0.87],
                    "per_class_scores": {"angry": 0.0, "neutral": 0.87},
                }
            ]

    caplog.set_level("INFO", logger="backend.providers.emotion2vec")
    provider = Emotion2VecPlusProvider(model=_FakeModel(), clipper=fake_clipper)
    audio_path = tmp_path / "source.wav"
    audio_path.write_text("audio", encoding="utf-8")

    provider.run(
        audio_path=audio_path,
        turns=[{"turn_id": "t_000001", "start_ms": 0, "end_ms": 1200}],
    )

    assert "top: neutral 0.87" in caplog.text


def test_yamnet_provider_merges_adjacent_patch_events():
    from backend.providers.yamnet import YAMNetProvider

    def fake_runner(*, audio_path):
        return [
            {"event_label": "Laughter", "start_ms": 1000, "end_ms": 1400, "confidence": 0.75},
            {"event_label": "Laughter", "start_ms": 1400, "end_ms": 1800, "confidence": 0.88},
            {"event_label": "Music", "start_ms": 2500, "end_ms": 2900, "confidence": 0.61},
        ]

    provider = YAMNetProvider(runner=fake_runner)
    payload = provider.run(audio_path=Path("/tmp/audio.wav"))

    assert payload["events"] == [
        {"event_label": "Laughter", "start_ms": 1000, "end_ms": 1800, "confidence": 0.88},
        {"event_label": "Music", "start_ms": 2500, "end_ms": 2900, "confidence": 0.61},
    ]
    assert provider.device == "gpu"


def test_vibevoice_vllm_payload_wires_sampling_and_beam_controls(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="base64",
        do_sample=True,
        temperature=0.2,
        top_p=0.91,
        repetition_penalty=0.97,
        num_beams=4,
    )

    payload = provider._build_payload(audio_path=audio, context="", duration_s=2.5)

    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.91
    assert payload["repetition_penalty"] == 0.97
    assert payload["use_beam_search"] is True
    assert payload["best_of"] == 4


def test_vibevoice_vllm_payload_disables_sampling_when_do_sample_off(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="base64",
        do_sample=False,
        temperature=0.8,
        top_p=0.6,
        repetition_penalty=0.97,
        num_beams=1,
    )

    payload = provider._build_payload(audio_path=audio, context="", duration_s=1.0)

    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["repetition_penalty"] == 0.97
    assert "use_beam_search" not in payload
    assert "best_of" not in payload


def test_vibevoice_vllm_payload_uses_url_audio_when_configured(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="url",
    )

    payload = provider._build_payload(
        audio_path=audio,
        context="",
        duration_s=1.0,
        audio_url="https://example.com/signed/audio.wav?sig=abc",
    )

    audio_part = payload["messages"][1]["content"][0]
    assert audio_part["audio_url"]["url"].startswith("https://example.com/signed/audio.wav")
    assert "base64" not in audio_part["audio_url"]["url"]


def test_build_gcs_uri_url_resolver_signs_uri():
    from backend.providers.vibevoice_vllm import build_gcs_uri_url_resolver

    class _FakeStorageClient:
        def __init__(self):
            self.expiry_hours = None

        def get_https_url(self, gcs_uri: str, expiry_hours: int = 24) -> str:
            self.expiry_hours = expiry_hours
            assert gcs_uri == "gs://bucket/key.wav"
            return "https://storage.googleapis.com/bucket/key.wav?sig=123"

    fake_storage = _FakeStorageClient()
    resolver = build_gcs_uri_url_resolver(storage_client=fake_storage, signed_url_expiry_hours=3)
    resolved_url = resolver("gs://bucket/key.wav")

    assert resolved_url.startswith("https://storage.googleapis.com/bucket/key.wav")
    assert fake_storage.expiry_hours == 3


def test_vibevoice_vllm_prefers_audio_gcs_uri_signing_when_available(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="url",
        audio_gcs_url_resolver=lambda gcs_uri: f"https://signed.example.com/{gcs_uri.rsplit('/', 1)[-1]}",
    )

    resolved = provider._resolve_audio_url(audio_gcs_uri="gs://bucket/path/canonical.wav")

    assert resolved == "https://signed.example.com/canonical.wav"


def test_vibevoice_vllm_url_mode_requires_audio_gcs_uri(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="url",
        audio_gcs_url_resolver=lambda gcs_uri: "https://signed.example.com/audio.wav",
    )

    try:
        provider._resolve_audio_url(audio_gcs_uri=None)
    except RuntimeError as exc:
        assert "requires audio_gcs_uri" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError when audio_gcs_uri is missing in URL mode")


def test_vibevoice_vllm_url_mode_requires_signer_for_gcs_uri(tmp_path: Path):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        audio_mode="url",
        audio_gcs_url_resolver=None,
    )

    try:
        provider._resolve_audio_url(audio_gcs_uri="gs://bucket/canonical.wav")
    except RuntimeError as exc:
        assert "no signer is configured" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError when signer is missing for gs:// URI")


def test_vibevoice_vllm_parse_failure_persists_full_raw_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    artifact_dir = tmp_path / "vibevoice-failures"
    monkeypatch.setenv("CLYPT_VIBEVOICE_FAILURE_DIR", str(artifact_dir))
    provider = VibeVoiceVLLMProvider(base_url="http://127.0.0.1:8000", audio_mode="base64")

    partial_content = (
        '[{"Start":0,"End":1.0,"Speaker":0,"Content":"I am not interested in being '
        'around grizzly bears. Did you s'
    )

    with pytest.raises(RuntimeError, match="artifact=") as exc_info:
        provider._parse_content(
            partial_content,
            finish_reason="length",
            generated_chars=len(partial_content),
            chunk_count=3,
        )

    msg = str(exc_info.value)
    assert "finish_reason=length" in msg
    assert f"generated_chars={len(partial_content)}" in msg
    artifact_path = Path(msg.split("artifact=", 1)[1].split()[0])
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == partial_content


def test_vibevoice_vllm_do_request_surfaces_stream_finish_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")

    provider = VibeVoiceVLLMProvider(base_url="http://127.0.0.1:8000", audio_mode="base64")

    class _FakeStreamingResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = [line.encode("utf-8") for line in lines]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        def __iter__(self):
            return iter(self._lines)

    partial_content = '[{"Start":0,"End":1.0,"Speaker":0,"Content":"Did you s'
    stream_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": partial_content},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        "data: [DONE]",
    ]

    def _fake_urlopen(req, timeout):
        _ = req, timeout
        return _FakeStreamingResponse(stream_lines)

    monkeypatch.setattr("backend.providers.vibevoice_vllm.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="finish_reason=length") as exc_info:
        provider._do_request(audio, "", 1.0, None)

    assert "generated_chars=" in str(exc_info.value)
