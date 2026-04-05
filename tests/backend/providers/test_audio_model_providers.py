from __future__ import annotations

from pathlib import Path


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
