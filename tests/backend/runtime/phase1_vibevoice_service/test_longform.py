from __future__ import annotations

import pytest

from backend.runtime.phase1_vibevoice_service.longform import (
    merge_shard_turns,
    plan_audio_shards,
    run_longform_vibevoice_asr,
)
from backend.runtime.phase1_vibevoice_service.models import (
    ShardAsrResult,
    ShardPlan,
)
from backend.runtime.phase1_vibevoice_service.speaker_stitch import (
    stitch_global_speakers,
)


def test_plan_audio_shards_returns_single_pass_for_60_minutes() -> None:
    shards = plan_audio_shards(duration_s=60 * 60)

    assert shards == [
        ShardPlan(index=0, shard_count=1, start_s=0.0, end_s=3600.0),
    ]


def test_plan_audio_shards_returns_two_equal_shards_through_90_minutes() -> None:
    shards = plan_audio_shards(duration_s=90 * 60)

    assert shards == [
        ShardPlan(index=0, shard_count=2, start_s=0.0, end_s=2700.0),
        ShardPlan(index=1, shard_count=2, start_s=2700.0, end_s=5400.0),
    ]


def test_plan_audio_shards_returns_three_equal_shards_through_180_minutes() -> None:
    shards = plan_audio_shards(duration_s=150 * 60)

    assert shards == [
        ShardPlan(index=0, shard_count=3, start_s=0.0, end_s=3000.0),
        ShardPlan(index=1, shard_count=3, start_s=3000.0, end_s=6000.0),
        ShardPlan(index=2, shard_count=3, start_s=6000.0, end_s=9000.0),
    ]


def test_plan_audio_shards_rejects_inputs_above_180_minutes() -> None:
    with pytest.raises(ValueError, match="180 minutes"):
        plan_audio_shards(duration_s=(180 * 60) + 1)


def test_plan_audio_shards_rejects_when_required_shards_exceed_max_shards() -> None:
    with pytest.raises(ValueError, match="requires 3 shards"):
        plan_audio_shards(duration_s=120 * 60, max_shards=2)


def test_merge_shard_turns_offsets_timestamps_into_global_time() -> None:
    shard_results = [
        ShardAsrResult(
            plan=ShardPlan(index=0, shard_count=2, start_s=0.0, end_s=1800.0),
            turns=[
                {"Speaker": 0, "Start": 1.0, "End": 4.0, "Content": "first"},
            ],
        ),
        ShardAsrResult(
            plan=ShardPlan(index=1, shard_count=2, start_s=1800.0, end_s=3600.0),
            turns=[
                {"Speaker": 0, "Start": 2.0, "End": 5.0, "Content": "second"},
            ],
        ),
    ]

    merged = merge_shard_turns(shard_results)

    assert merged == [
        {"Speaker": 0, "Start": 1.0, "End": 4.0, "Content": "first"},
        {"Speaker": 0, "Start": 1802.0, "End": 1805.0, "Content": "second"},
    ]


class _FakeVerifier:
    def __init__(self, scores: dict[tuple[int, int, int, int], float]) -> None:
        self.scores = scores

    def similarity(self, left: tuple[int, int], right: tuple[int, int]) -> float:
        return self.scores.get((left[0], left[1], right[0], right[1]), 0.0)


def test_stitch_global_speakers_matches_adjacent_shards_and_renumbers_by_first_appearance() -> None:
    shard_results = [
        ShardAsrResult(
            plan=ShardPlan(index=0, shard_count=2, start_s=0.0, end_s=1800.0),
            turns=[
                {"Speaker": 3, "Start": 10.0, "End": 12.0, "Content": "host intro"},
                {"Speaker": 7, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
            ],
        ),
        ShardAsrResult(
            plan=ShardPlan(index=1, shard_count=2, start_s=1800.0, end_s=3600.0),
            turns=[
                {"Speaker": 1, "Start": 5.0, "End": 7.0, "Content": "host follow-up"},
                {"Speaker": 2, "Start": 9.0, "End": 11.0, "Content": "guest follow-up"},
            ],
        ),
    ]
    verifier = _FakeVerifier(
        {
            (0, 3, 1, 1): 0.93,
            (0, 7, 1, 2): 0.91,
        }
    )

    merged = merge_shard_turns(stitch_global_speakers(shard_results, verifier, threshold=0.85))

    assert merged == [
        {"Speaker": 0, "Start": 10.0, "End": 12.0, "Content": "host intro"},
        {"Speaker": 1, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
        {"Speaker": 0, "Start": 1805.0, "End": 1807.0, "Content": "host follow-up"},
        {"Speaker": 1, "Start": 1809.0, "End": 1811.0, "Content": "guest follow-up"},
    ]


class _FakeStorageClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, *, local_path, object_name):
        self.uploads.append((str(local_path), object_name))
        return f"gs://bucket/{object_name}"


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, *, audio_path, audio_gcs_uri, context_info=None):  # noqa: ARG002
        self.calls.append(audio_gcs_uri)
        if audio_gcs_uri.endswith("shard_000_of_002.wav"):
            return [
                {"Speaker": 3, "Start": 10.0, "End": 12.0, "Content": "host intro"},
                {"Speaker": 7, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
            ]
        return [
            {"Speaker": 1, "Start": 5.0, "End": 7.0, "Content": "host follow-up"},
            {"Speaker": 2, "Start": 9.0, "End": 11.0, "Content": "guest follow-up"},
        ]


def test_run_longform_vibevoice_asr_splits_uploads_and_merges_two_shards(tmp_path) -> None:
    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_bytes(b"fake audio")
    storage = _FakeStorageClient()
    provider = _FakeProvider()
    verifier = _FakeVerifier(
        {
            (0, 3, 1, 1): 0.93,
            (0, 7, 1, 2): 0.91,
        }
    )

    def _fake_extract_shard_audio(*, source_audio_path, output_audio_path, start_s, end_s):  # noqa: ARG001
        output_audio_path.write_bytes(b"fake shard audio")

    result = run_longform_vibevoice_asr(
        audio_path=audio_path,
        canonical_audio_gcs_uri="gs://bucket/canonical.wav",
        run_id="run-123",
        vibevoice_provider=provider,
        storage_client=storage,
        speaker_verifier=verifier,
        duration_s=90 * 60,
        extract_shard_audio=_fake_extract_shard_audio,
    )

    assert provider.calls == [
        "gs://bucket/phase1/run-123/vibevoice_shards/shard_000_of_002.wav",
        "gs://bucket/phase1/run-123/vibevoice_shards/shard_001_of_002.wav",
    ]
    assert [object_name for _local_path, object_name in storage.uploads] == [
        "phase1/run-123/vibevoice_shards/shard_000_of_002.wav",
        "phase1/run-123/vibevoice_shards/shard_001_of_002.wav",
    ]
    assert result.turns == [
        {"Speaker": 0, "Start": 10.0, "End": 12.0, "Content": "host intro"},
        {"Speaker": 1, "Start": 20.0, "End": 22.0, "Content": "guest intro"},
        {"Speaker": 0, "Start": 2705.0, "End": 2707.0, "Content": "host follow-up"},
        {"Speaker": 1, "Start": 2709.0, "End": 2711.0, "Content": "guest follow-up"},
    ]
    assert any(event["stage_name"] == "vibevoice_longform_merge" for event in result.stage_events)


def test_run_longform_vibevoice_asr_respects_custom_shard_thresholds(tmp_path) -> None:
    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_bytes(b"fake audio")
    storage = _FakeStorageClient()
    provider = _FakeProvider()
    verifier = _FakeVerifier(
        {
            (0, 3, 1, 1): 0.93,
            (0, 7, 1, 2): 0.91,
        }
    )

    def _fake_extract_shard_audio(*, source_audio_path, output_audio_path, start_s, end_s):  # noqa: ARG001
        output_audio_path.write_bytes(b"fake shard audio")

    result = run_longform_vibevoice_asr(
        audio_path=audio_path,
        canonical_audio_gcs_uri="gs://bucket/canonical.wav",
        run_id="run-456",
        vibevoice_provider=provider,
        storage_client=storage,
        speaker_verifier=verifier,
        duration_s=91 * 60,
        two_shard_max_minutes=120,
        extract_shard_audio=_fake_extract_shard_audio,
    )

    assert len(provider.calls) == 2
    assert result.turns[2]["Start"] == 2735.0
