from __future__ import annotations

from pathlib import Path

import pytest

from backend.providers.forced_aligner import ForcedAlignmentProvider


def _make_turn(start_ms: int, end_ms: int, text: str, turn_id: str) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "speaker_id": "SPEAKER_0",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "transcript_text": text,
    }


def test_alignment_chunk_count_uses_duration_defaults() -> None:
    provider = ForcedAlignmentProvider()

    assert provider._alignment_chunk_count_for_duration_s(30 * 60) == 1
    assert provider._alignment_chunk_count_for_duration_s(40 * 60) == 1
    assert provider._alignment_chunk_count_for_duration_s((40 * 60) + 1) == 2
    assert provider._alignment_chunk_count_for_duration_s(60 * 60) == 2
    assert provider._alignment_chunk_count_for_duration_s((60 * 60) + 1) == 3
    assert provider._alignment_chunk_count_for_duration_s(120 * 60) == 3
    assert provider._alignment_chunk_count_for_duration_s((120 * 60) + 1) == 4
    assert provider._alignment_chunk_count_for_duration_s(150 * 60) == 4
    assert provider._alignment_chunk_count_for_duration_s((150 * 60) + 1) == 5
    assert provider._alignment_chunk_count_for_duration_s(180 * 60) == 5


def test_run_uses_duration_chunked_alignment_for_long_inputs(monkeypatch, tmp_path: Path) -> None:
    provider = ForcedAlignmentProvider()
    turns = [
        _make_turn(0, 60_000, "one", "t1"),
        _make_turn(2_400_000, 2_460_000, "two", "t2"),
        _make_turn(4_800_000, 4_860_000, "three", "t3"),
        _make_turn(7_200_000, 7_260_000, "four", "t4"),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(provider, "_check_available", lambda: True)
    monkeypatch.setattr(provider, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(provider, "_ensure_model", lambda device: None)
    monkeypatch.setattr(provider, "_audio_duration_s", lambda path: 130 * 60)
    monkeypatch.setattr(
        provider,
        "_slice_audio_window",
        lambda *, audio_path, start_ms, end_ms, tmpdir, chunk_index: audio_path,  # noqa: ARG005
    )

    seen_turn_groups: list[list[str]] = []

    def _fake_align_chunk(*, audio_path: Path, turns: list[dict[str, object]], device: str) -> list[dict[str, object]]:  # noqa: ARG001
        seen_turn_groups.append([str(turn["turn_id"]) for turn in turns])
        first_turn = turns[0]
        return [
            {
                "word_id": f"w_{len(seen_turn_groups):06d}",
                "text": str(first_turn["transcript_text"]),
                "start_ms": int(first_turn["start_ms"]),
                "end_ms": int(first_turn["end_ms"]),
                "speaker_id": str(first_turn["speaker_id"]),
            }
        ]

    monkeypatch.setattr(provider, "_align_global_transcript", _fake_align_chunk)

    words = provider.run(audio_path, turns)

    assert seen_turn_groups == [["t1"], ["t2"], ["t3"], ["t4"]]
    assert [word["text"] for word in words] == ["one", "two", "three", "four"]


def test_run_hard_fails_when_chunk_alignment_fails(monkeypatch, tmp_path: Path) -> None:
    provider = ForcedAlignmentProvider()
    turns = [
        _make_turn(0, 60_000, "one", "t1"),
        _make_turn(3_000_000, 3_060_000, "two", "t2"),
    ]
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(provider, "_check_available", lambda: True)
    monkeypatch.setattr(provider, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(provider, "_ensure_model", lambda device: None)
    monkeypatch.setattr(provider, "_audio_duration_s", lambda path: 50 * 60)
    monkeypatch.setattr(
        provider,
        "_slice_audio_window",
        lambda *, audio_path, start_ms, end_ms, tmpdir, chunk_index: audio_path,  # noqa: ARG005
    )

    calls = {"count": 0}

    def _failing_align_chunk(*, audio_path: Path, turns: list[dict[str, object]], device: str) -> list[dict[str, object]]:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("chunk oom")
        return [
            {
                "word_id": "w_000001",
                "text": "ok",
                "start_ms": 0,
                "end_ms": 100,
                "speaker_id": "SPEAKER_0",
            }
        ]

    monkeypatch.setattr(provider, "_align_global_transcript", _failing_align_chunk)

    with pytest.raises(RuntimeError, match="chunk oom"):
        provider.run(audio_path, turns)
