from __future__ import annotations

from pathlib import Path


def test_audio_branch_runs_vibevoice_then_alignment_then_emotion(tmp_path: Path):
    from backend.phase1_runtime.branch_models import BranchRequest, BranchKind
    from backend.phase1_runtime.branches.audio_branch import run_audio_branch
    from backend.phase1_runtime.models import Phase1Workspace

    call_order: list[str] = []

    class _FakeVibeVoice:
        def run(self, audio_path: Path, context_info=None):
            call_order.append(f"vibevoice:{audio_path.name}")
            return [
                {"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello world"},
                {"Start": 0.4, "End": 0.7, "Speaker": 1, "Content": "again"},
            ]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            call_order.append(f"forced_aligner:{audio_path.name}:{len(turns)}")
            assert turns == [
                {
                    "turn_id": "t_000001",
                    "speaker_id": "SPEAKER_0",
                    "start_ms": 0,
                    "end_ms": 300,
                    "transcript_text": "hello world",
                },
                {
                    "turn_id": "t_000002",
                    "speaker_id": "SPEAKER_1",
                    "start_ms": 400,
                    "end_ms": 700,
                    "transcript_text": "again",
                },
            ]
            return [
                {"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 150, "speaker_id": "SPEAKER_0"},
                {"word_id": "w_000002", "text": "world", "start_ms": 150, "end_ms": 300, "speaker_id": "SPEAKER_0"},
                {"word_id": "w_000003", "text": "again", "start_ms": 400, "end_ms": 700, "speaker_id": "SPEAKER_1"},
            ]

    class _FakeEmotionProvider:
        def run(self, *, audio_path: Path, turns: list[dict]):
            call_order.append(f"emotion:{audio_path.name}:{len(turns)}")
            assert turns[0]["turn_id"] == "t_000001"
            assert turns[0]["word_ids"] == ["w_000001", "w_000002"]
            assert turns[1]["transcript_text"] == "again"
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

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video", encoding="utf-8")
    workspace.audio_path.write_text("audio", encoding="utf-8")
    request = BranchRequest(
        job_id="job_123",
        run_id="run_001",
        branch=BranchKind.AUDIO,
        source_path="/tmp/source.mp4",
    )

    result = run_audio_branch(
        request=request,
        workspace=workspace,
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        emotion_provider=_FakeEmotionProvider(),
    )

    assert result["diarization_payload"]["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert result["diarization_payload"]["words"][0]["text"] == "hello"
    assert result["emotion2vec_payload"]["segments"][0]["labels"] == ["neutral"]
    assert call_order == [
        "vibevoice:source_audio.wav",
        "forced_aligner:source_audio.wav:2",
        "emotion:source_audio.wav:2",
    ]
