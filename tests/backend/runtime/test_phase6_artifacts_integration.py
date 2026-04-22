from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.phase1_runtime.payloads import (
    DiarizationPayload,
    EmotionSegmentsPayload,
    Phase1AudioAssets,
    Phase1SidecarOutputs,
    VisualPayload,
    YamnetPayload,
)
from backend.pipeline.config import V31Config
from backend.pipeline.orchestrator import V31Phase14Orchestrator, V31Phase14RunInputs
from backend.runtime.phase14_live import V31LivePhase14Runner


UTC = timezone.utc


def _phase1_outputs() -> Phase1SidecarOutputs:
    return Phase1SidecarOutputs(
        phase1_audio=Phase1AudioAssets(
            source_audio="https://youtube.com/watch?v=abc123",
            video_gcs_uri="gs://bucket/video.mp4",
            local_video_path="/tmp/source_video.mp4",
        ),
        diarization_payload=DiarizationPayload(words=[], turns=[]),
        phase1_visual=VisualPayload(video_metadata={"fps": 30.0}, shot_changes=[], tracks=[]),
        emotion2vec_payload=EmotionSegmentsPayload(segments=[]),
        yamnet_payload=YamnetPayload(events=[]),
    )


def test_orchestrator_run_summary_includes_phase6_artifact_outputs(monkeypatch, tmp_path: Path) -> None:
    orchestrator = V31Phase14Orchestrator(config=V31Config(output_root=tmp_path))

    monkeypatch.setattr(V31Phase14Orchestrator, "run_phase_1", lambda self, **kwargs: {"canonical_timeline": SimpleNamespace(turns=[]), "speech_emotion_timeline": SimpleNamespace(), "audio_event_timeline": SimpleNamespace()})
    monkeypatch.setattr(V31Phase14Orchestrator, "run_phase_2", lambda self, **kwargs: {"nodes": []})
    monkeypatch.setattr(V31Phase14Orchestrator, "run_phase_3", lambda self, **kwargs: {"edges": []})
    monkeypatch.setattr(V31Phase14Orchestrator, "run_phase_4", lambda self, **kwargs: None)

    summary = orchestrator.run(
        run_id="run_demo",
        source_url="https://youtube.com/watch?v=abc123",
        inputs=V31Phase14RunInputs(
            phase1_audio={
                "source_audio": "https://youtube.com/watch?v=abc123",
                "video_gcs_uri": "gs://bucket/video.mp4",
            },
            phase1_visual={
                "video_metadata": {"fps": 30.0},
                "shot_changes": [],
                "tracks": [],
            },
            diarization_payload={"words": [], "turns": []},
            emotion2vec_payload={"segments": []},
            yamnet_payload={"events": []},
            phase2_merge_responses={},
            phase2_boundary_responses={},
            phase3_local_edge_responses=[],
            phase3_long_range_response={"edges": []},
            phase4_subgraph_responses={},
            phase4_pool_response={"ranked_candidates": [], "dropped_candidate_temp_ids": []},
        ),
    )

    assert "caption_plan" in summary.artifact_paths
    assert "publish_metadata" in summary.artifact_paths
    assert "render_plan" in summary.artifact_paths
    assert "captions_clip_001.ass" not in summary.artifact_paths


def test_live_runner_summary_uses_real_phase6_artifact_outputs(tmp_path: Path) -> None:
    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        debug_snapshots=True,
    )
    paths = runner.build_run_paths(run_id="run_demo")
    started_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    started = 0.0
    phase6_result = {
        "artifact_paths": {
            "source_context": str(paths.source_context),
            "caption_plan": str(paths.caption_plan),
            "publish_metadata": str(paths.publish_metadata),
            "render_plan": str(paths.render_plan),
            "captions_clip_live.ass": str(paths.captions_ass("clip_live")),
        }
    }

    summary = runner._finish_phase24_success(
        run_id="run_demo",
        job_id="job_demo",
        attempt=1,
        paths=paths,
        started_at=started_at,
        started=started,
        phase2_nodes=[],
        phase3_edges=[],
        phase4_candidates=[],
        phase6_result=phase6_result,
    )

    assert "caption_plan" in summary.artifact_paths
    assert "publish_metadata" in summary.artifact_paths
    assert "render_plan" in summary.artifact_paths
    assert summary.artifact_paths["captions_clip_live.ass"] == str(paths.captions_ass("clip_live"))
    assert "captions_clip_001.ass" not in summary.artifact_paths
