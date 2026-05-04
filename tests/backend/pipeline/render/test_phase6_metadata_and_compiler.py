from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from backend.pipeline.artifacts import build_run_paths
from backend.pipeline.contracts import (
    CanonicalTimeline,
    CanonicalTurn,
    ClipCandidate,
    SemanticGraphNode,
    SemanticNodeEvidence,
    ShotTrackletDescriptor,
    ShotTrackletIndex,
    TrackletGeometry,
    TrackletGeometryEntry,
    TrackletGeometryPoint,
    TranscriptWord,
)


def _timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        words=[
            TranscriptWord(word_id="w1", text="Nobody", start_ms=0, end_ms=150, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w2", text="saw", start_ms=150, end_ms=260, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w3", text="this", start_ms=260, end_ms=380, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w4", text="coming", start_ms=380, end_ms=610, speaker_id="SPEAKER_0"),
        ],
        turns=[
            CanonicalTurn(
                turn_id="t1",
                speaker_id="SPEAKER_0",
                start_ms=0,
                end_ms=610,
                word_ids=["w1", "w2", "w3", "w4"],
                transcript_text="Nobody saw this coming",
            )
        ],
    )


def _candidate() -> ClipCandidate:
    return ClipCandidate(
        clip_id="clip_001",
        node_ids=["node_1"],
        start_ms=0,
        end_ms=610,
        score=9.1,
        rationale="Surprise payoff lands immediately.",
    )


def _nodes() -> list[SemanticGraphNode]:
    return [
        SemanticGraphNode(
            node_id="node_1",
            node_type="reveal",
            start_ms=0,
            end_ms=610,
            source_turn_ids=["t1"],
            word_ids=["w1", "w2", "w3", "w4"],
            transcript_text="Nobody saw this coming",
            node_flags=[],
            summary="A compact surprise reveal.",
            evidence=SemanticNodeEvidence(),
        )
    ]


def _shot_tracklets() -> ShotTrackletIndex:
    return ShotTrackletIndex(
        tracklets=[
            ShotTrackletDescriptor(
                tracklet_id="tracklet_1",
                shot_id="shot_1",
                start_ms=0,
                end_ms=610,
                representative_thumbnail_uris=[],
            )
        ]
    )


def _shot_tracklets_ineligible_then_eligible() -> ShotTrackletIndex:
    return ShotTrackletIndex(
        tracklets=[
            ShotTrackletDescriptor(
                tracklet_id="tracklet_bad",
                shot_id="shot_1",
                start_ms=0,
                end_ms=610,
                auto_follow_eligible=False,
                subject_quality={"head_evidence_ratio": 0.0},
                representative_thumbnail_uris=[],
            ),
            ShotTrackletDescriptor(
                tracklet_id="tracklet_good",
                shot_id="shot_1",
                start_ms=0,
                end_ms=610,
                auto_follow_eligible=True,
                subject_quality={"head_evidence_ratio": 1.0},
                representative_thumbnail_uris=[],
            ),
        ]
    )


def _tracklet_geometry() -> TrackletGeometry:
    return TrackletGeometry(
        tracklets=[
            TrackletGeometryEntry(
                tracklet_id="tracklet_1",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=0,
                        bbox_xyxy=[220.0, 420.0, 820.0, 1520.0],
                    )
                ],
            )
        ]
    )


def _tracklet_geometry_bad_and_good() -> TrackletGeometry:
    return TrackletGeometry(
        tracklets=[
            TrackletGeometryEntry(
                tracklet_id="tracklet_bad",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=0,
                        bbox_xyxy=[0.0, 600.0, 500.0, 1080.0],
                    )
                ],
            ),
            TrackletGeometryEntry(
                tracklet_id="tracklet_good",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=0,
                        bbox_xyxy=[220.0, 120.0, 820.0, 1020.0],
                    )
                ],
            ),
        ]
    )


def _shot_tracklets_two_segments_same_shot() -> ShotTrackletIndex:
    return ShotTrackletIndex(
        tracklets=[
            ShotTrackletDescriptor(
                tracklet_id="tracklet_left",
                shot_id="shot_1",
                start_ms=0,
                end_ms=1000,
                auto_follow_eligible=True,
                subject_quality={
                    "head_evidence_ratio": 0.45,
                    "upper_body_anchor_ratio": 0.25,
                    "median_rfdetr_confidence": 0.92,
                },
                representative_thumbnail_uris=[],
            ),
            ShotTrackletDescriptor(
                tracklet_id="tracklet_right",
                shot_id="shot_1",
                start_ms=0,
                end_ms=1000,
                auto_follow_eligible=True,
                subject_quality={
                    "head_evidence_ratio": 0.92,
                    "upper_body_anchor_ratio": 0.88,
                    "median_rfdetr_confidence": 0.91,
                },
                representative_thumbnail_uris=[],
            ),
        ]
    )


def _tracklet_geometry_two_segments_same_shot() -> TrackletGeometry:
    return TrackletGeometry(
        tracklets=[
            TrackletGeometryEntry(
                tracklet_id="tracklet_left",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=0,
                        bbox_xyxy=[100.0, 100.0, 500.0, 900.0],
                    ),
                    TrackletGeometryPoint(
                        frame_index=1,
                        timestamp_ms=1000,
                        bbox_xyxy=[160.0, 120.0, 560.0, 920.0],
                    ),
                ],
            ),
            TrackletGeometryEntry(
                tracklet_id="tracklet_right",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=0,
                        bbox_xyxy=[900.0, 110.0, 1320.0, 990.0],
                    ),
                    TrackletGeometryPoint(
                        frame_index=1,
                        timestamp_ms=500,
                        bbox_xyxy=[980.0, 130.0, 1400.0, 1010.0],
                    ),
                    TrackletGeometryPoint(
                        frame_index=2,
                        timestamp_ms=1000,
                        bbox_xyxy=[1040.0, 150.0, 1460.0, 1030.0],
                    ),
                ],
            ),
        ]
    )


def _caption_plan_two_segments() -> dict:
    return {
        "run_id": "run_phase6",
        "clips": [
            {
                "clip_id": "clip_001",
                "clip_start_ms": 0,
                "clip_end_ms": 1000,
                "preset_id": "karaoke_focus",
                "default_zone": "lower_safe",
                "segments": [
                    {
                        "segment_id": "clip_001_seg_001",
                        "start_ms": 0,
                        "end_ms": 500,
                        "text": "Nobody saw",
                        "word_ids": ["w1"],
                        "speaker_ids": ["SPEAKER_0"],
                        "turn_ids": ["t1"],
                        "placement_zone": "lower_safe",
                        "highlight_mode": "word_highlight",
                        "review_needed": False,
                        "review_reason": "",
                        "active_word_timings": [],
                    },
                    {
                        "segment_id": "clip_001_seg_002",
                        "start_ms": 500,
                        "end_ms": 1000,
                        "text": "this coming",
                        "word_ids": ["w2"],
                        "speaker_ids": ["SPEAKER_0"],
                        "turn_ids": ["t1"],
                        "placement_zone": "lower_safe",
                        "highlight_mode": "word_highlight",
                        "review_needed": False,
                        "review_reason": "",
                        "active_word_timings": [],
                    },
                ],
            }
        ],
    }


def _publish_metadata_basic() -> dict:
    return {
        "run_id": "run_phase6",
        "clips": [
            {
                "clip_id": "clip_001",
                "title_primary": "Nobody Saw This Coming",
                "title_alternates": [],
                "description_short": "A surprise reveal lands fast.",
                "thumbnail_text": "NOBODY SAW THIS",
                "topic_tags": [],
                "hashtags": [],
                "generation_inputs_summary": {},
            }
        ],
    }


def _camera_intent_follow(*, end_ms: int = 610, primary_tracklet_id: str = "tracklet_1") -> dict[str, object]:
    return {
        "segments": [
            {
                "start_ms": 0,
                "end_ms": end_ms,
                "intent": "follow",
                "primary_tracklet_id": primary_tracklet_id,
                "secondary_tracklet_id": None,
                "clip_candidate_id": "clip_001",
                "user_confirmed": True,
            }
        ]
    }


def _camera_intent_follow_then_split() -> dict[str, object]:
    return {
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 280,
                "intent": "follow",
                "primary_tracklet_id": "tracklet_1",
                "secondary_tracklet_id": None,
                "clip_candidate_id": "clip_001",
                "user_confirmed": True,
            },
            {
                "start_ms": 280,
                "end_ms": 610,
                "intent": "split",
                "primary_tracklet_id": "tracklet_1",
                "secondary_tracklet_id": "tracklet_2",
                "clip_candidate_id": "clip_001",
                "user_confirmed": True,
            },
        ]
    }


def _shot_tracklets_with_split() -> ShotTrackletIndex:
    return ShotTrackletIndex(
        tracklets=[
            ShotTrackletDescriptor(
                tracklet_id="tracklet_1",
                shot_id="shot_1",
                start_ms=0,
                end_ms=610,
                representative_thumbnail_uris=[],
            ),
            ShotTrackletDescriptor(
                tracklet_id="tracklet_2",
                shot_id="shot_1",
                start_ms=280,
                end_ms=610,
                representative_thumbnail_uris=[],
            ),
        ]
    )


def _tracklet_geometry_below_center() -> TrackletGeometry:
    return TrackletGeometry(
        tracklets=[
            TrackletGeometryEntry(
                tracklet_id="tracklet_1",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=120,
                        bbox_xyxy=[220.0, 1380.0, 820.0, 1820.0],
                    )
                ],
            ),
            TrackletGeometryEntry(
                tracklet_id="tracklet_2",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=1,
                        timestamp_ms=420,
                        bbox_xyxy=[120.0, 320.0, 460.0, 1180.0],
                    )
                ],
            ),
        ]
    )


def _tracklet_geometry_center_collision() -> TrackletGeometry:
    return TrackletGeometry(
        tracklets=[
            TrackletGeometryEntry(
                tracklet_id="tracklet_1",
                shot_id="shot_1",
                points=[
                    TrackletGeometryPoint(
                        frame_index=0,
                        timestamp_ms=120,
                        bbox_xyxy=[220.0, 640.0, 820.0, 1500.0],
                    )
                ],
            )
        ]
    )


def test_generate_publish_metadata_uses_source_context_and_clip_inputs() -> None:
    from backend.pipeline.render.contracts import SourceContext
    from backend.pipeline.render.phase6 import generate_publish_metadata

    metadata = generate_publish_metadata(
        run_id="run_phase6",
        candidates=[_candidate()],
        canonical_timeline=_timeline(),
        nodes=_nodes(),
        source_context=SourceContext(
            source_url="https://www.youtube.com/watch?v=abc123xyz00",
            youtube_video_id="abc123xyz00",
            source_title="The Long Interview",
            source_description="A long conversation about creator workflows.",
            channel_id="channel_123",
            channel_title="Clypt Clips",
            published_at="2026-04-19T00:00:00+00:00",
            default_audio_language="en",
            category_id="22",
            tags=["creators", "workflow"],
            thumbnails={"default": {"url": "https://example.com/thumb.jpg"}},
        ),
    )

    clip = metadata.clips[0]
    assert clip.clip_id == "clip_001"
    assert clip.title_primary
    assert 2 <= len(clip.title_alternates) <= 4
    assert clip.description_short
    assert clip.thumbnail_text
    assert 3 <= len(clip.topic_tags) <= 8
    assert 3 <= len(clip.hashtags) <= 6
    assert clip.generation_inputs_summary["youtube_video_id"] == "abc123xyz00"
    assert clip.generation_inputs_summary["source_title"] == "The Long Interview"


def test_run_phase6_writes_caption_metadata_render_and_ass_artifacts(tmp_path: Path) -> None:
    from backend.pipeline.render.contracts import SourceContext
    from backend.pipeline.render.phase6 import run_phase_6

    paths = build_run_paths(output_root=tmp_path, run_id="run_phase6")

    result = run_phase_6(
        paths=paths,
        canonical_timeline=_timeline(),
        shot_tracklet_index=_shot_tracklets(),
        tracklet_geometry=_tracklet_geometry(),
        candidates=[_candidate()],
        nodes=_nodes(),
        source_context=SourceContext(
            source_url="https://www.youtube.com/watch?v=abc123xyz00",
            youtube_video_id="abc123xyz00",
            source_title="The Long Interview",
            source_description="A long conversation about creator workflows.",
            channel_id="channel_123",
            channel_title="Clypt Clips",
            published_at="2026-04-19T00:00:00+00:00",
            default_audio_language="en",
            category_id="22",
            tags=["creators", "workflow"],
            thumbnails={"default": {"url": "https://example.com/thumb.jpg"}},
        ),
    )

    assert paths.caption_plan.exists()
    assert paths.publish_metadata.exists()
    assert paths.render_plan.exists()
    assert paths.source_context.exists()
    assert paths.captions_ass("clip_001").exists()

    render_plan = result["render_plan"]
    assert render_plan.caption_plan_ref.endswith("caption_plan.json")
    assert render_plan.publish_metadata_ref.endswith("publish_metadata.json")
    assert render_plan.clips[0].segments[0].caption_preset_id == "karaoke_focus"
    assert render_plan.clips[0].segments[0].caption_segment_ids

    ass_text = paths.captions_ass("clip_001").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "karaoke_focus" in ass_text
    assert ass_text.count("Dialogue:") >= 2
    assert "Dialogue: 0" not in ass_text
    assert "&H0000FF25" in ass_text


def test_compile_render_plan_selects_tracklet_crop_without_camera_intent() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 610,
                    "preset_id": "karaoke_focus",
                    "default_zone": "lower_safe",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 610,
                            "text": "Nobody saw this coming",
                            "word_ids": ["w1", "w2", "w3", "w4"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "lower_safe",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        }
                    ],
                }
            ],
        },
        publish_metadata={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "Nobody Saw This Coming",
                    "title_alternates": ["This Was Unexpected", "Nobody Saw It"],
                    "description_short": "A surprise reveal lands fast.",
                    "thumbnail_text": "NOBODY SAW THIS",
                    "topic_tags": ["surprise", "reveal", "interview"],
                    "hashtags": ["#surprise", "#reveal", "#interview"],
                    "generation_inputs_summary": {},
                }
            ],
        },
        shot_tracklet_index=_shot_tracklets().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry().model_dump(mode="json"),
    )

    clip = render_plan["clips"][0]
    segment = clip["segments"][0]
    assert segment["shot_id"] == "shot_1"
    assert segment["layout_mode"] == "auto_follow"
    assert segment["primary_tracklet_id"] == "tracklet_1"
    assert clip["crop_plan"]["mode"] == "tracklet_follow_9x16_smooth_inside_person"
    assert clip["crop_plan"]["segments"][0]["tracklet_id"] == "tracklet_1"
    assert clip["crop_plan"]["keyframes"][0]["bbox_xyxy"] == [220.0, 420.0, 820.0, 1520.0]


def test_compile_render_plan_skips_pose_ineligible_auto_tracklets() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 610,
                    "preset_id": "karaoke_focus",
                    "default_zone": "lower_safe",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 610,
                            "text": "Nobody saw this coming",
                            "word_ids": ["w1"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "lower_safe",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        }
                    ],
                }
            ],
        },
        publish_metadata={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "Nobody Saw This Coming",
                    "title_alternates": [],
                    "description_short": "A surprise reveal lands fast.",
                    "thumbnail_text": "NOBODY SAW THIS",
                    "topic_tags": [],
                    "hashtags": [],
                    "generation_inputs_summary": {},
                }
            ],
        },
        shot_tracklet_index=_shot_tracklets_ineligible_then_eligible().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_bad_and_good().model_dump(mode="json"),
    )

    segment = render_plan["clips"][0]["segments"][0]
    assert segment["layout_mode"] == "auto_follow"
    assert segment["primary_tracklet_id"] == "tracklet_good"
    assert render_plan["clips"][0]["crop_plan"]["segments"][0]["tracklet_id"] == "tracklet_good"


def test_compile_render_plan_locks_best_pose_subject_for_entire_shot() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan=_caption_plan_two_segments(),
        publish_metadata=_publish_metadata_basic(),
        shot_tracklet_index=_shot_tracklets_two_segments_same_shot().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_two_segments_same_shot().model_dump(mode="json"),
    )

    segments = render_plan["clips"][0]["segments"]
    assert [segment["primary_tracklet_id"] for segment in segments] == [
        "tracklet_right",
        "tracklet_right",
    ]
    assert {segment["shot_id"] for segment in segments} == {"shot_1"}


def test_compile_render_plan_builds_smooth_person_box_contained_crop_path() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan=_caption_plan_two_segments(),
        publish_metadata=_publish_metadata_basic(),
        shot_tracklet_index=_shot_tracklets_two_segments_same_shot().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_two_segments_same_shot().model_dump(mode="json"),
    )

    crop_plan = render_plan["clips"][0]["crop_plan"]
    assert crop_plan["mode"] == "tracklet_follow_9x16_smooth_inside_person"
    assert crop_plan["crop_width"] < 420
    assert crop_plan["crop_height"] < 880
    assert crop_plan["tracklet_ids"] == ["tracklet_right"]
    assert len(crop_plan["keyframes"]) >= 3
    for keyframe in crop_plan["keyframes"]:
        bbox = keyframe["bbox_xyxy"]
        assert bbox[0] <= keyframe["x"] <= bbox[2] - crop_plan["crop_width"]
        assert bbox[1] <= keyframe["y"] <= bbox[3] - crop_plan["crop_height"]
        assert keyframe["anchor_source"] in {"pose", "bbox_upper_third", "bbox_center"}


def test_run_phase6_fails_fast_for_partial_namespace_timelines_without_canonical_words(tmp_path: Path) -> None:
    from backend.pipeline.render.contracts import SourceContext
    from backend.pipeline.render.phase6 import run_phase_6

    paths = build_run_paths(output_root=tmp_path, run_id="run_phase6_partial")

    with pytest.raises(ValueError, match="cannot be resolved against canonical timeline"):
        run_phase_6(
            paths=paths,
            canonical_timeline=SimpleNamespace(
                turns=[SimpleNamespace(end_ms=610)],
                source_video_url="https://www.youtube.com/watch?v=abc123xyz00",
            ),
            shot_tracklet_index=_shot_tracklets(),
            tracklet_geometry=_tracklet_geometry(),
            candidates=[_candidate()],
            nodes=_nodes(),
            source_context=SourceContext(
                source_url="https://www.youtube.com/watch?v=abc123xyz00",
                youtube_video_id="abc123xyz00",
                source_title="The Long Interview",
                source_description="A long conversation about creator workflows.",
                channel_id="channel_123",
                channel_title="Clypt Clips",
                published_at="2026-04-19T00:00:00+00:00",
                default_audio_language="en",
                category_id="22",
                tags=["creators", "workflow"],
                thumbnails={"default": {"url": "https://example.com/thumb.jpg"}},
            ),
        )


def test_run_phase6_accepts_partial_source_context_dict(tmp_path: Path) -> None:
    from backend.pipeline.render.phase6 import run_phase_6

    paths = build_run_paths(output_root=tmp_path, run_id="run_phase6_source_context_partial")

    result = run_phase_6(
        paths=paths,
        canonical_timeline=_timeline(),
        shot_tracklet_index=_shot_tracklets(),
        tracklet_geometry=_tracklet_geometry(),
        candidates=[_candidate()],
        nodes=_nodes(),
        source_context={
            "source_url": "https://www.youtube.com/watch?v=abc123xyz00",
            "youtube_video_id": "abc123xyz00",
        },
    )

    assert result["source_context"].source_title == "Unknown Source"
    assert result["publish_metadata"].clips[0].generation_inputs_summary["youtube_video_id"] == "abc123xyz00"


def test_compile_render_plan_logs_zone_transitions_from_camera_intent_segments() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 610,
                    "preset_id": "karaoke_focus",
                    "default_zone": "center_band",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 280,
                            "text": "Nobody saw",
                            "word_ids": ["w1", "w2"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "center_band",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        },
                        {
                            "segment_id": "clip_001_seg_002",
                            "start_ms": 280,
                            "end_ms": 610,
                            "text": "this coming",
                            "word_ids": ["w3", "w4"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "center_band",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        },
                    ],
                }
            ],
        },
        publish_metadata={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "Nobody Saw This Coming",
                    "title_alternates": ["This Was Unexpected", "Nobody Saw It"],
                    "description_short": "A surprise reveal lands fast.",
                    "thumbnail_text": "NOBODY SAW THIS",
                    "topic_tags": ["surprise", "reveal", "interview"],
                    "hashtags": ["#surprise", "#reveal", "#interview"],
                    "generation_inputs_summary": {},
                }
            ],
        },
        camera_intent_timeline=_camera_intent_follow_then_split(),
        shot_tracklet_index=_shot_tracklets_with_split().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_below_center().model_dump(mode="json"),
    )

    segments = render_plan["clips"][0]["segments"]
    assert [segment["caption_zone"] for segment in segments] == ["center_band", "center_band"]
    assert segments[0]["review_needed"] is False
    assert segments[0]["review_reasons"] == []
    assert segments[0]["zone_transition_reason"] == ""
    assert segments[1]["review_needed"] is False
    assert segments[1]["review_reasons"] == []
    assert segments[1]["zone_transition_reason"] == ""
    assert segments[1]["layout_mode"] == "split"
    assert segments[1]["primary_tracklet_id"] == "tracklet_1"
    assert segments[1]["secondary_tracklet_id"] == "tracklet_2"


def test_compile_render_plan_keeps_lower_safe_default_even_with_camera_intent() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 610,
                    "preset_id": "karaoke_focus",
                    "default_zone": "lower_safe",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 280,
                            "text": "Nobody saw",
                            "word_ids": ["w1", "w2"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "lower_safe",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        },
                        {
                            "segment_id": "clip_001_seg_002",
                            "start_ms": 280,
                            "end_ms": 610,
                            "text": "this coming",
                            "word_ids": ["w3", "w4"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "lower_safe",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        },
                    ],
                }
            ],
        },
        publish_metadata={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "Nobody Saw This Coming",
                    "title_alternates": ["This Was Unexpected", "Nobody Saw It"],
                    "description_short": "A surprise reveal lands fast.",
                    "thumbnail_text": "NOBODY SAW THIS",
                    "topic_tags": ["surprise", "reveal", "interview"],
                    "hashtags": ["#surprise", "#reveal", "#interview"],
                    "generation_inputs_summary": {},
                }
            ],
        },
        camera_intent_timeline=_camera_intent_follow_then_split(),
        shot_tracklet_index=_shot_tracklets_with_split().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_below_center().model_dump(mode="json"),
    )

    segments = render_plan["clips"][0]["segments"]
    assert [segment["caption_zone"] for segment in segments] == ["lower_safe", "lower_safe"]
    assert segments[0]["layout_mode"] == "follow"
    assert segments[1]["layout_mode"] == "split"
    assert segments[0]["zone_transition_reason"] == ""
    assert segments[1]["zone_transition_reason"] == ""
    assert segments[1]["primary_tracklet_id"] == "tracklet_1"
    assert segments[1]["secondary_tracklet_id"] == "tracklet_2"


def test_compile_render_plan_logs_collision_fallback_and_segment_review_fields() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    render_plan = compile_render_plan(
        run_id="run_phase6",
        caption_plan={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 610,
                    "preset_id": "karaoke_focus",
                    "default_zone": "center_band",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 610,
                            "text": "Nobody saw this coming",
                            "word_ids": ["w1", "w2", "w3", "w4"],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t1"],
                            "placement_zone": "center_band",
                            "highlight_mode": "word_highlight",
                            "review_needed": False,
                            "review_reason": "",
                            "active_word_timings": [],
                        }
                    ],
                }
            ],
        },
        publish_metadata={
            "run_id": "run_phase6",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "Nobody Saw This Coming",
                    "title_alternates": ["This Was Unexpected", "Nobody Saw It"],
                    "description_short": "A surprise reveal lands fast.",
                    "thumbnail_text": "NOBODY SAW THIS",
                    "topic_tags": ["surprise", "reveal", "interview"],
                    "hashtags": ["#surprise", "#reveal", "#interview"],
                    "generation_inputs_summary": {},
                }
            ],
        },
        camera_intent_timeline=_camera_intent_follow(),
        shot_tracklet_index=_shot_tracklets().model_dump(mode="json"),
        tracklet_geometry=_tracklet_geometry_center_collision().model_dump(mode="json"),
    )

    clip = render_plan["clips"][0]
    segment = clip["segments"][0]
    assert segment["caption_zone"] == "lower_safe"
    assert segment["fallback_applied"] is True
    assert segment["zone_transition_reason"] == "collision_fallback:center_band->lower_safe"
    assert segment["review_needed"] is True
    assert "collision" in segment["review_reasons"][0]
    assert clip["review_needed"] is True
    assert clip["review_reasons"] == segment["review_reasons"]


def test_compile_render_plan_fails_fast_when_camera_intent_does_not_cover_segment_interval() -> None:
    from backend.pipeline.render.compiler import compile_render_plan

    with pytest.raises(ValueError, match="unresolved interval"):
        compile_render_plan(
            run_id="run_phase6",
            caption_plan={
                "run_id": "run_phase6",
                "clips": [
                    {
                        "clip_id": "clip_001",
                        "clip_start_ms": 0,
                        "clip_end_ms": 610,
                        "preset_id": "karaoke_focus",
                        "default_zone": "center_band",
                        "segments": [
                            {
                                "segment_id": "clip_001_seg_001",
                                "start_ms": 0,
                                "end_ms": 610,
                                "text": "Nobody saw this coming",
                                "word_ids": ["w1", "w2", "w3", "w4"],
                                "speaker_ids": ["SPEAKER_0"],
                                "turn_ids": ["t1"],
                                "placement_zone": "center_band",
                                "highlight_mode": "word_highlight",
                                "review_needed": False,
                                "review_reason": "",
                                "active_word_timings": [],
                            }
                        ],
                    }
                ],
            },
            publish_metadata={
                "run_id": "run_phase6",
                "clips": [
                    {
                        "clip_id": "clip_001",
                        "title_primary": "Nobody Saw This Coming",
                        "title_alternates": ["This Was Unexpected", "Nobody Saw It"],
                        "description_short": "A surprise reveal lands fast.",
                        "thumbnail_text": "NOBODY SAW THIS",
                        "topic_tags": ["surprise", "reveal", "interview"],
                        "hashtags": ["#surprise", "#reveal", "#interview"],
                        "generation_inputs_summary": {},
                    }
                ],
            },
            camera_intent_timeline=_camera_intent_follow(end_ms=120),
            shot_tracklet_index=_shot_tracklets().model_dump(mode="json"),
            tracklet_geometry=_tracklet_geometry_below_center().model_dump(mode="json"),
        )
