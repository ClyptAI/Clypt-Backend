from __future__ import annotations

import importlib


def test_phase6_caption_preset_registry_has_required_ids_and_schema() -> None:
    registry = importlib.import_module("backend.pipeline.render.presets.registry")

    required_ids = {
        "bold_center",
        "karaoke_focus",
        "clean_lower",
        "split_speaker",
    }
    required_fields = {
        "preset_id",
        "font_asset_id",
        "font_family",
        "font_weight",
        "font_case",
        "fill_color",
        "inactive_fill_color",
        "active_fill_color",
        "stroke_color",
        "stroke_width",
        "shadow",
        "highlight_mode",
        "default_zone",
        "max_words_per_segment",
        "line_break_policy",
        "speaker_label_mode",
        "font_size_px_1080x1920",
        "line_height",
        "letter_spacing",
        "max_lines",
        "safe_margin_bottom_px",
        "active_scale",
        "active_pop_in_ms",
        "active_pop_out_ms",
    }

    assert set(registry.CAPTION_PRESET_REGISTRY) == required_ids
    for preset_id, preset in registry.CAPTION_PRESET_REGISTRY.items():
        assert preset["preset_id"] == preset_id
        assert required_fields.issubset(preset)
        assert preset["highlight_mode"] in {"phrase_static", "phrase_pop", "word_highlight"}
        assert preset["default_zone"] in {"lower_safe", "center_band", "split_band"}


def test_phase6_caption_chunking_is_deterministic_and_preserves_word_highlights() -> None:
    registry = importlib.import_module("backend.pipeline.render.presets.registry")
    chunker = importlib.import_module("backend.pipeline.render.captions.chunker")

    canonical_timeline = {
        "words": [
            {"word_id": "w_000001", "text": "We", "start_ms": 0, "end_ms": 120, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000002", "text": "can", "start_ms": 120, "end_ms": 220, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000003", "text": "ship", "start_ms": 220, "end_ms": 320, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000004", "text": "this", "start_ms": 320, "end_ms": 420, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000005", "text": "today.", "start_ms": 420, "end_ms": 560, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000006", "text": "Then", "start_ms": 720, "end_ms": 820, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000007", "text": "we", "start_ms": 820, "end_ms": 920, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000008", "text": "document", "start_ms": 920, "end_ms": 1080, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000009", "text": "the", "start_ms": 1080, "end_ms": 1160, "speaker_id": "SPEAKER_0"},
            {"word_id": "w_000010", "text": "result.", "start_ms": 1160, "end_ms": 1320, "speaker_id": "SPEAKER_0"},
        ],
        "turns": [
            {
                "turn_id": "t_000001",
                "speaker_id": "SPEAKER_0",
                "start_ms": 0,
                "end_ms": 1320,
                "transcript_text": "We can ship this today. Then we document the result.",
                "word_ids": [
                    "w_000001",
                    "w_000002",
                    "w_000003",
                    "w_000004",
                    "w_000005",
                    "w_000006",
                    "w_000007",
                    "w_000008",
                    "w_000009",
                    "w_000010",
                ],
                "identification_match": None,
            }
        ],
    }
    finalist = {
        "clip_id": "clip_001",
        "clip_start_ms": 0,
        "clip_end_ms": 1320,
        "preset_id": "karaoke_focus",
        "default_zone": "center_band",
        "speaker_id": "SPEAKER_0",
    }

    plan_a = chunker.build_caption_plan(
        run_id="run_123",
        canonical_timeline=canonical_timeline,
        finalists=[finalist],
        preset_registry=registry.CAPTION_PRESET_REGISTRY,
    )
    plan_b = chunker.build_caption_plan(
        run_id="run_123",
        canonical_timeline=canonical_timeline,
        finalists=[finalist],
        preset_registry=registry.CAPTION_PRESET_REGISTRY,
    )

    assert plan_a == plan_b
    assert plan_a["run_id"] == "run_123"
    clip_plan = plan_a["clips"][0]
    assert clip_plan["clip_id"] == "clip_001"
    assert clip_plan["preset_id"] == "karaoke_focus"
    assert clip_plan["default_zone"] == "center_band"
    assert [segment["text"] for segment in clip_plan["segments"]] == [
        "We can ship this today.",
        "Then we document the result.",
    ]
    assert clip_plan["segments"][0]["highlight_mode"] == "word_highlight"
    assert clip_plan["segments"][0]["word_ids"] == [
        "w_000001",
        "w_000002",
        "w_000003",
        "w_000004",
        "w_000005",
    ]
    assert clip_plan["segments"][0]["active_word_timings"] == [
        {"word_id": "w_000001", "start_ms": 0, "end_ms": 120, "text": "We"},
        {"word_id": "w_000002", "start_ms": 120, "end_ms": 220, "text": "can"},
        {"word_id": "w_000003", "start_ms": 220, "end_ms": 320, "text": "ship"},
        {"word_id": "w_000004", "start_ms": 320, "end_ms": 420, "text": "this"},
        {"word_id": "w_000005", "start_ms": 420, "end_ms": 560, "text": "today."},
    ]
    assert clip_plan["segments"][0]["review_needed"] is False
    assert clip_plan["segments"][0]["review_reason"] == ""


def test_phase6_publish_metadata_is_deterministic_and_source_context_driven() -> None:
    generator = importlib.import_module("backend.pipeline.render.metadata.generator")

    source_context = {
        "source_url": "https://youtube.com/watch?v=abc123",
        "youtube_video_id": "abc123",
        "source_title": "How We Ship Faster",
        "source_description": "A long-form discussion about shortening iteration loops.",
        "channel_id": "channel_001",
        "channel_title": "Build Notes",
        "published_at": "2026-04-19T18:00:00Z",
        "default_audio_language": "en",
        "category_id": "28",
        "tags": ["shipping", "productivity", "engineering"],
        "thumbnails": {"default": {"url": "https://example.com/thumb.jpg"}},
    }
    finalist = {
        "clip_id": "clip_001",
        "transcript_excerpt": "We can ship this today.",
        "rationale": "Short, clear payoff line with a strong hook.",
        "semantic_node_summaries": ["ship today"],
        "external_attribution": {"comment_count": 12},
    }

    metadata_a = generator.build_publish_metadata(
        run_id="run_123",
        source_context=source_context,
        finalists=[finalist],
    )
    metadata_b = generator.build_publish_metadata(
        run_id="run_123",
        source_context=source_context,
        finalists=[finalist],
    )

    assert metadata_a == metadata_b
    assert metadata_a["run_id"] == "run_123"
    clip_metadata = metadata_a["clips"][0]
    assert clip_metadata["clip_id"] == "clip_001"
    assert clip_metadata["title_primary"]
    assert 2 <= len(clip_metadata["title_alternates"]) <= 4
    assert 1 <= len(clip_metadata["description_short"].split()) <= 40
    assert 1 <= len(clip_metadata["thumbnail_text"].split()) <= 8
    assert 3 <= len(clip_metadata["topic_tags"]) <= 8
    assert 3 <= len(clip_metadata["hashtags"]) <= 6
    assert clip_metadata["generation_inputs_summary"]["source_context"]["youtube_video_id"] == "abc123"
    assert clip_metadata["generation_inputs_summary"]["source_context"]["source_title"] == "How We Ship Faster"
    assert clip_metadata["generation_inputs_summary"]["finalist"]["clip_id"] == "clip_001"
    assert clip_metadata["generation_inputs_summary"]["finalist"]["rationale"] == "Short, clear payoff line with a strong hook."


def test_phase6_render_plan_references_caption_and_metadata_artifacts_and_compiles_ass() -> None:
    compiler = importlib.import_module("backend.pipeline.render.compiler")

    caption_plan = {
        "run_id": "run_123",
        "clips": [
            {
                "clip_id": "clip_001",
                "clip_start_ms": 0,
                "clip_end_ms": 1320,
                "preset_id": "karaoke_focus",
                "default_zone": "center_band",
                "segments": [
                    {
                        "segment_id": "clip_001_seg_001",
                        "start_ms": 0,
                        "end_ms": 560,
                        "text": "We can ship this today.",
                        "word_ids": [
                            "w_000001",
                            "w_000002",
                            "w_000003",
                            "w_000004",
                            "w_000005",
                        ],
                        "speaker_ids": ["SPEAKER_0"],
                        "turn_ids": ["t_000001"],
                        "placement_zone": "center_band",
                        "highlight_mode": "word_highlight",
                        "review_needed": False,
                        "review_reason": "",
                        "active_word_timings": [
                            {"word_id": "w_000001", "start_ms": 0, "end_ms": 120, "text": "We"},
                            {"word_id": "w_000002", "start_ms": 120, "end_ms": 220, "text": "can"},
                        ],
                    }
                ],
            }
        ],
    }
    publish_metadata = {
        "run_id": "run_123",
        "clips": [
            {
                "clip_id": "clip_001",
                "title_primary": "We Can Ship This Today",
                "title_alternates": ["Ship It Today", "The Fastest Path Wins"],
                "description_short": "A crisp reminder that the work is already ready to ship.",
                "thumbnail_text": "SHIP TODAY",
                "topic_tags": ["shipping", "execution", "productivity"],
                "hashtags": ["#shipping", "#buildinpublic", "#productivity"],
                "generation_inputs_summary": {
                    "source_context": {"youtube_video_id": "abc123", "source_title": "How We Ship Faster"},
                    "finalist": {"clip_id": "clip_001", "rationale": "Short, clear payoff line with a strong hook."},
                },
            }
        ],
    }
    render_inputs = {
        "run_id": "run_123",
        "clip_finalists": [
            {
                "clip_id": "clip_001",
                "start_ms": 0,
                "end_ms": 1320,
                "score": 9.2,
                "rationale": "Short, clear payoff line with a strong hook.",
                "pool_rank": 1,
            }
        ],
        "caption_plan": caption_plan,
        "publish_metadata": publish_metadata,
        "source_context": {"youtube_video_id": "abc123", "source_title": "How We Ship Faster"},
        "participation_timeline": {
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 1320,
                    "speaker_ids": ["SPEAKER_0"],
                    "tracklet_ids": ["shot_0001:Global_Person_0"],
                    "binding_source": "speaker_id_map",
                    "shot_id": "shot_0001",
                    "user_confirmed": True,
                }
            ]
        },
        "camera_intent_timeline": {
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 1320,
                    "intent": "follow",
                    "primary_tracklet_id": "shot_0001:Global_Person_0",
                    "secondary_tracklet_id": None,
                    "clip_candidate_id": "clip_001",
                    "user_confirmed": True,
                }
            ]
        },
    }

    render_plan = compiler.compile_render_plan(**render_inputs)
    ass_text = compiler.compile_ass_subtitles(
        run_id="run_123",
        clip_id="clip_001",
        caption_plan=caption_plan,
        publish_metadata=publish_metadata,
        render_plan=render_plan,
    )

    clip_render = render_plan["clips"][0]
    assert clip_render["caption_plan_ref"] == "caption_plan.json"
    assert clip_render["publish_metadata_ref"] == "publish_metadata.json"
    assert clip_render["caption_segment_ids"] == ["clip_001_seg_001"]
    assert clip_render["caption_zone"] == "center_band"
    assert clip_render["caption_preset_id"] == "karaoke_focus"
    assert clip_render["review_needed"] is False
    assert clip_render["review_reasons"] == []
    assert clip_render["overlays"]
    assert clip_render["segments"][0]["review_needed"] is False
    assert clip_render["segments"][0]["review_reasons"] == []
    assert clip_render["segments"][0]["fallback_applied"] is False
    assert clip_render["segments"][0]["zone_transition_reason"] == ""
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "karaoke_focus" in ass_text
    assert ass_text.count("Dialogue:") == 3


def test_phase6_render_plan_logs_zone_fallbacks_for_collision_review_cases() -> None:
    compiler = importlib.import_module("backend.pipeline.render.compiler")

    render_plan = compiler.compile_render_plan(
        run_id="run_123",
        caption_plan={
            "run_id": "run_123",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 0,
                    "clip_end_ms": 560,
                    "preset_id": "karaoke_focus",
                    "default_zone": "center_band",
                    "segments": [
                        {
                            "segment_id": "clip_001_seg_001",
                            "start_ms": 0,
                            "end_ms": 560,
                            "text": "We can ship this today.",
                            "word_ids": [
                                "w_000001",
                                "w_000002",
                                "w_000003",
                                "w_000004",
                                "w_000005",
                            ],
                            "speaker_ids": ["SPEAKER_0"],
                            "turn_ids": ["t_000001"],
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
            "run_id": "run_123",
            "clips": [
                {
                    "clip_id": "clip_001",
                    "title_primary": "We Can Ship This Today",
                    "title_alternates": ["Ship It Today", "The Fastest Path Wins"],
                    "description_short": "A crisp reminder that the work is already ready to ship.",
                    "thumbnail_text": "SHIP TODAY",
                    "topic_tags": ["shipping", "execution", "productivity"],
                    "hashtags": ["#shipping", "#buildinpublic", "#productivity"],
                    "generation_inputs_summary": {},
                }
            ],
        },
        camera_intent_timeline={
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 560,
                    "intent": "follow",
                    "primary_tracklet_id": "shot_0001:Global_Person_0",
                    "secondary_tracklet_id": None,
                    "clip_candidate_id": "clip_001",
                    "user_confirmed": True,
                }
            ]
        },
        shot_tracklet_index={
            "tracklets": [
                {
                    "tracklet_id": "shot_0001:Global_Person_0",
                    "shot_id": "shot_0001",
                    "start_ms": 0,
                    "end_ms": 560,
                    "representative_thumbnail_uris": [],
                }
            ]
        },
        tracklet_geometry={
            "tracklets": [
                {
                    "tracklet_id": "shot_0001:Global_Person_0",
                    "shot_id": "shot_0001",
                    "points": [
                        {
                            "frame_index": 0,
                            "timestamp_ms": 120,
                            "bbox_xyxy": [220.0, 680.0, 840.0, 1500.0],
                        }
                    ],
                }
            ]
        },
    )

    segment = render_plan["clips"][0]["segments"][0]
    assert segment["caption_zone"] == "lower_safe"
    assert segment["fallback_applied"] is True
    assert segment["review_needed"] is True
    assert "collision" in segment["review_reasons"][0]
    assert segment["zone_transition_reason"] == "collision_fallback:center_band->lower_safe"
