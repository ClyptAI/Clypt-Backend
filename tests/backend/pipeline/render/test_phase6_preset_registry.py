from __future__ import annotations


def test_load_caption_presets_exposes_required_ids_and_schema_fields() -> None:
    from backend.pipeline.render.presets import load_caption_presets

    presets = load_caption_presets()

    assert set(presets) >= {
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

    karaoke = presets["karaoke_focus"]
    assert required_fields.issubset(karaoke.model_dump().keys())
    assert karaoke.highlight_mode == "word_highlight"
    assert karaoke.default_zone in {"center_band", "lower_safe"}
