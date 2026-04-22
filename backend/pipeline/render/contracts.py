from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, NonNegativeInt, model_validator

from backend.pipeline.contracts import StrictModel


HighlightMode = Literal["phrase_static", "phrase_pop", "word_highlight"]
CaptionZone = Literal["lower_safe", "center_band", "split_band"]
FontCase = Literal["as_is", "upper", "title"]
LineBreakPolicy = Literal["balanced", "punctuation_first", "single_line"]
SpeakerLabelMode = Literal["none", "speaker_id"]


class SourceContext(StrictModel):
    source_url: str
    youtube_video_id: str
    source_title: str
    source_description: str
    channel_id: str
    channel_title: str
    published_at: str
    default_audio_language: str
    category_id: str
    tags: list[str] = Field(default_factory=list)
    thumbnails: dict[str, Any] = Field(default_factory=dict)


class CaptionPresetShadow(StrictModel):
    color: str
    blur_radius: int
    offset_x: int
    offset_y: int
    opacity: float


class CaptionPreset(StrictModel):
    preset_id: str
    font_asset_id: str
    font_family: str
    font_weight: int
    font_case: FontCase
    fill_color: str
    inactive_fill_color: str
    active_fill_color: str
    stroke_color: str
    stroke_width: float
    shadow: CaptionPresetShadow
    highlight_mode: HighlightMode
    default_zone: CaptionZone
    max_words_per_segment: int
    line_break_policy: LineBreakPolicy
    speaker_label_mode: SpeakerLabelMode
    font_size_px_1080x1920: int
    line_height: float
    letter_spacing: float
    max_lines: int
    safe_margin_bottom_px: int
    active_scale: float
    active_pop_in_ms: int
    active_pop_out_ms: int


class ActiveWordTiming(StrictModel):
    word_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    text: str

    @model_validator(mode="after")
    def _check_time_order(self) -> "ActiveWordTiming":
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class CaptionSegment(StrictModel):
    segment_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    text: str
    word_ids: list[str] = Field(default_factory=list)
    speaker_ids: list[str] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    placement_zone: CaptionZone
    highlight_mode: HighlightMode
    review_needed: bool = False
    review_reason: str = ""
    fallback_applied: bool = False
    zone_transition_reason: str = ""
    active_word_timings: list[ActiveWordTiming] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self) -> "CaptionSegment":
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class CaptionPlanClip(StrictModel):
    clip_id: str
    clip_start_ms: NonNegativeInt
    clip_end_ms: NonNegativeInt
    preset_id: str
    default_zone: CaptionZone
    segments: list[CaptionSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self) -> "CaptionPlanClip":
        if self.clip_start_ms > self.clip_end_ms:
            raise ValueError("clip_start_ms must be <= clip_end_ms")
        return self


class CaptionPlan(StrictModel):
    run_id: str
    clips: list[CaptionPlanClip] = Field(default_factory=list)


class PublishMetadataClip(StrictModel):
    clip_id: str
    title_primary: str
    title_alternates: list[str] = Field(default_factory=list)
    description_short: str
    thumbnail_text: str
    topic_tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    generation_inputs_summary: dict[str, Any] = Field(default_factory=dict)


class PublishMetadata(StrictModel):
    run_id: str
    clips: list[PublishMetadataClip] = Field(default_factory=list)


class RenderPlanSegment(StrictModel):
    segment_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    caption_segment_ids: list[str] = Field(default_factory=list)
    caption_preset_id: str
    caption_zone: CaptionZone
    highlight_mode: HighlightMode
    shot_id: str | None = None
    layout_mode: str | None = None
    primary_tracklet_id: str | None = None
    secondary_tracklet_id: str | None = None
    semantic_reason: str = ""
    review_needed: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    fallback_applied: bool = False
    zone_transition_reason: str = ""
    overlays: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self) -> "RenderPlanSegment":
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class RenderPlanClip(StrictModel):
    clip_id: str
    clip_start_ms: NonNegativeInt
    clip_end_ms: NonNegativeInt
    caption_plan_ref: str
    publish_metadata_ref: str
    caption_segment_ids: list[str] = Field(default_factory=list)
    caption_zone: CaptionZone
    caption_preset_id: str
    review_needed: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    overlays: list[dict[str, Any]] = Field(default_factory=list)
    segments: list[RenderPlanSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self) -> "RenderPlanClip":
        if self.clip_start_ms > self.clip_end_ms:
            raise ValueError("clip_start_ms must be <= clip_end_ms")
        return self


class RenderPlan(StrictModel):
    run_id: str
    source_context_ref: str
    caption_plan_ref: str
    publish_metadata_ref: str
    clips: list[RenderPlanClip] = Field(default_factory=list)
