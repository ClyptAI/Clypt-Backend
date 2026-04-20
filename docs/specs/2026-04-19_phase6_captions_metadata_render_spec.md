# Clypt V3.1 Spec: Phase 6 Captions, Publish Metadata, and Render Packaging

**Status:** Active (planned)
**Date:** 2026-04-19
**Scope:** Extend the planned Phase 6 render pipeline so Clypt can compile retention-oriented burned-in captions, lightweight publish metadata, and deterministic final render instructions from existing Phase 1-5 artifacts.

---

## 1. Relationship To Existing Specs

This document is a **Phase 6 supplement** to:

- [2026-04-10_phase5_6_spec.md](2026-04-10_phase5_6_spec.md)

That earlier spec remains the high-level source of truth for:

- Phase 5 participation grounding
- camera intent
- the existence of a deterministic `render_plan.json`

This supplement adds the missing product/output details for:

- caption planning
- caption presets and placement rules
- lightweight publish metadata generation
- the ordering of packaging vs render execution inside Phase 6

If this document conflicts with the earlier Phase 5-6 spec, this document controls the **caption, metadata, and render-packaging** portions of Phase 6.

---

## 2. Locked Decisions

1. **Captions are a Phase 6 responsibility, not a Phase 1 responsibility.** Phase 1 produces the canonical transcript/timing backbone; Phase 6 decides how those words should be chunked, styled, placed, and rendered.
2. **NFA-backed canonical word timings are the primary timing source for captions.** Caption timing must come from Phase 1 `canonical_timeline.words`, not from ad-hoc subtitle regeneration during render.
3. **Phase 6 must not rerun ASR or forced alignment.** Caption rerenders must remain possible from stored artifacts alone.
4. **Caption planning is a distinct artifact from render planning.** Phase 6 emits `caption_plan.json` first, then references it from `render_plan.json`.
5. **Publish metadata generation is also a distinct artifact.** Titles, short descriptions, thumbnail text, and lightweight tags/hashtags are emitted into `publish_metadata.json`, not buried inside `render_plan.json`.
6. **Phase 6 metadata generation happens after Phase 4 final candidate selection, not during ranking.** Ranking remains focused on clip quality and evidence; packaging copy is generated only for finalists.
7. **Caption placement uses named safe zones, not arbitrary freeform coordinates in MVP.** This keeps render behavior deterministic and easier to QA.
8. **Phase 5 camera intent influences caption placement, but does not define caption content.** Participation/camera intent answers where and whom to emphasize visually; transcript words still define caption text.
9. **Phase 6 ships with a small preset library first.** MVP supports a constrained set of caption presets rather than open-ended style synthesis.
10. **Bottom captions must respect mobile-safe UI margins.** “Bottom captions” means a lower-safe zone above platform UI overlays, not text flush against the literal frame bottom.
11. **Word-by-word highlighting is supported as a rendering mode, but the underlying caption chunk still references full word spans.**
12. **Lightweight publish metadata is sufficient for MVP.** Phase 6 should generate high-signal, low-ceremony packaging outputs rather than full social-media campaign copy.

---

## 3. Problem Statement

The current pipeline can:

- extract canonical word- and turn-level transcript timing in Phase 1,
- rank strong clip finalists in Phase 4,
- and persist reasoning/evidence for those finalists.

But it still cannot produce “OpusClip-like” final outputs because it lacks:

- a formal caption compilation stage,
- a small reusable preset system,
- placement rules that adapt to camera intent and shot layout,
- and a packaging step for titles, short metadata, and lightweight tags.

Without those additions, Phase 6 can technically render crops, but it cannot produce creator-ready short-form outputs with burned-in captions and publish-ready packaging.

---

## 4. Current Inputs Already Available

This spec intentionally builds on artifacts the pipeline already produces.

### 4.1 Phase 1 transcript/timing backbone

Phase 1 already yields:

- `canonical_timeline.json`
- `DiarizationPayload.words`
- `DiarizationPayload.turns`

Those artifacts contain:

- `word_id`
- `text`
- `start_ms`
- `end_ms`
- `speaker_id`
- `turn_id`
- `transcript_text`
- `word_ids`

This is enough to build caption chunks without rerunning transcription.

### 4.2 Phase 1 visual geometry backbone

Phase 1 already yields:

- `shot_tracklet_index.json`
- `tracklet_geometry.json`
- shot boundaries
- tracklet bounding boxes over time

This is enough to estimate whether a caption zone would collide with a face/body crop or split-screen layout.

### 4.3 Phase 4 finalist backbone

Phase 4 already persists finalist `clip_candidates` with:

- `clip_id`
- `start_ms`
- `end_ms`
- `score`
- `rationale`
- `pool_rank`
- external attribution/explanation fields when available

This is enough to generate lightweight titles and tags after ranking is complete.

### 4.4 Phase 5 grounding backbone

The base Phase 5/6 spec already defines:

- `participation_timeline.json`
- `camera_intent_timeline.json`

These determine which speaker/tracklet is visually central and whether the clip is operating in `follow`, `reaction`, or `split` mode.

---

## 5. Phase 6 Substep Ordering

Phase 6 should be split into the following deterministic substeps for each finalist clip:

1. `caption_plan`
2. `publish_metadata`
3. `render_plan_compile`
4. `render_execute`
5. optional `export_sidecars`

Where:

- `caption_plan` converts canonical words/turns into styled caption segments.
- `publish_metadata` generates lightweight packaging text.
- `render_plan_compile` resolves camera layout, crops, caption placements, and output assets into one renderer-facing contract.
- `render_execute` produces final media outputs.
- `export_sidecars` optionally emits SRT/VTT/transcript sidecars for downstream tooling.

This ordering is mandatory because:

- captions depend on transcript timing and visual layout,
- metadata depends on the final selected clip interval,
- and the renderer should consume stable, precomputed packaging/caption artifacts rather than mixing business logic into the render loop.

---

## 6. Caption Planning Model

## 6.1 Caption source of truth

The caption planner must consume:

- `CanonicalTimeline.words`
- `CanonicalTimeline.turns`
- clip interval (`start_ms`, `end_ms`)

The planner must slice only the words/turns overlapping the candidate interval and convert them into clip-local time.

## 6.2 Caption chunking unit

The MVP chunking unit is a **caption segment**, defined as:

- a contiguous set of words,
- with one rendered text string,
- one display interval,
- one placement decision,
- and one preset assignment.

Each caption segment must reference the canonical `word_ids` it contains.

## 6.3 Caption chunking heuristics

The planner should use simple deterministic heuristics first:

1. Do not cross a speaker boundary inside a single caption segment.
2. Prefer splitting on punctuation boundaries.
3. Prefer splitting on natural pauses between words.
4. Prefer 5-8 words per segment for mobile readability.
5. Hard cap at 10-12 words per segment.
6. If a sentence is long, split into sequential segments rather than one dense block.
7. Keep caption segments aligned to actual word timings; do not invent timing outside the clip interval.

## 6.4 Timing policy

Each caption segment includes:

- `display_start_ms`
- `display_end_ms`
- referenced `word_ids`

The planner may optionally lead the first word slightly for readability:

- target lead-in: `100-150 ms`
- hard cap: `200 ms`

Word-by-word highlighting, when enabled, must still use the underlying per-word timing from the canonical timeline.

## 6.5 Word highlight mode

Each caption preset may choose one of:

- `phrase_static`
- `phrase_pop`
- `word_highlight`

For `word_highlight`, the caption segment remains one textual phrase, but the renderer also receives ordered word timing spans for highlight animation.

---

## 7. Caption Placement Model

## 7.1 Named safe zones

MVP supports these named zones:

- `lower_safe`
- `center_band`
- `split_band`

### `lower_safe`

Use for:

- generic talking-head clips
- visually dense clips where center text would cover important action
- fallback placement

Behavior:

- horizontally centered
- placed above mobile UI-safe margin
- not flush with literal frame bottom

### `center_band`

Use for:

- single-speaker talking-head segments
- hook-driven clips
- direct-to-camera moments where center placement improves retention and readability

Behavior:

- centered vertically within a designated middle band
- should not obscure the dominant face crop if collision threshold is exceeded

### `split_band`

Use for:

- `split` camera-intent segments
- dual-speaker conversational moments with persistent split composition

Behavior:

- reserved band positioned to coexist with split layout
- may optionally include speaker label/color accents

## 7.2 Placement selection rules

Phase 6 chooses a zone in this order:

1. derive a preferred zone from camera intent and clip composition,
2. test for collision with important visual regions,
3. if collision is too high, fallback to the next safe zone,
4. record the fallback reason in the render plan.

## 7.3 Camera-intent defaults

- `follow` -> prefer `center_band`, fallback `lower_safe`
- `reaction` -> prefer `lower_safe`
- `split` -> prefer `split_band`, fallback `lower_safe`

## 7.4 Collision policy

The planner should reject a preferred zone when it significantly overlaps:

- the primary target tracklet box,
- a split-screen seam,
- or another reserved overlay element.

Exact thresholds can be tuned later; MVP only needs deterministic, logged decisions.

---

## 8. Caption Preset Library

Phase 6 ships with a small preset library instead of freeform styling.

## 8.1 Required preset IDs

1. `bold_center`
2. `karaoke_focus`
3. `clean_lower`
4. `split_speaker`

## 8.2 Preset goals

### `bold_center`

Intended for:

- hooks
- punchy statements
- authority-driven talking points

Characteristics:

- heavy sans-serif
- high contrast
- center-band default
- phrase-level animation

### `karaoke_focus`

Intended for:

- educational content
- interviews
- explanatory speech

Characteristics:

- readable sans-serif
- word-by-word highlight mode
- accent-color active word
- center-band or lower-safe depending on layout

### `clean_lower`

Intended for:

- visually busy footage
- clips where the image should dominate
- safer generic fallback

Characteristics:

- lighter visual treatment
- lower-safe default
- minimal animation

### `split_speaker`

Intended for:

- two-person dialogue
- podcast conversation clips
- persistent split compositions

Characteristics:

- split-band default
- phrase-level captions
- optional speaker tint/label support

## 8.3 Preset schema

Each preset should be represented in a registry with:

- `preset_id`
- `font_family`
- `font_weight`
- `font_case`
- `fill_color`
- `stroke_color`
- `stroke_width`
- `shadow`
- `highlight_mode`
- `default_zone`
- `max_words_per_segment`
- `line_break_policy`
- `speaker_label_mode`

The registry should live as data, not hardcoded scattered constants.

---

## 9. Publish Metadata Generation

## 9.1 When metadata is generated

Titles, short descriptions, tags, hashtags, and thumbnail text should be generated:

- after Phase 4 finalists are locked,
- before render execution,
- as a Phase 6 packaging substep.

They should **not** be generated during Phase 4 ranking itself.

## 9.2 Why Phase 6 owns this

Phase 4 answers:

- which clips are best
- why they are strong

Phase 6 answers:

- how each finalist should be published and rendered

This keeps ranking logic separate from platform/package formatting.

## 9.3 Inputs for metadata generation

The generator should consume:

- finalist clip transcript excerpt
- clip rationale
- semantic node summaries
- external attribution/explanation when present
- source URL / source context

## 9.4 MVP outputs

For each clip, Phase 6 generates:

- `title_primary`
- `title_alternates`
- `description_short`
- `thumbnail_text`
- `topic_tags`
- `hashtags`

## 9.5 Output style guidance

The generator should remain lightweight:

- 1 strong primary title
- 2-4 alternates
- 1 short description
- 3-8 topic tags
- 3-6 hashtags
- 1 concise thumbnail text line

This is enough for publishing workflows without turning Phase 6 into a full social copywriting system.

## 9.6 Important separation rule

`topic_tags` and `hashtags` are packaging outputs, not ranking features.

They must not feed back into Phase 4 scoring in MVP.

---

## 10. New Phase 6 Artifacts

## 10.1 `caption_plan.json`

Top-level shape:

```json
{
  "run_id": "run_123",
  "clips": [
    {
      "clip_id": "clip_001",
      "preset_id": "karaoke_focus",
      "segments": []
    }
  ]
}
```

Each clip entry must include:

- `clip_id`
- `clip_start_ms`
- `clip_end_ms`
- `preset_id`
- `default_zone`
- `segments`

Each caption segment must include:

- `segment_id`
- `start_ms`
- `end_ms`
- `text`
- `word_ids`
- `speaker_ids`
- `turn_ids`
- `placement_zone`
- `highlight_mode`
- `review_needed`
- `review_reason`

Optional per-segment fields:

- `speaker_label`
- `active_word_timings`
- `line_count`
- `fallback_applied`

## 10.2 `publish_metadata.json`

Top-level shape:

```json
{
  "run_id": "run_123",
  "clips": [
    {
      "clip_id": "clip_001",
      "title_primary": "string",
      "title_alternates": [],
      "description_short": "string",
      "thumbnail_text": "string",
      "topic_tags": [],
      "hashtags": []
    }
  ]
}
```

Each clip entry must include:

- `clip_id`
- `title_primary`
- `title_alternates`
- `description_short`
- `thumbnail_text`
- `topic_tags`
- `hashtags`
- `generation_inputs_summary`

## 10.3 Expanded `render_plan.json`

The existing `render_plan.json` from the base Phase 5-6 spec must now also reference:

- `caption_plan_ref`
- `publish_metadata_ref`
- resolved caption zone/preset for each segment
- review-required caption fallbacks

Each render segment should include, at minimum:

- existing camera/tracklet layout fields
- `caption_segment_ids`
- `caption_zone`
- `caption_preset_id`
- `overlays`
- `review_needed`
- `review_reasons`

---

## 11. Renderer Responsibilities

The renderer should not decide caption wording or metadata copy.

It should consume compiled plans and execute them:

1. read `render_plan.json`
2. resolve video crop/layout
3. render caption overlays from referenced `caption_segment_ids`
4. apply preset styling
5. render active-word highlights when enabled
6. export final video
7. optionally export sidecars such as `.srt` / `.vtt`

This keeps planning deterministic and render execution mechanical.

---

## 12. Persistence Requirements

Phase 6 outputs should be persisted so rerender is possible without recomputing earlier phases:

- `caption_plan.json`
- `publish_metadata.json`
- `render_plan.json`
- final rendered asset URIs
- optional subtitle sidecars

These should be addressable per `run_id` and `clip_id`.

---

## 13. Error Handling

## 13.1 Fail-fast cases

Phase 6 should fail fast when:

- canonical transcript words are missing for a selected clip
- clip interval cannot be resolved against canonical timeline
- a requested preset does not exist
- camera intent references unresolved tracklets
- required output artifacts cannot be written

## 13.2 Review-needed cases

Phase 6 should continue but flag `review_needed` when:

- preferred caption zone collides and a fallback zone is used
- a split clip degrades to lower-safe caption placement
- metadata generation produces low-confidence or duplicate titles
- caption chunking hits a hard readability cap and must split awkwardly

These cases should be surfaced in the render plan, not hidden.

---

## 14. Non-Goals

1. No open-ended WYSIWYG caption editor in MVP.
2. No arbitrary draggable caption positioning in MVP.
3. No per-word animated typography beyond supported preset behaviors.
4. No full social post body / thread / description pack generation.
5. No automated thumbnail image generation in this spec.
6. No caption translation or multilingual subtitle support in MVP.

---

## 15. Acceptance Criteria

1. For every Phase 4 finalist, Phase 6 can produce a deterministic `caption_plan.json` using stored canonical word timings.
2. Caption rerendering does not require rerunning VibeVoice or forced alignment.
3. Phase 6 emits `publish_metadata.json` for finalists with titles, alternates, thumbnail text, topic tags, and hashtags.
4. `render_plan.json` references caption and metadata artifacts rather than recomputing them during render execution.
5. MVP supports the four caption presets defined in this spec.
6. Caption placement decisions are deterministic, zone-based, and reviewable.
7. Split-screen clips can render with an explicit `split_band` caption mode or a logged fallback.
8. Bottom captions respect mobile-safe margins rather than using the literal bottom edge.

---

## 16. Recommended Implementation Surfaces

Likely new package additions:

- `backend/pipeline/render/captions/`
- `backend/pipeline/render/metadata/`
- `backend/pipeline/render/presets/`

Likely responsibilities:

- caption chunker
- caption placement resolver
- preset registry
- packaging metadata generator
- render-plan compiler updates

This spec does not force exact filenames yet, but the implementation should keep:

- caption planning
- metadata generation
- and render compilation

as distinct modules rather than one large Phase 6 function.
