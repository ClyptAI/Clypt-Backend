# SPECS INDEX

Active specification documents for current and planned backend work.

## Active Specs

- [2026-05-02_scribe_v2_modal_l40s_phase1_phase26_refactor_spec.md](2026-05-02_scribe_v2_modal_l40s_phase1_phase26_refactor_spec.md)
  Active topology refactor: Phase1 uses ElevenLabs Scribe v2 and Modal RF-DETR, Phase26 keeps Qwen on one MI300X, node-media-prep/render share a second Modal L40S, and Scribe word timings are canonical for caption timing.

- [2026-05-02_phase26_amd_mi300x_sglang_qwen_spec.md](2026-05-02_phase26_amd_mi300x_sglang_qwen_spec.md)
  Phase26 MI300X SGLang/Qwen runtime contract: local SQLite queue, local OpenAI-compatible generation, ROCm SGLang acceptance profiles, and no Nebius or alternate GPU-host fallback.

- [2026-04-20_phase24_node_media_prep_batched_l40s_spec.md](2026-04-20_phase24_node_media_prep_batched_l40s_spec.md)
  Implemented timeline-batched node-media-prep with per-batch submit/poll jobs, pipelined multimodal embedding, hybrid seek/trim extraction, and Modal L40S as the active media-prep target.

- [2026-04-19_phase6_captions_metadata_render_spec.md](2026-04-19_phase6_captions_metadata_render_spec.md)
  Planned Phase6 caption, metadata, and render/export behavior, updated for Scribe-backed canonical words and the shared Modal media worker.

- [2026-04-09_comments_trends_augment_spec.md](2026-04-09_comments_trends_augment_spec.md)
  Comment/trend signal augmentation, hard-join behavior, fail-fast policy, and attribution scoring.

- [2026-04-10_phase5_6_spec.md](2026-04-10_phase5_6_spec.md)
  Planned Phase5-6 grounding and render pipeline.

## Deleted Historical Specs

The older H200, VibeVoice, and Phase1 MI300X switchover specs were removed from this branch as part of the atomic Scribe/Modal + Phase26 MI300X cleanup. Historical debugging context remains in [ERROR_LOG.md](/Users/rithvik/Clypt-Backend/docs/ERROR_LOG.md).
