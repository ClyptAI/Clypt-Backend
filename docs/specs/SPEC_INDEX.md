# SPECS INDEX

Active specification documents for current and planned backend work.

## Active Specs

- [2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md](2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md)  
  Current source of truth for the Phase 1-4 concurrency revamp, L4 ASR/media offload topology, and removal of the global concurrency knob.

- [2026-04-15_qwen_sglang_full_cutover_spec.md](2026-04-15_qwen_sglang_full_cutover_spec.md)  
  Current cutover spec; tracks Qwen service migration to SGLang-compatible local OpenAI path and fail-fast queue policy.

- [2026-04-09_comments_trends_augment_spec.md](2026-04-09_comments_trends_augment_spec.md)  
  Comment/trend signal augmentation, hard-join behavior, fail-fast policy, attribution scoring.

- [2026-04-10_phase5_6_spec.md](2026-04-10_phase5_6_spec.md)  
  Planned Phase 5-6 grounding and render pipeline.

## Inactive Specs

- [2026-04-14_single_h200_qwen_phase24_local_spec.md](2026-04-14_single_h200_qwen_phase24_local_spec.md)  
  Superseded vLLM-Qwen plan retained for historical context.

- [2026-04-08_gcp_phase24_spanner_decoupling_spec.md](2026-04-08_gcp_phase24_spanner_decoupling_spec.md)  
  Historical decoupling plan; marked inactive now that the queue/Spanner migration is already implemented.

## Status Vocabulary

- **Active:** current source of truth for implementation.
- **Inactive:** historical context retained for reference, not used as current implementation source of truth.
- **Superseded:** replaced by a newer spec; retained for history.
- **Archived:** historical context, no longer used for implementation decisions.
