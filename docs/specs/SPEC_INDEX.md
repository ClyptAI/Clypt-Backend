# SPECS INDEX

Active specification documents for current and planned backend work.

## Active Specs

- [2026-04-16_qwen36_swap_and_sglang_tuning_spec.md](2026-04-16_qwen36_swap_and_sglang_tuning_spec.md)  
  Current source of truth for the Phase 2-4 generation model (`Qwen/Qwen3.6-35B-A3B` on SGLang ≥ 0.5.10), SGLang launch flags (FP8 KV, NextN MTP, radix cache, `mem-fraction-static=0.78`), strict-JSON sampler defaults, and the full-transition doctrine (no Qwen3.5-27B fallback retained). Supersedes `2026-04-15_qwen_sglang_full_cutover_spec.md` on Qwen serving.

- [2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md](2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md)  
  Current source of truth for the Phase 1-4 concurrency revamp, L4 ASR/media offload topology, and removal of the global concurrency knob.

- [2026-04-09_comments_trends_augment_spec.md](2026-04-09_comments_trends_augment_spec.md)  
  Comment/trend signal augmentation, hard-join behavior, fail-fast policy, attribution scoring.

- [2026-04-10_phase5_6_spec.md](2026-04-10_phase5_6_spec.md)  
  Planned Phase 5-6 grounding and render pipeline.

## Inactive Specs

- [2026-04-15_qwen_sglang_full_cutover_spec.md](2026-04-15_qwen_sglang_full_cutover_spec.md)  
  Superseded by `2026-04-16_qwen36_swap_and_sglang_tuning_spec.md`. Documented the vLLM→SGLang cutover on the prior Qwen3.5 generation model; the cutover itself is landed and the model has moved to Qwen3.6-35B-A3B.

- [2026-04-14_single_h200_qwen_phase24_local_spec.md](2026-04-14_single_h200_qwen_phase24_local_spec.md)  
  Superseded vLLM-Qwen plan retained for historical context.

- [2026-04-08_gcp_phase24_spanner_decoupling_spec.md](2026-04-08_gcp_phase24_spanner_decoupling_spec.md)  
  Historical decoupling plan; marked inactive now that the queue/Spanner migration is already implemented.

## Status Vocabulary

- **Active:** current source of truth for implementation.
- **Inactive:** historical context retained for reference, not used as current implementation source of truth.
- **Superseded:** replaced by a newer spec; retained for history.
- **Archived:** historical context, no longer used for implementation decisions.
