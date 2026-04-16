# FRONTEND INTEGRATION NOTES

**Last updated:** 2026-04-15

Integration notes for backend contract alignment with the current frontend repo:

- Frontend repo path: `/Users/rithvik/Clypt-Frontend`

## Current Backend Outputs Relevant To Frontend

- Run-level status and phase timing from Spanner (`runs`, `phase_metrics`).
- Candidate outputs and ranks (`clip_candidates` + attribution/provenance fields).
- Phase 1 timeline artifacts (transcript turns/words, tracklets, emotion/audio timelines).
- External signal overlays and links (comments/trends attribution tables).
- Local queue telemetry exists in SQLite (`phase24_jobs`) for operator/debug workflows; this is not currently a frontend API contract.

## Integration Priorities (Near-Term)

1. Keep run-status polling contract stable (queued/running/failed/completed).
2. Keep candidate ranking payload stable while comments/trends scoring evolves.
3. Expose provenance fields needed for UI explainability.
4. Preserve deterministic ordering fields (`pool_rank`, score tie-break behavior).

## Phase 5-6 Forward Contract Considerations

- Participation timeline schema should be frontend-friendly for interactive editing.
- Camera intent timeline should support timeline and transcript-driven UI editing.
- Render plan should expose review-required segments for actionable UX.

## Notes

- This file is intentionally high-level for now.
- Deep frontend contract shaping should happen alongside Phase 5-6 implementation.
- Current code path for Phase 2-4 generation is local OpenAI-compatible backend, but persisted output contracts consumed by frontend remain Spanner-based and unchanged.
