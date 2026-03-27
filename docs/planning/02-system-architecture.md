# System Architecture

See also: [Planning Index](./README.md), [Product and Demo](./01-product-and-demo.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

For Phase 1, the real source of truth is code:
- `backend/do_phase1_worker.py`
- `backend/do_phase1_service/`
- `backend/pipeline/phase_1_do_pipeline.py`

## High-Level Architecture

```text
Video URL
  -> DO Phase 1 API
  -> DO worker service
  -> phase_1 manifest + artifacts in GCS
  -> local compatibility materialization
  -> Gemini semantic graph phases
  -> Spanner + GCS
  -> clip selection
  -> render output
```

## Phase Responsibilities

### Phase 1
The active extraction system:
1. downloads source media on the DO side
2. converts audio to 16kHz mono WAV
3. runs Parakeet ASR
4. runs YOLO26s + BoT-SORT tracking
5. builds canonical face observations and identity features
6. clusters tracklets into global identities
7. runs LR-ASD or heuristic speaker binding
8. emits `phase_1_visual` / `phase_1_audio`
9. persists a manifest and artifacts to GCS

### Phase 2A / 2B / 3 / 4 / 5
These phases remain local-orchestrated and Gemini-backed. They consume the Phase 1 ledgers rather than calling the old Google extraction APIs.

## Current Phase 1 Runtime Model

| Layer | Responsibility |
|---|---|
| `backend/pipeline/phase_1_do_pipeline.py` | submit DO jobs, poll, fetch manifest, materialize local compatibility artifacts |
| `backend/do_phase1_service/app.py` | async job API |
| `backend/do_phase1_service/worker.py` | persistent job-claiming worker processes |
| `backend/do_phase1_service/extract.py` | media download, GPU slot control, local extractor execution, manifest persistence |
| `backend/do_phase1_worker.py` | actual ASR + tracking + identity + speaker-binding implementation |

## Important Phase 1 Reality Checks

- No Google Video Intelligence in the active path
- No `phase_1a_reconcile` stage in the active path
- No TalkNet in the active path
- No requirement for Modal in the active path
- `phase_1_visual` / `phase_1_audio` are the contract outputs; `phase_1a_*` aliases remain for compatibility only

## Compatibility Bridge

After a DO job succeeds, the local pipeline still re-downloads the source media and rewrites local compatibility artifacts for downstream tools that assume local files. That is current behavior, even though the canonical source video is already uploaded by the DO service.
