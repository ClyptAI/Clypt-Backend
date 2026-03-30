# Clypt Planning Docs

These planning docs have been trimmed to match the active architecture more closely.

For **Phase 1**, the real source of truth is code, not planning prose:
- `backend/do_phase1_worker.py`
- `backend/do_phase1_service/`
- `backend/pipeline/phase_1_do_pipeline.py`

## Read Order
1. [Product and Demo](./01-product-and-demo.md)
2. [System Architecture](./02-system-architecture.md)
3. [Agents and Clipping](./03-agents-and-clipping.md)
4. [Data, Integrations, and Reference](./04-data-integrations-and-reference.md)

## Doc Map
- `01-product-and-demo.md`: current product framing and demo narrative.
- `02-system-architecture.md`: current runtime architecture and phase responsibilities.
- `03-agents-and-clipping.md`: implemented agent/model roles and clipping behavior.
- `04-data-integrations-and-reference.md`: integrations, artifacts, and storage/runtime reference.

## Legacy Material
- Older planning docs that described Google Video Intelligence, `phase_1a_reconcile`, TalkNet, or YOLO11 are no longer authoritative for Phase 1.
- For current implementation decisions, use `docs/superpowers/specs/clypt_v3_refactor_spec.md`.
