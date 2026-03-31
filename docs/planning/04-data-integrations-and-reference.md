# Data, Integrations, and Reference

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md)

## Primary Integrations

| Integration | Usage |
|---|---|
| YouTube + `yt-dlp` | source media acquisition |
| DigitalOcean Phase 1 service | async extraction API + worker runtime |
| Parakeet TDT 1.1B | transcription with word-level timing |
| YOLOv26-seg + ByteTrack | person/segment tracking (`yolo26m-seg.pt` default, ByteTrack-only backend in worker) |
| SCRFD + ArcFace/InsightFace face-track pipeline | face observations, identity stabilization, and canonical face stream |
| LR-ASD (+ optional heuristic paths per worker env) | speaker binding |
| Gemini | node decomposition, narrative edges, clip scoring |
| Vertex embeddings | node and query vectorization |
| Cloud Spanner | graph + vector storage |
| Cloud Storage | artifact persistence |
| Remotion / QA renderers | final clips and inspection renders |

## Core Data Artifacts

| File | Produced By | Consumed By |
|---|---|---|
| `phase_1_visual.json` | Phase 1 DO pipeline materialization | later phases, renderers, QA tools |
| `phase_1_audio.json` | Phase 1 DO pipeline materialization | later phases, clip scoring, QA tools |
| `phase_1_visual.ndjson` | Phase 1 DO pipeline materialization | compatibility consumers |
| `phase_2a_nodes.json` | Phase 2A | 2B, 3, 4, 5 |
| `phase_2b_narrative_edges.json` | Phase 2B | 4, 5 |
| `phase_3_embeddings.json` | Phase 3 | 4, 5 |
| `remotion_payloads_array.json` | Phase 5 auto-curate | renderers |
| `remotion_payload.json` | Phase 5 retrieve | renderers |

Artifact notes:
- `phase_1_visual.json` includes `tracks`, `shot_changes`, `person_detections`, `face_detections`, `object_tracking`, `label_detections`, and `tracking_metrics` (v3 contract in `backend/pipeline/phase1_contract.py`; lists are coerced to `[]` in `backend/pipeline/phase_1_do_pipeline.py` when materializing).
- `phase_1_audio.json` includes word timings, speaker bindings (including local/follow variants when emitted), and optional overlap-follow fields validated against the same contract.
- `proxy_face_detections` may exist as a compatibility bridge even though real `face_detections` are preferred.
- No `phase_1a_speaker_map.json` is part of the active path.

## Manifest Flow

The DO service persists a contract `v3` manifest plus uploaded artifacts. The local pipeline fetches the manifest from the DO API and materializes local copies for downstream compatibility.

## Infrastructure Summary

| Resource | Service | Purpose |
|---|---|---|
| DO Phase 1 endpoint | DigitalOcean | async extraction job target |
| `clypt-v3` | GCP project | non-extraction cloud resources |
| `clypt-storage-v3` | Cloud Storage | artifact storage |
| `clypt-spanner-v3` | Spanner instance | database host |
| `clypt-graph-db-v3` | Spanner database | graph + vectors |

## Runtime Dependencies

Python:
- Phase 1: `requirements-do-phase1.txt`
- broader local pipeline: `requirements.txt`

Node:
- `remotion-render/` for render-time dependencies
