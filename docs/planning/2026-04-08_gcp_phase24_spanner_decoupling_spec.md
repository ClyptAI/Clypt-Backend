# Clypt V3.1 Spec: GCP Decoupling Of Phases 2-4 With Spanner-Backed Graph

**Status:** Draft v1  
**Date:** 2026-04-08  
**Scope:** Production migration of Phases 2-4 from in-process/on-GPU-host execution to GCP CPU workers with Spanner persistence and graph traversal  
**Decision Constraints (locked):**
- Keep Phase 1 on the GPU droplet.
- Move Phase 2-4 orchestration/ranking off the GPU host.
- Use Spanner as the system of record.
- Keep multimodal embeddings.
- Execute Phase 4 graph traversal and deterministic scoring fully in Spanner (GQL/SQL), not in Python app logic.
- No historical backfill required.
- Remove JSON artifact files as primary persistence (no durable filesystem artifact dependency in production).

---

## 1. Why This Migration

Current Phase 1-4 runs are tightly coupled at runtime:
- Phase 1 executes on the GPU host.
- Phase 2-4 can be started from the Phase 1 callback path in the same process/host.
- Phase outputs are persisted as JSON files under `backend/outputs/v3_1/{run_id}/...`.

This is fast for local iteration, but it blocks production goals:
- no durable multi-run graph store,
- weak multi-user/query surface,
- hard to scale Phase 2-4 independently,
- no database-native traversal/search.

Target state: Phase 1 remains GPU-specialized; Phase 2-4 becomes a GCP-native, CPU-scaled, Spanner-backed pipeline.

---

## 2. Current Code Surfaces (Migration Anchors)

### 2.1 Phase 1 -> Phase 2-4 coupling
- `/Users/rithvik/Clypt-V3/backend/phase1_runtime/runner.py`
  - `Phase1JobRunner.run_job(...)` currently invokes `phase14_runner.run(...)` after audio-chain callback.

### 2.2 Phase 2-4 orchestration
- `/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py`
  - `V31LivePhase14Runner.run()` and `run_phase_2/3/4()`.

### 2.3 JSON artifact persistence (to remove as durable store)
- `/Users/rithvik/Clypt-V3/backend/pipeline/artifacts.py`
  - `V31RunPaths`, `save_json(...)`, `load_json(...)`.
- `/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py`
  - multiple `save_json(...)` calls for timeline, graph, and candidate outputs.

---

## 3. Target Architecture

## 3.1 Runtime split

1. **GPU tier (existing droplet)**
- Runs Phase 1 sidecars (ASR, forced alignment, emotion2vec, YAMNet, RF-DETR).
- Publishes Phase 1 outputs to GCS/Spanner.
- Enqueues a `phase24` job to GCP.

2. **CPU tier (new, GCP Cloud Run service)**
- Pulls `phase24` work via authenticated HTTP task dispatch (Cloud Tasks -> Cloud Run).
- Runs Phase 2-4 orchestration.
- Calls Vertex Gemini + Vertex Embeddings.
- Writes nodes/edges/candidates/metrics to Spanner.

3. **Persistence tier (GCP Spanner)**
- Source of truth for runs, graph nodes/edges, embeddings, candidate clips, phase timings, and status.
- Supports graph traversal and vector-driven seed retrieval.

4. **Blob/media tier (GCS)**
- Source video and extracted node media clip URIs.

## 3.2 Control-plane components

- **Cloud Tasks queue:** `clypt-phase24`
  - task payload includes `run_id`, `project_id`, `source_uri`, and pointers needed for Phase 2-4 input hydration.
- **Cloud Run service:** `clypt-phase24-worker`
  - authenticated endpoint receives one job request, processes end-to-end, writes status.
- **Optional future:** Cloud Run Jobs for large backfills/reprocess only.

---

## 4. Spanner Data Model (System Of Record)

Use GoogleSQL dialect (required for Spanner Graph and vector/ANN features).

## 4.1 Core tables

1. `runs`
- `run_id STRING(MAX) NOT NULL`
- `source_url STRING(MAX)`
- `source_video_gcs_uri STRING(MAX)`
- `status STRING(64)` (`PHASE1_DONE`, `PHASE24_QUEUED`, `PHASE24_RUNNING`, `PHASE24_DONE`, `FAILED`)
- `created_at TIMESTAMP`, `updated_at TIMESTAMP`

2. `timeline_turns`
- `(run_id, turn_id)` PK
- speaker/time/transcript + minimal evidence pointers for deterministic replay.

3. `semantic_nodes`
- `(run_id, node_id)` PK
- node timing/type/flags/summary/transcript excerpt
- semantic evidence rollups
- `semantic_embedding ARRAY<FLOAT32>`
- `multimodal_embedding ARRAY<FLOAT32>`

4. `semantic_edges`
- `(run_id, edge_id)` PK (or `(run_id, source_node_id, target_node_id, edge_type)` if deterministic)
- `source_node_id`, `target_node_id`, `edge_type`, confidence/metadata.

5. `clip_candidates`
- `(run_id, clip_id)` PK
- start/end, node membership, pool rank, score, rationale, score breakdown JSON/string.

6. `phase_metrics`
- `(run_id, phase_name)` PK
- `started_at`, `ended_at`, `duration_ms`, `status`, error payload.

7. `phase24_jobs`
- `(run_id)` PK
- dedupe/idempotency lock, attempt_count, last_error, worker metadata.

## 4.2 Spanner Graph mapping

Define graph schema over node/edge tables so traversal can run in-database.

Illustrative DDL pattern (adapt to exact column names):

```sql
CREATE OR REPLACE PROPERTY GRAPH ClyptSemanticGraph
  NODE TABLES (semantic_nodes)
  EDGE TABLES (
    semantic_edges
      SOURCE KEY (run_id, source_node_id) REFERENCES semantic_nodes (run_id, node_id)
      DESTINATION KEY (run_id, target_node_id) REFERENCES semantic_nodes (run_id, node_id)
      LABEL Connects
  );
```

Notes:
- Confirm final `SOURCE KEY`/`DESTINATION KEY` syntax against production DDL validation.
- Keep `run_id` in node+edge keys to guarantee per-run isolation.

## 4.3 Vector search indexes

Create ANN vector indexes on:
- `semantic_nodes.semantic_embedding`
- `semantic_nodes.multimodal_embedding`

Seed retrieval strategy:
- query top-K semantic ANN,
- query top-K multimodal ANN,
- fuse scores with current weighting (`0.75 semantic + 0.25 multimodal`),
- then run graph traversal expansion from selected seeds.

---

## 5. End-To-End Data Flow

1. **Phase 1 completes audio chain on droplet.**
2. Droplet writes required Phase 1-derived canonical payloads to Spanner/GCS.
3. Droplet enqueues `phase24` Cloud Task with `run_id` and pointers.
4. Cloud Tasks calls authenticated Cloud Run worker endpoint.
5. Worker acquires idempotency lease (`phase24_jobs` row) and marks `PHASE24_RUNNING`.
6. Worker executes Phase 2:
- merge/classify,
- boundary reconciliation,
- semantic + multimodal embeddings,
- write `semantic_nodes`.
7. Worker executes Phase 3:
- structural + semantic + long-range edges,
- reconcile,
- write `semantic_edges`.
8. Worker executes Phase 4:
- prompt generation,
- prompt embeddings,
- seed retrieval,
- local subgraph expansion,
- candidate review/ranking,
- write `clip_candidates`.
9. Worker writes phase timings + run status, marks `PHASE24_DONE`.

---

## 6. JSON Removal Plan

## 6.1 Principle

JSON artifact files stop being a required persistence mechanism.

## 6.2 Code changes

1. Introduce repository interface for Phase 1-4 persistence:
- `Phase14Repository` with methods like:
  - `upsert_run(...)`
  - `write_timeline(...)`
  - `write_nodes(...)`
  - `write_edges(...)`
  - `write_candidates(...)`
  - `write_phase_metric(...)`

2. Implement Spanner repository:
- `SpannerPhase14Repository` for all durable writes and reads.

3. Refactor `/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py`:
- remove direct filesystem dependency on `save_json(...)` and `V31RunPaths` for core outputs,
- persist through repository methods.

4. Keep optional debug snapshots only behind explicit flag:
- `CLYPT_DEBUG_SNAPSHOTS=1` -> write non-authoritative payload dumps to GCS (not local JSON dependency).

5. Deprecate/remove `/Users/rithvik/Clypt-V3/backend/pipeline/artifacts.py` from production path.

Result: all production reads/writes come from Spanner (+ GCS for media blobs), not local JSON files.

---

## 7. Graph Traversal In Spanner

Traversal logic should be database-backed for persistent graph operations.

## 7.1 Retrieval + traversal pattern

1. **Seeding:** vector ANN over node embeddings.
2. **Traversal:** hop-constrained traversal over `semantic_edges` (edge-type-aware).
3. **Subgraph scoring:** edge-type weights + temporal coherence + closure bonuses.
4. **Candidate extraction:** contiguous or connected node sets subject to duration/overlap constraints.

## 7.2 Execution location

- Phase 2-4 worker remains orchestrator.
- Seed retrieval, neighborhood expansion, hop-constrained traversal, and deterministic candidate scoring execute in Spanner queries (GQL/SQL).
- Application code is responsible for orchestration, query dispatch, validation of returned payloads, and downstream Gemini review calls.
- Query versions should be tracked explicitly (named query set + rollout version) so ranking changes are auditable.

---

## 8. Queueing, Idempotency, And Failure Policy

## 8.1 Idempotency

- `run_id` is the global idempotency key.
- Worker first writes/locks `phase24_jobs(run_id)` in a transaction.
- If a completed row exists, worker returns success without reprocessing.

## 8.2 Retry policy

- For initial production rollout, configure conservative retries: low max attempts, bounded backoff, and no unbounded retry loops.
- Suggested baseline for `phase24` queue: `maxAttempts=2-3`, short exponential backoff (for transient failures only), then dead-letter/manual replay.
- If you want current behavior parity (`no auto-retry`), queue retry policy can be set to minimal/disabled and manual replay can be used.
- Regardless of queue retry setting, writes remain idempotent.

## 8.3 Failure handling

- On phase failure: write error details in `phase_metrics` + `runs.status=FAILED`.
- Surface failures in Cloud Logging + Monitoring alerts.

---

## 9. Service Boundaries And Infra Placement

## 9.1 Keep on GPU droplet

- Phase 1 heavy inference only.
- Do not execute Phase 2-4 there in production mode.

## 9.2 Move to GCP CPU worker

- `phase24` Cloud Run service (`concurrency=1` recommended at start).
- Tune max instances + concurrency based on Vertex call pressure and per-job memory profile.

## 9.3 IAM

- Droplet enqueuer service account: enqueue tasks only.
- Cloud Tasks invoker SA: invoke Cloud Run worker.
- Worker runtime SA: Spanner read/write + Vertex invoke + GCS read/write.

---

## 10. Migration Rollout (No Backfill)

## Phase A: Schema + repository foundation
- Create Spanner tables and indexes.
- Build repository abstraction.
- Add integration tests for repository writes/reads.

## Phase B: Decouple orchestration
- Modify Phase 1 runner to enqueue `phase24` instead of local direct call.
- Add Cloud Run Phase 2-4 worker entrypoint.

## Phase C: Spanner-first write path
- Refactor `phase14_live` to persist via repository.
- Disable local JSON persistence in production code path.

## Phase D: Staging validation
- Run staged jobs and verify:
  - phase timing integrity,
  - node/edge/candidate counts,
  - rank consistency against expected behavior,
  - traversal correctness.

## Phase E: Production cutover
- Enable queue-driven flow from droplet.
- Monitor SLOs and failure rates.
- Keep manual replay command for failed run_ids.

No historical run backfill is required.

---

## 11. Cutover Validation (No Dual-Read Comparator)

This migration does not require a read-only dual-path comparator.

Validation before production cutover should use:
- documented baseline run data in [v3.1_baseline_reference.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_baseline_reference.md),
- canary rollout percentages,
- run-level guardrails (node/edge/candidate count thresholds, empty-output guards, latency/error SLOs),
- query versioning with fast rollback to prior query set.

Spanner remains the only authoritative read path in production.

---

## 12. File-Level Implementation Plan (First Pass)

### Modify
- `/Users/rithvik/Clypt-V3/backend/phase1_runtime/runner.py`
  - replace in-process Phase 2-4 launch with enqueue call.
- `/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py`
  - remove JSON persistence coupling; write via repository.
- `/Users/rithvik/Clypt-V3/backend/providers/config.py`
  - add GCP queue/Spanner/worker config knobs.

### Add
- `/Users/rithvik/Clypt-V3/backend/repository/phase14_repository.py`
- `/Users/rithvik/Clypt-V3/backend/repository/spanner_phase14_repository.py`
- `/Users/rithvik/Clypt-V3/backend/repository/models.py`
- `/Users/rithvik/Clypt-V3/backend/runtime/run_phase24_worker.py`
- `/Users/rithvik/Clypt-V3/backend/runtime/phase24_worker_app.py`
- `/Users/rithvik/Clypt-V3/backend/providers/task_queue.py` (Cloud Tasks client)

### Decommission from production path
- `/Users/rithvik/Clypt-V3/backend/pipeline/artifacts.py` (or keep only test helper role)

---

## 13. Observability And SLOs

Track per run:
- queue latency (enqueue -> worker start),
- Phase 2/3/4 duration,
- total Phase 2-4 duration,
- Vertex call counts/errors/429s,
- Spanner query latency (ANN + traversal + writes),
- candidate output cardinality.

Initial SLO targets:
- 99% successful Phase 2-4 completion without manual intervention.
- P95 Phase 2-4 wall time within current validated budget envelope.

---

## 14. Phase 2-4 Logging Contract

Phase 2-4 worker logs must be structured JSON logs in Cloud Logging.

Required top-level fields on every log line:
- `timestamp`
- `severity`
- `service` (`clypt-phase24-worker`)
- `environment` (`staging` or `prod`)
- `run_id`
- `job_id` (Cloud Tasks task name or worker-generated id)
- `phase` (`phase2`, `phase3`, `phase4`, or `phase24`)
- `event`
- `attempt`
- `query_version` (for traversal/scoring query bundle)
- `duration_ms` (when applicable)
- `status` (`start`, `success`, `error`, `retrying`, `terminal_failure`)
- `error_code` and `error_message` (error events only)

Required event taxonomy:

| Event | When emitted | Required extras |
|---|---|---|
| `phase_start` | Phase begins | `phase` |
| `phase_success` | Phase completes | `phase`, `duration_ms` |
| `phase_error` | Phase fails | `phase`, `error_code`, `error_message` |
| `phase_retry_scheduled` | Retry planned | `phase`, `attempt`, `next_retry_at` |
| `vertex_call` | Vertex request completes | `provider`, `model`, `latency_ms`, `http_status` |
| `spanner_query` | Traversal/ANN/write query completes | `query_name`, `latency_ms`, `rows_returned` |
| `candidate_summary` | Phase 4 output persisted | `seed_count`, `subgraph_count`, `candidate_count` |
| `run_terminal` | Run reaches terminal state | `final_status`, `total_duration_ms` |

Spanner query telemetry requirements:
- Emit `spanner_query` for ANN seed lookup, traversal expansion, deterministic scoring query, candidate write batch, and status/metrics write.
- Include `query_name` and `query_version` for every query event.
- Include `rows_scanned` when available from query stats.
- Include `is_retry` to distinguish first-attempt vs retry-path query behavior.

Alerting and dashboard minimums:
- Alert if `run_terminal` failure rate exceeds 1% in a rolling 30-minute window.
- Alert if P95 `phase24` duration exceeds baseline envelope by 30% for 30 minutes.
- Alert if `candidate_count=0` exceeds 5% of completed runs in a rolling 1-hour window.
- Alert on repeated Spanner query timeout/error bursts by `query_name`.
- Dashboard slices must support filtering by `run_id`, `phase`, `query_version`, and `attempt`.

Logging quality rules:
- No transcript text, prompt text, or user PII in error logs.
- Truncate all error payloads to bounded size (for example, 2 KB per field).
- Keep stable event names; additive-only changes preferred for parsers/alerts.

---

## 15. Risks And Mitigations

1. **Graph query complexity/perf drift**
- Mitigation: keep deterministic scoring in Spanner from day one, but ship versioned queries, benchmark with representative run sizes, and tune indexes/query plans before full traffic cutover.

2. **Vector retrieval behavior shift**
- Mitigation: lock fusion weights and run staged rank-diff checks.

3. **Idempotency gaps under retries**
- Mitigation: transactional run lock table + upsert-only writes.

4. **Operational coupling between droplet and GCP**
- Mitigation: explicit queue boundary and replay tooling.

---

## 16. External References (for implementation)

- Spanner Graph overview: [docs.cloud.google.com/spanner/docs/graph/overview](https://docs.cloud.google.com/spanner/docs/graph/overview)
- Spanner Graph setup/schema examples: [docs.cloud.google.com/spanner/docs/graph/set-up](https://docs.cloud.google.com/spanner/docs/graph/set-up)
- Spanner ANN/vector search: [docs.cloud.google.com/spanner/docs/find-approximate-nearest-neighbors](https://docs.cloud.google.com/spanner/docs/find-approximate-nearest-neighbors)
- Cloud Tasks -> authenticated HTTP target (Cloud Run): [docs.cloud.google.com/tasks/docs/creating-http-target-tasks](https://docs.cloud.google.com/tasks/docs/creating-http-target-tasks)
- Cloud Run concurrency tuning: [docs.cloud.google.com/run/docs/about-concurrency](https://docs.cloud.google.com/run/docs/about-concurrency)
