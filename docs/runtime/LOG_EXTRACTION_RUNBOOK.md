# LOG EXTRACTION RUNBOOK

Canonical operator flow for tracing a single run across:

- Phase 1 API SQLite
- Phase26 local queue SQLite
- Spanner (`runs`, `phase24_jobs`, `phase_metrics`, `phase_substeps`)
- journald on the host(s)

Use this before ad-hoc `journalctl`, manual SQLite inspection, or one-off Spanner queries.

## 1) Preferred Entry Point

From a deployed host with the repo synced and the runtime env loaded:

```bash
cd /opt/clypt-phase26/repo
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --json-output /tmp/<run_id>.diagnostics.json
```

What it collects:

- durable Spanner run + job state
- all persisted `phase_metrics`
- all persisted `phase_substeps`
- Phase 1 API job row from `/var/lib/clypt/phase1/jobs.db`
- Phase26 queue row from the configured local SQLite queue
- Phase 1 job log tail when `log_path` exists
- filtered journald snippets for:
  - `clypt-phase1-worker.service`
  - `clypt-phase1-vibevoice.service`
  - `clypt-phase26-worker.service`
  - `clypt-phase26-dispatch.service`

## 2) Known Noisy Lines

The extractor hides the recurring non-blocking Cloud Monitoring exporter noise by default:

- `Failed to export metrics to Cloud Monitoring`
- `monitoring.timeSeries.create`
- `Permission monitoring.timeSeries.create denied`

To include those lines again:

```bash
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --include-noise
```

Treat those warnings as background noise unless they are the only lines explaining a failure. They have not been the root cause in recent Clypt Phase 2-4 failures.

## 3) Useful Flags

Override queue path if needed:

```bash
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --phase24-queue-path /opt/clypt-phase26/repo/backend/outputs/phase24_local_queue.sqlite
```

Skip journald entirely when you only need durable state:

```bash
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --no-journal
```

Add extra units:

```bash
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --journal-unit clypt-phase1-visual.service \
  --journal-unit clypt-phase1-api.service
```

Pin a manual `--since` window:

```bash
python scripts/extract_run_diagnostics.py \
  --run-id <run_id> \
  --journal-since "2026-04-22 01:45:00 UTC"
```

## 4) Reading The Output

Start with:

1. `run status`
2. `phase24 status`
3. `Per-phase metrics`
4. `Phase1 API job`
5. `Phase26 local queue`
6. `Filtered journald`

Key interpretation rules:

- If `run_record.status=FAILED` but `phase1_visual_extraction` later succeeds, that visual work was off the critical path.
- `phase_substeps` is the authoritative place for detailed Phase 2 media-prep batch timings, pooled review diagnostics, and per-subgraph review latency.
- For node-media-prep, the important batch list lives at:
  - `phase_substeps`
  - `phase_name="phase2"`
  - `step_name="media_prep"`
  - `step_key="node_media"`
  - metadata field: `batches`
- `phase1_jobs.status` can lag behind downstream completion because it reflects the Phase 1 API/worker boundary, not the full Phase24 lifecycle.

## 5) Replay Phase 4 Only

For a failure that happened after Phase 3 already succeeded, use the ad-hoc replay helper:

```bash
cd /opt/clypt-phase26/repo
python scripts/tmp_replay_phase4_from_persisted_run.py \
  --source-run-id <failed_run_id> \
  --json-output /tmp/<failed_run_id>.phase4-replay.json
```

What it does:

1. loads `phase1_outputs.json` from the source run’s persisted GCS handoff
2. rebuilds the canonical timeline via `run_phase_1`
3. loads persisted Phase 2 nodes and Phase 3 edges from the source run
4. clones those nodes/edges into a fresh replay run id by default
5. reruns Phase 4 only

Important constraint:

- this is accurate for the recent long-form test-bank runs where comments/trends augmentation was disabled
- if a future source run depended on non-persisted signal inputs, validate signal reproducibility before treating replay output as canonical

## 6) When To Drop To Manual Queries

Use manual inspection only if the extractor does not answer the question.

Typical follow-ups:

- `journalctl -u clypt-phase26-worker.service --since "<timestamp>" --no-pager -o cat | rg <run_id>`
- `sqlite3 /var/lib/clypt/phase1/jobs.db`
- `sqlite3 /opt/clypt-phase26/repo/backend/outputs/phase24_local_queue.sqlite`
- targeted Spanner SQL against `phase_metrics` / `phase_substeps`

If you had to do that, update the extractor instead of relying on memory next time.
