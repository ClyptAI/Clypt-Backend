# ERROR LOG

Persistent record of major runtime/deployment/pipeline errors and their recoveries.

## Entry Template

- **Date/Time (UTC):**
- **Subsystem:**
- **Environment:**
- **Symptom / Error signature:**
- **Root cause:**
- **Fix applied:**
- **Verification evidence:**
- **Follow-up guardrails:**

---

## 2026-04-08 - vLLM model ID mismatch

- **Subsystem:** Phase 1 ASR
- **Environment:** Droplet + vLLM service
- **Symptom / Error signature:** HTTP 404 when ASR requests used model ID `microsoft/VibeVoice-ASR`.
- **Root cause:** vLLM served model name is `vibevoice` (`--served-model-name`), not HF repo ID.
- **Fix applied:** Standardized env/docs on `VIBEVOICE_VLLM_MODEL=vibevoice`.
- **Verification evidence:** `/v1/models` includes `id: vibevoice`; ASR smoke tests complete.
- **Follow-up guardrails:** Keep explicit model-id checks in startup/runbook.

## 2026-04-08 - ASR/audio-chain callback delayed behind visual completion

- **Subsystem:** Phase 1 orchestration
- **Environment:** `backend/phase1_runtime/extract.py`
- **Symptom / Error signature:** Audio sidecars started late, only after RF-DETR completion.
- **Root cause:** Callback scheduling depended on an `as_completed` pattern that effectively waited on both futures.
- **Fix applied:** Start audio chain immediately after `asr_future.result()`.
- **Verification evidence:** Audio artifacts become available significantly before visual completion.
- **Follow-up guardrails:** Preserve this concurrency invariant in runtime docs/tests.

## 2026-04-08 - GPU ffmpeg unavailable on queue worker

- **Subsystem:** Phase 2 node media extraction
- **Environment:** Cloud Run Phase 2-4 worker
- **Symptom / Error signature:** `GPU ffmpeg unavailable ... falling back to CPU encoder`.
- **Root cause:** Worker runtime lacked usable GPU ffmpeg path.
- **Fix applied:** Provision tuned worker profile on `us-east4` L4 with GPU ffmpeg configuration.
- **Verification evidence:** Tuned replay reduced Phase 2-4 duration from ~13m41s to ~2m24s on reference clip.
- **Follow-up guardrails:** Keep ffmpeg device/runtime checks in worker startup and runbook.

## 2026-04-09 - Phase 3 long-range strict validation failure

- **Subsystem:** Phase 3 graph long-range edges
- **Environment:** Queue worker runs
- **Symptom / Error signature:** `Gemini returned an edge for a non-shortlisted candidate long-range pair`.
- **Root cause:** Strict validation rejected model output outside shortlisted pair set.
- **Fix applied:** Treat as hard failure with retry/rerun strategy; no silent accept.
- **Verification evidence:** Replays succeed after transient retries/reruns.
- **Follow-up guardrails:** Keep shortlist/output validation strict and observable.

## 2026-04-11 - Spanner schema drift for signal provenance tables

- **Subsystem:** Phase 4 persistence / Spanner repository
- **Environment:** Cloud Run Phase 2-4 worker + `clypt-spanner-v3/clypt-graph-db-v3`
- **Symptom / Error signature:** `404 Table not found: subgraph_provenance` during Phase 4 writes.
- **Root cause:** Worker code expected comments/trends provenance schema, but live Spanner DB was missing newly introduced tables/columns.
- **Fix applied:** Added and executed `scripts/spanner/ensure_phase24_signal_schema.py` (idempotent schema sync), and documented it in runtime/deploy checklists.
- **Verification evidence:** Post-migration runs reached `PHASE24_DONE` with Phase 4 success and provenance/candidate writes.
- **Follow-up guardrails:** Run schema sync after backend changes that touch Phase 4 signal/provenance persistence.

## 2026-04-11 - Cloud Tasks/Cloud Run dispatch instability

- **Subsystem:** Queue handoff (Cloud Tasks -> Cloud Run GPU worker)
- **Environment:** `clypt-phase24` queue + `clypt-phase24-worker` (`us-east4` L4)
- **Symptom / Error signature:** repeated `POST 429 ... no available instance`, intermittent dispatch delays, and confusing cross-region service behavior.
- **Root cause:** Combined factors: stale duplicate Cloud Run service in `us-central1`, missing Cloud Tasks service identity role bindings, and tight L4 quota/capacity pressure in `us-east4`.
- **Fix applied:** removed stale `us-central1` service, restored Cloud Tasks service identity + IAM (`roles/cloudtasks.serviceAgent` project binding and token creator on worker SA), enforced serial queue dispatch (`maxConcurrentDispatches=1`), and redeployed clean worker in `us-east4`.
- **Verification evidence:** post-redeploy runs (`...postredeploy`, `...next`) completed with `PHASE24_DONE`.
- **Follow-up guardrails:** keep only one active worker region, keep queue serial for single-GPU profile, and monitor L4 quota/capacity before burst replays.
