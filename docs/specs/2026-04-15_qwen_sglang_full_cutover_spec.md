# Clypt V3.1 Spec: Full Qwen Cutover from vLLM to SGLang (VibeVoice Unchanged)

**Status:** Active (service/env split implemented; full Phase 1-4 validation pending)  
**Date:** 2026-04-15  
**Owner:** Backend runtime / inference  
**Scope:** Replace Qwen serving runtime from vLLM to SGLang for Phase 2-4 generation only. Keep VibeVoice ASR on existing vLLM path.

---

## 1. Locked Decisions

1. Qwen serving is migrated to **SGLang only** (no vLLM fallback for Qwen).
2. VibeVoice ASR remains on `clypt-vllm-vibevoice.service` unchanged.
3. API contract remains OpenAI-compatible (`/v1/chat/completions`) for pipeline callsites.
4. Existing typed validation (Pydantic + schema checks) remains fail-fast.
5. For SGLang compatibility, remove dynamic `oneOf` schema branches in Phase 4 for now; keep other schema tightening.
6. Target model remains `Qwen/Qwen3.5-27B`.

---

## 2. Why This Change

Recent production failure in Phase 4 was caused by the Qwen vLLM engine crashing while compiling a structured-output schema (xgrammar path), not by a normal model response mismatch.

Observed failure class:

- vLLM EngineCore fatal error during `compile_json_schema`
- xgrammar converter error:
  - `minItems is greater than the number of prefixItems, but additional items are not allowed`
- downstream effects:
  - `/v1/chat/completions` 500s
  - connection refused during retries
  - Phase 2-4 worker restart and stale queue lease

This makes structured-output robustness the priority axis for runtime selection.

---

## 3. Source Guidance Incorporated

### 3.1 Qwen HF Serving Guidance (`Qwen/Qwen3.5-27B`)

From model card deployment section:

- SGLang from current mainline is recommended/supported for Qwen3.5.
- Qwen3.5 defaults to thinking mode; non-thinking responses should be requested via:
  - `extra_body.chat_template_kwargs.enable_thinking=false` on OpenAI-compatible serving.
- Context recommendations:
  - default 262,144 context
  - avoid dropping below 128K when preserving complex reasoning behavior.

### 3.2 SGLang Structured Outputs

SGLang docs indicate:

- structured constraints available for JSON schema / regex / EBNF
- configurable grammar backends:
  - `xgrammar` (default)
  - `outlines`
  - `llguidance`

### 3.3 Practical Compatibility Note

Even with SGLang, schema complexity can still surface backend limitations. We therefore explicitly constrain schema shape for the first migration cut.

---

## 4. Current vs Target Runtime Topology

### 4.1 Current

- `clypt-vllm-vibevoice.service` (ASR) on localhost.
- `clypt-vllm-qwen.service` serving Qwen at `127.0.0.1:8001`.
- `GENAI_GENERATION_BACKEND=local_openai` in Phase 2-4 path.

### 4.2 Target

- Keep: `clypt-vllm-vibevoice.service` (unchanged).
- Replace: `clypt-vllm-qwen.service` with `clypt-sglang-qwen.service` on the same endpoint (`127.0.0.1:8001`).
- Keep provider contract: existing local OpenAI client path continues to call `/v1/chat/completions`.

No app-level fallback to vLLM Qwen remains after cutover.

---

## 5. Migration Streams

## 5.1 Stream A - Service and Runtime Replacement

### A1. New Systemd Unit for Qwen SGLang

Add:

- `scripts/do_phase1/systemd/clypt-sglang-qwen.service`

Requirements:

- bind `127.0.0.1:8001`
- model: `Qwen/Qwen3.5-27B`
- pass Qwen parser settings:
  - `--reasoning-parser qwen3`
- include explicit grammar backend flag (default initial target: `xgrammar`, with documented switch to `outlines` if needed)
- keep restart policy (`Restart=always`) and startup timeout.

### A2. Deployment Script Update

Update deployment automation to install/start SGLang service for Qwen while leaving VibeVoice vLLM flow untouched.

Expected script-level changes:

- add SGLang install/bootstrap steps (from official install guidance)
- split SGLang and Phase 1 into separate host virtualenvs so serving installs cannot mutate the Phase 1 runtime
- disable/remove Qwen vLLM unit management in cutover path
- verify `clypt-sglang-qwen.service` health endpoint before Phase 2-4 worker start.

### A3. Endpoint Stability

Keep `CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1` so Phase 2-4 provider wiring does not require broad callsite changes.

## 5.2 Stream B - Provider and Request Policy

### B1. Keep OpenAI-compatible Client Contract

`backend/providers/openai_local.py` remains the primary client surface.

### B2. Qwen Non-Thinking Default

Enforce per-request payload default:

- `extra_body.chat_template_kwargs.enable_thinking=false`

for deterministic structured-output tasks unless callpoint explicitly requires thinking.

### B3. Structured Output Transport

Continue sending `response_format: {"type":"json_schema", ...}` with strict object normalization already in client path.

---

## 5.3 Stream C - SGLang-Compatible Schema Porting

Goal: preserve strictness where robust, remove known high-risk constructs for first cut.

### C1. Keep

- `type`, `properties`, `required`
- scalar constraints (`minLength`, numeric `minimum`/`maximum`)
- `enum` restrictions (critical for node/edge fields)
- array item typing and simple `minItems` where not tied to tuple-style semantics

### C2. Remove/Defer (for initial SGLang cut)

- dynamic `oneOf` branches used for conditional object semantics in Phase 4
  - specifically the newly added reject-all branch logic in subgraph review schema
- any cross-field conditional schema logic that can be enforced in deterministic Python post-validation instead.

### C3. Replace with Deterministic App Validation

Implement conditional checks in runtime validation layer:

- if `reject_all=true` -> `reject_reason` must be non-empty and `candidates=[]`
- if `reject_all=false` -> `1 <= len(candidates) <= 3`

This keeps behavior strict while reducing guided-decoding schema complexity.

### C4. Compatibility Rulebook

Add a short schema authoring guideline document for generation schemas:

- prefer "flat-object + enums + scalar bounds"
- avoid dynamic branching keywords in serving-time schema
- put relational constraints in Python validators.

---

## 5.4 Stream D - Runtime Admission and Crash Handling (Fail-Fast Mode)

The recent stuck run showed stale `running` lease after service crash. For this phase, prefer debuggability over auto-recovery:

1. Disable automatic lease reclaim/requeue for Qwen backend crash scenarios.
2. On generation-service crash or connection-refused burst, mark the active run as terminal failed immediately with explicit crash signature.
3. Emit high-signal diagnostics (service state, recent stderr signature, queue row snapshot, run_id/job_id correlation).
4. Require manual operator intervention to requeue after root-cause analysis.

This mode is intentionally strict so crash patterns are surfaced early and are not masked by automated recovery.

---

## 6. File-Level Plan and Implementation State

## 6.1 Infra / Service

- Implemented: `scripts/do_phase1/systemd/clypt-sglang-qwen.service`
- Implemented: `scripts/do_phase1/systemd/clypt-v31-phase24-local-worker.service`
- Implemented: `scripts/do_phase1/deploy_sglang_qwen_service.sh`
- Implemented: `scripts/do_phase1/deploy_vllm_service.sh` now provisions a dedicated Phase 1 env and installs TensorRT host/runtime deps when the env selects `tensorrt_fp16`
- Implemented: SGLang deploy now provisions a separate `/opt/clypt-phase1/venvs/sglang` env and installs `ninja-build`
- Implemented: deployment/runtime docs updated to include SGLang Qwen path, separate envs, and TensorRT host requirements

## 6.2 Runtime / Provider

- Implemented: `backend/providers/openai_local.py` non-thinking + strict schema policy retained
- Implemented: `backend/providers/config.py` fail-fast/local queue settings wired
- Pending during host cutover: runtime service verification on fresh droplet

## 6.3 Schemas / Validation

- Implemented: `backend/pipeline/candidates/prompts.py` (`oneOf` removed)
- Implemented: strict enums/bounds retained in:
  - `backend/pipeline/semantics/prompts.py`
  - `backend/pipeline/graph/prompts.py`
  - `backend/pipeline/signals/llm_runtime.py`
- Implemented: deterministic Phase 4 checks moved into Python validation path.

## 6.4 Tests

- Implemented tests:
  - Phase 4 conditional validation moved from schema to Python checks
  - local_openai client payload includes non-thinking defaults
  - fail-fast queue behavior and schema-compat checks in runtime/pipeline test suites.

---

## 7. Cutover Procedure (No Qwen vLLM Fallback)

1. Deploy SGLang Qwen service to `8001`.
   - Use dedicated env `/opt/clypt-phase1/venvs/sglang`.
2. Stop and disable `clypt-vllm-qwen.service`.
3. Keep VibeVoice service running (`clypt-vllm-vibevoice.service`) on existing port.
4. Restart:
   - `clypt-v31-phase1-api.service`
   - `clypt-v31-phase1-worker.service`
   - `clypt-v31-phase24-local-worker.service`
   - All three Phase 1/local-worker services run from `/opt/clypt-phase1/venvs/phase1`.
5. Execute smoke run with `--run-phase14` on test-bank mapped input.
6. Validate:
   - no schema compile crash
   - Phase 2/3/4 terminal completion
   - queue row terminal state (no stale running lock).

No dual-serving or runtime fallback phase is planned in this spec.

---

## 8. Verification Matrix

## 8.1 Functional

1. Phase 2 schema-constrained outputs remain valid (`node_flags`, `node_type`).
2. Phase 3 local + long-range edges persist without schema regressions.
3. Phase 4 candidate generation/review completes with deterministic conditional checks.

## 8.2 Resilience

1. If Qwen service crashes/restarts mid-run, the run is terminally failed immediately with explicit error classification (no auto-reclaim in this phase).
2. No indefinite `running` rows after backend crash.
3. Crash evidence is sufficient to triage one-off vs recurrent failure causes.

## 8.3 Performance

Track for same canonical test video set:

- TTFT / p95 call latency by callpoint
- run wall time (Phase 2-4)
- token throughput
- failure rate by callpoint.

---

## 9. Risks and Mitigations

1. **Risk:** SGLang grammar backend can still reject complex schemas.
   - **Mitigation:** keep schemas in portable subset; move dynamic relations to Python validators.
2. **Risk:** Qwen non-thinking switch behavior differs by stack.
   - **Mitigation:** enforce `chat_template_kwargs.enable_thinking=false`; add request/response conformance tests.
3. **Risk:** Service-level crash causes stale queue lock.
   - **Mitigation:** fail fast and hard: disable auto-reclaim, force terminal failure with crash telemetry, and require manual requeue after debugging.

---

## 10. Acceptance Criteria

This cutover is complete only when all are true:

1. Qwen generation is served exclusively by SGLang in production runtime.
2. VibeVoice remains on existing vLLM ASR service and is unaffected.
3. Phase 2-4 complete successfully on golden test videos without schema-compile crashes.
4. Phase 4 conditional semantics are preserved via deterministic post-schema validation.
5. Documentation and ops runbooks reflect SGLang-first Qwen serving.

---

## 11. Out of Scope (This Cut)

1. Migrating VibeVoice from vLLM.
2. Reintroducing advanced schema branching (`oneOf`) before proving backend stability.
3. Multi-node distributed inference changes.

