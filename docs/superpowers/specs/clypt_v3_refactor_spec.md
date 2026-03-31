# Phase 1 Refactor - Clypt v3 Implementation Spec

**Status:** Proposed  
**Date:** 2026-03-30  
**Scope:** `backend/do_phase1_worker.py`, `backend/overlap_follow.py`, `backend/test-render/render_speaker_follow_clips.py`, `backend/pipeline/phase1_contract.py`, `backend/do_phase1_service/storage.py`, `backend/do_phase1_service/models.py`, `backend/do_phase1_service/.env.example`  
**Primary objective:** Maximize accuracy and speed.  
**Constraints:** No requirement to preserve v2 compatibility, no detect-only rollback path, no full heuristic-binding path.

## Phase 1 Refactor - Goal
Build a Phase 1 pipeline optimized for highest practical assignment quality and throughput on NVIDIA H200 hardware, with architecture and defaults aligned to a v3 output contract.

## Phase 1 Refactor - General Build Policy
- Build for production-grade operability: robust, clean, and organized logging is a required baseline across all stages.
- Logging must be structured and stage-scoped (tracking, faces, clustering, binding, overlap-follow, persistence), with consistent identifiers (`job_id`, stage name, span/window identifiers) for traceability.
- Emit concise summary logs at stage boundaries and actionable error logs at failure points; avoid noisy per-frame spam by default.
- Fail-fast policy is mandatory for required runtime dependencies and contract-critical invariants: do not silently degrade to hidden fallback paths.
- Any fallback that remains must be explicit in logs with reason codes, confidence context, and counters, so quality regressions are diagnosable.
- Required structured fields in all stage logs: `job_id`, `stage`, `event`, `decision_source`, `reason_code`, `elapsed_ms`, `worker_id`.
- Overlap-follow and fallback events must emit counters and reason codes (for example: `gemini_unavailable`, `gemini_invalid_response`, `low_overlap_evidence`, `deterministic_selected`).

## Phase 1 Refactor - Hard Product Decisions

### Phase 1 Refactor - Decision 1 (Tracking): Option A + D
- **Selected:** ByteTrack + camera-cut awareness + post-tracking ReID merge.
- **Not selected:** BoT-SORT fallback flag path (explicitly removed from strategy).
- **Implementation requirement:** Remove BoT-SORT runtime branch and unknown-backend fallback behavior from the worker code path.
- **Rationale:** Keep ByteTrack throughput while recovering identity continuity with post-pass ReID across cuts.

### Phase 1 Refactor - Decision 2 (Clustering): Option C
- **Selected:** Hybrid clustering (face-first global signal, same-shot signature fallback only).
- **Implementation requirement:** Wave 3 signature fallback scoring must include shot ID + temporal proximity cues (no mask-overlap term yet).
- **Implementation requirement:** Wave 4 extends the same scoring with mask-overlap cues after seg masks are integrated.
- **Implementation requirement:** Signature-only attachment must use a global min-cost assignment (Hungarian) with confidence gating and explicit `defer-to-unknown` when ambiguous.
- **Rationale:** Preserve camera-angle robustness while still handling faceless fragments within shot-local constraints.

### Phase 1 Refactor - Decision 3 (Binding): Option B + D, no full heuristic route
- **Selected:** Improve canonical face coverage + enable/require GPU face pipeline.
- **Selected policy:** Remove auto route that can force full heuristic binding; no video should run Phase 1 on full heuristic binding path.
- **Implementation requirement:** Canonical face completion must be shot-bounded and use optical-flow or tracklet-assisted propagation with hard quality cutoffs.
- **Implementation requirement:** Local fallback scoring must be calibrated with LR-ASD confidence + turn-level consistency + hysteresis penalties against rapid speaker switches.
- **Rationale:** Accuracy-first on H200 means invest in stronger LR-ASD candidate coverage rather than routing to weaker fallback logic.

### Phase 1 Refactor - Decision 4 (Architecture): Options A + B
- **Selected:** Modular extraction + shared decode/cache + stage overlap.
- **Rationale:** A provides maintainability/testability; B delivers structural speedup by reducing repeated decode and idle gaps.

### Phase 1 Refactor - Decision 5 (Performance Defaults)
- **Selected:**
  - GPU face pipeline required, fail fast on runtime init failure.
  - Raise YOLO `imgsz` target for 1080p handling.
  - Increase LR-ASD batching/inflight utilization for H200.
  - Implement shared decode/cache as a core architecture feature.
- **Policy:** No automatic fallback path for face pipeline in nominal execution; failures should surface immediately.

### Phase 1 Refactor - Decision 6 (Detector Family): Option C
- **Selected:** Replace detect with segmentation and consume masks (`-seg`) in tracking/postprocess.
- **Model lock:** Use **`yolo26m-seg.pt`** (medium), not small.
- **Policy:** No detect-only rollback track in this refactor plan.
- **Implementation requirement:** Renderer and overlap-follow target selection must use mask-aware target stability scoring (IoU continuity + face-presence bonus + motion smoothness) with short-term memory.

## Phase 1 Refactor - Explicit Non-Goals
- Backward compatibility with v2 manifest/schema.
- Preserve legacy heuristic-only behavior for long/complex videos.
- BoT-SORT as a tracking backend (implemented behavior: ByteTrack-only; invalid `CLYPT_TRACKER_BACKEND` values fail fast in `backend/do_phase1_worker.py`).

## Phase 1 Refactor - Future Note (Out of Scope for This Phase)
- Spanner adoption instead of local graph storage is acknowledged but deferred; this document only defines Phase 1 refactor scope and interfaces.

## Phase 1 Refactor - Target Runtime Defaults

### Phase 1 Refactor - Tracking/Detection Defaults
- `YOLO_WEIGHTS_PATH=yolo26m-seg.pt`
- `CLYPT_YOLO_IMGSZ=1080` (for 1080p-oriented quality/speed target)
- Tracking backend path centers on ByteTrack + cut detection + ReID merge pass

### Phase 1 Refactor - Face Pipeline Defaults
- `CLYPT_FACE_PIPELINE_GPU=1` (required)
- `CLYPT_FACE_PIPELINE_START_FRAME=0`
- `CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE=1280`
- `CLYPT_FULLFRAME_FACE_MIN_SIZE=14`
- Fail-fast behavior when required face runtime cannot initialize

### Phase 1 Refactor - LR-ASD Defaults
- Increase batch/inflight scheduling for H200 capacity
- Maintain overlap-enabled prep/infer pipeline
- Remove routing mode that sends entire jobs to heuristic-only binding

## Phase 1 Refactor - Current-to-Target Runtime Delta
- Tracking model path changes from detect default (`yolo26s.pt`) to segmentation medium (`yolo26m-seg.pt`).
- Default `CLYPT_YOLO_IMGSZ` target changes from legacy lower defaults to `1080`.
- Binding mode policy changes from auto heuristic-route capable behavior to LR-ASD-first with narrow local fallback only.
- Face runtime policy changes from permissive downgrade behavior to explicit fail-fast when GPU-required face runtime is unavailable.
- Source-of-truth implementation requirement: worker-level constants/defaults must be updated to match v3 defaults (no policy-only documentation drift).

## Phase 1 Refactor - v3 Contract Direction
- Treat this refactor as a v3 boundary.
- Optimize internal and output structures for quality/speed and downstream simplicity.
- Downstream phase handoff updates are expected and acceptable.

## Phase 1 Refactor - Contract and Persistence Migration
- v3 migration is mandatory and includes both schema and persistence layers.
- Update `Phase1Manifest.contract_version` and all strict validators in `backend/pipeline/phase1_contract.py`.
- Update persistence payload assembly in `backend/do_phase1_service/storage.py` and model wrappers in `backend/do_phase1_service/models.py` to emit/validate v3-compliant manifest/artifact shapes.
- Add explicit migration tests for:
  - manifest validate -> persist -> reload round-trip,
  - artifact contract integrity,
  - downstream compatibility adapters where required.
- No partial state is allowed where implementation claims v3 behavior while persistence still writes v2 manifests.

### Phase 1 Refactor - v3 Schema Appendix (Minimum Required Delta)
- `contract_version`: move from `v2` to `v3` across schema + persistence.
- Transcript artifact additions:
  - fallback/decision reason metadata for overlap policy and binding abstentions.
- Visual artifact additions:
  - fields required for mask-aware stability logic (or documented references to persisted mask-derived metrics/signals).
- Debug/calibration additions:
  - per-word top-1/top-2 margin,
  - calibrated confidence,
  - ambiguity + abstention reason.
- Overlap-follow additions:
  - explicit `decision_source` reason codes and confidence/evidence context.

## Phase 1 Refactor - Authority Boundaries (Label Accuracy vs Camera Follow)

### Phase 1 Refactor - Speaker Label Authority
- Word-level speaker labels (`speaker_track_id`, `speaker_local_track_id`) are owned by LR-ASD pipeline + calibrated local fallback logic only.
- Overlap-follow adjudication must not directly overwrite strong LR-ASD word assignments.
- Audio diarization/turn information is assistive prior/context, not an authoritative replacement for LR-ASD evidence.
- In uncertain conditions, abstention (`unknown`) is preferred over low-confidence forced assignment.

### Phase 1 Refactor - Camera Follow Authority
- Camera target selection during overlap windows is owned by overlap-follow logic (`active_speakers_local` + `overlap_follow_decisions`).
- Gemini overlap adjudication is a camera-policy intelligence layer, not a word-label assignment layer.
- Deterministic fallback remains mandatory for invalid/unavailable model responses.

### Phase 1 Refactor - Interaction Contract
- LR-ASD emits per-word labels and candidate confidence/ambiguity diagnostics.
- Overlap-follow consumes span-level context and candidate diagnostics to select camera behavior.
- Any camera decision that conflicts with strong word-level LR-ASD evidence is constrained by confidence gates and hysteresis.

## Phase 1 Refactor - Architecture

### Phase 1 Refactor - Stage Modules
- `backend/pipeline/phase1/tracking.py`
- `backend/pipeline/phase1/faces.py`
- `backend/pipeline/phase1/clustering.py`
- `backend/pipeline/phase1/binding.py`
- `backend/pipeline/phase1/postprocess.py`
- `backend/pipeline/phase1/config.py`

### Phase 1 Refactor - Shared Decode/Cache
- Single decode authority with reusable frame access API.
- Tracking, face extraction, and binding prep consume shared frame cache.
- Remove repeated independent full-video decode passes.

### Phase 1 Refactor - Fail-Fast Runtime Policy
- GPU-required components must initialize before stage execution.
- If mandatory GPU components fail, terminate job with explicit error class + diagnostics.
- Do not silently switch to degraded full-heuristic path.
- Fail-fast matrix:
  - `CLYPT_FACE_PIPELINE_GPU=1` + face runtime unavailable -> hard fail.
  - LR-ASD runtime required but unavailable -> hard fail (not whole-job heuristic downgrade).
  - Optional overlap adjudication model unavailable -> deterministic overlap policy with explicit reason-code logs.

## Phase 1 Refactor - Algorithmic Workstreams

### Phase 1 Refactor - Workstream A (Tracking Continuity)
- Add true editorial shot boundary detection (not chunk-window pseudo shots).
- Reset or gate continuity assumptions at cuts.
- Run post-tracking ReID merge for fragmented identities.
- Feed merged identities into clustering as first-class inputs.

### Phase 1 Refactor - Workstream B (Hybrid Clustering)
- Face embeddings remain global primary signal.
- Restrict spatial signature attachment to same-shot contexts.
- Replace median-geometry-only signature scoring with composite scoring:
  - shot constraint (hard gate),
  - temporal proximity,
  - spatial signature as secondary term.
- Replace greedy attachment with global assignment (Hungarian/min-cost).
- Add confidence-gated `defer-to-unknown` for low-margin candidates.
- Preserve covisibility conflict rejection and repair passes.
- Improve deterministic tie-breaking to reduce greedy attachment errors.
- Defer mask-overlap clustering terms until segmentation masks are available in the same execution path (Wave 4 dependency).

### Phase 1 Refactor - Workstream C (LR-ASD Coverage)
- Improve canonical face stream density/quality.
- Expand short-gap and medium-gap face continuity handling with shot-bounded interpolation.
- Add optical-flow or tracklet-assisted propagation for medium gaps.
- Apply hard cutoff when propagated face quality drops below threshold.
- Tighten candidate pruning to preserve quality while reducing waste.
- Keep heuristic binder only as narrow local fallback, not whole-job execution mode.
- Calibrate fallback scoring with:
  - LR-ASD confidence,
  - turn-level consistency priors,
  - hysteresis penalties for rapid speaker switching.
- Add explicit abstention policy:
  - if top candidate confidence/margin is below threshold, assign `unknown`,
  - do not force assignment in overlap-heavy low-evidence windows.

### Phase 1 Refactor - Workstream C1 (Speaker Binding Rehaul Scope)
- Rework binding into explicit sub-stages with testable interfaces:
  - candidate generation,
  - LR-ASD scoring,
  - confidence calibration,
  - decision policy (assign vs unknown),
  - turn-consistency smoothing.
- Introduce calibration artifacts:
  - per-word top-1/top-2 margin,
  - calibrated confidence score,
  - ambiguity flag and abstention reason.
- Add overlap-aware decision policy:
  - maintain label consistency within high-confidence turns,
  - allow controlled switches when visual evidence decisively changes,
  - penalize one-word oscillations.

### Phase 1 Refactor - Workstream D (Segmentation Integration)
- Replace `detect` model path with `seg` model path (`yolo26m-seg.pt`).
- Keep box outputs for tracker compatibility.
- Integrate masks for overlap disambiguation and fragment suppression.
- Extend v3 visual artifact contract/persistence as needed so mask-derived stability signals are available to renderer/overlap consumers.
- Add mask-aware postprocess hooks for camera-follow stability:
  - IoU continuity scoring across recent frames,
  - face-presence bonus,
  - motion smoothness regularization,
  - short-term memory to avoid single-frame target jumps.

### Phase 1 Refactor - Workstream E (Overlap and Diarization Integration)
- Keep pyannote diarization supported as turn/overlap context for label smoothing and camera policy.
- Production profile policy: enable diarization where overlap-follow adjudication is expected; when diarization is disabled/unavailable, overlap follows deterministic policy with explicit logs.
- Build explicit span-level overlap confidence from:
  - diarization overlap signals,
  - visible speaker count,
  - LR-ASD candidate competition.
- Gate Gemini overlap adjudication:
  - run only when span overlap confidence and evidence density are above threshold,
  - otherwise use deterministic policy directly.
- Preserve deterministic overlap fallback for all failure/invalid-response cases with reason-coded logs and counters.
- Logging context wiring requirement: overlap-follow logs must be emitted through call-site context that injects `job_id`/`worker_id` fields (or equivalent structured context object).

## Phase 1 Refactor - Implementation Program

### Phase 1 Refactor - Wave 1 (Config + Runtime Enforcement)
- Land typed config surface with new defaults.
- Enforce fail-fast runtime checks for required GPU dependencies.
- Remove full-job heuristic auto-routing behavior.
- Align env and backend policy with implementation:
  - BoT-SORT runtime path removed (worker is ByteTrack-only; see `_select_tracker_backend`),
  - remove/deprecate auto full-job heuristic binding route,
  - remove/deprecate `CLYPT_SPEAKER_BINDING_MODE=auto` semantics that can select whole-job `heuristic`,
  - deprecate `CLYPT_SPEAKER_BINDING_AUTO_*` knobs for full-job fallback routing,
  - lock defaults for seg medium + imgsz + face GPU requirements.

**Exit criteria**
- Job boot fails fast with actionable errors when required GPU stack is unavailable.
- Config paths are typed, centralized, and used by all stages.
- No worker execution path silently downgrades to whole-job heuristic mode.

### Phase 1 Refactor - Wave 2 (Module Extraction + Shared Decode)
- Extract stage modules from monolith.
- Introduce shared decode/cache seam and migrate consumers.
- Preserve stable orchestration entrypoint while internals move to modules.

**Exit criteria**
- Tracking/faces/binding consume shared decode API.
- No duplicate full-video decode loops remain in stage code.

### Phase 1 Refactor - Wave 3 (Core Accuracy Upgrades)
- Implement shot-aware tracking controls.
- Implement post-track ReID merge.
- Ship hybrid clustering Option C (without mask-overlap scoring terms).
- Improve canonical face stream for LR-ASD scoring coverage.
- Ship speaker binding rehaul sub-stages (candidate -> calibration -> decision -> smoothing).

**Exit criteria**
- Fragmentation materially reduced.
- `with_scored_candidate` and assignment coverage increase.
- Unknown-rate is reduced without increasing high-confidence mis-assignments.

### Phase 1 Refactor - Wave 4 (Seg + Throughput Tuning)
- Switch model baseline to `yolo26m-seg.pt`.
- Enable mask-aware association/postprocess logic.
- Enable mask-overlap scoring terms in clustering once segmentation outputs are integrated.
- Tune LR-ASD batch/inflight against H200 profile.
- Integrate overlap+diarization gating and Gemini camera-policy guardrails.

**Exit criteria**
- Throughput and assignment quality both improve vs baseline.
- No detect-only rollback dependency in execution path.
- Camera-follow overlap stability improves without degrading word-label accuracy.

## Phase 1 Refactor - Validation

### Phase 1 Refactor - Quality Metrics
- Speaker assignment coverage
- `with_scored_candidate` ratio
- Canonical face stream coverage
- Identity fragmentation (pre/post merge)
- Overlap-follow stability metrics
- Unknown assignment rate (overall and overlap-only)
- High-confidence mis-assignment rate (manual spot-check set)
- Overlap-window camera target consistency score

### Phase 1 Refactor - Speed Metrics
- End-to-end wall clock for 10-15 minute content
- Stage-level timings (tracking/faces/clustering/binding)
- Decode overhead before/after shared cache
- GPU utilization during LR-ASD and face stages

### Phase 1 Refactor - Required Test Coverage
- Unit tests for each extracted stage module.
- Integration tests on representative podcast/interview footage:
  - frequent cuts,
  - overlap-heavy sections,
  - low-face-visibility segments,
  - long-form single-session recordings.
- Contract tests that enforce authority boundaries:
  - overlap-follow modules cannot mutate `words[*].speaker_*` assignments,
  - camera-policy decisions consume span/candidate context without writing label fields,
  - low-evidence overlap windows default to deterministic camera policy.

## Phase 1 Refactor - Risks and Mitigations
- **Risk:** Seg model latency increase.  
  **Mitigation:** H200-tuned batch/overlap and shared decode/cache to reclaim throughput.

- **Risk:** ReID over-merge errors.  
  **Mitigation:** Covisibility gates, shot-aware constraints, and confidence-scored merge policies.

- **Risk:** Fail-fast policy reduces robustness to bad host setup.  
  **Mitigation:** Strong preflight diagnostics and deployment checks, not silent fallback.

## Phase 1 Refactor - Immediate Next Actions
1. Lock defaults (`yolo26m-seg.pt`, `imgsz=1080`, GPU face required, no full-job heuristic mode).
2. Land module extraction scaffolding and shared decode/cache API.
3. Implement shot boundary detection + ReID merge seam.
4. Integrate mask-aware logic into tracking/postprocess path.
5. Tune LR-ASD batching/inflight specifically for H200.

## Phase 1 Refactor - Execution Readiness Checklist

### Phase 1 Refactor - File-Level Change Map
- `backend/do_phase1_worker.py`
  - lock seg-medium defaults and remove whole-job heuristic route,
  - enforce fail-fast runtime behavior for required GPU paths,
  - implement shot-aware tracking controls + post-track ReID merge,
  - implement speaker-binding rehaul stages and abstention policy.
- `backend/overlap_follow.py`
  - add overlap-confidence/evidence-density gating for Gemini calls,
  - emit structured reason-coded fallback logs and counters,
  - keep deterministic overlap policy as guaranteed fallback.
- `backend/test-render/render_speaker_follow_clips.py`
  - consume mask-aware overlap-follow outputs and short-term stability memory signals,
  - enforce camera policy authority boundaries (no word-label mutation paths).
- `backend/pipeline/phase1_contract.py`
  - migrate manifest/artifact contract version and strict schemas to v3.
- `backend/do_phase1_service/storage.py`
  - emit v3 manifest payloads and v3 artifact shapes,
  - preserve round-trip validation against strict contract models.
- `backend/do_phase1_service/.env.example`
  - align defaults with v3 policy (seg model, imgsz target, GPU face required, no full-job heuristic route).

### Phase 1 Refactor - Required Config Surface (Typed + Centralized)
- Tracking + seg:
  - `YOLO_WEIGHTS_PATH=yolo26m-seg.pt`
  - `CLYPT_YOLO_IMGSZ=1080`
- Binding policy:
  - remove/deprecate full-job heuristic auto route controls,
  - keep narrow local fallback controls with calibrated thresholds.
- Face runtime:
  - `CLYPT_FACE_PIPELINE_GPU=1`
  - `CLYPT_FACE_PIPELINE_START_FRAME=0`
  - `CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE=1280`
  - `CLYPT_FULLFRAME_FACE_MIN_SIZE=14`
- Overlap-follow:
  - explicit gating thresholds for overlap confidence/evidence density,
  - explicit deterministic fallback reason-code keys.

### Phase 1 Refactor - Hard Sequencing Rules
1. Contract/persistence migration plan is defined before implementation starts.
2. Wave 3 ships hybrid clustering without mask-overlap scoring terms.
3. Wave 4 introduces seg mask integration and then enables mask-overlap clustering terms.
4. Authority-boundary tests must pass before enabling overlap-follow model gating in production.

### Phase 1 Refactor - Verification Commands (Per PR)
- Unit/integration scope:
  - `python3 -m pytest tests/backend/do_phase1_service/test_extract.py -q`
  - `python3 -m pytest tests/backend/pipeline -q`
  - `python3 -m pytest tests/backend -k "speaker_binding or overlap_follow or phase1_contract" -q`
- Contract/persistence:
  - run manifest validation tests for v3 round-trip (`phase1_contract` + storage payload builder).
- Runtime profiling:
  - run fixed-corpus benchmark on H200 and capture:
    - assignment coverage,
    - `with_scored_candidate`,
    - unknown-rate + high-confidence mis-assignment,
    - overlap camera consistency,
    - total wall-clock and stage timings.

### Phase 1 Refactor - Definition of Ready to Implement
- v3 authority boundaries are accepted as non-negotiable.
- Contract + persistence v3 migration path is explicit and tested.
- Config defaults and fail-fast matrix are approved and documented.
- Wave dependencies (especially mask-enabled sequencing) are locked.
- Benchmark corpus and scorecard are defined before algorithm changes land.

### Phase 1 Refactor - Benchmark Corpus and Scorecard Definition
- Minimum corpus:
  - 10 overlap-heavy podcast/interview clips,
  - 10 frequent-cut clips,
  - 10 low-face-visibility clips,
  - 10 long-form single-session clips.
- Per-clip required outputs:
  - assignment coverage,
  - `with_scored_candidate`,
  - unknown-rate,
  - high-confidence mis-assignment counts,
  - overlap camera consistency score,
  - total/stage wall-clock.
- Scorecard acceptance:
  - compare against frozen baseline run manifest set before any behavior-changing merge.
