# VibeVoice vLLM Phase 1 Design

**Date:** 2026-04-06  
**Branch:** `v3.1-refactor-vLLM`

## Goal

Replace the current per-job native VibeVoice subprocess path with a **persistent, Docker-managed vLLM ASR service** on the GPU droplet, then overlap ASR with visual extraction in Phase 1.

This migration must:

- keep the Phase 1 output contract stable
- keep the serial branch's working visual / aligner / emotion / YAMNet stack intact
- avoid starting a Docker image per request
- **fail fast and fail hard** if the vLLM service is unavailable or returns invalid output
- require **VibeVoice-only validation first**, before any full Phase 1 rerun

There is **no native fallback** in the new target architecture.

## Non-Goals

This spec does **not**:

- switch the visual pipeline away from `RF-DETR Small`
- make TensorRT the default visual backend
- change forced alignment, `emotion2vec+`, or `YAMNet` semantics
- add multi-request job scheduling or queueing beyond what vLLM already provides
- preserve the current native subprocess path as a production fallback

## Current Baseline

Today, on `v3.1-refactor-serial`, Phase 1 is:

1. media prep
2. visual extraction
3. VibeVoice ASR via a native subprocess in a second venv
4. NeMo forced alignment
5. `emotion2vec+`
6. `YAMNet`

The current serial branch is stable enough to use as the migration base because:

- visual extraction is working
- logging is working
- the native subprocess Unicode boundary issue was debugged and fixed
- VibeVoice-only validation succeeded on multiple clips and slices

The pain points that motivate this change are:

- per-job ASR process lifecycle complexity
- separate native venv maintenance
- lack of persistent model residency
- inability to cleanly start ASR at the same time as visual extraction

## Recommended Architecture

Use a **persistent sidecar ASR service** on the droplet:

- one long-lived Docker container
- one long-lived vLLM server process
- model remains resident on GPU between requests
- worker talks to the service over localhost HTTP

### Runtime Shape

The new intended Phase 1 flow is:

1. prepare media
2. fan out immediately:
   - `visual branch`: RF-DETR Small + ByteTrack
   - `audio branch`: call local vLLM ASR HTTP API
3. when ASR completes:
   - run NeMo forced alignment
   - run `emotion2vec+`
4. run `YAMNet`
5. join outputs and build `Phase1SidecarOutputs`

This keeps one clear boundary:

- the worker still orchestrates the job
- the vLLM service only owns ASR inference

## Why Docker-Managed Persistent vLLM

This is the best fit because it gives us:

- persistent GPU residency without per-request startup
- closer parity with VibeVoice's documented vLLM path
- isolation from the main worker environment
- simpler rollout and rollback than an in-process vLLM integration
- easier systemd health checks and logs

We explicitly do **not** want:

- a fresh Docker container per request
- an in-process vLLM runtime inside the worker
- a second per-job subprocess boundary like the current native path

## Service Contract

The worker will call a persistent local endpoint, likely:

- `POST http://127.0.0.1:8000/v1/chat/completions`

The request shape should follow the upstream VibeVoice vLLM API format as closely as possible.

### Input contract

The worker sends:

- model identifier
- audio payload
- hotwords/context text
- generation controls

For V1, prefer the same request pattern VibeVoice's own vLLM tests use:

- OpenAI-compatible request
- audio embedded in the request payload
- prompt/context passed as text content

### Output contract

The worker-side provider must normalize the service response into the existing turn format:

```python
[
    {
        "Start": float,
        "End": float,
        "Speaker": int,
        "Content": str,
    }
]
```

This preserves compatibility with:

- `vibevoice_merge.py`
- forced alignment
- `emotion2vec+`
- downstream Phase 1 contracts

## Fail-Fast Policy

The new `vllm` backend has **no native fallback**.

If any of the following happen, the job must fail immediately:

- vLLM service health check fails
- request connection fails
- request times out
- response is malformed
- response JSON cannot be normalized into turns
- response contains zero usable turns for a clip that should have speech

This is intentional. Silent backend switching would make validation ambiguous and would hide migration bugs.

## Configuration Model

Extend provider config with a new backend:

- `VIBEVOICE_BACKEND=vllm`

Add new settings for the service:

- `VIBEVOICE_VLLM_BASE_URL`
- `VIBEVOICE_VLLM_MODEL`
- `VIBEVOICE_VLLM_TIMEOUT_S`
- `VIBEVOICE_VLLM_HEALTHCHECK_PATH`
- `VIBEVOICE_VLLM_MAX_RETRIES`
- `VIBEVOICE_VLLM_AUDIO_MODE`

The current VibeVoice generation knobs remain relevant and should still flow through:

- `VIBEVOICE_MAX_NEW_TOKENS`
- `VIBEVOICE_DO_SAMPLE`
- `VIBEVOICE_TEMPERATURE`
- `VIBEVOICE_TOP_P`
- `VIBEVOICE_REPETITION_PENALTY`
- `VIBEVOICE_NUM_BEAMS`
- `VIBEVOICE_HOTWORDS_CONTEXT`

### Configuration rules

- when `VIBEVOICE_BACKEND=vllm`, the worker must **not** require `VIBEVOICE_NATIVE_VENV_PYTHON`
- when `VIBEVOICE_BACKEND=vllm`, any attempt to use the native subprocess path is a bug
- if `VIBEVOICE_VLLM_BASE_URL` is missing, fail at worker startup

## Code Changes

### New provider

Add a dedicated provider module, likely:

- `backend/providers/vibevoice_vllm.py`

Responsibilities:

- health check
- request building
- HTTP call
- structured response parsing
- turn normalization
- latency / turn-count logging

### Existing provider/config changes

Modify:

- `backend/providers/config.py`
- `backend/providers/__init__.py`
- `backend/phase1_runtime/factory.py`

Goals:

- support `backend == "vllm"`
- construct the correct provider
- remove any assumption that only `native|hf` exist

### Phase 1 orchestration changes

Modify:

- `backend/phase1_runtime/extract.py`

Goals:

- split Phase 1 into explicit visual and audio execution lanes
- start visual extraction and vLLM ASR immediately after media prep
- keep aligner + `emotion2vec+` downstream of ASR
- keep `YAMNet` as a distinct final stage unless benchmarking later suggests otherwise

The initial parallel model should be:

- `visual`
- `ASR -> aligner -> emotion2vec+`
- `YAMNet`

with `YAMNet` still run after the main overlap unless profiling later justifies changing that.

### Deployment changes

Add/update droplet deployment artifacts:

- a Dockerfile or image-pull strategy for the vLLM service
- a systemd unit for the persistent vLLM container
- deploy scripts that:
  - start/restart the container
  - wait for readiness
  - fail if health checks do not pass

Likely files:

- `scripts/do_phase1/deploy_vllm_service.sh`
- `scripts/do_phase1/systemd/clypt-vllm-vibevoice.service`
- possibly a `docker/` directory or service-specific Dockerfile

## Docker / Service Strategy

The service must be **persistent**, not ephemeral.

### Required behavior

- started once at deploy or boot
- kept hot under systemd
- bound to localhost only
- not started from inside the Phase 1 request path

### Preferred deployment pattern

Systemd owns the container lifecycle:

- pull or build image during deploy
- `ExecStart` runs the long-lived vLLM container
- `Restart=always`
- health check confirms readiness before the worker relies on it

### Avoid

- `docker run` from inside the worker
- rebuilding the image on every ASR request
- using the upstream convenience launcher as the steady-state service command if it performs installation/setup work on each launch

## Logging and Observability

We already have better worker logs on the serial branch, and the vLLM migration should keep that standard.

### Worker logs must show

- ASR request start
- service URL / model tag
- audio duration
- request latency
- returned turn count
- parse errors
- health check failures

### Service logs must be accessible via

- Docker logs
- systemd journal

We should be able to debug:

- request accepted but stalled
- service unavailable
- malformed output
- long-context performance degradation

## Validation Plan

The rollout must happen in this order:

### Stage 1: service-only bring-up

Validate:

- container starts
- health endpoint is reachable
- model stays resident
- a simple transcription request works

### Stage 2: VibeVoice-only validation

Required before full Phase 1:

- run short clip
- run medium clip
- run long clip
- compare turn count and qualitative output vs the current native path

Suggested validation set:

- `60s` Joe Rogan slice
- `300s` Joe Rogan slice
- full `392.9s` MrBeast clip
- `540s` Joe Rogan slice
- full `788.7s` Joe Rogan clip

Only after VibeVoice-only validation is acceptable should we proceed to full Phase 1.

### Stage 3: full Phase 1 validation

Then validate:

- visual + ASR overlap works
- forced aligner receives correct turns
- `emotion2vec+` still consumes aligned turns
- final `Phase1SidecarOutputs` are intact

## Risks

### Output drift

The user prefers the current 7B path because it feels closer to the Gradio/playground behavior. vLLM may still change behavior enough to matter, even if it is faster.

Mitigation:

- VibeVoice-only validation first
- no hidden fallback
- compare outputs before switching full jobs

### Long-context latency

Long-form audio may still be expensive under vLLM, especially on single requests.

Mitigation:

- benchmark on the real validation clips
- log prompt/context size where possible

### Plugin/runtime fragility

The custom VibeVoice vLLM plugin is an extra moving piece.

Mitigation:

- isolate it in Docker
- keep the worker environment separate
- fail fast on health / parsing problems

## Success Criteria

This migration is successful when:

1. the droplet runs one persistent vLLM ASR service under Docker + systemd
2. `VIBEVOICE_BACKEND=vllm` works without any native subprocess path
3. VibeVoice-only validation passes on the agreed clips
4. full Phase 1 runs with:
   - visual starting in parallel with ASR
   - stable downstream artifacts
5. failures are explicit and immediate, not silently rerouted to another backend

## Implementation Note

After this spec is approved, the next artifact should be a concrete implementation plan that breaks the migration into:

- provider/config work
- deployment/service work
- Phase 1 orchestration changes
- validation tasks
