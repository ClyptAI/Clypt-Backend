# Clypt V3.1 Spec: Phase1 VibeVoice Long-Form Chunking on H200

**Status:** Active (implemented)
**Date:** 2026-04-19
**Owner:** Phase1 runtime / VibeVoice ASR
**Scope:** Extend the Phase1 H200 VibeVoice service to transcribe podcast-style inputs longer than the current safe single-pass window by splitting canonical audio into at most 2-3 shards, issuing shard ASR requests in parallel through the local vLLM sidecar, and reconciling shard-local speaker IDs into one global speaker space before the rest of Phase1 continues.

---

## 1. Locked Decisions

1. **Safe single-pass cap stays 60 minutes.** Upstream VibeVoice documentation still describes the model as handling about 60 minutes of long-form audio in a single request. Clypt will continue to treat 60 minutes as the conservative per-request ceiling rather than trying to raise the bound in-place.
2. **Shard counts are fixed by total duration:**
   - `<= 60 minutes`: existing single-request path
   - `> 60 and <= 90 minutes`: exactly 2 shards
   - `> 90 and <= 180 minutes`: exactly 3 shards
   - `> 180 minutes`: fail fast with a clear validation/runtime error
3. **Chunking happens inside the Phase1 VibeVoice service, not in the runner.** The outer Phase1 runner and `RemoteVibeVoiceAsrClient` contract remain one request per job.
4. **We split canonical audio, not source video.** The VibeVoice service already downloads `audio_gcs_uri` and only consumes audio for ASR. Visual extraction continues to use the full original video path and is not chunked.
5. **Parallel shard requests go to the existing local vLLM sidecar.** No new remote surface is introduced. The known-good H200 env already sets `VIBEVOICE_VLLM_MAX_NUM_SEQS=4`, which is sufficient for up to 3 parallel shard requests plus service overhead.
6. **Speaker stitching is additive, not a replacement diarizer.** VibeVoice remains the per-shard source of transcript text, timestamps, and shard-local speaker labels. A lightweight verification layer only reconciles identities across shards.
7. **Global speaker IDs are deterministic.** Final speaker numbering is assigned by first appearance in the merged transcript after cross-shard reconciliation.
8. **The Phase1 enqueue boundary remains “merged ASR complete,” not “first shard complete.”** Phase1 may still enqueue downstream before visual finishes, but only after all ASR shards, timestamp offsets, and global speaker stitching are complete.
9. **Fail-fast remains the default.** Unsupported durations, shard orchestration errors, broken merge invariants, or verifier infrastructure failures terminate the Phase1 ASR step rather than silently degrading into inconsistent speaker IDs.

---

## 2. Problem Statement

The current Phase1 topology is optimized around a single VibeVoice request per job:

- the Phase1 runner calls `POST /tasks/vibevoice-asr` once,
- the service downloads canonical audio from GCS,
- the provider sends one long-form ASR request to the local vLLM sidecar,
- audio-post work (NFA -> emotion2vec+ -> YAMNet) starts only after that ASR response returns.

That works well for the current single-pass envelope, but it breaks down for long podcasts in the 60-180 minute range:

1. upstream VibeVoice guidance still treats ~60 minutes as the safe single-pass budget,
2. Clypt currently has no built-in sharding/orchestration path for longer content,
3. naive chunking would create independent speaker namespaces per shard (`Speaker 0`, `Speaker 1`, etc.) and would corrupt downstream participation grounding if those local IDs were treated as global truth.

The Phase1 H200 should therefore gain a long-form ASR mode that:

- preserves the current external service contract,
- parallelizes only inside the service boundary,
- returns one merged transcript payload,
- and reconciles speakers globally before the rest of Phase1 proceeds.

---

## 3. Goals

1. Support podcast-style jobs up to 180 minutes without changing the outer Phase1 runner contract.
2. Keep the existing <=60 minute path unchanged and cheap.
3. Use the local vLLM sidecar for intra-job parallelism instead of serializing 2-3 long ASR passes.
4. Preserve a single merged `turns` list and `stage_events` response shape from `/tasks/vibevoice-asr`.
5. Produce one global speaker space across all shards so downstream Phase1/Phase26 artifacts remain coherent.
6. Preserve the current Phase1 invariant that audio-post begins as soon as final ASR is ready and does not wait for visual completion.

---

## 4. Non-Goals

1. No support for podcasts longer than 180 minutes in this phase.
2. No replacement of VibeVoice diarization with a standalone end-to-end diarization stack.
3. No chunking of the visual pipeline, source video upload path, or Phase26 queue contract.
4. No streaming partial transcripts to downstream Phases 2-4.
5. No attempt to “overclock” VibeVoice into 90+ minute single-pass requests by increasing vLLM context or token budgets.

---

## 5. Chosen Approach

### 5.1 Summary

For long inputs, the Phase1 VibeVoice service will:

1. download the canonical audio object from GCS into service scratch,
2. probe duration,
3. choose `1`, `2`, or `3` shards based on §1.2,
4. split the local audio file into equal-duration shard WAVs,
5. upload shard WAVs back to ephemeral GCS paths under the current `run_id`,
6. issue up to 3 parallel VibeVoice requests against those shard GCS URIs,
7. shift shard-local timestamps into one global timeline,
8. reconcile shard-local speaker IDs using speaker embeddings + cosine similarity,
9. renumber speakers deterministically by first appearance,
10. return one merged transcript payload to the runner.

### 5.2 Why this approach

This approach keeps the live architecture aligned with current runtime truths:

- the service already owns scratch space and GCS download,
- the provider already prefers URL/GCS transport instead of giant inline payloads,
- the H200 env already supports sidecar request concurrency,
- the outer runner contract stays simple and stable,
- and downstream systems continue to receive a single `VibeVoiceAsrResponse`.

### 5.3 Rejected alternatives

1. **Raise the single-pass cap above 60 minutes and keep one request.** Rejected because it directly fights the upstream model envelope and would create the least predictable failure mode.
2. **Chunk locally and send shard audio inline as base64.** Rejected as the default because 2-3 hour podcast shards create very large HTTP payloads; Clypt’s current VibeVoice path is intentionally GCS/URL-driven.
3. **Run a completely separate diarization system for the whole file and discard VibeVoice speakers.** Rejected because the desired change is specifically global speaker reconciliation on top of VibeVoice, not a parallel diarization rewrite.

---

## 6. Duration and Shard Planning

## 6.1 Planning rules

The service computes `audio_duration_s` from the downloaded canonical audio file and applies:

- `duration <= 3600s` -> `shard_count = 1`
- `3600s < duration <= 5400s` -> `shard_count = 2`
- `5400s < duration <= 10800s` -> `shard_count = 3`
- `duration > 10800s` -> hard failure

## 6.2 Shard boundaries

For `N in {2, 3}`, the service computes equal-length time windows:

- shard 0: `[0, shard_len)`
- shard 1: `[shard_len, 2*shard_len)`
- shard 2: `[2*shard_len, duration)` if present

The first implementation intentionally uses **non-overlapping** shards to preserve the simple “2 shards through 90 minutes / 3 shards through 180 minutes” contract the product request calls for. Boundary clipping risk is accepted as a known tradeoff for V1 and is tracked in §15.

## 6.3 Unsupported inputs

The service returns a clear error when:

- the canonical audio exceeds 180 minutes,
- shard extraction fails,
- the service cannot create shard GCS objects,
- or shard metadata cannot be reconciled into a complete merged transcript.

---

## 7. Service-Orchestration Design

## 7.1 Boundary of responsibility

Long-form orchestration belongs in `backend/runtime/phase1_vibevoice_service/`, not in the provider or runner:

- `RemoteVibeVoiceAsrClient` should still send one HTTP request,
- `VibeVoiceVLLMProvider` should still mean “run one VibeVoice ASR request against one audio object,”
- and `Phase1JobRunner` should remain unaware of sharding details.

## 7.2 Outer request lifecycle

`POST /tasks/vibevoice-asr` becomes:

1. authenticate request,
2. download canonical audio from `audio_gcs_uri`,
3. probe duration and choose `shard_count`,
4. if `shard_count == 1`, use the existing fast path,
5. otherwise, run the long-form orchestration path,
6. return the same top-level response shape:
   - `run_id`
   - `turns`
   - `stage_events`
   - `elapsed_ms`

## 7.3 Service-level serialization

The existing service-level `asr_lock` remains in place. This preserves the current “one active Phase1 ASR job at a time” operational posture while still allowing **intra-job parallelism** across 2-3 shard requests.

This means:

- Clypt does **not** add multi-job concurrency in this change,
- but a single long-form job can exploit vLLM’s `max-num-seqs` parallel capacity.

---

## 8. Shard Audio Transport

## 8.1 Why shard GCS objects are required

The current provider is URL-first:

- `audio_mode=url` is the default,
- `audio_gcs_uri` is required in the Phase1 sidecar path,
- and the provider already knows how to resolve canonical GCS URIs into requestable URLs.

That makes ephemeral shard GCS objects the cleanest transport for long-form requests.

## 8.2 Shard object layout

For a long-form job, the service writes shard audio objects under a run-scoped prefix:

```text
gs://<bucket>/phase1/<run_id>/vibevoice_shards/
  shard_000_of_002.wav
  shard_001_of_002.wav
  shard_000_of_003.wav
  shard_001_of_003.wav
  shard_002_of_003.wav
```

These are **ephemeral runtime artifacts**, not test-bank canonical assets.

## 8.3 Lifecycle policy

Initial implementation keeps shard objects after completion for debuggability and replay. Cleanup can be added later behind an explicit retention policy if bucket pressure becomes material.

---

## 9. Parallel VibeVoice Request Plan

## 9.1 Request fan-out

For each shard, the service issues one provider call with:

- local shard audio path (for duration/probing if needed),
- shard-specific `audio_gcs_uri`,
- the existing model id (`vibevoice`),
- and the existing VibeVoice transport mode (`url`).

The service fans those requests out via a thread pool sized to `shard_count`.

## 9.2 Concurrency budget

Known-good Phase1 H200 env already sets:

- `VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION=0.82`
- `VIBEVOICE_VLLM_MAX_NUM_SEQS=4`

That is sufficient for:

- 1 long-form service job,
- up to 3 concurrent shard requests,
- and no additional sidecar tuning in the first rollout.

The spec still requires an explicit canary/benchmark gate in §14 before raising concurrency beyond the 3-shard design maximum.

## 9.3 Stage telemetry

The service emits shard-level stage events in addition to the existing top-level `vibevoice_asr` event. New stage names:

- `vibevoice_longform_plan`
- `vibevoice_shard_asr`
- `vibevoice_speaker_stitch`
- `vibevoice_longform_merge`

Each shard event includes:

- `shard_index`
- `shard_count`
- `start_s`
- `end_s`
- `duration_s`
- `turn_count`

The final top-level `vibevoice_asr` event remains the authoritative summary event for the whole service call.

---

## 10. Timestamp Merge Rules

## 10.1 Shard-local to global time

VibeVoice outputs shard-local timestamps. After all shard requests complete, the service converts each turn to global time by adding the shard start offset:

- `global_start = shard_start_s + local_start`
- `global_end = shard_start_s + local_end`

## 10.2 Sort and normalize

The merged turn list is then:

1. sorted by `global_start`,
2. secondarily sorted by `global_end`,
3. normalized to the existing output schema (`Start`, `End`, `Speaker`, `Content`).

## 10.3 Validation invariants

The merge step fails if:

- any turn has `End < Start`,
- any shard returns malformed speaker/timestamp fields,
- or any turn falls materially outside the expected shard time window before offsetting.

---

## 11. Global Speaker Stitching

## 11.1 Goal

Each shard may restart speaker numbering from zero. The stitcher’s job is to determine whether, for example, `shard 0 / speaker 3` and `shard 1 / speaker 1` are actually the same person, then collapse them into one global speaker ID.

## 11.2 Representative audio selection

For each `(shard_index, local_speaker_id)` pair:

1. collect that speaker’s turns from the shard transcript,
2. find the longest contiguous voiced region for that speaker,
3. extract a representative clip with target duration in the `15-30s` range,
4. if no single continuous region reaches 15 seconds, use the longest available region,
5. if the speaker has too little total speech to produce a meaningful clip, mark them “low confidence” and allow them to remain unmatched.

The extracted representative audio comes from the local shard WAV file already present in service scratch.

## 11.3 Verification model

Phase1 adds a lightweight speaker verification provider, with **ECAPA-TDNN as the default baseline choice** for implementation. The provider computes one embedding per representative clip and exposes cosine similarity scoring.

Design intent:

- keep this separate from VibeVoice itself,
- keep it lightweight enough to run without materially perturbing RF-DETR + vLLM,
- prefer CPU-by-default execution on the H200 host to avoid unnecessary GPU contention.

## 11.4 Matching algorithm

Speaker matching is only attempted across **adjacent shards**:

- shard 0 <-> shard 1
- shard 1 <-> shard 2

The service computes cosine similarities for every candidate pair across adjacent shards and applies:

1. a minimum similarity threshold of `0.85`,
2. one-to-one greedy matching in descending similarity order,
3. transitive union across the shard chain to construct global identities.

If a speaker has no match above threshold, they become a new global speaker.

## 11.5 Global speaker numbering

After match groups are built, global speaker IDs are assigned by earliest first appearance in the merged transcript:

- first appearing identity -> `Speaker 0`
- next unseen identity -> `Speaker 1`
- and so on

This keeps IDs deterministic across reruns with identical transcripts/embeddings.

## 11.6 Failure behavior

The stitcher does **not** fail just because a speaker is unmatched. That is normal and simply means “new speaker.”

The stitcher **does** fail when:

- the verifier model cannot load,
- clip extraction for the comparison graph crashes,
- embeddings cannot be produced,
- or the merge graph becomes structurally inconsistent.

---

## 12. Phase1 Runtime Semantics After the Change

## 12.1 What stays the same

1. The runner still calls `RemoteVibeVoiceAsrClient.run(...)` once.
2. The service still returns one `VibeVoiceAsrResponse`.
3. NFA -> emotion2vec+ -> YAMNet still begin only after the final merged ASR payload returns.
4. Visual extraction still runs concurrently with the whole ASR step.
5. Phase24 enqueue still happens only after ASR + local audio-post are complete, and may still happen before visual finishes.

## 12.2 What changes

The internal meaning of the service call changes from:

- “run one VibeVoice request”

to:

- “produce one globally stitched ASR result, possibly via 2-3 parallel VibeVoice shard requests.”

This is intentionally an internal service detail. Downstream code should not see a contract change.

---

## 13. Code and File Plan

## 13.1 New modules

Add focused helpers under `backend/runtime/phase1_vibevoice_service/`:

- `longform.py`
  - duration planning
  - shard boundary calculation
  - shard WAV extraction
  - shard GCS upload planning
  - parallel shard request orchestration
  - timestamp offset merge
- `speaker_stitch.py`
  - representative clip extraction
  - verifier calls
  - adjacent-shard matching
  - global speaker renumbering
- `models.py`
  - typed internal data classes for shard plans, shard results, speaker match groups

## 13.2 Existing files to modify

- `backend/runtime/phase1_vibevoice_service/app.py`
  - route requests into single-pass vs long-form orchestration
  - emit new stage events
  - preserve response shape
- `backend/runtime/phase1_vibevoice_service/deps.py`
  - provide speaker verifier dependency and new settings
- `backend/providers/config.py`
  - add env-backed settings for long-form thresholds and verifier behavior
- `docs/runtime/ENV_REFERENCE.md`
  - document new `VIBEVOICE_LONGFORM_*` envs
- `docs/runtime/known-good-phase1-h200.env`
  - add the new env defaults
- `docs/runtime/RUNTIME_GUIDE.md`
  - mention long-form sharding support and 180-minute cap
- `docs/deployment/PHASE1_HOST_DEPLOY.md`
  - mention the verifier dependency and canary expectations
- `docs/ERROR_LOG.md`
  - append only if rollout uncovers a runtime/deploy failure that gets diagnosed and fixed

## 13.3 Files intentionally not modified

- `backend/phase1_runtime/runner.py`
- `backend/phase1_runtime/extract.py`
- `backend/providers/audio_host_client.py` response shape
- Phase26 worker/runtime code

No outer contract change should force downstream rewrites.

---

## 14. Configuration Additions

Add Phase1 env knobs with conservative defaults:

- `VIBEVOICE_LONGFORM_ENABLED=1`
- `VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES=60`
- `VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES=90`
- `VIBEVOICE_LONGFORM_THREE_SHARD_MAX_MINUTES=180`
- `VIBEVOICE_LONGFORM_MAX_SHARDS=3`
- `VIBEVOICE_LONGFORM_SPEAKER_MATCH_THRESHOLD=0.85`
- `VIBEVOICE_LONGFORM_REP_CLIP_MIN_SECONDS=15`
- `VIBEVOICE_LONGFORM_REP_CLIP_MAX_SECONDS=30`
- `VIBEVOICE_LONGFORM_VERIFIER_BACKEND=ecapa_tdnn`
- `VIBEVOICE_LONGFORM_VERIFIER_DEVICE=cpu`

Notes:

- these are Phase1-host-only knobs,
- the `vibevoice` served model id stays unchanged,
- and `VIBEVOICE_VLLM_MAX_NUM_SEQS` remains the sidecar-level concurrency control.

---

## 15. Testing and Eval Plan

## 15.1 Unit tests

Add focused tests for:

- duration -> shard-count planning
- 2-shard and 3-shard boundary generation
- timestamp offset merge correctness
- deterministic speaker renumbering
- one-to-one adjacent-shard speaker matching
- unmatched speaker behavior
- >180 minute rejection

## 15.2 Service tests

Add runtime tests around `phase1_vibevoice_service` that validate:

1. `<=60 minute` requests still call the provider exactly once,
2. `61-90 minute` requests fan out exactly 2 shard calls,
3. `91-180 minute` requests fan out exactly 3 shard calls,
4. the response shape remains unchanged,
5. shard stage events are present,
6. merged turns return global timestamps and global speaker IDs.

## 15.3 Pipeline/runtime checks

Required verification commands before shipping:

```bash
python -m pytest tests/backend/runtime -q
python -m pytest tests/backend/pipeline -q
python -m pytest tests/backend/phase1_runtime -q
python -m pytest tests/backend/providers -q
```

## 15.4 Canary inputs

Long-form canary set should include at minimum:

1. one 70-90 minute input -> 2-shard path
2. one 100-150 minute input -> 3-shard path
3. one input near the 180 minute cap

Each canary must confirm:

- the service chooses the expected shard count,
- the total service response is successful,
- global speaker IDs are stable across shard boundaries,
- and Phase1 still enqueues downstream before visual completion when audio finishes first.

---

## 16. Acceptance Criteria

1. Inputs up to 60 minutes remain on the existing single-pass code path with no behavior change.
2. Inputs in `(60, 90]` minutes are transcribed via exactly 2 parallel shard requests.
3. Inputs in `(90, 180]` minutes are transcribed via exactly 3 parallel shard requests.
4. Inputs above 180 minutes fail fast with a clear error message.
5. `/tasks/vibevoice-asr` still returns one merged `turns` list and one `stage_events` list.
6. The merged output uses global speaker IDs rather than shard-local IDs.
7. Phase1 audio-post still starts only after the final merged ASR result is ready.
8. No change is required in the outer Phase1 runner call pattern or the Phase26 queue contract.

---

## 17. Risks and Watchouts

1. **Boundary loss without overlap:** non-overlapping shards are simpler and match the requested shard-count policy, but they can clip speech near exact cut points.
2. **Verifier false positives:** a threshold that is too low will collapse distinct speakers across shards; too high will over-split the same speaker into multiple globals.
3. **Service wall-clock spikes:** total elapsed time should improve versus serial multi-pass ASR, but shard upload + merge work is still non-zero overhead.
4. **CPU pressure on the H200 host:** CPU-based verification is safer for GPU contention, but representative clip extraction and embedding still need profiling alongside RF-DETR + local audio-post.
5. **Scratch/GCS artifact growth:** long-form jobs create more temporary audio files and runtime GCS objects than current short-form jobs.

---

## 18. Follow-Ups (Out of Scope for This Spec)

1. Add optional shard overlap + deterministic boundary dedupe if no-overlap quality proves insufficient.
2. Add cleanup/retention policy for ephemeral shard GCS objects.
3. Add benchmark-driven sidecar tuning if 3-way parallelism needs more than the current `max-num-seqs=4`.
4. Consider averaging multiple representative clips per speaker if one-clip verification is not stable enough in evals.
