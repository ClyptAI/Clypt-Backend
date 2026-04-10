# RUNTIME GUIDE

**Status:** Active  
**Last updated:** 2026-04-10

This is the canonical runtime reference for implemented backend behavior (Phases 1-4) plus comments/trends augmentation in Phase 2-4.

## 1) Current Runtime Topology

- **Phase 1 host:** GPU droplet (`run_phase1`, API service, worker loop).
- **ASR path:** `VibeVoiceVLLMProvider` against local `clypt-vllm-vibevoice.service`.
- **Phase 2-4 execution path:** Cloud Tasks dispatch to Cloud Run worker (`us-east4`), defaulting to an L4 GPU-accelerated worker profile.
- **Generation backend:** Gemini Developer API only (`GENAI_GENERATION_BACKEND=developer`).
- **Embedding backend:** Vertex (`VERTEX_EMBEDDING_BACKEND=vertex`).

## 2) Phase 1 Execution Semantics

Phase 1 runs visual extraction and ASR concurrently, then starts audio sidecars immediately after ASR completes:

```text
visual(RFDETR+ByteTrack) -----------------------\
                                                 +--> both done
asr(vLLM HTTP) ---------------------------------/
   |
   +--> forced-aligner -> emotion2vec+ -> YAMNet
```

Critical rule: the audio chain starts from `asr_future.result()` and must not be delayed behind RF-DETR completion.

## 3) Comments + Trends Augment Integration (Phase 2-4)

When enabled:

- comments future starts after Phase24 preflight
- trends future starts after Phase 2 summaries are available
- both run in parallel with core work where applicable
- **hard join** happens before Phase 4 prompt seeding
- invalid/failed enabled signal pipeline is terminal (fail-fast)

When both signals are disabled, pipeline proceeds with general prompts only.

## 4) Canonical Repro Checklist

### 4.1 Load env

```bash
cd /opt/clypt-phase1/repo
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
```

### 4.2 Verify services

```bash
sudo systemctl is-active clypt-vllm-vibevoice clypt-v31-phase1-api clypt-v31-phase1-worker
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

### 4.3 Run pipeline

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

### 4.4 Observe logs

```bash
tail -f /var/log/clypt/v3_1_phase1/<job_log>.log
gcloud run services logs read clypt-phase24-worker --region=us-east4 --project=clypt-v3 --follow
```

### 4.5 Verify run in Spanner

```bash
gcloud spanner databases execute-sql clypt-graph-db-v3 \
  --instance=clypt-spanner-v3 --project=clypt-v3 \
  --sql="SELECT run_id,status,updated_at FROM runs WHERE run_id='<run_id>'"

gcloud spanner databases execute-sql clypt-graph-db-v3 \
  --instance=clypt-spanner-v3 --project=clypt-v3 \
  --sql="SELECT phase_name,status,duration_ms,started_at,ended_at FROM phase_metrics WHERE run_id='<run_id>' ORDER BY started_at ASC"
```

## 5) Key Env Contract

Required regardless of run type:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `VIBEVOICE_BACKEND=vllm`
- `VIBEVOICE_VLLM_BASE_URL`
- `VIBEVOICE_VLLM_MODEL=vibevoice`

Important defaults:

- `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1`
- `CLYPT_PHASE1_YAMNET_DEVICE=cpu`
- `CLYPT_GEMINI_MAX_CONCURRENT=8`
- `CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY=8`
- `CLYPT_PHASE3_TARGET_BATCH_COUNT=3`
- `CLYPT_PHASE3_MAX_NODES_PER_BATCH=24`

### 5.1 Phase 2-4 worker defaults (future runs)

The default Phase 2-4 Cloud Run worker profile for production runs is `us-east4` L4 GPU-accelerated with the following env baseline:

```bash
CLYPT_PHASE24_FFMPEG_DEVICE=gpu
CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY=8
CLYPT_GEMINI_MAX_CONCURRENT=8
CLYPT_PHASE3_TARGET_BATCH_COUNT=3
CLYPT_PHASE3_MAX_NODES_PER_BATCH=24

CLYPT_PHASE2_MERGE_THINKING_LEVEL=low
CLYPT_PHASE2_BOUNDARY_THINKING_LEVEL=minimal
CLYPT_PHASE3_LOCAL_THINKING_LEVEL=minimal
CLYPT_PHASE3_LONG_RANGE_THINKING_LEVEL=low
CLYPT_PHASE4_META_THINKING_LEVEL=low
CLYPT_PHASE4_SUBGRAPH_THINKING_LEVEL=medium
CLYPT_PHASE4_POOL_THINKING_LEVEL=medium
```

CPU encoder execution is a fallback/degraded path, not the default.

### 5.2 Comments/Trends signal defaults (future runs)

#### Signal LLM routing defaults

Use this mapping for the implemented signal pipeline. Purpose is listed first, then model + thinking.

1. Purpose: comment/trend cluster -> clip-seeking retrieval prompt generation  
   Model: `gemini-3-flash`  
   Thinking: `low`
2. Purpose: trend relevance adjudication against video context  
   Model: `gemini-3-flash`  
   Thinking: `minimal`
3. Purpose: comment/reply quality classification (`useful` vs `noise`)  
   Model: `gemini-3.1-flash-lite`  
   Thinking: `low`
4. Purpose: cluster-to-node moment span resolution  
   Model: `gemini-3-flash`  
   Thinking: `minimal`
5. Purpose: trend query synthesis from video context  
   Model: `gemini-3-flash`  
   Thinking: `low`
6. Purpose: top-level + full-reply thread consolidation  
   Model: `gemini-3-flash`  
   Thinking: `minimal`
7. Purpose: candidate attribution explanation text for UI/debug  
   Model: `gemini-3.1-flash-lite`  
   Thinking: `low`

Equivalent env contract:

```bash
CLYPT_SIGNAL_LLM_FAIL_FAST=1
CLYPT_SIGNAL_LLM_MODEL_1=gemini-3-flash
CLYPT_SIGNAL_LLM_THINKING_1=low
CLYPT_SIGNAL_LLM_MODEL_2=gemini-3-flash
CLYPT_SIGNAL_LLM_THINKING_2=minimal
CLYPT_SIGNAL_LLM_MODEL_3=gemini-3.1-flash-lite
CLYPT_SIGNAL_LLM_THINKING_3=low
CLYPT_SIGNAL_LLM_MODEL_5=gemini-3-flash
CLYPT_SIGNAL_LLM_THINKING_5=minimal
CLYPT_SIGNAL_LLM_MODEL_9=gemini-3-flash
CLYPT_SIGNAL_LLM_THINKING_9=low
CLYPT_SIGNAL_LLM_MODEL_10=gemini-3-flash
CLYPT_SIGNAL_LLM_THINKING_10=minimal
CLYPT_SIGNAL_LLM_MODEL_11=gemini-3.1-flash-lite
CLYPT_SIGNAL_LLM_THINKING_11=low
```

#### Signal pipeline env defaults

```bash
CLYPT_ENABLE_COMMENT_SIGNALS=1
CLYPT_ENABLE_TREND_SIGNALS=1
CLYPT_SIGNAL_MODE=augment
CLYPT_SIGNAL_FAIL_FAST=1
CLYPT_SIGNAL_MAX_HOPS=2
CLYPT_SIGNAL_TIME_WINDOW_MS=30000
CLYPT_COMMENT_ORDER=relevance
CLYPT_COMMENT_TOP_THREADS_MIN=15
CLYPT_COMMENT_TOP_THREADS_MAX=40
CLYPT_SIGNAL_EPSILON=1e-6
CLYPT_SIGNAL_ENGAGEMENT_TOP_LIKE_WEIGHT=0.65
CLYPT_SIGNAL_ENGAGEMENT_TOP_REPLY_WEIGHT=0.35
CLYPT_SIGNAL_ENGAGEMENT_REPLY_LIKE_WEIGHT=0.85
CLYPT_SIGNAL_ENGAGEMENT_REPLY_PARENT_WEIGHT=0.15
CLYPT_SIGNAL_CLUSTER_MEAN_WEIGHT=0.45
CLYPT_SIGNAL_CLUSTER_MAX_WEIGHT=0.25
CLYPT_SIGNAL_CLUSTER_FREQ_WEIGHT=0.30
CLYPT_SIGNAL_CLUSTER_FREQ_REF=30
CLYPT_SIGNAL_HOP_DECAY_1=0.75
CLYPT_SIGNAL_HOP_DECAY_2=0.55
CLYPT_SIGNAL_COVERAGE_WEIGHT=0.30
CLYPT_SIGNAL_DIRECT_RATIO_WEIGHT=0.15
CLYPT_SIGNAL_CLUSTER_CAP=0.12
CLYPT_SIGNAL_TOTAL_CAP=0.20
CLYPT_SIGNAL_AGREEMENT_CAP=0.10
CLYPT_SIGNAL_MEANINGFUL_MIN_SOURCE_COVERAGE=0.15
CLYPT_SIGNAL_AGREEMENT_BONUS_TIER1=0.04
CLYPT_SIGNAL_AGREEMENT_BONUS_TIER2=0.07
```

Set explicitly per deployment (no single locked numeric default in runtime docs):

- `CLYPT_COMMENT_MAX_REPLIES_PER_THREAD`
- `CLYPT_COMMENT_CLUSTER_SIM_THRESHOLD`
- `CLYPT_TREND_MAX_ITEMS`
- `CLYPT_TREND_RELEVANCE_THRESHOLD`
- `CLYPT_SIGNAL_MEANINGFUL_MIN_CLUSTER_CONTRIB`

## 6) Timing/Health Markers

Expect these log gates in successful Phase 2-4 runs:

- `[phase14] Phase 2 done in ...`
- `[phase14] Phase 3 done in ...`
- `[phase14] Phase 4 done in ...`
- `[phase14] Phases 2-4 done in ...`
- `[phase2] merge+boundary done in ...`
- `[phase3] local semantic edges done in ...`
- `[phase4] pooled review done in ...`

## 7) Known Operational Failure Modes

1. `RESOURCE_EXHAUSTED` on larger runs (especially Phase 3 local-edge batches).
2. `GPU ffmpeg unavailable ... falling back to CPU encoder` on queue worker (major Phase 2 slowdown).
3. `Gemini returned an edge for a non-shortlisted candidate long-range pair` strict validation failure.
4. `Budget 0 is invalid...` if thinking budget is misconfigured for affected models.

### 7.1 Detailed Phase 2-4 worker caveats

1. `embed_content` input shape caveat (`gemini-embedding-2-preview`):
   - a list of texts is treated as one document (returns one embedding, not N).
   - at most one video/media part is accepted per call.
   - batching must be done by looping per item in application code.
2. Vertex Priority PayGo caveat:
   - generation is locked to Developer API, so Vertex generation Priority PayGo is not part of the active runtime path.
   - embedding endpoint remains `us-central1`; keep embedding traffic on Vertex as configured.
3. Quota pressure on longer runs:
   - Phase 3 local-edge concurrency can exhaust RPM/TPM (`RESOURCE_EXHAUSTED`).
   - retries can recover transient cases; sustained pressure may still fail the run.
4. Strict long-range validation:
   - non-shortlisted long-range edge proposals are hard failures by design.
   - queue retries may recover; repeated failures usually require rerun.

See [ERROR_LOG.md](../ERROR_LOG.md) for incident history and recoveries.

## 8) Implemented vs Planned Boundary

- Implemented: Phase 1-4 + comments/trends augmentation + Spanner persistence.
- Planned: Phase 5 participation grounding and Phase 6 render/9:16 path (documented in specs + architecture).
