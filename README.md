# Clypt V3.1 Backend

This branch contains the V3.1 backend implementation for Phases 1–4 of the Clypt pipeline, with the vLLM VibeVoice ASR sidecar.

Phases implemented:

1. `Phase 1: Timeline Foundation` — visual extraction, ASR, forced alignment, emotion, audio events
2. `Phase 2: Node Construction, Classification, And Embedding`
3. `Phase 3: Graph Construction`
4. `Phase 4: Candidate Retrieval, Subgraph Selection, And Ranking`

Phases 5–6 (speaker participation grounding, render/9:16 output) are not yet implemented.

## Current Working Setup

### Phase 1 — Concurrent extraction

Phase 1 runs visual extraction and ASR **concurrently**. The audio chain (NFA → emotion2vec+ → YAMNet) starts immediately when ASR finishes — without waiting for RF-DETR:

```
visual (RF-DETR + ByteTrack) ───────────────────────┐
                               ThreadPoolExecutor    ├── both done
ASR (VibeVoice vLLM HTTP) ───┘  (max_workers=3)     │
    ↓ immediately (RF-DETR still running)            │
NFA → emotion2vec+ → YAMNet ────────────────────────┘
    (serial with each other; concurrent with RF-DETR)
```

Audio artifacts ready ~230s before RF-DETR finishes on a 13-min clip. The audio chain callback fires immediately after `asr_future.result()` — **not** inside an `as_completed` loop (that pattern caused the callback to wait for RF-DETR too, see [Known Gotchas in the deployment doc](docs/deployment/v3.1_phase1_digitalocean.md)).

- **ASR:** `VibeVoiceVLLMProvider` — HTTP calls to `clypt-vllm-vibevoice.service` (Docker-managed vLLM container)
- **Visual:** RF-DETR Small + ByteTrack (`pytorch_cuda_fp16` default)
- **vLLM version:** `v0.14.1` (pinned — tested compatible with the VibeVoice plugin)
- **Served model name:** `vibevoice` (not `microsoft/VibeVoice-ASR`)

### Phase 1 transcript assembly (current default)

- VibeVoice returns turn-level output first: `{Start, End, Speaker, Content}`.
- NFA then aligns the transcript to audio and produces word-level timings.
- `merge_vibevoice_outputs()` assigns aligned `word_ids` back onto each VibeVoice turn.
- Downstream Gemini phases consume turn/node payloads (not raw canonical `words[]` objects).

### Phase 2–4 — Current backend + performance notes

Phases 2–4 now run in queue mode by default (`Cloud Tasks -> Cloud Run worker`) when `--run-phase14` is used. Backend split:

- Generation calls: Gemini **Developer API** (`GENAI_GENERATION_BACKEND=developer`)
- Embeddings (text + multimodal URI): **Vertex AI** (`GENAI_EMBEDDING_BACKEND=vertex`)
- Runtime target for queue worker: Cloud Run `us-east4` with L4 GPU and ffmpeg GPU mode (`CLYPT_PHASE24_FFMPEG_DEVICE=gpu`)

Current optimizations:

| Optimization | Effect |
|---|---|
| **Developer API generation path** | Uses AI Studio/Developer API for generate-content calls; avoids prior Vertex DSQ tail-latency pattern for this workload. |
| **Per-stage Flash thinking levels** | Phase 2A `low`, Phase 2B `minimal`, Phase 3 local `minimal`, Phase 3 long-range `low`, Phase 4 meta `low`, Phase 4 subgraph `medium`, Phase 4 pool `medium`. |
| **`responseSchema` constrained decoding** | Per-phase JSON Schema passed to `GenerateContentConfig.response_schema` — faster generation, no hallucinated fields, no parse failures. |
| **`max_output_tokens` caps (`32768`)** | Applied for all Phase 2–4 Gemini JSON calls. |
| **Parallel embeddings** (ThreadPoolExecutor) | `embed_texts` and `embed_media_uris` each spawn one `embed_content` call per item in parallel (API limit: 1 text doc or 1 video part per call). |
| **Batch/concurrency tuning** | `CLYPT_GEMINI_MAX_CONCURRENT=8`, `CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY=8`, Phase 3 geometry tuned to fewer/larger batches (`target=3`, `max_nodes=24`). |
| **Dynamic meta-prompts** (Phase 4) | Gemini Flash reads Phase 2 node summaries and generates video-specific retrieval queries. Count scales with duration: 2/4/6/8/10 prompts. No static fallback — fails fast if Gemini returns nothing. |
| **Sub-step timers** | Added around merge/boundary, node media extraction+upload, semantic vs multimodal embeddings, local vs long-range vs persistence, and Phase 4 review stages. |

### Validated runs (2026-04-07/08/09)

| Clip | Duration | Turns | Phase 1 | Phases 2–4 | Total | Mode |
|------|----------|-------|---------|-----------|-------|------|
| mrbeastflagrant.mp4 | 392.9s (6.5 min) | 104 | 153s | 98s | **251s (4m11s)** | Full Phase 1–4 ✓ |
| mrbeastflagrant.mp4 | 392.9s (6.5 min) | 104 | — | **820.8s (13m41s)** | — | Phase 2–4 Cloud Run queue worker (slow but successful baseline) |
| mrbeastflagrant.mp4 (verified Phase 1 handoff replay) | 392.9s (6.5 min) | 104 | — | **143.8s (2m24s)** | — | Phase 2–4 Cloud Run queue worker (us-east4 L4, tuned Flash thinking profile) |
| joeroganflagrant.mp4 | 788.7s (13.1 min) | 201 | 285s | — | — | Phase 1 only (Phase 3 hit 429) |

ASR RTF ~0.07x on the validated GPU droplets. Full Phase 1–4 pipeline is **0.64× real-time** on a 6.5-min clip.

Operational updates (2026-04-08):
- Fresh `ai/ml` base-image redeploy run (`run_20260408_095543_mrbeastflagrant`) completed in ~4m34s wall time with **Phases 2-4 = 271.8s**.
- Live logs now include explicit timing gates: `Phase 2 done`, `Phase 3 done`, `Phase 4 done`, and `Phases 2-4 done`.
- emotion2vec progress logging now reports the true top class via score argmax (older logs could show misleading entries like `top: angry 0.00`).
- Baseline run metrics and final clip candidate snapshots are consolidated in [docs/runtime/v3.1_baseline_reference.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_baseline_reference.md).

Operational updates (2026-04-09):
- Generation backend default switched to Developer API; embeddings remain on Vertex.
- Added `VERTEX_THINKING_BUDGET` (default `128`) as a compatibility knob if a Pro generation path is re-enabled.
- Current tuned worker profile uses Flash for all Phase 2-4 generation calls with stage-specific thinking levels and GPU ffmpeg on Cloud Run (`us-east4` L4).

**429 note:** Longer videos (200+ turns) can hit `RESOURCE_EXHAUSTED` on Phase 3 local-edge batches. Transient retries/backoff are enabled, but sustained quota pressure may still require rerun. See [Known Issues](#known-issues).

## Environment

Canonical copy/paste repro checklist (authoritative):
- [Canonical Repro Checklist](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md#canonical-repro-checklist) in `docs/runtime/v3.1_runtime_guide.md`

Core env vars (summary):

```bash
CLYPT_V31_OUTPUT_ROOT=backend/outputs/v3_1
CLYPT_PHASE1_WORK_ROOT=backend/outputs/v3_1_phase1_work
CLYPT_PHASE1_KEEP_WORKDIR=0
CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1   # fail hard on 0-word alignment for non-empty turns
VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice             # NOT "microsoft/VibeVoice-ASR" — returns 404
GOOGLE_CLOUD_PROJECT=clypt-v3              # required even for VibeVoice-only runs
GCS_BUCKET=clypt-storage-v3               # required even for VibeVoice-only runs
GOOGLE_CLOUD_LOCATION=global              # used when generation backend is Vertex
VERTEX_GEMINI_LOCATION=global
VERTEX_EMBEDDING_LOCATION=us-central1     # embedding endpoint (cannot use global)
GENAI_GENERATION_BACKEND=developer        # default generation backend (AI Studio/Developer API)
GENAI_EMBEDDING_BACKEND=vertex            # keep embeddings on Vertex (required for media URI embeddings)
GEMINI_API_KEY=<from secret manager/env>
VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview
VERTEX_FLASH_MODEL=gemini-3-flash-preview # current default model for all Phase 2-4 generation calls
VERTEX_EMBEDDING_MODEL=gemini-embedding-2-preview
VERTEX_THINKING_BUDGET=128               # keep for compatibility if a Pro call path is re-enabled
VERTEX_API_MAX_RETRIES=6                 # retry transient Vertex 429/5xx without lowering parallelism
VERTEX_API_INITIAL_BACKOFF_S=1.0
VERTEX_API_MAX_BACKOFF_S=30.0
VERTEX_API_BACKOFF_MULTIPLIER=2.0
VERTEX_API_JITTER_RATIO=0.2
CLYPT_PHASE1_YAMNET_DEVICE=cpu            # TF GPU path is unstable on current droplets — keep cpu
CLYPT_PHASE1_CACHE_HOME=/opt/clypt-phase1/.cache
XDG_CACHE_HOME=/opt/clypt-phase1/.cache
TORCH_HOME=/opt/clypt-phase1/.cache/torch
HF_HOME=/opt/clypt-phase1/.cache/huggingface
FUNASR_MODEL_SOURCE=hf
CLYPT_PHASE24_FFMPEG_DEVICE=gpu
CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY=8
CLYPT_GEMINI_MAX_CONCURRENT=8
CLYPT_PHASE3_TARGET_BATCH_COUNT=3
CLYPT_PHASE3_MAX_NODES_PER_BATCH=24
CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS=32768
CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS=32768
CLYPT_PHASE2_MERGE_THINKING_LEVEL=low
CLYPT_PHASE2_BOUNDARY_THINKING_LEVEL=minimal
CLYPT_PHASE3_LOCAL_THINKING_LEVEL=minimal
CLYPT_PHASE3_LONG_RANGE_THINKING_LEVEL=low
CLYPT_PHASE4_META_THINKING_LEVEL=low
CLYPT_PHASE4_SUBGRAPH_THINKING_LEVEL=medium
CLYPT_PHASE4_POOL_THINKING_LEVEL=medium
CLYPT_PHASE24_PROJECT=clypt-v3
CLYPT_PHASE24_TASKS_LOCATION=us-central1
CLYPT_PHASE24_TASKS_QUEUE=clypt-phase24
CLYPT_PHASE24_WORKER_URL=https://clypt-phase24-worker-m64xv2dm7a-uk.a.run.app/tasks/phase24
CLYPT_PHASE24_WORKER_SERVICE_ACCOUNT_EMAIL=clypt-phase24-worker@clypt-v3.iam.gserviceaccount.com
CLYPT_SPANNER_PROJECT=clypt-v3
CLYPT_SPANNER_INSTANCE=clypt-spanner-v3
CLYPT_SPANNER_DATABASE=clypt-graph-db-v3
```

Full config in [.env.example](/Users/rithvik/Clypt-V3/.env.example).

## Setup

```bash
# Primary venv (vLLM pipeline)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For Phase 1 GPU deployment
pip install -r requirements-do-phase1.txt
```

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

Pipeline tests (Phases 1–4) run fully offline — no API keys, GPU, or vLLM service required.

## VibeVoice-Only Smoke Test

```bash
source .venv/bin/activate
export PYTHONPATH=.
export VIBEVOICE_BACKEND=vllm
export VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
export VIBEVOICE_VLLM_MODEL=vibevoice
export GOOGLE_CLOUD_PROJECT=clypt-v3
export GCS_BUCKET=clypt-storage-v3
python scripts/run_vibevoice_only.py --audio /path/to/clip.mp4
```

Accepts MP4, WAV, MP3, etc. MP4 files are automatically extracted to MP3 before sending.

## Production Deployment

Target: DigitalOcean GPU droplet (currently validated on `atl1` H200 / Ubuntu 22.04 + CUDA 12 base image).

```bash
# Sync repo to droplet
rsync -az --delete --exclude='.git' --exclude='.venv' --exclude='outputs' \
  -e "ssh -i ~/.ssh/clypt_do_ed25519" \
  /Users/rithvik/Clypt-V3/ root@<DROPLET_IP>:/opt/clypt-phase1/repo/

# Deploy vLLM service (run once on fresh droplet)
ssh -i ~/.ssh/clypt_do_ed25519 root@<DROPLET_IP>
bash /opt/clypt-phase1/repo/scripts/do_phase1/deploy_vllm_service.sh
```

First start downloads the ~17GB model — takes 15–30 minutes. Subsequent starts complete in ~2–3 minutes (model is cached).

See [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md) for the full deployment reference including all gotchas.

## Runtime Paths

### Injected / test path

- [backend/pipeline/orchestrator.py](/Users/rithvik/Clypt-V3/backend/pipeline/orchestrator.py) — `V31Phase14Orchestrator` / `V31Phase14RunInputs`

### Live / provider-backed path

- [backend/runtime/phase14_live.py](/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py) — `V31LivePhase14Runner`

### Phase 1 sidecar orchestration

- [backend/phase1_runtime/extract.py](/Users/rithvik/Clypt-V3/backend/phase1_runtime/extract.py) — vLLM concurrent orchestration

### Phase 1 CLI

```bash
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
python -m backend.runtime.run_phase1 \
  --job-id demo_run \
  --source-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Known Issues

### 429 RESOURCE_EXHAUSTED (longer videos)

Longer videos (200+ speaker turns) generate more Phase 3 edge batches, which can still exhaust model quota/rate windows.

**Current behavior:** transient API retries/backoff are implemented, and queue-mode tasks can be retried by Cloud Tasks; sustained quota exhaustion can still require rerun after quota recovery.

### Phase 3 local-edge batching vs rate limits

Phase 3 sends N concurrent batches of local edge proposals to Gemini Flash. With 200+ nodes the batch count can exceed the rate limit window. Consider reducing `phase3_target_batch_count` in `V31Config` if you consistently hit 429 on Phase 3.

### Cloud Run queue-mode throughput can be CPU-bound

If Phase 2 logs show `GPU ffmpeg unavailable ... falling back to CPU encoder`, node clip extraction runs on CPU in the worker and Phase 2-4 duration will be significantly slower than the inline GPU baseline.

### Strict long-range edge validation can fail a run

Rarely, Phase 3 can fail with `Gemini returned an edge for a non-shortlisted candidate long-range pair`. This is a strict validation guard in the long-range edge path; queue retries may recover, but repeated failures require rerun.

## Key Docs

- [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md) — deployment, env setup, all gotchas
- [docs/runtime/v3.1_runtime_guide.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md) — provider surface, execution model
- [docs/runtime/v3.1_baseline_reference.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_baseline_reference.md) — canonical baseline runs/candidates for migration and cutover validation
- [docs/superpowers/specs/2026-04-06-vllm-vibevoice-phase1-design.md](/Users/rithvik/Clypt-V3/docs/superpowers/specs/2026-04-06-vllm-vibevoice-phase1-design.md) — vLLM migration design spec

## Repository Structure

```
backend/pipeline/          — Phase 1-4 contracts, transforms, orchestrator
backend/providers/         — VibeVoice vLLM, Vertex AI, GCS, emotion2vec+, YAMNet
backend/phase1_runtime/    — Phase 1 sidecar orchestration, visual pipeline, job store
backend/runtime/           — Live Phase 1-4 runner, CLI entrypoints
docker/vibevoice-vllm/     — Dockerfile for the vLLM sidecar (based on vllm/vllm-openai:v0.14.1)
scripts/do_phase1/         — Deployment scripts and systemd units
docs/                      — Deployment reference, runtime guide, specs
```
