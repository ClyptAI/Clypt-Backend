# Clypt LLM Startup

## What Clypt Is

Clypt V3.1 is a long-form video analysis pipeline.

- Implemented now: Phases 1-4.
- Not implemented yet: Phases 5-6 (speaker grounding + final render path).
- Main/default ASR path on `main`: **vLLM VibeVoice only** (no legacy/native fallback path).

## Read These First (Rule Of Thumb)

1. [README.md](/Users/rithvik/Clypt-V3/README.md)  
   Fast overview, current architecture, baseline commands.
2. [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md)  
   Droplet truth, env contract, deployment gotchas, health checks.
3. [docs/runtime/v3.1_runtime_guide.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md)  
   Runtime/provider behavior, payload flow, Phase 1-4 execution details.
4. [.env.example](/Users/rithvik/Clypt-V3/.env.example)  
   Canonical env keys and defaults.

## Current Default Flow

Phase 1:

- Visual: RF-DETR + ByteTrack
- ASR: VibeVoice via local vLLM service
- Then (audio chain): Forced aligner -> emotion2vec+ -> YAMNet

Transcript/timeline contract:

- VibeVoice outputs **turn-level** items first (`Start/End/Speaker/Content`).
- Forced aligner adds word-level timing.
- Merge step attaches `word_ids` onto turns.
- Gemini phases consume turn/node payloads (not raw canonical `words[]` objects).

Phase 2-4 backend split (current default):

- Generation: Gemini Developer API (`GENAI_GENERATION_BACKEND=developer`)
- Embeddings: Vertex (`GENAI_EMBEDDING_BACKEND=vertex`)
- Queue mode: Cloud Tasks -> Cloud Run worker when `--run-phase14` (default unless `--inline-phase24`)
- Current worker target: `us-east4` Cloud Run GPU revision (L4), with ffmpeg GPU mode enabled

Phase 2-4 Gemini thinking profile (current default):

- Phase 2A merge/classify: Flash + `low`
- Phase 2B boundary reconciliation: Flash + `minimal`
- Phase 3 local semantic edges: Flash + `minimal`
- Phase 3 long-range adjudication: Flash + `low`
- Phase 4 meta prompt generation: Flash + `low`
- Phase 4 subgraph reviews: Flash + `medium`
- Phase 4 pooled candidate review: Flash + `medium`

## Start Runs

### On Droplet: full Phase 1 (and optional 1-4)

```bash
cd /opt/clypt-phase1/repo
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a

# Local video:
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14

# YouTube source:
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --run-phase14
```

### On Droplet: VibeVoice-only smoke test

```bash
cd /opt/clypt-phase1/repo
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
export PYTHONPATH=.
python scripts/run_vibevoice_only.py --audio /opt/clypt-phase1/videos/<video>.mp4
```

### Service mode (API + worker)

```bash
# service health
sudo systemctl is-active clypt-vllm-vibevoice clypt-v31-phase1-api clypt-v31-phase1-worker
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

### Log Tailing

```bash
# worker service logs
sudo journalctl -u clypt-v31-phase1-worker -f

# phase1 job logs
tail -f /var/log/clypt/v3_1_phase1/<job_log>.log

# phase timing gates only
tail -f /var/log/clypt/v3_1_phase1/<job_log>.log | grep --line-buffered -E "Phase 2 done|Phase 3 done|Phase 4 done|Phases 2-4 done|Phase 1-4 run complete"

# Cloud Run Phase 2-4 worker logs (queue mode, GPU worker)
gcloud run services logs read clypt-phase24-worker --region=us-east4 --project=clypt-v3 --follow
```

## Key Paths

- Droplet env file: `/etc/clypt-phase1/v3_1_phase1.env`
- Repo on droplet: `/opt/clypt-phase1/repo`
- Input videos on droplet: `/opt/clypt-phase1/videos`
- Phase 1 service logs: `/var/log/clypt/v3_1_phase1`
- Artifacts: `backend/outputs/v3_1/<run_id>/...`

## Canonical Repro Source

Use this as the single source of truth for copy/paste runtime setup and verification:
- [Canonical Repro Checklist](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md#canonical-repro-checklist) in `docs/runtime/v3.1_runtime_guide.md`

## Important Gotchas

- Do not use legacy/non-vLLM ASR code paths.
- Do not use `file://` with `--source-url`; use `--source-path` for local files.
- First vLLM start may take 15-30 min due model download; later starts are much faster (cached).
- If deployment pip resolver backtracks/fails on old `datasets/pyarrow`, deploy script auto-fallbacks to legacy resolver; this is expected.
- emotion2vec progress logs now report true argmax top labels; old logs with `top: angry 0.00` were a logging artifact, not necessarily inference failure.
- Queue-mode Phase 2 can be much slower when Cloud Run worker lacks ffmpeg GPU; logs show `GPU ffmpeg unavailable ... falling back to CPU encoder`.
- If you see `Budget 0 is invalid. This model only works in thinking mode.`, set `VERTEX_THINKING_BUDGET` to a low positive value (default now `128`).
- If you see `Gemini returned an edge for a non-shortlisted candidate long-range pair`, it is a strict Phase 3 validation guard; queue retries may recover, otherwise rerun.
