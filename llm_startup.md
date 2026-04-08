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
```

## Key Paths

- Droplet env file: `/etc/clypt-phase1/v3_1_phase1.env`
- Repo on droplet: `/opt/clypt-phase1/repo`
- Input videos on droplet: `/opt/clypt-phase1/videos`
- Phase 1 service logs: `/var/log/clypt/v3_1_phase1`
- Artifacts: `backend/outputs/v3_1/<run_id>/...`

## Critical Env Facts

- Must be `VIBEVOICE_BACKEND=vllm`
- Must be `VIBEVOICE_VLLM_MODEL=vibevoice` (not `microsoft/VibeVoice-ASR`)
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required even for VibeVoice-only smoke runs
- Keep `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1` (strict)
- Keep `CLYPT_PHASE1_YAMNET_DEVICE=cpu`
- Keep cache env aligned to `/opt/clypt-phase1/.cache` roots (`TORCH_HOME`, `HF_HOME`; `MODELSCOPE_CACHE` is optional compatibility only)
- Keep `VIBEVOICE_VLLM_MAX_RETRIES=1` (single-attempt behavior; no automatic retry policy yet)
- Keep deploy prewarm enabled (`PREWARM_PHASE1_MODELS=1`, default)

## Important Gotchas

- Do not use legacy/non-vLLM ASR code paths.
- Do not use `file://` with `--source-url`; use `--source-path` for local files.
- First vLLM start may take 15-30 min due model download; later starts are much faster (cached).
- If deployment pip resolver backtracks/fails on old `datasets/pyarrow`, deploy script auto-fallbacks to legacy resolver; this is expected.
- emotion2vec progress logs now report true argmax top labels; old logs with `top: angry 0.00` were a logging artifact, not necessarily inference failure.

## Session Checklist For Any LLM

1. Read the 4 docs above.
2. Confirm active services + `/health` + `/v1/models` (`vibevoice` present).
3. Confirm env loaded from `/etc/clypt-phase1/v3_1_phase1.env`.
4. Run VibeVoice-only smoke test first.
5. Then run Phase 1/1-4 job and tail logs.
