# Phase1 Host Deploy

Deploy the **Phase1 H200** host.

This host owns:

- Phase 1 runner/orchestrator
- local VibeVoice service
- local visual service
- co-located VibeVoice vLLM sidecar
- in-process NFA -> emotion2vec+ -> YAMNet

## 1) Bootstrap

On the target H200:

```bash
ssh root@<phase1-host>
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/bootstrap_phase1_h200.sh
```

That prepares:

- `/opt/clypt-phase1`
- `/opt/clypt-phase1/venvs`
- `/etc/clypt-phase1`
- `/var/lib/clypt/phase1`
- `/var/log/clypt/phase1`

## 2) Env File

Copy the baseline:

- [known-good-phase1-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h200.env)

Optional overlay for H100 only:

- [known-good-phase1-h100-backup.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h100-backup.env)

Install to:

```bash
/etc/clypt-phase1/phase1.env
```

Required values to set:

- service account key path
- bearer tokens
- Phase26 dispatch URL
- GCS / Spanner project-specific values

Long-form ASR defaults already live in the baseline env:

- `<= 40 minutes`: single-pass VibeVoice
- `> 40 and <= 80 minutes`: 2 shard VibeVoice fan-out
- `> 80 and <= 160 minutes`: 4 shard VibeVoice fan-out
- `> 160 minutes`: fail fast

Topology note:

- The current working split runs across two DigitalOcean teams, so Phase1 talks
  to Phase26 over the public endpoint.
- The live known-good Phase26 dispatch URL is `http://107.170.33.122:9300`.

Credential requirement:

- `GOOGLE_APPLICATION_CREDENTIALS` must point to a real service-account JSON key with `type=service_account` and a private key.
- Do not copy `~/.config/gcloud/application_default_credentials.json` onto the host. That file is typically `authorized_user` ADC and will fail when Phase 1 needs to generate signed GCS URLs for VibeVoice.

## 3) Deploy Services

```bash
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/deploy_phase1_services.sh
```

Before running the deploy:

- exclude repo-root `.env` and `.env.local` from the copy/sync step
- keep host runtime config only in `/etc/clypt-phase1/phase1.env`
- expect `deploy_phase1_services.sh` to fail fast if `.env` or `.env.local` exists on the droplet
- expect `deploy_phase1_services.sh` to fail fast if `GOOGLE_APPLICATION_CREDENTIALS` is not a signing-capable service-account key

This deploys:

- `clypt-phase1-vllm-vibevoice.service`
- `clypt-phase1-vibevoice.service`
- `clypt-phase1-visual.service`
- `clypt-phase1-api.service`
- `clypt-phase1-worker.service`

The Phase1 Python environment now also needs the ECAPA-TDNN speaker verifier dependency used for cross-shard speaker stitching on long-form jobs. That dependency is installed from `requirements-do-phase1-h200.txt`; do not omit it when rebuilding the host venv.

## 4) Health Checks

```bash
systemctl status clypt-phase1-vllm-vibevoice.service --no-pager
systemctl status clypt-phase1-vibevoice.service --no-pager
systemctl status clypt-phase1-visual.service --no-pager
systemctl status clypt-phase1-api.service --no-pager
systemctl status clypt-phase1-worker.service --no-pager
curl -sf http://127.0.0.1:9100/health
curl -sf http://127.0.0.1:9200/health
curl -sf http://127.0.0.1:8080/healthz
curl -sf http://127.0.0.1:8000/health
curl -sf http://127.0.0.1:8000/v1/models
```

## 5) Notes

- Preserve the RF-DETR / ByteTrack settings in the Phase1 env unless explicitly retuning.
- The H100 overlay may only change memory-sensitive VibeVoice knobs.
- Phase1 does not own the downstream SQLite queue anymore.
- The canonical per-run debugging flow now starts with [LOG_EXTRACTION_RUNBOOK.md](/Users/rithvik/Clypt-Backend/docs/runtime/LOG_EXTRACTION_RUNBOOK.md) and [scripts/extract_run_diagnostics.py](/Users/rithvik/Clypt-Backend/scripts/extract_run_diagnostics.py); use that before ad-hoc `journalctl` / SQLite / Spanner spelunking.
- The VibeVoice sidecar must report the downloaded `microsoft/VibeVoice-ASR` snapshot via `/v1/models`; a green outer service health check alone is not enough.
- The VibeVoice service now supports 40-160 minute podcast inputs by splitting canonical audio into 2-4 temporary shard WAVs, uploading them to run-scoped GCS paths, and stitching shard-local speakers back into one global speaker space before Phase1 audio-post begins.
- If `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`, the deploy script now installs and verifies both `trtexec` and the TensorRT Python package automatically.
- The deploy script also prewarms NFA, emotion2vec+, and YAMNet so the first live job does not stall on model downloads.
- That prewarm must run with the same cache env as the live services:
  - `HOME=/opt/clypt-phase1`
  - `CLYPT_PHASE1_CACHE_HOME=/opt/clypt-phase1/.cache`
  - `XDG_CACHE_HOME=/opt/clypt-phase1/.cache`
  - `TORCH_HOME=/opt/clypt-phase1/.cache/torch`
  - `HF_HOME=/opt/clypt-phase1/.cache/huggingface`
- `deploy_phase1_services.sh` now hard-fails if the NeMo forced-aligner model does not land in the service cache tree after prewarm.
