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

- service account path
- bearer tokens
- Phase26 private URL
- GCS / Spanner project-specific values

## 3) Deploy Services

```bash
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/deploy_phase1_services.sh
```

This deploys:

- `clypt-phase1-vllm-vibevoice.service`
- `clypt-phase1-vibevoice.service`
- `clypt-phase1-visual.service`
- `clypt-phase1-api.service`
- `clypt-phase1-worker.service`

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
```

## 5) Notes

- Preserve the RF-DETR / ByteTrack settings in the Phase1 env unless explicitly retuning.
- The H100 overlay may only change memory-sensitive VibeVoice knobs.
- Phase1 does not own the downstream SQLite queue anymore.
