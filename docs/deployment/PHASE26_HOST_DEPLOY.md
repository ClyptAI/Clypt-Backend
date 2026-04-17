# Phase26 Host Deploy

Deploy the **Phase26 H200** host.

This host owns:

- `POST /tasks/phase26-enqueue`
- local SQLite queue
- local Phase 2-4 worker/runtime
- SGLang Qwen on `:8001`
- future Phase 5-6 orchestration

## 1) Bootstrap

On the target H200:

```bash
ssh root@<phase26-host>
cd /opt/clypt-phase26/repo
bash scripts/do_phase26/bootstrap_phase26_h200.sh
```

## 2) Env File

Use:

- [known-good-phase26-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase26-h200.env)

Install to:

```bash
/etc/clypt-phase26/phase26.env
```

Required values to set:

- service account path
- bearer tokens
- Modal endpoint URL
- GCS / Spanner project-specific values

## 3) Deploy Services

```bash
cd /opt/clypt-phase26/repo
bash scripts/do_phase26/deploy_phase26_services.sh
```

This deploys:

- `clypt-phase26-sglang-qwen.service`
- `clypt-phase26-dispatch.service`
- `clypt-phase26-worker.service`

## 4) Health Checks

```bash
systemctl status clypt-phase26-sglang-qwen.service --no-pager
systemctl status clypt-phase26-dispatch.service --no-pager
systemctl status clypt-phase26-worker.service --no-pager
curl -sf http://127.0.0.1:8001/health
curl -sf http://127.0.0.1:9300/health
curl -sf http://127.0.0.1:8001/v1/models
```

## 5) Notes

- Queue backend must remain `local_sqlite`.
- Node-media-prep must point at Modal, not the Phase1 host.
- The host-level entrypoint is `run_phase26_worker`, even though the underlying business logic still lives under current `phase24` modules.
