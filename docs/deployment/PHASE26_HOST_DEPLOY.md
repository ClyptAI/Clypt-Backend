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

- service account key path
- bearer tokens
- Modal endpoint URL
- GCS / Spanner project-specific values

Current known-good Modal endpoint:

- `https://rithuuu--clypt-node-media-prep-node-media-prep.modal.run/tasks/node-media-prep`

Observed live non-secret Phase26 values on 2026-04-20:

- host: `clypt-phase26-h200-ming-nyc2` (`162.243.208.185`)
- `GENAI_GENERATION_MODEL=Qwen/Qwen3.6-35B-A3B`
- `CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1`
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_PHASE24_LOCAL_MAX_INFLIGHT=1`

`CLYPT_PHASE24_NODE_MEDIA_PREP_URL` may be set to either the Modal base URL or the full task endpoint URL. The current known-good env uses the full endpoint URL.

The Phase26 worker now uses a submit-and-poll contract against Modal:

- `POST /tasks/node-media-prep` returns a `call_id`
- `GET /tasks/node-media-prep/result/{call_id}` is polled until completion

Credential requirement:

- `GOOGLE_APPLICATION_CREDENTIALS` must point to a real service-account JSON key with `type=service_account` and a private key.
- Do not copy `~/.config/gcloud/application_default_credentials.json` onto the host. Token-only `authorized_user` ADC files are not a supported host credential shape.

## 3) Deploy Services

```bash
cd /opt/clypt-phase26/repo
bash scripts/do_phase26/deploy_phase26_services.sh
```

Before running the deploy:

- exclude repo-root `.env` and `.env.local` from the copy/sync step
- keep host runtime config only in `/etc/clypt-phase26/phase26.env`
- expect `deploy_phase26_services.sh` to fail fast if `.env` or `.env.local` exists on the droplet
- expect `deploy_phase26_services.sh` to fail fast if `GOOGLE_APPLICATION_CREDENTIALS` is not a service-account key

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
