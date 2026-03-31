# Environment Variables

Authoritative lists live in the templates (they mirror what code reads):

- Root / local orchestration: **`.env.example`**
- DigitalOcean Phase 1 API + worker host: **`backend/do_phase1_service/.env.example`**

Do not commit real secrets.

## Team-Critical Vars

| Variable | Required for | Notes |
| --- | --- | --- |
| `SENSO_API_KEY` | Creator onboarding | Required for real Senso-backed profiles |
| `SENSO_CREATOR_PROFILE_PROMPT_ID` | Creator onboarding | Set locally/outside git |
| `YOUTUBE_API_KEY` | YouTube metadata/trends/comments | Optional for some local flows; required for YouTube API routes in `backend/api/app.py` |
| `CLYPT_FRONTEND_ORIGINS` | Frontend → backend local API calls | Comma-separated allowed origins |
| `DO_PHASE1_BASE_URL` | Remote Phase 1 extraction | Same role as `PHASE1_SERVICE_URL` / `DIGITALOCEAN_PHASE1_BASE_URL`; see `get_phase1_service_base_url()` in `backend/pipeline/phase_1_do_pipeline.py` |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS / Vertex / Google SDKs | Host-level path to service account JSON |
| `GCS_BUCKET` | Artifact persistence | `backend/do_phase1_service/storage.py` defaults to `clypt-storage-v3` when unset |

## Recommended Local Backend Setup

```env
SENSO_API_KEY=...
SENSO_CREATOR_PROFILE_PROMPT_ID=...
YOUTUBE_API_KEY=...
CLYPT_FRONTEND_ORIGINS=http://localhost:8080,http://localhost:3000,http://localhost:5173
```

If running extraction via remote Phase 1 service:

```env
DO_PHASE1_BASE_URL=http://<droplet-ip>:8080
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-sa.json
GOOGLE_CLOUD_PROJECT=clypt-v3
GCS_BUCKET=clypt-storage-v3
```

## Local Phase 1 client (orchestration) env

Read by `backend/pipeline/phase_1_do_pipeline.py` (and related tests):

- URL: `DO_PHASE1_BASE_URL`, `PHASE1_SERVICE_URL`, `DIGITALOCEAN_PHASE1_BASE_URL`
- Polling/timeouts: `DO_PHASE1_POLL_INTERVAL_SECONDS`, `DO_PHASE1_TIMEOUT_SECONDS`
- Runtime controls snapshot: `PHASE1_RUNTIME_PROFILE`, `PHASE1_FORCE_LRASD`, `PHASE1_SPEAKER_BINDING_MODE`, `PHASE1_TRACKING_MODE`, `PHASE1_TRACKER_BACKEND` (ByteTrack-only), `PHASE1_SHARED_ANALYSIS_PROXY`, `PHASE1_HEURISTIC_BINDING_ENABLED`
- yt-dlp / video: `YTDLP_*`, `ALLOW_LOW_RES_VIDEO`, etc. (see `.env.example`)

## Phase 1 Service Env (DigitalOcean)

Prefer editing:

- `backend/do_phase1_service/.env.example` (template)
- `/etc/clypt-phase1/do-phase1.env` (host runtime file)

**Uvicorn:** `UVICORN_HOST` and `UVICORN_PORT` are consumed by `scripts/do_phase1/systemd/clypt-phase1-api.service` (`ExecStart=... --host ${UVICORN_HOST} --port ${UVICORN_PORT}`).

Notable service groups (non-exhaustive; see template + worker):

- Service/runtime: `DO_PHASE1_*`, `UVICORN_*`, `DO_REGION`
- Tracking: `CLYPT_TRACKING_MODE`, `CLYPT_TRACKER_BACKEND` (ByteTrack-only in `backend/do_phase1_worker.py`), `CLYPT_TRACK_CHUNK_WORKERS`, `YOLO_WEIGHTS_PATH`, `CLYPT_YOLO_IMGSZ`
- Speaker binding: `CLYPT_SPEAKER_BINDING_MODE`, `CLYPT_SPEAKER_BINDING_HEURISTIC_FALLBACK`, `CLYPT_SPEAKER_BINDING_PROXY_ENABLE`, `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE`, `CLYPT_LRASD_*`
- Face pipeline: `CLYPT_FACE_*`, `CLYPT_FULLFRAME_FACE_MIN_SIZE`
- Diarization: `CLYPT_AUDIO_DIARIZATION_*`, `HF_TOKEN`

## Related Docs

- `README.md`
- `docs/deployment/do-phase1-digitalocean.md`
- `docs/do_phase1_worker.md`