# Environment Variables

This repo uses two primary env templates:

- Root/local development: `.env.example`
- DigitalOcean Phase 1 service: `backend/do_phase1_service/.env.example`

Use the template that matches your runtime. Do not commit real secrets.

## Team-Critical Vars

| Variable | Required for | Notes |
| --- | --- | --- |
| `SENSO_API_KEY` | Creator onboarding | Required for real Senso-backed profiles |
| `SENSO_CREATOR_PROFILE_PROMPT_ID` | Creator onboarding | Set locally/outside git |
| `YOUTUBE_API_KEY` | YouTube metadata/trends/comments | Optional for some local flows, required for YouTube API routes |
| `CLYPT_FRONTEND_ORIGINS` | Frontend → backend local API calls | Comma-separated allowed origins |
| `DO_PHASE1_BASE_URL` | Remote Phase 1 extraction | Set when using DO extraction service |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS / Vertex / Google SDKs | Usually host-level path |
| `GCS_BUCKET` | Artifact persistence | Defaults to `clypt-storage-v2` in service code |

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
GOOGLE_CLOUD_PROJECT=clypt-v2
GCS_BUCKET=clypt-storage-v2
```

## Phase 1 Service Env (DigitalOcean)

Prefer editing:

- `backend/do_phase1_service/.env.example` (template)
- `/etc/clypt-phase1/do-phase1.env` (host runtime file)

Notable service groups:

- Service/runtime: `DO_PHASE1_*`, `UVICORN_*`, `DO_REGION`
- Speaker binding: `CLYPT_SPEAKER_BINDING_*`, `CLYPT_LRASD_*`
- Face pipeline: `CLYPT_FACE_*`, `CLYPT_FULLFRAME_FACE_MIN_SIZE`, `CLYPT_YOLO_IMGSZ`
- Diarization: `CLYPT_AUDIO_DIARIZATION_*`, `HF_TOKEN`

## Related Docs

- `README.md`
- `docs/deployment/do-phase1-digitalocean.md`
- `docs/superpowers/specs/clypt_v3_refactor_spec.md`