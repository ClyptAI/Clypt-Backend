# Senso Backend Integration

This repo now includes a small backend integration foundation for Senso:

- `backend/integrations/senso_client.py`
- `backend/services/creator_knowledge.py`
- `backend/services/creator_onboarding.py`
- `backend/services/youtube_channel_service.py`
- `backend/api/app.py`

## What It Covers

- Senso API key auth via `X-API-Key`
- Category lookup and creation
- Topic lookup and creation
- Raw text content ingestion
- Search
- Prompt-based generation
- Creator workspace orchestration for onboarding
- Frontend-facing onboarding API endpoints with polling-friendly job storage

## Required Env Vars

```bash
SENSO_API_KEY=...
SENSO_API_BASE_URL=https://sdk.senso.ai/api/v1
SENSO_TIMEOUT_SECONDS=30
```

## Current Assumptions

The client is implemented against Senso’s indexed public docs and examples for:

- `GET /categories`
- `POST /categories/batch-create`
- `GET /categories/{category_id}/topics`
- `POST /topics`
- `POST /content/raw`
- `GET /content/{content_id}`
- `POST /search`
- `POST /generate/prompt`

The category/topic and prompt-based generate endpoints are directly supported by the official indexed docs.
The topic creation and topic listing paths are inferred from the public examples and endpoint naming patterns and should be verified once your team has live credentials.

## Intended Flow

1. Resolve a creator channel from YouTube
2. `ensure_creator_workspace(...)`
3. Ingest transcripts and metadata with `ingest_video_documents(...)`
4. Build the creator profile with `build_creator_profile(...)`
5. Use `search_creator_context(...)` at clip time to retrieve creator-specific context

## Next Backend Step

## Onboarding API Env Vars

```bash
YOUTUBE_API_KEY=...
SENSO_API_KEY=...
SENSO_CREATOR_PROFILE_PROMPT_ID=...
SENSO_CREATOR_PROFILE_TEMPLATE_ID=...   # optional
SENSO_CREATOR_PROFILE_MAX_VIDEOS=6      # optional
SENSO_CREATOR_PROFILE_MAX_RESULTS=8     # optional
CLYPT_ONBOARDING_STATE_ROOT=backend/outputs/onboarding_jobs
```

## Current API Routes

- `GET /healthz`
- `POST /api/v1/onboarding/channel/resolve`
- `POST /api/v1/onboarding/channel/analyze`
- `GET /api/v1/onboarding/channel/analyze/{job_id}`
- `GET /api/v1/creators/{creator_id}/profile`
- `GET /api/v1/creators/{creator_id}/preferences`
- `PUT /api/v1/creators/{creator_id}/preferences`
- `POST /api/v1/runs/{run_id}/clips/retrieve`

## Notes

- Creator profiles are persisted locally under `backend/outputs/creators`.
- Creator preferences are persisted locally under `backend/outputs/creators`.
- Retrieve mode currently composes a final query from creator profile, saved preferences, and the current request, then uses local clip candidates as a reliable fallback backend for frontend integration.
- The retrieve bridge is intentionally swappable so the live Spanner/Gemini Phase 5 path can replace the fallback without changing the frontend contract.

Run locally with:

```bash
uvicorn backend.api.app:app --reload --port 8000
```
