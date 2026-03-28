# Frontend / Backend API Contract

This document defines the API contract between:

- Frontend repo: `rithm84/clypt-cortexfinal`
- Backend repo: `Clypt-V2`

The frontend is currently mock-driven. The goal of this contract is to give the frontend stable request/response shapes while the backend implementation catches up.

## Principles

- The frontend and backend remain in separate repos.
- The frontend should only depend on HTTP contracts, never local backend files.
- Backend internals such as Senso ingestion, YouTube APIs, Gemini, Spanner, and GCS remain implementation details behind these endpoints.
- Long-running work uses job-style APIs with polling.

## Base URL

Frontend should read a configurable backend base URL, for example:

```ts
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
```

Examples:

- local: `http://localhost:8000`
- staging: `https://api-staging.clypt.ai`

## Auth

For the first integration pass, auth can be omitted or mocked.

Recommended future shape:

- `Authorization: Bearer <token>`
- `X-Clypt-Creator-Id: <creator_id>` optional for internal testing

## Route Map

The frontend currently expects these product areas:

1. Onboarding
2. Run overview
3. Graph view
4. Clip review
5. Artifact explorer

Those map to the following backend endpoints.

---

## 1. Onboarding

### `POST /api/v1/onboarding/channel/resolve`

Resolve a YouTube channel from a pasted URL, handle, or freeform query.

Request:

```json
{
  "query": "https://youtube.com/@t3dotgg"
}
```

Response:

```json
{
  "channel": {
    "channel_id": "UC123",
    "channel_name": "Theo - t3.gg",
    "channel_url": "https://youtube.com/@t3dotgg",
    "handle": "@t3dotgg",
    "avatar_url": "https://...",
    "banner_url": "https://...",
    "description": "Full-stack development, opinions, and TypeScript.",
    "category": "Science & Technology",
    "subscriber_count": 420000,
    "subscriber_count_label": "420K",
    "total_views": 89000000,
    "total_views_label": "89M",
    "upload_frequency_label": "~5 videos/week",
    "joined_date_label": "2020"
  },
  "recent_shorts": [
    {
      "video_id": "short_1",
      "title": "React Server Components in 60 seconds",
      "views": 2100000,
      "views_label": "2.1M",
      "duration_seconds": 58,
      "duration_label": "58s",
      "likes": 89000,
      "likes_label": "89K",
      "thumbnail_url": "https://..."
    }
  ],
  "recent_videos": [
    {
      "video_id": "video_1",
      "title": "The T3 Stack in 2024: Complete Guide",
      "views": 1200000,
      "views_label": "1.2M",
      "duration_seconds": 1934,
      "duration_label": "32:14",
      "likes": 48000,
      "likes_label": "48K",
      "thumbnail_url": "https://..."
    }
  ]
}
```

Frontend mapping:

- `mockChannelResult`
- `mockTopShorts`
- `mockTopVideos`

### `POST /api/v1/onboarding/channel/analyze`

Start creator-profile analysis for a resolved channel.

Request:

```json
{
  "channel_id": "UC123"
}
```

Response:

```json
{
  "job_id": "creator_job_001",
  "status": "queued"
}
```

### `GET /api/v1/onboarding/channel/analyze/{job_id}`

Poll creator-profile analysis progress.

Response while running:

```json
{
  "job_id": "creator_job_001",
  "status": "running",
  "progress_pct": 56,
  "current_stage": "detecting_brand_signals",
  "stage_label": "Detecting brand signals",
  "stage_detail": "Extracting tone, hooks, recurring themes...",
  "channel": {
    "channel_id": "UC123",
    "channel_name": "Theo - t3.gg"
  },
  "recent_items_scanned": [
    {
      "video_id": "short_1",
      "title": "React Server Components in 60 seconds",
      "views_label": "2.1M"
    }
  ]
}
```

Response when complete:

```json
{
  "job_id": "creator_job_001",
  "status": "succeeded",
  "progress_pct": 100,
  "profile": {
    "creator_id": "creator_001",
    "creator_archetype": "Educator-Entertainer",
    "archetype_description": "Blends deep technical expertise with conversational energy and humor.",
    "dominant_mechanisms": {
      "humor": { "intensity": 0.7, "style": "Ironic commentary, self-deprecating asides" },
      "emotion": { "intensity": 0.3, "style": "Occasional genuine passion" },
      "social": { "intensity": 0.6, "style": "Hot takes that drive debate" },
      "expertise": { "intensity": 0.9, "style": "Counterintuitive truths, elegant simplification" }
    },
    "audience_signature": "Developers who want to stay current with the JS ecosystem.",
    "brand_voice": [
      "Opinionated",
      "Fast-paced",
      "Technically deep",
      "Meme-literate",
      "Conversational"
    ],
    "recurring_themes": [
      "TypeScript",
      "React ecosystem",
      "Developer tooling"
    ],
    "hook_style": "Provocative statement or contrarian claim in the first 3 seconds",
    "payoff_style": "Technical revelation backed by evidence"
  }
}
```

Frontend mapping:

- `mockBrandProfile`
- onboarding analyzing stage UI

### `PUT /api/v1/creators/{creator_id}/preferences`

Save creator clip preferences selected in onboarding.

Request:

```json
{
  "preferred_duration_range": { "min_seconds": 30, "max_seconds": 90 },
  "target_platforms": ["YouTube Shorts", "TikTok", "Twitter/X"],
  "tone_preferences": ["Educational", "Entertaining", "Opinionated"],
  "avoid_topics": [],
  "caption_style": "Bold, white with black outline",
  "hook_importance": 0.9,
  "payoff_importance": 0.8,
  "default_retrieve_queries": [
    "Find the strongest contrarian hot take",
    "Find the clearest beginner explanation"
  ]
}
```

Response:

```json
{
  "creator_id": "creator_001",
  "saved": true,
  "preferences": {
    "preferred_duration_range": { "min_seconds": 30, "max_seconds": 90 },
    "target_platforms": ["YouTube Shorts", "TikTok", "Twitter/X"],
    "tone_preferences": ["Educational", "Entertaining", "Opinionated"],
    "avoid_topics": [],
    "caption_style": "Bold, white with black outline",
    "hook_importance": 0.9,
    "payoff_importance": 0.8,
    "default_retrieve_queries": [
      "Find the strongest contrarian hot take",
      "Find the clearest beginner explanation"
    ]
  }
}
```

Frontend mapping:

- `mockClipPreferences`

### `GET /api/v1/creators/{creator_id}/profile`

Return the persisted creator profile after onboarding analysis completes.

### `GET /api/v1/creators/{creator_id}/preferences`

Return persisted creator defaults for the onboarding review/edit screens.

---

## 2. Pipeline Runs

### `POST /api/v1/runs`

Create a new run for a source video.

Request:

```json
{
  "source_url": "https://www.youtube.com/watch?v=xR4FC5jEMtQ",
  "creator_id": "creator_001",
  "mode": "auto_curate"
}
```

Response:

```json
{
  "run_id": "run_001",
  "status": "queued"
}
```

Notes:

- This is the frontend-facing wrapper around backend job orchestration.
- Internally, the backend may call the existing Phase 1 `POST /jobs` service.

### `GET /api/v1/runs`

List recent runs for the landing page and dashboard.

Response:

```json
{
  "runs": [
    {
      "run_id": "run_001",
      "status": "completed",
      "started_at": "2024-12-10T14:32:00Z",
      "completed_at": "2024-12-10T14:38:22Z",
      "video": {
        "video_id": "xR4FC5jEMtQ",
        "url": "https://www.youtube.com/watch?v=xR4FC5jEMtQ",
        "title": "The Problem With Being Too Honest",
        "channel": "Colin and Samir",
        "duration": "18:42",
        "duration_seconds": 1122,
        "thumbnail": "https://img.youtube.com/vi/xR4FC5jEMtQ/maxresdefault.jpg",
        "published_at": "2024-11-15"
      },
      "metrics": {
        "nodes": 15,
        "edges": 22,
        "clips": 6
      }
    }
  ]
}
```

Frontend mapping:

- `sampleRuns`

### `GET /api/v1/runs/{run_id}`

Get run overview metadata plus pipeline phases and activity log.

Response:

```json
{
  "run_id": "run_001",
  "status": "completed",
  "video": {
    "video_id": "xR4FC5jEMtQ",
    "url": "https://www.youtube.com/watch?v=xR4FC5jEMtQ",
    "title": "The Problem With Being Too Honest",
    "channel": "Colin and Samir",
    "duration": "18:42",
    "duration_seconds": 1122,
    "thumbnail": "https://img.youtube.com/vi/xR4FC5jEMtQ/maxresdefault.jpg",
    "published_at": "2024-11-15",
    "description": "..."
  },
  "runtime": {
    "gpu": "H100",
    "llm_model": "Gemini 1.5 Pro"
  },
  "phases": [
    {
      "id": "phase-1",
      "name": "Deterministic Grounding",
      "short_name": "Ground",
      "description": "GPU-accelerated video extraction...",
      "status": "completed",
      "artifacts": ["phase_1_visual.json", "phase_1_audio.json"],
      "metrics": {
        "tracks": 4,
        "personDetections": 287
      },
      "started_at": "2024-12-10T14:32:00Z",
      "completed_at": "2024-12-10T14:34:12Z",
      "duration_ms": 132000
    }
  ],
  "activity_log": [
    {
      "id": "log-01",
      "timestamp": "14:32:00",
      "phase": "phase-1",
      "message": "Starting deterministic grounding pipeline",
      "level": "info"
    }
  ]
}
```

Frontend mapping:

- `mockVideo`
- `mockPipeline`
- `mockActivityLog`

### `GET /api/v1/runs/{run_id}/status`

Lightweight polling endpoint for in-progress runs.

Response:

```json
{
  "run_id": "run_001",
  "status": "running",
  "progress_pct": 42,
  "current_phase": "phase-2a",
  "current_step": "semantic_nodes",
  "message": "Extracting semantic nodes"
}
```

---

## 3. Graph View

### `GET /api/v1/runs/{run_id}/graph`

Return graph nodes and edges for the Cortex graph page.

Response:

```json
{
  "nodes": [
    {
      "id": "n10",
      "type": "hook",
      "label": "The Authenticity Paradox",
      "speaker": "Colin",
      "start_time": 370.0,
      "end_time": 400.0,
      "score": 0.95,
      "clip_worthy": true,
      "transcript": "The most dishonest thing you can do...",
      "mechanisms": {
        "humor": 0.2,
        "emotion": 0.4,
        "social": 0.3,
        "expertise": 0.8
      },
      "position": { "x": 100, "y": 200 }
    }
  ],
  "edges": [
    {
      "id": "e10",
      "source": "n9",
      "target": "n10",
      "relation": "setup_payoff",
      "strength": 0.91,
      "label": "setup → payoff"
    }
  ],
  "summary": {
    "node_count": 15,
    "edge_count": 20
  }
}
```

Notes:

- `position` may be precomputed server-side or generated client-side.
- The frontend’s current graph data model is driven by `mockNodes` and `mockEdges`.

---

## 4. Clip Review

### `GET /api/v1/runs/{run_id}/clips`

Return ranked clip candidates for the clip review page.

Response:

```json
{
  "clips": [
    {
      "id": "clip-1",
      "rank": 1,
      "title": "The Authenticity Paradox",
      "start_time": 370,
      "end_time": 400,
      "duration": 30,
      "score": 0.95,
      "transcript": "The most dishonest thing you can do...",
      "justification": "Peak narrative density...",
      "framing_type": "single_person",
      "speaker": "Colin",
      "scores": {
        "hook": 0.93,
        "payoff": 0.97,
        "pacing": 0.91,
        "narrative_arc": 0.94,
        "clip_worthiness": 0.95
      },
      "node_ids": ["n10"],
      "pinned": false,
      "best_cut": true,
      "render_payload_uri": "gs://..."
    }
  ]
}
```

Frontend mapping:

- `mockClips`

### `POST /api/v1/runs/{run_id}/clips/retrieve`

Run retrieve mode for a user-specified type of clip.

Request:

```json
{
  "query": "Find the strongest contrarian hot take",
  "creator_id": "creator_001",
  "preferences_override": {
    "preferred_duration_range": { "min_seconds": 20, "max_seconds": 45 }
  }
}
```

Response:

```json
{
  "query": "Find the strongest contrarian hot take",
  "anchor_node_id": "n10",
  "clip": {
    "id": "clip_retrieve_001",
    "title": "The Authenticity Paradox",
    "start_time": 370,
    "end_time": 400,
    "duration": 30,
    "score": 0.95,
    "transcript": "The most dishonest thing you can do...",
    "justification": "Best match for contrarian hot take query",
    "framing_type": "single_person",
    "speaker": "Colin",
    "scores": {
      "hook": 0.93,
      "payoff": 0.97,
      "pacing": 0.91,
      "narrative_arc": 0.94,
      "clip_worthiness": 0.95
    },
    "node_ids": ["n10"]
  }
}
```

Notes:

- This is the frontend-facing wrapper around the existing backend retrieve logic in `backend/pipeline/phase_5_retrieve.py`.
- The backend composes a `final_query` from creator profile, saved preferences, and the current freeform ask before running retrieval.
- The response includes both the original `query` and the composed `final_query` for debugging and demo transparency.

### `PUT /api/v1/runs/{run_id}/clips/{clip_id}`

Update clip review flags such as pin or best cut.

Request:

```json
{
  "pinned": true,
  "best_cut": false
}
```

Response:

```json
{
  "id": "clip-1",
  "pinned": true,
  "best_cut": false
}
```

---

## 5. Artifact Explorer

### `GET /api/v1/runs/{run_id}/artifacts`

Return structured previews of pipeline artifacts.

Response:

```json
{
  "artifacts": [
    {
      "id": "art-1",
      "name": "phase_1_visual.json",
      "phase": "Phase 1 - Deterministic Grounding",
      "description": "Visual extraction results...",
      "size": "2.4 MB",
      "item_count": 4,
      "preview": {
        "tracks": [
          {
            "trackId": "T001",
            "label": "Person A (Colin)"
          }
        ]
      }
    }
  ]
}
```

Frontend mapping:

- `mockArtifacts`

---

## Existing Backend Surfaces

The backend already has a real Phase 1 job service in:

- `backend/do_phase1_service/app.py`

Existing endpoints:

- `GET /healthz`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/logs`
- `GET /jobs/{job_id}/result`

This means the fastest path is:

1. Keep those internal endpoints for backend orchestration.
2. Add frontend-facing `/api/v1/runs` routes in a separate API layer.
3. Let `/api/v1/runs` translate frontend needs into the internal Phase 1 / pipeline jobs.

## Recommended Implementation Order

### Phase A: Minimal real integration

Implement first:

1. `POST /api/v1/runs`
2. `GET /api/v1/runs/{run_id}/status`
3. `GET /api/v1/runs/{run_id}`
4. `GET /api/v1/runs/{run_id}/graph`
5. `GET /api/v1/runs/{run_id}/clips`
6. `GET /api/v1/runs/{run_id}/artifacts`

Use local files in `backend/outputs` and existing pipeline artifacts where possible.

### Phase B: Onboarding

Implement next:

1. `POST /api/v1/onboarding/channel/resolve`
2. `POST /api/v1/onboarding/channel/analyze`
3. `GET /api/v1/onboarding/channel/analyze/{job_id}`
4. `PUT /api/v1/creators/{creator_id}/preferences`

This is where YouTube channel analysis and Senso fit naturally.

### Phase C: Retrieve mode

Implement:

1. `POST /api/v1/runs/{run_id}/clips/retrieve`
2. `PUT /api/v1/runs/{run_id}/clips/{clip_id}`

This enables the clip-intent query flow.

## Open Questions

These need product/backend decisions before final implementation:

1. Should a `run_id` wrap the entire pipeline, or should it mirror the existing Phase 1 `job_id`?
2. Where should creator profiles and preferences be persisted?
3. Should channel analysis be synchronous for hackathon demo purposes, or always job/polling based?
4. Should graph node layout be produced server-side or client-side?
5. Should clip pin/best-cut state be persisted per user, per creator, or only in frontend session state?
