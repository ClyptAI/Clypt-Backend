# Environment Variables

This repo uses a mix of required credentials, local-path overrides, and advanced runtime tuning flags.

Start with [.env.example](/c:/Users/chess/Desktop/Clypt/Clypt-V2/.env.example). For most teammates, only a small subset is needed.

## Team-Critical Vars

These are the ones most likely to block local development:

| Variable | Required for | Notes |
|---|---|---|
| `SENSO_API_KEY` | Creator onboarding analyze step | Required for real Senso-backed creator profiles. |
| `SENSO_CREATOR_PROFILE_PROMPT_ID` | Creator onboarding analyze step | Current prompt id lives outside git; set locally. |
| `YOUTUBE_API_KEY` | Rich YouTube metadata/comments/trends | Optional for handle/url channel resolve, but required for YouTube search/comments/statistics paths. |
| `CLYPT_FRONTEND_ORIGINS` | Separate frontend calling backend API | Defaults to localhost ports used by the frontend. |
| `DO_PHASE1_BASE_URL` | Remote Phase 1 extraction | Needed when using the active DigitalOcean extraction path. |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS / Vertex / other Google SDK usage | Usually set on the machine or deployment host, not in git. |
| `GCS_BUCKET` | Phase 1 service storage | Defaults to `clypt-storage-v2` in service code. |

## Recommended Local Backend Setup

For the current onboarding flow, a teammate can usually get by with:

```env
SENSO_API_KEY=...
SENSO_CREATOR_PROFILE_PROMPT_ID=...
YOUTUBE_API_KEY=...
CLYPT_FRONTEND_ORIGINS=http://localhost:8080,http://localhost:3000,http://localhost:5173
```

If they are also running the full extraction pipeline through the remote service:

```env
DO_PHASE1_BASE_URL=http://<droplet-ip>:8080
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-sa.json
GOOGLE_CLOUD_PROJECT=clypt-v2
GCS_BUCKET=clypt-storage-v2
```

## By Subsystem

### Backend API / onboarding

Defined in [app.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/api/app.py) and onboarding services.

- `CLYPT_FRONTEND_ORIGINS`
- `CLYPT_ONBOARDING_STATE_ROOT`
- `CLYPT_CREATOR_STORE_ROOT`
- `CLYPT_RETRIEVE_CANDIDATES_PATH`
- `SENSO_API_KEY`
- `SENSO_CREATOR_PROFILE_PROMPT_ID`
- `SENSO_CREATOR_PROFILE_TEMPLATE_ID`
- `SENSO_CREATOR_PROFILE_MAX_VIDEOS`
- `SENSO_CREATOR_PROFILE_MAX_RESULTS`
- `SENSO_API_BASE_URL`
- `SENSO_TIMEOUT_SECONDS`
- `YOUTUBE_API_KEY`

### Full pipeline / orchestrator

- `START_FROM`
- `FFMPEG_REENCODE_CRF`
- `FFMPEG_REENCODE_PRESET`
- `REMOTION_CRF`
- `REMOTION_X264_PRESET`

### Shared GCS / video inputs

- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `VIDEO_GCS_URI`
- `GCS_BUCKET`

### Phase 1 local client / bridge

Defined mainly in [phase_1_do_pipeline.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/pipeline/phase_1_do_pipeline.py).

- `DO_PHASE1_BASE_URL`
- `PHASE1_SERVICE_URL`
- `DIGITALOCEAN_PHASE1_BASE_URL`
- `DO_PHASE1_POLL_INTERVAL_SECONDS`
- `DO_PHASE1_TIMEOUT_SECONDS`
- `YTDLP_VIDEO_FORMAT`
- `YTDLP_H264_PREFERRED_FORMAT`
- `YTDLP_MIN_LONG_EDGE`
- `ALLOW_LOW_RES_VIDEO`
- `PHASE1_RUNTIME_PROFILE`
- `PHASE1_FORCE_LRASD`
- `PHASE1_SPEAKER_BINDING_MODE`
- `PHASE1_TRACKING_MODE`
- `PHASE1_TRACKER_BACKEND`
- `PHASE1_SHARED_ANALYSIS_PROXY`
- `PHASE1_HEURISTIC_BINDING_ENABLED`
- `CLYPT_RUN_TALKNET_REFINEMENT`

### DigitalOcean Phase 1 service

Defined in:
- [app.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/do_phase1_service/app.py)
- [worker.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/do_phase1_service/worker.py)
- [extract.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/do_phase1_service/extract.py)

- `DO_PHASE1_STATE_ROOT`
- `DO_PHASE1_DB_PATH`
- `DO_PHASE1_OUTPUT_ROOT`
- `DO_PHASE1_LOG_ROOT`
- `DO_PHASE1_LOG_LEVEL`
- `DO_PHASE1_WORKER_CONCURRENCY`
- `DO_PHASE1_WORKER_ID`
- `DO_PHASE1_GPU_SLOTS`
- `DO_PHASE1_GPU_SLOT_POLL_INTERVAL_S`
- `DO_PHASE1_RUNNING_STALE_AFTER_SECONDS`
- `DO_PHASE1_HEARTBEAT_INTERVAL_SECONDS`
- `DO_PHASE1_LOOP_ERROR_BACKOFF_SECONDS`
- `DO_PHASE1_DASHBOARD_REMOTE_BASE_URL`
- `DO_PHASE1_HOST_LOCK_PATH`
- `DO_REGION`

### Advanced Phase 1 worker tuning

Defined in [do_phase1_worker.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/do_phase1_worker.py).

Core runtime selection:
- `CLYPT_ENABLE_LEGACY_SERVERLESS_SDK`
- `CLYPT_PHASE1_EVAL_PROFILE`
- `CLYPT_SPEAKER_BINDING_MODE`
- `CLYPT_TRACKING_MODE`
- `CLYPT_TRACKER_BACKEND`
- `CLYPT_TRACK_CHUNK_WORKERS`
- `CLYPT_SHARED_ANALYSIS_PROXY`
- `CLYPT_ANALYSIS_PROXY_ENABLE`
- `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE`
- `CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS`

LR-ASD / diarization:
- `CLYPT_AUDIO_DIARIZATION_ENABLE`
- `CLYPT_AUDIO_DIARIZATION_MODEL`
- `CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS`
- `CLYPT_LRASD_BATCH_SIZE`
- `CLYPT_LRASD_PIPELINE_OVERLAP`
- `CLYPT_LRASD_MAX_INFLIGHT`
- `CLYPT_SPEAKER_BINDING_PROXY_ENABLE`
- `CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_DURATION_S`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_LONG_EDGE`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_WORDS`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_TRACKS`
- `CLYPT_SPEAKER_FOLLOW_MIN_SEGMENT_MS`


Face / clustering / tracking heuristics:
- `CLYPT_CLUSTER_DISABLE_COVISIBILITY`
- `CLYPT_CLUSTER_ATTACH_MAX_GAP_FRAMES`
- `CLYPT_CLUSTER_ATTACH_GAP_WEIGHT`
- `CLYPT_CLUSTER_ATTACH_AMBIGUITY_MARGIN`
- `CLYPT_CLUSTER_GROUP_ATTACH_SIG_RELAX`
- `CLYPT_CLUSTER_GROUP_ATTACH_MIN_SUPPORT_SHARE`
- `CLYPT_CLUSTER_GROUP_ATTACH_MIN_SUPPORT_COUNT`
- `CLYPT_CLUSTER_HIST_LONG_GAP_FRAMES`
- `CLYPT_CLUSTER_HIST_LONG_GAP_MAX_SIG`
- `CLYPT_FACE_TRACK_MIN_ASSOC_COUNT`
- `CLYPT_FACE_TRACK_MIN_ASSOC_SHARE`
- `CLYPT_FACE_TRACK_MIN_DOMINANT_RATIO`
- `CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC`
- `CLYPT_FACE_TRACK_SECONDARY_MIN_RATIO`
- `CLYPT_FACE_TRACK_SECONDARY_MAX_SIG`
- `CLYPT_FACE_TRACK_MAX_GAP`
- `CLYPT_FACE_TRACK_MATCH_COST`
- `CLYPT_FACE_TRACK_STITCH_MAX_GAP_FRAMES`
- `CLYPT_FACE_TRACK_STITCH_MAX_COS`
- `CLYPT_FACE_TRACK_STITCH_MAX_SIG`
- `CLYPT_FACE_TRACK_STITCH_SAME_TRACK_MAX_GAP_FRAMES`
- `CLYPT_FACE_TRACK_SEED_MIN_SHARE`
- `CLYPT_FACE_TRACK_SEED_MIN_MARGIN`
- `CLYPT_FACE_TRACK_PROPAGATE_MAX_GAP_FRAMES`
- `CLYPT_FACE_TRACK_PROPAGATE_MAX_SIG_DIST`
- `CLYPT_FACE_TRACK_PROPAGATE_AMBIGUITY_MARGIN`
- `CLYPT_FACE_ASSOC_MIN_SCORE`
- `CLYPT_FACE_DETECTOR_INPUT_SIZE`
- `CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE`
- `CLYPT_FACE_PIPELINE_WORKERS`
- `CLYPT_FACE_LEDGER_WORKERS`
- `CLYPT_FACE_PIPELINE_GPU_WORKERS`
- `CLYPT_FACE_PIPELINE_START_FRAME`
- `CLYPT_FACE_PIPELINE_SEGMENT_FRAMES`
- `CLYPT_FACE_LEDGER_SEGMENT_FRAMES`
- `CLYPT_FULLFRAME_FACE_MIN_SIZE`
- `CLYPT_YOLO_IMGSZ`

Debug / rollout gates:
- `CLYPT_MODEL_DEBUG`
- `CLYPT_MODEL_DEBUG_EVERY`
- `CLYPT_GATE_MIN_IDF1_PROXY`
- `CLYPT_GATE_MIN_MOTA_PROXY`
- `CLYPT_GATE_MAX_FRAGMENTATION`
- `CLYPT_GATE_MIN_THROUGHPUT_FPS`
- `CLYPT_GATE_MAX_WALLCLOCK_S`
- `CLYPT_GATE_MIN_SCHEMA_PASS_RATE`
- `CLYPT_ENFORCE_ROLLOUT_GATES`

### Phase 2 / graph generation

- `PHASE_2A_CHUNK_REQUEST_DELAY_S`
- `PHASE_2A_MAX_RETRIES_429`
- `PHASE_2A_RETRY_BASE_S`
- `PHASE_2A_RETRY_MAX_S`
- `CLYPT_MIN_NODE_DURATION_S`
- `VIDEO_GCS_URI`

### Crowd Clip / audience signals

Defined across the files in [backend/pipeline/audience](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/pipeline/audience).

- `YOUTUBE_API_KEY`
- `CROWD_CLIP_RELEVANCE_PAGES`
- `CROWD_CLIP_TIME_PAGES`
- `CROWD_AUDIO_LEDGER_PATH`
- `CROWD_NODES_PATH`

### Trend Trim

Defined in [trend_1_ingest_external.py](/c:/Users/chess/Desktop/Clypt/Clypt-V2/backend/pipeline/trends/trend_1_ingest_external.py).

- `YOUTUBE_API_KEY`
- `TREND_TRIM_WATCHLIST`
- `TREND_TRIM_REGION`
- `TREND_TRIM_GOOGLE_TRENDS_LIMIT`
- `TREND_TRIM_YOUTUBE_POPULAR_LIMIT`
- `TREND_TRIM_WATCHLIST_SEARCH_LIMIT`

### Caption payload tools

- `CAPTION_MAX_WORDS`
- `CAPTION_MAX_CHARS`
- `CAPTION_GAP_MS`
- `CAPTION_PAYLOAD_INPUT_PATH`
- `CAPTION_AUDIO_LEDGER_PATH`
- `CAPTION_PAYLOAD_OUTPUT_PATH`
- `TOP_CAPTION_PAYLOAD_INPUT_PATH`
- `TOP_CAPTION_NODES_PATH`
- `TOP_CAPTION_PAYLOAD_OUTPUT_PATH`
- `TOP_CAPTION_STYLE_HINT`
- `TOP_CAPTION_STYLE_PATH`

### Test / debug render tooling

- `CLYPT_RENDER_NUM_CLIPS`
- `CLYPT_RENDER_CLIP_DURATION_S`
- `CLYPT_RENDER_DEBUG_MODE`
- `CLYPT_RENDER_DEBUG_SHOW_FACES`

## Team Workflow Recommendation

For teammates:

1. Copy [.env.example](/c:/Users/chess/Desktop/Clypt/Clypt-V2/.env.example) to a local untracked env file.
2. Fill in only the vars needed for the subsystem they are working on.
3. Keep secrets out of git and out of screenshots/chat logs.

For onboarding/frontend work, the minimum useful local env is the Senso block plus optional `YOUTUBE_API_KEY`.

## Related Docs

- [README.md](/c:/Users/chess/Desktop/Clypt/Clypt-V2/README.md)
- [senso-integration.md](/c:/Users/chess/Desktop/Clypt/Clypt-V2/docs/planning/senso-integration.md)
- [do-phase1-digitalocean.md](/c:/Users/chess/Desktop/Clypt/Clypt-V2/docs/deployment/do-phase1-digitalocean.md)