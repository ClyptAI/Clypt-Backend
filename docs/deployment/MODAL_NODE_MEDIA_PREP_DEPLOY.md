# Modal Node-Media-Prep Deploy

Deploy the Modal `node-media-prep` service.

This service is the active remote backend for:

- `POST /tasks/node-media-prep`
- `GET /tasks/node-media-prep/result/{call_id}`

Target shape:

- CPU ASGI submit/poll surface
- one warm `L40S` worker via `node_media_prep_job min_containers=1`
- ffmpeg must expose both `h264_nvenc` and `h264_cuvid` inside the worker runtime

## 1) App Source

- [scripts/modal/node_media_prep_app.py](/Users/rithvik/Clypt-Backend/scripts/modal/node_media_prep_app.py)

Future sibling:

- [scripts/modal/render_video_app.py](/Users/rithvik/Clypt-Backend/scripts/modal/render_video_app.py)

## 2) Deploy

```bash
cd /opt/clypt-phase26/repo
modal deploy scripts/modal/node_media_prep_app.py
```

## 3) Required Secrets / Env

At minimum, provide:

- `GCS_BUCKET`
- `NODE_MEDIA_PREP_AUTH_TOKEN`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`

Current deployed Modal wiring:

- App name: `clypt-node-media-prep`
- App id from `modal app list`: `ap-FV1hNRaPzXUIV72NIsFNSk`
- Secret name: `clypt-node-media-prep`
- Deployed endpoint:
  `https://testifytestprep--clypt-node-media-prep-node-media-prep.modal.run/tasks/node-media-prep`
- Working bucket value: `GCS_BUCKET=clypt-storage-v3`

The deployed ASGI app exposes:

- `GET /health`
- `POST /tasks/node-media-prep` -> returns `202 Accepted` with `call_id`
- `GET /tasks/node-media-prep/result/{call_id}` -> returns `202` while pending, `200` with final `media` payload on completion

`GOOGLE_APPLICATION_CREDENTIALS_JSON` should be a real service-account JSON key
blob. Avoid token-only `authorized_user` ADC documents for production deploys;
they are less predictable in headless/serverless environments and do not match
the host deploy credential standard.

## 4) Smoke Checks

The web app starts on CPU and the spawned worker verifies GPU codec availability before processing. In addition, validate:

```bash
modal app list
modal logs clypt-node-media-prep
```

And confirm the downstream host can reach the deployed endpoint with its configured bearer token.

Expected contract:

```bash
curl -X POST "$CLYPT_PHASE24_NODE_MEDIA_PREP_URL" \
  -H "Authorization: Bearer $CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"run_id":"smoke","video_gcs_uri":"gs://bucket/video.mp4","nodes":[]}'
```

This should return a `call_id` immediately. Poll:

```bash
curl -H "Authorization: Bearer $CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN" \
  "https://.../tasks/node-media-prep/result/<call_id>"
```

Important:

- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` may be set to either the Modal app base URL or the full `POST /tasks/node-media-prep` endpoint URL.
- The current deploy/env records use the full endpoint URL. Do not append `/tasks/node-media-prep` a second time when validating or wiring callers.

## 5) Notes

- Each submitted job now represents one timeline-local batch. The worker downloads the source video once into worker scratch space, extracts a shared local batch window, emits exact per-node clips from that window, uploads them to GCS, and returns the resulting `file_uri` descriptors plus optional batch timing metadata.
- `RemoteNodeMediaPrepClient` now implements submit-and-poll so Phase26 can pipeline batch completion into immediate multimodal embedding while still producing one final ordered result per node.
- Only `node_media_prep_job` keeps a warm GPU. The public `node_media_prep` route should not reserve an `L40S`.
- This is a warm serverless surface, not a permanently pinned VM.
- Future Phase 6 render/export should follow the same submit-and-poll pattern if render duration can cross Modal's webhook timeout boundary, but do not wire that endpoint into runtime until the Phase 6 contract is finalized.
