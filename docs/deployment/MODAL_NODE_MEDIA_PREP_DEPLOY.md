# Modal Node-Media-Prep Deploy

Deploy the Modal `node-media-prep` service.

This service is the active remote backend for:

- `POST /tasks/node-media-prep`

Target shape:

- GPU: `L4`
- `min_containers=1`
- ffmpeg must expose both `h264_nvenc` and `h264_cuvid`

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
- Google credentials for GCS access
- bearer token used by the public endpoint wrapper

## 4) Smoke Checks

The app already verifies codec availability at request time. In addition, validate:

```bash
modal app list
modal logs clypt-node-media-prep
```

And confirm the downstream host can reach the deployed endpoint with its configured bearer token.

## 5) Notes

- Each request downloads the source video once into worker scratch space, extracts all requested clips, uploads them to GCS, and returns only the resulting `file_uri` descriptors.
- The request/response contract is intentionally unchanged so `RemoteNodeMediaPrepClient` continues to work.
- This is a warm serverless surface, not a permanently pinned VM.
- Future Phase 6 render/export should follow the same pattern, but do not wire that endpoint into runtime until the Phase 6 contract is finalized.
