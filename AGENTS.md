# AGENTS

Operational startup and maintenance guide for coding agents.

## Project Overview

- Product: Clypt V3.1 backend
- Implemented: Phases 1-4
- Planned: Phases 5-6
- Active topology: **Colocated Phase1 vCPU orchestrator + Phase26 MI300X GPU + Modal L40S x2**
  - **Phase1 orchestrator**: Phase 1 runner, test-bank media ingress, signed HTTPS GCS URL generation for ElevenLabs Scribe v2, Modal RF-DETR submit, and Phase26 dispatch. It runs on the Phase26 MI300X droplet's vCPUs. It does not run local ASR, forced alignment, emotion2vec+, YAMNet, RF-DETR, VibeVoice, vLLM, SGLang, or any local GPU service.
  - **Phase26 host (MI300X)**: the same droplet also owns `POST /tasks/phase26-enqueue`, local SQLite queue + worker, SGLang ROCm Qwen on `:8001`, current Phase 2-4 runtime, future Phase 5-6 orchestration.
  - **Modal visual L40S**: CPU `POST /tasks/visual-extract` submit/poll surface plus one warm `L40S` `visual_extract_job` worker using TensorRT/NVDEC RF-DETR-Seg Nano.
  - **Modal media L40S**: CPU submit/poll surfaces for `POST /tasks/node-media-prep` and `POST /tasks/render-video`; both dispatch to one shared warm `L40S` `media_gpu_job` worker.
- Current Phase 2-4 local runtime: SQLite queue + local worker + local OpenAI-compatible generation endpoint on the Phase26 host.

## Read Order (Required - You MUST read these before reporting back to the user.)

1. [README.md](README.md)
2. [docs/runtime/RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
3. [docs/deployment/PHASE1_HOST_DEPLOY.md](docs/deployment/PHASE1_HOST_DEPLOY.md)
4. [docs/deployment/PHASE26_HOST_DEPLOY.md](docs/deployment/PHASE26_HOST_DEPLOY.md)
5. [docs/deployment/MODAL_NODE_MEDIA_PREP_DEPLOY.md](docs/deployment/MODAL_NODE_MEDIA_PREP_DEPLOY.md)
6. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
7. [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)
8. [docs/EVALS.md](docs/EVALS.md)
9. [docs/ERROR_LOG.md](docs/ERROR_LOG.md)

## Documentation-First Rule (Required)

When in doubt, read documentation before making changes or running operational commands.

- Start with the required read order above.
- Always use GitNexus for codebase retrieval/search before falling back to ad-hoc file scanning.
- If this is a brand new agent session, run `npx gitnexus analyze` as soon as you finish reading this doc.
- Always double-check `.gitignore` coverage before committing to avoid accidental local artifact commits.
- Treat docs as the source of truth for runtime/deploy behavior unless code has clearly diverged.
- If behavior is unclear, re-check `RUNTIME_GUIDE.md`, `PHASE1_HOST_DEPLOY.md`, and `PHASE26_HOST_DEPLOY.md` before proceeding.
- If you discover a docs/code mismatch, call it out and tell the user immediately.
- If you fix a deploy/runtime issue on a live host or remote worker, you must also fix the local source of truth in this repo during the same task (code, scripts, env records, docs, or tests as needed) so a fresh droplet does not hit the same failure again.

## Resolver-First Rule (Required)

Before starting any task, read [docs/EVALS.md](docs/EVALS.md) and load the relevant skill pointers for that task type. If all the skills listed there are not installed into project-local `.agents/skills/` and `.claude/skills/` then do so ASAP.

- Keep context lightweight: load only the skills needed for the current task.
- For prompt/scoring/behavior changes, run pipeline evals and compare against baseline before shipping.

## Canonical Run Commands

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

### Phase 1 runtime deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1-orchestrator.txt
```

### Phase26 runtime deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase26-mi300x.txt
```

### Pipeline tests (offline)

```bash
python -m pytest tests/backend/pipeline -q
```

### Phase 1 + downstream enqueue

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase26/videos/<video>.mp4 \
  --run-phase14
```

### Phase26 local worker

```bash
python -m backend.runtime.run_phase26_worker --worker-id phase26-worker-1
```

## Runtime Truths To Preserve

- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required on the colocated Phase1/Phase26 MI300X host.
- The `Phase26` host name is intentionally retained as semantic shorthand even though Phase1 now runs on the same droplet's vCPUs; do not infer a separate Phase1 host from that name.
- Phase 1 audio backend is ElevenLabs Scribe v2:
  - `CLYPT_PHASE1_AUDIO_BACKEND=elevenlabs_scribe_v2`
  - `ELEVENLABS_API_KEY` is required.
  - Scribe uses a signed HTTPS GCS audio URL with `source_url` by default.
  - `CLYPT_PHASE1_SCRIBE_LANGUAGE_CODE=en`
  - `num_speakers` and `keyterms` are omitted by default unless later supplied by frontend controls.
  - entity detection/redaction stay disabled.
- Phase 1 visual extraction is a Modal future:
  - Phase1 submits `POST /tasks/visual-extract` to the dedicated Modal visual L40S service.
  - Phase1 enqueues Phase26 immediately after Scribe audio adaptation completes.
  - Phase26 may run Phase2-4 while RF-DETR-Seg continues, but must join/fail-hard on the visual future before Phase5/frontend grounding or Phase6 visual use.
  - Modal visual poll results stay lightweight: the worker uploads the full `phase1_visual` payload as `phase1_visual.json.gz` to GCS and returns pointer fields (`phase1_visual_gcs_uri`, `phase1_visual_encoding=json_gzip_v1`) instead of inlining the full payload over the HTTP result surface.
  - The colocated host-side `RemoteVisualExtractClient` must hydrate that GCS artifact back into `phase1_visual` before Phase26 consumes the joined result.
  - Do not document or draw the MI300X host as a generic fan-out hub. The event flow is: canonical media upload -> parallel Scribe + Modal visual submit -> Scribe audio adaptation -> Phase26 enqueue -> Phase2-4 audio/text work while visual remains pending -> visual hard join before Phase5/visual use -> Modal media prep/render when those stages need it.
- Current Modal visual settings must stay intact unless the user explicitly approves retuning:
  - `CLYPT_PHASE1_VISUAL_MODEL=seg_nano`
  - Phase1 orchestrator route: `CLYPT_PHASE1_VISUAL_BACKEND=modal_rfdetr`
  - Modal worker detector route: `CLYPT_MODAL_VISUAL_BACKEND=tensorrt` and internal `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
  - batch size `16`
  - threshold `0.85`
  - shape `648`
  - RF-DETR-Seg masks are retained as one compressed low-resolution `.npz` sidecar artifact per visual job. JSON payload rows carry `mask_ref` pointers using `lowres_mask_ref_v1`; full-frame `mask_rle` rows are not emitted on the active path. ByteTrack remains box-only; mask refs are associated back to tracked rows after ID assignment and are not used for identity decisions.
  - TensorRT RF-DETR-Seg postprocess should stay close to RF-DETR upstream semantics: decode logits, threshold, filter to `person`, and retain surviving person queries. Do **not** add an extra hard box-IoU NMS stage on the active path unless the user explicitly approves it.
  - Segmentation was added to support future person-aware caption placement, motion-graphics/overlay placement, and short/reel crop decisions. Current Phase6 crop planning and caption placement do **not** consume masks yet; those integrations are future-sprint work.
  - sampled YOLO11s-pose TensorRT subject validation enabled for auto-follow eligibility
  - ByteTrack buffer `30`
  - ByteTrack match threshold `0.7`
  - GPU decode through `CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND=nvdec`
  - Live timing runs must start only after the Modal visual worker has completed a real person-containing warmup that builds/loads both the RF-DETR-Seg TensorRT engine and the YOLO11s-pose TensorRT engine. A blank/synthetic warmup is insufficient because it can skip pose validation and leave the pose engine build in the measured run.
- Phase5-less render auto-follow is implemented but currently experimental and **not production-quality**:
  - active mode is `tracklet_follow_9x16_pose_x_dynamic_inside_person`: the compiler emits pose-x anchored, bbox-top anchored, dynamic inside-person 9:16 keyframes with hard crop cuts at shot/tracklet boundaries
  - the active Modal FFmpeg renderer does **not** apply dynamic crop `w/h` inside a single ffmpeg pass; it renders per-run/per-tracklet fixed-size cropped video segments, stitches them into one clip, and applies subtitles in a final pass
  - latest reviewed clips still had poor tracking/subject selection and insufficiently smooth crop motion
  - do not treat generated auto-follow renders as accepted visual-quality baselines
  - manual Phase5 grounding remains the production-quality path until tracking/crop planning is repaired and reviewed again
- There is **no local fallback**:
  - Phase1 requires `CLYPT_PHASE1_VISUAL_SERVICE_*`
  - Phase1 requires `CLYPT_PHASE24_DISPATCH_*`
  - Phase1 requires `ELEVENLABS_API_KEY`
  - Phase26 requires `CLYPT_PHASE24_NODE_MEDIA_PREP_*`
- Legacy Phase1 VibeVoice/NFA/emotion2vec+/YAMNet envs are deleted on this branch and must not be reintroduced as active runtime requirements.
- Phase 1 runtime requires `CLYPT_PHASE1_INPUT_MODE=test_bank`.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; use explicit per-stage concurrency envs.
- Phase26 local runtime requires `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`.
- Phase26 worker enforces `GENAI_GENERATION_BACKEND=local_openai`.
- Default crash handling mode remains fail-fast:
  - `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`
  - `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`
- Comments/trends augmentation is hard-join + fail-fast before Phase 4.
- Phase26 retry/resume must not mark a run terminal solely because persisted metrics say Phase 4 succeeded. If a run resumes after a visual-join failure, it must reload persisted Phase2/Phase3 artifacts, rerun Phase4, and then attempt the visual join again.
- Qwen serving target is the SGLang ROCm service on Phase26 `127.0.0.1:8001`.
- Node-media prep and render/export are always delegated remotely to the shared Modal media L40S. Do not re-introduce an in-process ffmpeg fallback on the Phase26 MI300X host.
- Historical H200/H100 and Phase1 MI300X/VibeVoice overlays are deleted on AMD-refactor; do not reintroduce or reference `known-good-phase1-h100-backup.env` unless the user explicitly asks for a fresh replacement.

## Critical Maintenance Rule

Whenever a major runtime/deploy/pipeline error is diagnosed and resolved, update:

- [docs/ERROR_LOG.md](docs/ERROR_LOG.md)

Each entry must include:

- date/time
- affected subsystem
- error signature
- root cause
- fix
- verification evidence

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Clypt-Backend** (4158 symbols, 10524 relationships, 251 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/Clypt-Backend/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Clypt-Backend/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Clypt-Backend/clusters` | All functional areas |
| `gitnexus://repo/Clypt-Backend/processes` | All execution flows |
| `gitnexus://repo/Clypt-Backend/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
