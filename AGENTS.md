# AGENTS

Operational startup and maintenance guide for coding agents.

## Project Overview

- Product: Clypt V3.1 backend
- Implemented: Phases 1-4
- Planned: Phases 5-6
- Active topology: **Phase1 H200 + Phase26 H200 + Modal**
  - **Phase1 host (H200 default)**: Phase 1 runner/orchestrator, persistent local VibeVoice service, persistent local visual service, co-located VibeVoice vLLM sidecar, in-process NFA -> emotion2vec+ -> YAMNet.
  - **Phase26 host (H200)**: `POST /tasks/phase26-enqueue`, local SQLite queue + worker, SGLang Qwen on `:8001`, current Phase 2-4 runtime, future Phase 5-6 orchestration.
  - **Modal**: `POST /tasks/node-media-prep` on `L4` with `min_containers=1`; future render/export will follow the same external-worker pattern.
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
python -m pip install -r requirements-do-phase1-h200.txt
```

### Phase26 runtime deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase26-h200.txt
```

### Pipeline tests (offline)

```bash
python -m pytest tests/backend/pipeline -q
```

### Phase 1 + downstream enqueue

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

### Phase26 local worker

```bash
python -m backend.runtime.run_phase26_worker --worker-id phase26-worker-1
```

## Runtime Truths To Preserve

- `VIBEVOICE_VLLM_MODEL` must be `vibevoice` on the Phase1 host.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required on both H200 hosts.
- Phase 1 now runs on one host with two hot local services:
  - VibeVoice at `POST /tasks/vibevoice-asr`
  - visual extraction at `POST /tasks/visual-extract`
- Current Phase 1 visual settings must stay intact unless the user explicitly approves retuning:
  - `tensorrt_fp16`
  - batch size `16`
  - threshold `0.35`
  - shape `640`
  - ByteTrack buffer `30`
  - ByteTrack match threshold `0.7`
  - GPU decode
- NFA / emotion2vec+ / YAMNet remain persistent in-process on the Phase1 host.
- There is **no local fallback**:
  - Phase1 requires `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_*`
  - Phase1 requires `CLYPT_PHASE1_VISUAL_SERVICE_*`
  - Phase1 requires `CLYPT_PHASE24_DISPATCH_*`
  - Phase26 requires `CLYPT_PHASE24_NODE_MEDIA_PREP_*`
- Phase 1 audio-post must launch immediately after the VibeVoice HTTP call returns, not after visual completion.
- Phase 1 runtime requires `CLYPT_PHASE1_INPUT_MODE=test_bank`.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; use explicit per-stage concurrency envs.
- Phase26 local runtime requires `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`.
- Phase26 worker enforces `GENAI_GENERATION_BACKEND=local_openai`.
- Default crash handling mode remains fail-fast:
  - `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`
  - `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`
- Comments/trends augmentation is hard-join + fail-fast before Phase 4.
- Qwen serving target is the SGLang service on Phase26 `127.0.0.1:8001`.
- Node-media prep is always delegated remotely. Do not re-introduce an in-process ffmpeg fallback on either H200 host.
- Phase 1 H100 backup settings belong only in [docs/runtime/known-good-phase1-h100-backup.env](docs/runtime/known-good-phase1-h100-backup.env), and that file must only adjust memory-sensitive knobs.

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

This project is indexed by GitNexus as **Clypt-Backend** (3940 symbols, 9755 relationships, 229 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
