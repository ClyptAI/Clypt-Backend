# AGENTS

Operational startup and maintenance guide for coding agents and maintainers.

## Project Snapshot

- Product: Clypt V3.1 backend
- Implemented: Phases 1-4
- Planned: Phases 5-6
- Phase 1 topology: two-host, no local fallback.
  - **RTX 6000 Ada (sole tenant)**: VibeVoice vLLM ASR + ffmpeg NVENC
    node-media prep, served as one FastAPI app at
    `POST /tasks/vibevoice-asr` (ASR only — NFA / emotion2vec+ / YAMNet
    are **not** on this host) and `POST /tasks/node-media-prep`.
  - **H200**: visual chain (RF-DETR + ByteTrack) **plus the post-ASR
    audio chain in-process (NFA → emotion2vec+ → YAMNet CPU)**, SGLang
    Qwen on `:8001`, Phase 1 orchestrator, Phase 2-4 SQLite queue and
    local worker.
  - H200 calls the RTX host over HTTP via `RemoteVibeVoiceAsrClient`
    (legacy alias: `RemoteAudioChainClient`) and
    `RemoteNodeMediaPrepClient`. Config load fails fast if the ASR /
    node-media prep URL or bearer token is unset.
- Current Phase 2-4 local runtime: SQLite queue + local worker + local OpenAI-compatible generation endpoint

## Read Order (Required - You MUST read these before reporting back to the user.)

1. [README.md](README.md)
2. [docs/runtime/RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
3. [docs/deployment/P1_DEPLOY.md](docs/deployment/P1_DEPLOY.md)
4. [docs/deployment/P1_AUDIO_HOST_DEPLOY.md](docs/deployment/P1_AUDIO_HOST_DEPLOY.md)
5. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
6. [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)
7. [docs/EVALS.md](docs/EVALS.md)
8. [docs/ERROR_LOG.md](docs/ERROR_LOG.md)

## Documentation-First Rule (Required)

When in doubt, read documentation before making changes or running operational commands.

- Start with the required read order above.
- Always use GitNexus for codebase retrieval/search before falling back to ad-hoc file scanning. 
- If this is a brand new agent session, run `npx gitnexus analyze` as soon as you finish reading this doc.
- IF YOU ARE OPENAI CODEX, you may NOT at any cost do "invocation": { "server": "codebase-retrieval", "tool": "codebase-retrieval" }. Use GitNexus for retrieval ALWAYS.
- Always double-check `.gitignore` coverage before committing to avoid accidental local artifact commits.
- Treat docs as the source of truth for runtime/deploy behavior unless code has clearly diverged.
- If behavior is unclear, re-check `RUNTIME_GUIDE.md` and `P1_DEPLOY.md` before proceeding.
- If you discover a docs/code mismatch, call it out and tell the user immediately!

## Resolver-First Rule (Required)

Before starting any task, read [docs/EVALS.md](docs/EVALS.md) and load the relevant skill pointers for that task type. If all the skills listed there are not installed into project-local .agents/skills/ and .claude/skills/ then do so ASAP.

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

H200 (visual + audio-post + Phase 2-4 worker):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1-visual.txt
```

RTX 6000 Ada (VibeVoice ASR + node-media prep, sole tenant):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1-audio.txt
```

Notes:
- `requirements-local.txt` is the local/dev dependency set.
- `requirements-do-phase1-visual.txt` is the standalone GPU runtime set for the H200. It now includes the post-ASR audio chain deps (NFA, emotion2vec+, YAMNet) that previously lived on the RTX host.
- `requirements-do-phase1-audio.txt` is the standalone GPU runtime set for the RTX 6000 Ada VibeVoice ASR host. It explicitly does **not** include NeMo / FunASR / TensorFlow / librosa / resampy.

### Pipeline tests (offline)

```bash
python -m pytest tests/backend/pipeline -q
```

### Phase 1 + Phase 2-4 queue mode

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

### Phase 2-4 local worker

```bash
python -m backend.runtime.run_phase24_local_worker --worker-id local-worker-1
```

## Runtime Truths To Preserve

- `VIBEVOICE_VLLM_MODEL` must be `vibevoice` **on the RTX 6000 Ada**. VibeVoice envs must not be set on the H200.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required on both hosts.
- Phase 1 is split across two hosts:
  - **H200**: visual chain (RF-DETR + ByteTrack), **post-ASR audio chain in-process (NFA → emotion2vec+ → YAMNet CPU)**, Phase 1 orchestrator, Phase 2-4 queue + worker, SGLang Qwen on `:8001`. Current SGLang flags: `--context-length 65536`, `--kv-cache-dtype fp8_e4m3`, `--mem-fraction-static 0.78`, speculative NEXTN (3 steps, eagle-topk 1, 4 draft tokens), `--mamba-scheduler-strategy extra_buffer`, `--schedule-policy lpm`, `--chunked-prefill-size 8192`, `--grammar-backend xgrammar`, `--reasoning-parser qwen3`, plus systemd env `HF_HUB_OFFLINE=1` and `SGLANG_ENABLE_SPEC_V2=1`.
  - **RTX 6000 Ada (sole tenant)**: VibeVoice vLLM ASR + ffmpeg NVENC/NVDEC node-media prep, served via one FastAPI app with `POST /tasks/vibevoice-asr` and `POST /tasks/node-media-prep`. Current vLLM flags: `--gpu-memory-utilization 0.77 --max-num-seqs 2 --dtype bfloat16`, CUDA graph capture enabled, no speculative decoding.
- There is **no local fallback**. `backend/providers/config.py` requires `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`, `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`, `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`, and `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` on the H200. Missing any of these fails fast at startup. The legacy `CLYPT_PHASE1_AUDIO_HOST_URL` / `CLYPT_PHASE1_AUDIO_HOST_TOKEN` names are still accepted as deprecated aliases for one release.
- The H200 Python client is `RemoteVibeVoiceAsrClient` (legacy alias: `RemoteAudioChainClient`); the response type is `VibeVoiceAsrResponse` (legacy alias: `PhaseOneAudioResponse`); the settings type is `VibeVoiceAsrServiceSettings` (legacy alias: `AudioHostSettings`).
- Phase 1 audio chain (in-process NFA → emotion2vec+ → YAMNet) must launch immediately after the VibeVoice ASR HTTP call returns, not after RF-DETR finishes.
- Phase 1 local runtime requires `CLYPT_PHASE1_INPUT_MODE=test_bank`.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; use explicit per-stage concurrency envs instead.
- Phase 2-4 local runtime requires `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`.
- Phase 2-4 local worker enforces `GENAI_GENERATION_BACKEND=local_openai`.
- Default crash handling mode is fail-fast (`CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`, `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`).
- Comments/trends augmentation is hard-join + fail-fast before Phase 4.
- Qwen serving target is the SGLang service on H200 `127.0.0.1:8001`.
- `CLYPT_PHASE1_ASR_BACKEND` (if set at all) only accepts `vllm`. The active ASR path is a remote HTTP call to the RTX VibeVoice ASR service; the H200 has no in-process VibeVoice provider.
- Node-media prep is always delegated to the RTX host. Do not re-introduce an in-process ffmpeg fallback on the H200 — H200 NVENC returns `unsupported device (2)`. NVDEC clip extraction uses `-c:v h264_cuvid` (explicit, not just `-hwaccel cuda`). Max concurrency is 8 (`CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY`); 16 OOMs NVENC input buffers at the current vLLM footprint.
- NFA / emotion2vec+ / YAMNet are installed from `requirements-do-phase1-visual.txt` on the H200 only. Do not add them to `requirements-do-phase1-audio.txt` — the RTX host must stay a narrow VibeVoice + ffmpeg box.

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

This project is indexed by GitNexus as **Clypt-Backend** (2782 symbols, 7052 relationships, 211 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
