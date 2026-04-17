# AGENTS

Operational startup and maintenance guide for coding agents and maintainers.

## Project Snapshot

- Product: Clypt V3.1 backend
- Implemented: Phases 1-4
- Planned: Phases 5-6
- Current Phase 1 ASR backend: local vLLM VibeVoice on the Phase 1 GPU host
- Current Phase 2-4 local runtime: SQLite queue + local worker + local OpenAI-compatible generation endpoint

## Read Order (Required - You MUST read these before reporting back to the user.)

1. [README.md](README.md)
2. [docs/runtime/RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
3. [docs/deployment/P1_DEPLOY.md](docs/deployment/P1_DEPLOY.md)
4. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
5. [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)
6. [docs/EVALS.md](docs/EVALS.md)
7. [docs/ERROR_LOG.md](docs/ERROR_LOG.md)

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

### Phase 1 runtime deps (DO GPU host only)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1.txt
```

Notes:
- `requirements-local.txt` is the local/dev dependency set.
- `requirements-do-phase1.txt` is the standalone GPU runtime set for Phase 1.

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

- `VIBEVOICE_VLLM_MODEL` must be `vibevoice`.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required even for VibeVoice-only runs.
- Phase 1 audio chain must launch immediately after ASR completion.
- Phase 1 local runtime requires `CLYPT_PHASE1_INPUT_MODE=test_bank`.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; use explicit per-stage concurrency envs instead.
- Phase 2-4 local runtime requires `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`.
- Phase 2-4 local worker currently enforces `GENAI_GENERATION_BACKEND=local_openai`.
- Default crash handling mode is fail-fast (`CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`, `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`).
- Comments/trends augmentation is hard-join + fail-fast before Phase 4.
- Qwen serving target is the SGLang service on `127.0.0.1:8001`.
- `CLYPT_PHASE1_ASR_BACKEND` only accepts `vllm`. `VIBEVOICE_VLLM_BASE_URL` is required.
- Node-media prep for Phase 2 runs in-process on the Phase 2-4 worker host; there is no remote media-prep offload.

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
