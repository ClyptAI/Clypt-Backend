# EVALS Resolver Pointers

Purpose: keep context lean and load the right skills only when needed.

## Resolver Rule (Required)

Before starting implementation work:

1. Classify the task type.
2. Load the matching skill pointers from this file.
3. Run the required checks for that task class before shipping.

For prompt/scoring/behavior changes, run the pipeline eval suite and compare against the latest baseline. If accuracy drops by more than 2%, revert and investigate before shipping.

Canonical eval command:

```bash
python -m pytest tests/backend/pipeline -q
```

## Task Router

| Task Type | Load First |
|---|---|
| Architecture/approach planning | `superpowers/skills/brainstorming`, `superpowers/skills/writing-plans`, `gstack/plan-eng-review`, `gstack/plan-ceo-review`, `gstack/office-hours` |
| Executing a multi-step approved plan | `superpowers/skills/executing-plans`, `superpowers/skills/subagent-driven-development`, `superpowers/skills/dispatching-parallel-agents` |
| Debugging regressions | `superpowers/skills/systematic-debugging`, `gstack/investigate`, `gitnexus/gitnexus-debugging`, `gstack/review` |
| Refactors/renames with blast-radius checks | `gitnexus/gitnexus-refactoring`, `gitnexus/gitnexus-impact-analysis`, `superpowers/skills/receiving-code-review` |
| Code review readiness | `superpowers/skills/requesting-code-review`, `superpowers/skills/verification-before-completion`, `gstack/review`, `gstack/codex` |
| QA test/report/fix loops | `gstack/qa`, `gstack/qa-only`, `gstack/browse`, `gstack/setup-browser-cookies` |
| Frontend UX/design quality | `gstack/design-review`, `gstack/design-consultation` |
| Deploy/ship operations | `do-app-platform-skills/skills/deployment`, `do-app-platform-skills/skills/troubleshooting`, `gstack/ship`, `gstack/setup-deploy`, `gstack/careful`, `gstack/guard` |
| DigitalOcean infra/data concerns | `do-app-platform-skills/skills/networking`, `do-app-platform-skills/skills/spaces`, `do-app-platform-skills/skills/managed-db-services`, `do-app-platform-skills/skills/postgres`, `do-app-platform-skills/skills/migration` |
| Gemini / Vertex API work | `gemini-skills/skills/gemini-api-dev`, `gemini-skills/skills/vertex-ai-api-dev`, `gemini-skills/skills/gemini-live-api-dev`, `gemini-skills/skills/gemini-interactions-api` |
| Hugging Face model/training/data work | `hf-skills/skills/hf-cli`, `hf-skills/skills/huggingface-datasets`, `hf-skills/skills/huggingface-llm-trainer`, `hf-skills/skills/huggingface-vision-trainer`, `hf-skills/skills/huggingface-community-evals`, `hf-skills/hf-mcp/skills/hf-mcp` |
| External docs/API lookup | `context-hub/cli/skills/get-api-docs`, `context-hub/content/tavily/skills/tavily-best-practices`, `hf-skills/hf-mcp/skills/hf-mcp` |
| Branch/worktree/release workflow | `superpowers/skills/using-git-worktrees`, `superpowers/skills/finishing-a-development-branch`, `gstack/land-and-deploy`, `gstack/document-release` |

## Installed Skill Pointers (Newly Installed)

These are pointer links only. Load on demand per task type.
If the active agent is Claude, use the same relative paths but swap `../.agents/` with `../.claude/`.

### Superpowers (obra/superpowers)

- [`brainstorming`](../.agents/skills/superpowers/skills/brainstorming/SKILL.md)
- [`dispatching-parallel-agents`](../.agents/skills/superpowers/skills/dispatching-parallel-agents/SKILL.md)
- [`executing-plans`](../.agents/skills/superpowers/skills/executing-plans/SKILL.md)
- [`finishing-a-development-branch`](../.agents/skills/superpowers/skills/finishing-a-development-branch/SKILL.md)
- [`receiving-code-review`](../.agents/skills/superpowers/skills/receiving-code-review/SKILL.md)
- [`requesting-code-review`](../.agents/skills/superpowers/skills/requesting-code-review/SKILL.md)
- [`subagent-driven-development`](../.agents/skills/superpowers/skills/subagent-driven-development/SKILL.md)
- [`systematic-debugging`](../.agents/skills/superpowers/skills/systematic-debugging/SKILL.md)
- [`test-driven-development`](../.agents/skills/superpowers/skills/test-driven-development/SKILL.md)
- [`using-git-worktrees`](../.agents/skills/superpowers/skills/using-git-worktrees/SKILL.md)
- [`using-superpowers`](../.agents/skills/superpowers/skills/using-superpowers/SKILL.md)
- [`verification-before-completion`](../.agents/skills/superpowers/skills/verification-before-completion/SKILL.md)
- [`writing-plans`](../.agents/skills/superpowers/skills/writing-plans/SKILL.md)
- [`writing-skills`](../.agents/skills/superpowers/skills/writing-skills/SKILL.md)

### Gstack (garrytan/gstack)

- [`gstack`](../.agents/skills/gstack/SKILL.md)
- [`autoplan`](../.agents/skills/gstack/autoplan/SKILL.md)
- [`benchmark`](../.agents/skills/gstack/benchmark/SKILL.md)
- [`browse`](../.agents/skills/gstack/browse/SKILL.md)
- [`canary`](../.agents/skills/gstack/canary/SKILL.md)
- [`careful`](../.agents/skills/gstack/careful/SKILL.md)
- [`checkpoint`](../.agents/skills/gstack/checkpoint/SKILL.md)
- [`codex`](../.agents/skills/gstack/codex/SKILL.md)
- [`cso`](../.agents/skills/gstack/cso/SKILL.md)
- [`design-consultation`](../.agents/skills/gstack/design-consultation/SKILL.md)
- [`design-html`](../.agents/skills/gstack/design-html/SKILL.md)
- [`design-review`](../.agents/skills/gstack/design-review/SKILL.md)
- [`design-shotgun`](../.agents/skills/gstack/design-shotgun/SKILL.md)
- [`devex-review`](../.agents/skills/gstack/devex-review/SKILL.md)
- [`document-release`](../.agents/skills/gstack/document-release/SKILL.md)
- [`freeze`](../.agents/skills/gstack/freeze/SKILL.md)
- [`gstack-upgrade`](../.agents/skills/gstack/gstack-upgrade/SKILL.md)
- [`guard`](../.agents/skills/gstack/guard/SKILL.md)
- [`health`](../.agents/skills/gstack/health/SKILL.md)
- [`investigate`](../.agents/skills/gstack/investigate/SKILL.md)
- [`land-and-deploy`](../.agents/skills/gstack/land-and-deploy/SKILL.md)
- [`learn`](../.agents/skills/gstack/learn/SKILL.md)
- [`office-hours`](../.agents/skills/gstack/office-hours/SKILL.md)
- [`open-gstack-browser`](../.agents/skills/gstack/open-gstack-browser/SKILL.md)
- [`gstack-openclaw-ceo-review`](../.agents/skills/gstack/openclaw/skills/gstack-openclaw-ceo-review/SKILL.md)
- [`gstack-openclaw-investigate`](../.agents/skills/gstack/openclaw/skills/gstack-openclaw-investigate/SKILL.md)
- [`gstack-openclaw-office-hours`](../.agents/skills/gstack/openclaw/skills/gstack-openclaw-office-hours/SKILL.md)
- [`gstack-openclaw-retro`](../.agents/skills/gstack/openclaw/skills/gstack-openclaw-retro/SKILL.md)
- [`pair-agent`](../.agents/skills/gstack/pair-agent/SKILL.md)
- [`plan-ceo-review`](../.agents/skills/gstack/plan-ceo-review/SKILL.md)
- [`plan-design-review`](../.agents/skills/gstack/plan-design-review/SKILL.md)
- [`plan-devex-review`](../.agents/skills/gstack/plan-devex-review/SKILL.md)
- [`plan-eng-review`](../.agents/skills/gstack/plan-eng-review/SKILL.md)
- [`qa-only`](../.agents/skills/gstack/qa-only/SKILL.md)
- [`qa`](../.agents/skills/gstack/qa/SKILL.md)
- [`retro`](../.agents/skills/gstack/retro/SKILL.md)
- [`review`](../.agents/skills/gstack/review/SKILL.md)
- [`setup-browser-cookies`](../.agents/skills/gstack/setup-browser-cookies/SKILL.md)
- [`setup-deploy`](../.agents/skills/gstack/setup-deploy/SKILL.md)
- [`ship`](../.agents/skills/gstack/ship/SKILL.md)
- [`unfreeze`](../.agents/skills/gstack/unfreeze/SKILL.md)

### Gemini Skills (google-gemini/gemini-skills)

- [`gemini-api-dev`](../.agents/skills/gemini-skills/skills/gemini-api-dev/SKILL.md)
- [`gemini-interactions-api`](../.agents/skills/gemini-skills/skills/gemini-interactions-api/SKILL.md)
- [`gemini-live-api-dev`](../.agents/skills/gemini-skills/skills/gemini-live-api-dev/SKILL.md)
- [`vertex-ai-api-dev`](../.agents/skills/gemini-skills/skills/vertex-ai-api-dev/SKILL.md)

### Context Hub (andrewyng/context-hub)

- [`get-api-docs`](../.agents/skills/context-hub/cli/skills/get-api-docs/SKILL.md)
- [`document-extraction`](../.agents/skills/context-hub/content/landingai/skills/ade/document-extraction/SKILL.md)
- [`document-workflows`](../.agents/skills/context-hub/content/landingai/skills/ade/document-workflows/SKILL.md)
- [`integrate`](../.agents/skills/context-hub/content/olakai/skills/integrate/SKILL.md)
- [`new-project`](../.agents/skills/context-hub/content/olakai/skills/new-project/SKILL.md)
- [`login-flows`](../.agents/skills/context-hub/content/playwright-community/skills/login-flows/SKILL.md)
- [`electronics-sourcing`](../.agents/skills/context-hub/content/sourceparts/skills/electronics-sourcing/SKILL.md)
- [`tavily-best-practices`](../.agents/skills/context-hub/content/tavily/skills/tavily-best-practices/SKILL.md)

### Hugging Face Skills (huggingface/skills)

- [`hf-mcp`](../.agents/skills/hf-skills/hf-mcp/skills/hf-mcp/SKILL.md)
- [`hf-cli`](../.agents/skills/hf-skills/skills/hf-cli/SKILL.md)
- [`huggingface-community-evals`](../.agents/skills/hf-skills/skills/huggingface-community-evals/SKILL.md)
- [`huggingface-datasets`](../.agents/skills/hf-skills/skills/huggingface-datasets/SKILL.md)
- [`huggingface-gradio`](../.agents/skills/hf-skills/skills/huggingface-gradio/SKILL.md)
- [`huggingface-llm-trainer`](../.agents/skills/hf-skills/skills/huggingface-llm-trainer/SKILL.md)
- [`huggingface-paper-publisher`](../.agents/skills/hf-skills/skills/huggingface-paper-publisher/SKILL.md)
- [`huggingface-papers`](../.agents/skills/hf-skills/skills/huggingface-papers/SKILL.md)
- [`huggingface-tool-builder`](../.agents/skills/hf-skills/skills/huggingface-tool-builder/SKILL.md)
- [`huggingface-trackio`](../.agents/skills/hf-skills/skills/huggingface-trackio/SKILL.md)
- [`huggingface-vision-trainer`](../.agents/skills/hf-skills/skills/huggingface-vision-trainer/SKILL.md)
- [`transformers-js`](../.agents/skills/hf-skills/skills/transformers-js/SKILL.md)

### DigitalOcean Skills (digitalocean-labs/do-app-platform-skills)

- [`do-app-platform-skills`](../.agents/skills/do-app-platform-skills/SKILL.md)
- [`ai-services`](../.agents/skills/do-app-platform-skills/skills/ai-services/SKILL.md)
- [`deployment`](../.agents/skills/do-app-platform-skills/skills/deployment/SKILL.md)
- [`designer`](../.agents/skills/do-app-platform-skills/skills/designer/SKILL.md)
- [`devcontainers`](../.agents/skills/do-app-platform-skills/skills/devcontainers/SKILL.md)
- [`managed-db-services`](../.agents/skills/do-app-platform-skills/skills/managed-db-services/SKILL.md)
- [`migration`](../.agents/skills/do-app-platform-skills/skills/migration/SKILL.md)
- [`networking`](../.agents/skills/do-app-platform-skills/skills/networking/SKILL.md)
- [`planner`](../.agents/skills/do-app-platform-skills/skills/planner/SKILL.md)
- [`postgres`](../.agents/skills/do-app-platform-skills/skills/postgres/SKILL.md)
- [`sandbox`](../.agents/skills/do-app-platform-skills/skills/sandbox/SKILL.md)
- [`spaces`](../.agents/skills/do-app-platform-skills/skills/spaces/SKILL.md)
- [`troubleshooting`](../.agents/skills/do-app-platform-skills/skills/troubleshooting/SKILL.md)
