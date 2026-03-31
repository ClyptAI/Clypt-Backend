# Clypt - Skill Documentation

## Current Installed Skill Sets
- `garrytan/gstack` set (22 skills): Browser QA + dogfooding + end-to-end engineering workflow commands (plan/review/debug/ship/safety modes).
- `obra/superpowers` set (14 skills): Process and quality discipline pack for planning, testing, debugging, code review, and verification-first execution.
- `nextlevelbuilder/ui-ux-pro-max-skill` set (7 skills): Full UI/UX + branding + design-system + visual asset/presentation skill suite.
- `understand-anything` plugin set (6 skills): Knowledge-graph codebase understanding tools for architecture analysis, diff risk analysis, deep explanation, and onboarding.
- `google-gemini/gemini-skills` set (2 skills): Gemini API and Vertex AI implementation guidance for multimodal/structured-output workflows.
- `wshobson/agents` set (2 skills): Python testing patterns and GitHub Actions CI/CD workflow templates.
- `preinstalled local toolbox` set (6 skills): Core local helpers for browser automation, skill discovery, React/Remotion guidance, and UI/accessibility review.
- `vercel-labs/next-skills@next-best-practices`: Next.js architecture and performance best-practices reference.
- `github/awesome-copilot@playwright-generate-test`: Scenario-to-test Playwright generation workflow using Playwright MCP.
- `elastic/agent-skills@observability-edot-python-instrument`: Python observability setup using Elastic EDOT auto-instrumentation.
- `bobmatnyc/claude-mpm-skills@digitalocean-management`: DigitalOcean monitoring/uptime/operations management guidance.
- `upstash/context7@context7-mcp`: Up-to-date library/framework/API reference grounding with practical code examples.

## Repository Skill Catalog

### vercel-labs/next-skills

- `next-best-practices`: Next.js best-practices guidance for file conventions, RSC boundaries, data patterns, async APIs, metadata, error handling, route handlers, image/font optimization, and bundling.

### google-gemini/gemini-skills

- `gemini-api-dev`: Build with Gemini API and multimodal content (text/image/audio/video), function calling, structured outputs, model selection, and official SDK usage patterns.
- `vertex-ai-api-dev`: Use Gemini via Vertex AI for enterprise/GCP workloads, including SDK setup (Python/JS/Go/Java/C#), Live API, tools, multimedia generation, caching, and batch prediction.

### wshobson/agents

- `python-testing-patterns`: Implement comprehensive Python testing with pytest, fixtures, mocking, and TDD.
- `github-actions-templates`: Create production-ready GitHub Actions workflows for automated test/build/deploy CI/CD.

### github/awesome-copilot

- `playwright-generate-test`: Generate a Playwright test from a scenario using Playwright MCP, then execute and iterate until it passes.

### elastic/agent-skills

- `observability-edot-python-instrument`: Instrument Python services with Elastic EDOT auto-instrumentation for tracing/metrics/logs when no existing APM agent is present.

### bobmatnyc/claude-mpm-skills

- `digitalocean-management`: Manage DigitalOcean operations for monitoring, uptime checks, alerts, and project/resource organization.

### upstash/context7

- `context7-mcp`: Use when library/framework/API references or concrete code examples are needed (React/Vue/Next/Prisma/Supabase/etc.), especially for up-to-date grounding.

### garrytan/gstack

- `gstack`: Top-level orchestrator for fast headless-browser QA and dogfooding plus proactive workflow-skill suggestions.
- `browse`: Headless browser testing and site interaction for verification, screenshots, and reproducible evidence.
- `careful`: Safety guardrails that warn before destructive operations (`rm -rf`, force-push, destructive DB/K8s commands).
- `codex`: OpenAI Codex CLI wrapper for independent review/challenge/consult workflows.
- `design-consultation`: Design-system creation workflow (aesthetic, typography, color, spacing, motion) and DESIGN.md generation.
- `design-review`: Live visual QA + fix loop for hierarchy, spacing, consistency, and interaction polish (with before/after verification).
- `document-release`: Post-ship documentation sync across README/ARCHITECTURE/CONTRIBUTING/CLAUDE/changelog/todos.
- `freeze`: Restrict edits to a specified directory (hard block on out-of-scope writes/edits).
- `gstack-upgrade`: Upgrade gstack, handling global/vendored install modes and rollout flow.
- `guard`: Maximum safety mode combining destructive-command warnings (`careful`) and edit-boundary enforcement (`freeze`).
- `investigate`: Root-cause-first debugging workflow (investigate -> analyze -> hypothesize -> implement).
- `office-hours`: Product ideation/diagnostic workflow (startup and builder modes) before coding begins.
- `plan-ceo-review`: Founder/CEO strategy review to challenge assumptions and tune ambition/scope mode.
- `plan-design-review`: Plan-mode UI/UX review that rates design dimensions and revises plans toward 10/10 quality.
- `plan-eng-review`: Engineering architecture review for data flow, edge cases, tests, and execution rigor.
- `qa-only`: Report-only QA pass with bugs/repro/screenshots but no code changes.
- `qa`: QA plus iterative fix-and-reverify loop with health scoring and ship-readiness summary.
- `retro`: Weekly engineering retrospective with contribution patterns and trend tracking.
- `review`: Pre-landing diff review focused on structural and production-risk issues.
- `setup-browser-cookies`: Import browser cookies into the headless session for authenticated QA flows.
- `ship`: End-to-end ship workflow (base sync, tests/review, version/changelog, commit/push/PR).
- `unfreeze`: Remove freeze restrictions to restore full edit scope.

### obra/superpowers

- `brainstorming`: Mandatory pre-implementation idea/requirements/design exploration for creative tasks.
- `dispatching-parallel-agents`: Use for 2+ independent tasks that can run concurrently.
- `executing-plans`: Execute a written implementation plan with checkpoints in a separate session.
- `finishing-a-development-branch`: Structured completion workflow once implementation/tests are done (merge/PR/cleanup options).
- `receiving-code-review`: Rigorously validate review feedback before applying changes.
- `requesting-code-review`: Request review for completed tasks, major features, and pre-merge gates.
- `subagent-driven-development`: Execute implementation plans through independent subtasks in-session.
- `systematic-debugging`: Enforce disciplined debugging before proposing fixes.
- `test-driven-development`: Require tests-first implementation for features and bugfixes.
- `using-git-worktrees`: Isolated feature-work setup with safe worktree workflows.
- `using-superpowers`: Entry-point skill for discovering/using superpowers skills correctly.
- `verification-before-completion`: Verify with command evidence before claiming completion/fix/pass.
- `writing-plans`: Create multi-step implementation plans from requirements/specs.
- `writing-skills`: Create/edit/verify skills before deployment.

### nextlevelbuilder/ui-ux-pro-max-skill

- `ui-ux-pro-max`: Broad UI/UX intelligence for web/mobile design and implementation across multiple stacks, styles, accessibility, typography, and interaction systems.
- `ckm:design`: Full-spectrum design operations (brand identity, tokens, logo/icon generation, CIP, presentations, banners, and social assets).
- `ckm:ui-styling`: Accessible, consistent UI styling with shadcn/ui + Radix + Tailwind, responsive theming, and component-level design quality.
- `ckm:design-system`: Token architecture (primitive -> semantic -> component), component specifications, and systematic presentation-ready design.
- `ckm:brand`: Brand voice, visual identity, messaging frameworks, and consistency enforcement.
- `ckm:banner-design`: Multi-platform banner/hero/ad/print visual design with multiple art directions and AI-assisted generation.
- `ckm:slides`: Strategic HTML slide generation with design tokens, responsive layout, and chart-driven storytelling.

### understand-anything plugin (`~/.codex/understand-anything/understand-anything-plugin`)

- `understand`: Build a codebase knowledge graph to understand architecture/components/relationships.
- `understand-chat`: Query and reason over the knowledge graph for targeted codebase questions.
- `understand-dashboard`: Launch an interactive dashboard to visualize the knowledge graph.
- `understand-diff`: Analyze git diffs/PRs to understand affected components and risk.
- `understand-explain`: Deep-dive explanation for a specific file/function/module.
- `understand-onboard`: Generate onboarding guidance for new team members from codebase context.

### Preinstalled Local Skills (Origin Repo Not Exposed by `skills ls --json`)

- `agent-browser`: Browser automation for navigation, form fills, screenshots, web testing, and extraction.
- `find-skills`: Discover/install skills via `npx skills find` and `npx skills add`.
- `react-doctor`: Diagnose React codebase health, performance, security, and quality issues.
- `remotion-best-practices`: Remotion video creation best-practices in React.
- `vercel-react-best-practices`: React/Next performance guidance from Vercel engineering.
- `web-design-guidelines`: UI review against Web Interface Guidelines and accessibility/UX checks.

## Recommended Usage by Clypt Phase

### 1) Frontend (Vite + React in `frontend/`)

Use: `next-best-practices` (React performance patterns still apply), `vercel-react-best-practices` where relevant

When:
- Refactoring `frontend/*`
- Improving routing, data fetching, caching, rendering boundaries
- Reducing unnecessary re-renders and bundle overhead

Prompt pattern:
- "Use `next-best-practices` / React performance guidance to review and optimize `frontend/src/pages/CortexGraph.tsx` for performance and maintainability."

### 2) Gemini Integration (Phases 2A/2B/3/5)

Use: `gemini-api-dev`

When:
- Building prompts, request shaping, response parsing, and retries
- Hardening semantic-node and edge-generation calls

Prompt pattern:
- "Use `gemini-api-dev` to harden API calls in `backend/pipeline/phase_2a_make_nodes.py` with retries, timeouts, and structured output validation."

### 3) Vertex AI Alignment (GCP Runtime Path)

Use: `vertex-ai-api-dev`

When:
- Running Gemini through Vertex AI
- Standardizing auth/env config and endpoint usage
- Migrating direct Gemini usage to Vertex conventions

Prompt pattern:
- "Use `vertex-ai-api-dev` to align our Phase 3 embedding calls with Vertex AI best practices in `backend/pipeline/phase_3_multimodal_embeddings.py`."

### 4) Python Test Coverage (Backend + Pipeline)

Use: `python-testing-patterns`

When:
- Adding unit/integration tests for modal worker and phase scripts
- Creating stable fixtures for pipeline outputs
- Preventing regressions in schema/contract handling

Prompt pattern:
- "Use `python-testing-patterns` to add tests for `backend/do_phase1_worker.py` contract validation and track schema output."

### 5) Browser E2E for the web app (`frontend/`)

Use: `playwright-generate-test`

When:
- Generating tests for graph interactions and UI flows
- Regressions around `frontend/src/pages/CortexGraph.tsx` and component panels

Prompt pattern:
- "Use `playwright-generate-test` to generate E2E tests for selecting a node, opening details, and rendering clip previews."

### 6) CI/CD Workflows

Use: `github-actions-templates`

When:
- Setting up CI for lint/test/build in both `backend` and `frontend`
- Running Playwright tests and Python tests on pull requests

Prompt pattern:
- "Use `github-actions-templates` to create workflows for backend pytest, frontend build, and Playwright E2E on PRs."

### 7) Observability (Python Services)

Use: `observability-edot-python-instrument`

When:
- Adding telemetry to long-running pipeline jobs
- Tracing latency bottlenecks in phase execution
- Correlating errors across ingestion, reasoning, and storage steps

Prompt pattern:
- "Use `observability-edot-python-instrument` to instrument `backend/pipeline/run_pipeline.py` with traces and structured spans per phase."

### 8) Infrastructure on DigitalOcean

Use: `digitalocean-management`

When:
- Planning deployment targets on Droplets/managed services
- Standardizing environment rollout and operational checks

Prompt pattern:
- "Use `digitalocean-management` to draft a deployment plan for Clypt services and required environment variables."

### 9) Dependency and API Reference Grounding

Use: `context7-mcp`

When:
- You need current docs/examples for libraries before coding
- You want fewer hallucinations in implementation details
- You are updating dependencies and need version-accurate usage

Prompt pattern:
- "Use `context7-mcp` to fetch current docs/examples for the library used in `frontend/src/pages/CortexGraph.tsx` before refactoring."

### 10) Fast QA and Dogfooding Flows

Use: `gstack`

When:
- Running fast browser QA loops and reproducing UI bugs with evidence
- Validating form flows, responsive behavior, and state transitions
- Doing pre-ship manual verification before merge/release

Prompt pattern:
- "Use `gstack` to run a QA pass on the Cortex graph flow and capture reproducible bug evidence for any failures."

### 11) Structured Delivery Workflow

Use: `superpowers`

When:
- You want a guided flow from idea -> plan -> implementation -> QA -> ship
- You need explicit checkpoints for review, risk reduction, and release notes
- You want repeatable prompts for planning and execution rituals

Prompt pattern:
- "Use `superpowers` to run planning + engineering review before implementing changes to `backend/pipeline/run_pipeline.py`, then generate a ship checklist."

### 12) UI/UX Design Suite

Use: `ui-ux-pro-max`, `ckm-design`, `ckm-ui-styling`, `ckm-design-system`, `ckm-brand`, `ckm-banner-design`, `ckm-slides`

When:
- You want to upgrade the web app (`frontend/`) visual quality and consistency
- You need a tighter design-system foundation for reusable components
- You need branded visual assets and presentation artifacts for demos/pitches

Prompt patterns:
- "Use `ckm-design-system` + `ckm-ui-styling` to standardize tokens, spacing, and component styling for `frontend/src/components/*`."
- "Use `ckm-design` + `ui-ux-pro-max` to redesign the graph detail and clip panels for clearer hierarchy and better usability."
- "Use `ckm-brand` + `ckm-banner-design` to generate launch-ready visuals for Clypt announcements."
- "Use `ckm-slides` to prepare a polished product walkthrough deck from our current docs."

## Suggested End-to-End Sequence

1. `python-testing-patterns` to stabilize backend tests.
2. `gemini-api-dev` and `vertex-ai-api-dev` to harden model calls.
3. `next-best-practices` (or React-focused skills) for `frontend/` optimization.
4. `ui-ux-pro-max` + `ckm-*` skills for visual design/system passes.
5. `playwright-generate-test` for UI regression coverage.
6. `github-actions-templates` to enforce CI gates.
7. `observability-edot-python-instrument` for runtime insight.
8. `context7-mcp` during implementation when APIs or libs are uncertain.
9. `gstack` for manual browser QA and pre-ship checks.
10. `superpowers` for plan/review/ship guardrails across the workflow.
11. `digitalocean-management` if/when DO deployment is active.

## Quick "What Skill Do I Use?" Map

- UI performance issue -> `next-best-practices`
- Gemini output quality/reliability issue -> `gemini-api-dev`
- Vertex AI config/runtime issue -> `vertex-ai-api-dev`
- Backend regression bug -> `python-testing-patterns`
- Broken user flow in app -> `playwright-generate-test`
- Need CI pipeline setup -> `github-actions-templates`
- Hard-to-debug runtime failures -> `observability-edot-python-instrument`
- Need up-to-date package or API docs while coding -> `context7-mcp`
- Need rapid browser dogfooding/QA evidence -> `gstack`
- Need a structured plan->review->ship workflow -> `superpowers`
- Need full UI redesign pass -> `ui-ux-pro-max` + `ckm-design`
- Need stronger component styling consistency -> `ckm-ui-styling`
- Need a reusable design token/component foundation -> `ckm-design-system`
- Need brand direction and assets -> `ckm-brand` + `ckm-banner-design`
- Need a slide deck for demos/pitches -> `ckm-slides`
- Deployment planning on DO -> `digitalocean-management`
