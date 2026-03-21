# Clypt Skill Cheat Sheet

This cheat sheet maps your installed skills to Clypt's end-to-end workflow.

## Current Installed Set

- `next-best-practices` (official, Vercel)
- `gemini-api-dev` (official, Google Gemini)
- `vertex-ai-api-dev` (official, Google Gemini)
- `python-testing-patterns`
- `github-actions-templates`
- `playwright-generate-test` (official, GitHub)
- `observability-edot-python-instrument` (official, Elastic)
- `digitalocean-management`
- `context7-mcp` (official, Upstash)
- `gstack`
- `superpowers`
- `ui-ux-pro-max`
- `ckm-design`
- `ckm-ui-styling`
- `ckm-design-system`
- `ckm-brand`
- `ckm-banner-design`
- `ckm-slides`

## Recommended Usage by Clypt Phase

### 1) Frontend (Cortex UI / Next.js)

Use: `next-best-practices`

When:
- Refactoring `cortex-ui/*`
- Improving app-router patterns, data fetching, caching, rendering boundaries
- Reducing unnecessary re-renders and bundle overhead

Prompt pattern:
- "Use `next-best-practices` to review and optimize `cortex-ui/app/graph/page.tsx` for performance and maintainability."

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
- "Use `python-testing-patterns` to add tests for `backend/modal_worker.py` contract validation and track schema output."

### 5) Browser E2E for Cortex UI

Use: `playwright-generate-test`

When:
- Generating tests for graph interactions and UI flows
- Regressions around `cortex-ui/app/graph/page.tsx` and component panels

Prompt pattern:
- "Use `playwright-generate-test` to generate E2E tests for selecting a node, opening details, and rendering clip previews."

### 6) CI/CD Workflows

Use: `github-actions-templates`

When:
- Setting up CI for lint/test/build in both `backend` and `cortex-ui`
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
- "Use `context7-mcp` to fetch current docs/examples for the library used in `cortex-ui/app/components/CortexGraph.tsx` before refactoring."

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
- You want to upgrade Cortex UI visual quality and consistency
- You need a tighter design-system foundation for reusable components
- You need branded visual assets and presentation artifacts for demos/pitches

Prompt patterns:
- "Use `ckm-design-system` + `ckm-ui-styling` to standardize tokens, spacing, and component styling for `cortex-ui/app/components/*`."
- "Use `ckm-design` + `ui-ux-pro-max` to redesign the graph detail and clip panels for clearer hierarchy and better usability."
- "Use `ckm-brand` + `ckm-banner-design` to generate launch-ready visuals for Clypt announcements."
- "Use `ckm-slides` to prepare a polished product walkthrough deck from our current docs."

## Suggested End-to-End Sequence

1. `python-testing-patterns` to stabilize backend tests.
2. `gemini-api-dev` and `vertex-ai-api-dev` to harden model calls.
3. `next-best-practices` for `cortex-ui` optimization.
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
