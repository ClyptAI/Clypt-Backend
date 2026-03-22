# Clypt Balanced Hybrid Hackathon Stack Design

Date: 2026-03-21
Project: Clypt-V2
Status: Approved in chat, pending spec review and user review

## Summary

For the Multimodal Frontier Hackathon, Clypt should keep the product behavior and differentiators described in [Clypt-V2.html](/Users/rithvik/CascadeProjects/Clypt-V2/docs/planning/Clypt-V2.html) while reshaping the stack into a sponsor-aligned balanced hybrid:

- Keep Google DeepMind / GCP central for multimodal reasoning, semantic graph construction, embeddings, and existing graph storage.
- Replace Modal with a DigitalOcean GPU Droplet-hosted Phase 1 extraction service.
- Use Lovable as the primary path for the creator-facing UI refresh, especially around Cortex and demo flows.
- Only add secondary sponsors if they improve an existing product capability instead of forcing extra scope.

The target story is:

`DigitalOcean runs deterministic multimodal extraction, DeepMind powers reasoning and embeddings, and Lovable powers the creator interface.`

## Product Constraints

The product surface must remain consistent with the current spec:

- Content Clip, Crowd Clip, Trend Trim, and Cortex remain core user-facing features.
- Deterministic object tracking, speaker-aware rendering, semantic graph reasoning, and editable graph logic remain intact.
- The system must still support graph-backed clip reasoning, retrieval, and Remotion rendering.
- Any stack changes must preserve demo reliability for a hackathon setting.

## Goals

- Strengthen sponsor alignment without degrading the product.
- Make DigitalOcean a real infrastructure dependency, not a cosmetic integration.
- Make Lovable part of a visible product improvement, not a side artifact.
- Preserve the current strongest technical moat: multimodal reasoning, graph semantics, and deterministic extraction quality.
- Avoid a rewrite of storage and retrieval infrastructure that would consume hackathon time without strengthening the demo.

## Non-Goals

- Replatform the full backend onto DigitalOcean during the hackathon.
- Replace Gemini / Vertex AI reasoning and embeddings.
- Replace Spanner property graph storage or GCS artifact storage during the hackathon.
- Force WorkOS, Nexla, Vori, Railtracks, or similar sponsors into the product without a clear functional benefit.

## Recommended Architecture

### 1. Frontend and User Experience

Lovable should be used to rebuild or significantly improve the creator-facing app shell and Cortex experience.

Scope:

- Improve the Cortex UI flow and demoability.
- Improve onboarding and graph interaction surfaces if time allows.
- Preserve the core feature set and interactions promised by the current spec.

Preferred implementation posture:

- Treat Lovable as an accelerator for the Cortex-facing app shell and product flows, not as a requirement to do a risky full rewrite from scratch.
- A Lovable-assisted rebuild of the primary surfaces or a Lovable-assisted refactor of the current UI are both acceptable outcomes.
- The team should choose the path that produces the strongest visible product result with the least hackathon risk.

Reasoning:

- Cortex is one of Clypt's most differentiated features.
- Using Lovable on the primary user-facing surface is more credible than using it for a landing page or disposable prototype.

### 2. Phase 1 Extraction on DigitalOcean

The current Modal-based Phase 1 extraction stack should move to a DigitalOcean GPU Droplet-hosted service.

Scope:

- ASR
- Object tracking
- Face identity stabilization
- Active speaker binding
- Related extraction orchestration and artifact emission

Recommended runtime shape:

- A GPU Droplet-hosted extraction backend with explicit job lifecycle management.
- The orchestrator submits work and receives either a synchronous result for short tasks or a job handle for longer-running tasks.
- Artifacts are written to durable storage and returned through a canonical extraction manifest.

Reasoning:

- This is the cleanest and most meaningful sponsor-driven migration in the current stack.
- Modal is infrastructure, not product identity.
- A GPU Droplet approach is closer to the current worker model than a larger platform rewrite.

### 3. Keep GCP for Intelligence and Graph Memory

The following should remain on the current GCP-aligned path:

- Gemini-based node generation
- Gemini-based edge generation
- Gemini multimodal embeddings
- Spanner graph and vector-backed retrieval
- GCS-backed media and tracking artifacts

Affected scripts include:

- [phase_2a_make_nodes.py](/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase_2a_make_nodes.py)
- [phase_2b_draw_edges.py](/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase_2b_draw_edges.py)
- [phase_3_multimodal_embeddings.py](/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase_3_multimodal_embeddings.py)
- [phase_4_store_graph.py](/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase_4_store_graph.py)
- [phase_5_retrieve.py](/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase_5_retrieve.py)

Reasoning:

- These pieces already align strongly with the DeepMind/GCP story.
- Replacing them would be high-risk and would not materially improve the product during the hackathon.
- Spanner + GCS are already embedded in the current graph and retrieval design.

## Migration Boundary

The preferred migration boundary is:

- Move Phase 1 extraction off Modal and onto DigitalOcean.
- Keep downstream graph, retrieval, and rendering behavior intact.
- Allow the exact handoff artifact format to evolve if a cleaner, infrastructure-agnostic contract improves the system.

This means the team should stop thinking in terms of "Modal outputs" and instead define a versioned "Phase 1 extraction contract."

## Phase 1 Extraction Contract

The DigitalOcean service should expose a canonical Phase 1 contract that is infrastructure-agnostic.

Required semantic outputs:

- Transcript artifact
- Visual / tracking artifact
- Optional NDJSON or event-stream artifact if still useful downstream
- Canonical source video reference
- Extraction metadata
- Optional quality / rollout metrics
- Stable storage URIs for downstream consumption

Preferred shape:

- A single manifest object describing the full Phase 1 run and all produced artifacts.
- Downstream phases consume the manifest or a thin adapter around it, rather than assuming Modal-specific file conventions.

Minimum manifest fields:

- `contract_version`
- `job_id`
- `status`
- `source_video`
- `artifacts.transcript`
- `artifacts.visual_tracking`
- `artifacts.events` if produced
- `metadata.runtime`
- `metadata.timings`
- `metadata.quality_metrics` if produced

This contract should preserve what the rest of the app needs, not necessarily how Modal happened to package it.

## Sponsor Evaluation

### Primary Sponsors

#### Google DeepMind / GCP

Keep central.

Why:

- Already core to multimodal reasoning quality.
- Already deeply integrated in the graph-building and embedding pipeline.
- Strongest fit for the product's intelligence layer.

#### Lovable

Use visibly in the creator-facing frontend.

Why:

- Strong fit for the demo surface.
- Best used on Cortex or the broader app shell.
- Helps the judges see a sponsor contribution directly in the product.

#### DigitalOcean

Use as the replacement for Modal-hosted extraction.

Why:

- Real infrastructure ownership.
- Strong sponsor-story value without forcing non-essential changes.
- Operationally coherent with the current Phase 1 workload.

### Secondary Sponsors

#### Assistant UI

Optional but strong if used to make Cortex's prompt/voice editing or AI copilot workflow more concrete.

Good fit if:

- The team wants a stronger chat-based editing or reasoning interface.

Not required if:

- The UI rewrite already has enough scope with Lovable alone.

#### Unkey

Optional and useful if the team exposes protected internal APIs, public demo APIs, or rate-limited endpoints.

Good fit if:

- The team wants clean API gating around retrieval, clip generation, or graph-editing actions.

Not required if:

- The demo remains mostly internal or single-user.

#### WorkOS

Not recommended for the hackathon unless org-aware auth becomes real product scope.

Why not now:

- No strong natural dependency in the current spec.
- Would likely be forced rather than product-strengthening.

#### Other Sponsors

Nexla, Vori, Railtracks, and similar sponsors are not natural fits for the current Clypt scope and should be skipped unless the product direction changes substantially.

## Why Not a Full DigitalOcean Reshuffle

A full-stack migration off GCP is not recommended for the hackathon.

Reasons:

- Spanner property graph usage is a real architectural dependency, not a disposable implementation detail.
- GCS-backed tracking/media artifact flow is already wired into the existing render and retrieval path.
- Rebuilding graph storage, vector retrieval, and data contracts would consume time that should go toward demo reliability and product polish.
- The sponsor story is already strong without forcing storage and retrieval migrations.

## Operational Model

Target flow:

1. User submits a YouTube URL or source media.
2. The orchestrator sends Phase 1 work to the DigitalOcean extraction service.
3. The extraction service runs deterministic multimodal extraction on a GPU Droplet.
4. The extraction service emits a canonical Phase 1 manifest and associated artifacts.
5. Existing Gemini / Vertex-backed downstream phases produce nodes, edges, and embeddings.
6. Existing Spanner + GCS-backed storage and retrieval continue serving Cortex and rendering.
7. Lovable-built or Lovable-assisted frontend presents the creator-facing experience.

## Risks

### 1. GPU Environment and Dependency Parity

The current worker stack may depend on runtime assumptions that were previously hidden by Modal.

Impact:

- Build and deployment friction on DigitalOcean.

Mitigation:

- Treat the DO service as a clean runtime target with explicit environment setup and smoke tests.

### 2. Long-Running Job Reliability

Phase 1 workloads may exceed comfortable synchronous request lifetimes.

Impact:

- Timeouts, lost progress, or weak operational visibility.

Mitigation:

- Design around explicit job lifecycle, resumability, and artifact manifests.

### 3. Contract Drift

Changing Phase 1 outputs without a stable interface could break downstream phases.

Impact:

- Hidden regressions in graph-building or rendering.

Mitigation:

- Define and version the extraction contract intentionally.
- Update downstream once against the contract instead of allowing ad hoc assumptions.

### 4. Hackathon Scope Creep

Optional sponsor integrations can distract from the core migration.

Impact:

- Too many moving parts and a weaker final demo.

Mitigation:

- Prioritize DeepMind + Lovable + DigitalOcean.
- Only add secondary sponsors if they strengthen an already-planned feature.

## Success Criteria

- Phase 1 runs end-to-end on DigitalOcean instead of Modal.
- The core features in [Clypt-V2.html](/Users/rithvik/CascadeProjects/Clypt-V2/docs/planning/Clypt-V2.html) still work as promised.
- The Cortex / creator-facing frontend is visibly improved and credibly associated with Lovable.
- The architecture can be explained simply:
  - DigitalOcean powers extraction.
  - DeepMind powers multimodal intelligence.
  - Lovable powers the creator experience.
- The team avoids unnecessary storage or graph migrations during the hackathon.

## Deferred Work

These should be explicitly deferred until after the hackathon unless new evidence makes them necessary:

- Replacing Spanner
- Replacing GCS
- Rebuilding retrieval around a new vector / graph stack
- Adding WorkOS just to have enterprise auth in the story
- Adding secondary sponsors with no direct product benefit

## Final Recommendation

Proceed with the balanced hybrid design:

- Lovable for the frontend / Cortex experience
- DigitalOcean GPU Droplet-hosted Phase 1 extraction service instead of Modal
- DeepMind / GCP retained for semantic reasoning, embeddings, graph storage, and downstream intelligence

This is the highest-probability path to a credible, sponsor-aligned, technically strong hackathon demo.
