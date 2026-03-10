# System Architecture

See also: [Planning Index](./README.md), [Product and Demo](./01-product-and-demo.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

---
## Platform: Web App

Clypt is a **Next.js web app** deployed on **Vercel**. The primary use case is desktop (Chrome), where creators act on clip opportunities surfaced by the backend. The agent pipeline runs in TypeScript (Google ADK), triggered by Convex actions. Video synthesis uses **Remotion** (React + Tailwind CSS) for code-driven clip rendering with programmatic captions, brand-matched styling, and **Lyria**-generated audio production.

### Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Frontend** | Next.js (App Router) | SSR, real-time Convex subscriptions, Vercel AI SDK |
| **Hosting** | Vercel | Native Next.js support, preview deployments, instant CI/CD |
| **Auth** | Convex Auth + Google/YouTube OAuth | Handles YouTube OAuth scopes, 80+ providers |
| **Backend / Database** | Convex | Real-time reactive database, TypeScript-native, serverless |
| **Vector Search** | Convex Vector Search + Gemini Embeddings | Built-in vector index, 1536-dimensional embeddings (MRL-truncated from 3072d). Primary search target is now `narrativeNodes` (multimodal-bound graph nodes) rather than flat transcript chunks. |
| **File Storage** | Convex File Storage | Clips, thumbnails, music files, Lyria audio assets |
| **Scheduled Jobs** | Convex Scheduled Functions | Trend polling crons, periodic catalog scans |
| **Orchestrator Agent** | Google ADK + Gemini 3.1 Pro | Routing, coordination, sub-agent delegation |
| **Crowd Clip / Trend Trim Subagents** | Gemini 3.1 Pro | Deep reasoning: comment analysis, heatmap fusion, trend/catalog matching |
| **Content Clip Subagents** | Vertex AI STT v2 + SenseVoice + Video Intelligence API + Gemini 3.1 Pro | 3-stream pipeline: transcript (STT v2: word timestamps + speaker tags in single pass) + audio events (SenseVoice: per-second emotion + event detection) + video annotations (Video Intelligence API: shot changes, labels, objects, faces with exact ms timestamps) → Multimodal Semantic Graph (Interval Overlap Binding maps audio/video data onto transcript nodes → cross-modal Few-Shot CoT NLI narrative edge classification: contradictions, causal links, thematic continuations, tension/release arcs) → ClipScoringAgent evaluates Semantic Sub-graphs with Dynamic Dimension Weighting (tone-specific multipliers) → deterministic clip boundaries from node timestamps → hook scoring + clip ranking (Pro) |
| **Narrative Analyst** | Gemini 3.1 Pro (Few-Shot CoT NLI) + Gemini Embeddings + Convex | Transcript → semantic graph nodes in Convex → **Interval Overlap Binding** maps audio events (SenseVoice) and video annotations (Video Intelligence API) directly onto nodes by timestamp overlap → **Multimodal Semantic Graph**. Cross-modal Few-Shot CoT NLI classifies narrative edges across the full taxonomy: contradictions (comedic juxtaposition, cross-modal clashes), causal links (setup/resolution for tutorials), thematic continuations (entity tracking across tangents), and tension/release arcs (challenge/payoff for vlogs). Two-stage vector loop optimization (cosine similarity bouncer + edge de-duplication) avoids O(n²) LLM calls. ClipScoringAgent evaluates **Semantic Sub-graphs** (clusters of bound nodes) instead of arbitrary windows — clip boundaries are deterministic from node timestamps. |
| **Embeddings** | Gemini Embeddings (gemini-embedding-001) | Semantic search across multimodal graph nodes, catalog, comments |
| **Transcription + Diarization** | Vertex AI Speech-to-Text v2 (`latest_long` model) | Unified single-pass: `enableWordTimeOffsets` + `diarizationConfig` → returns `{word, startMs, endMs, speakerTag}` per token. No multi-model merging. Feeds Remotion Caption components directly. |
| **Audio Event Detection + Pacing** | SenseVoice | Per-second classification of speech emotion (happy/sad/angry/excited/neutral), laughter, cheering, silence, music, sound effects. 70ms latency per 10s of audio. Events are bound onto graph nodes via Interval Overlap Binding — beat-aware pacing is inherent in the Multimodal Semantic Graph (reactions bound to punchline nodes naturally extend clip out-points). |
| **Sound Event Detection (SED)** | Gemini 3.1 Pro + Gemini File API | Open-vocabulary SED — when comment clusters flag non-verbal cues (e.g., a specific chime or sound effect), raw audio is uploaded to Gemini File API; Gemini 3.1 Pro returns exact start/end timestamps of that acoustic signature |
| **Audio Production** | Lyria | Custom music bed generation (tone-matched to clip energy/mood via Gemini 3.1 Pro tone classification), foley/sound design (transition sounds, impacts, whooshes synced to visual cuts), audio mixing/ducking (intelligently layers music under speech using transcript word timestamps), transformational audio (reverb on punchlines, bass enhancement on reactions, audio stingers for hooks). |
| **Video Annotation** | Google Cloud Video Intelligence API | Single `annotateVideo` call on GCS-uploaded video: SHOT_CHANGE_DETECTION (exact scene cuts → Remotion edit points), LABEL_DETECTION + OBJECT_TRACKING (entities/objects/locations with temporal bounding boxes → graph node `visualContext[]`), FACE_DETECTION (face tracking → speaker identification support), EXPLICIT_CONTENT_DETECTION (safety moderation for brand deals). Structured JSON output — no LLM hallucination, no token cost. Replaces Gemini Flash 1fps frame extraction. |
| **Generative Video (B-Roll + Extension)** | Veo on Vertex AI (veo-3.1-generate-001) | B-roll generation: VeoGenerationAgent reads semantic graph nodes, Gemini writes Veo prompt → 4–8s of 1080p 9:16 footage overlaid in Remotion. Clip extension: final frame passed as `firstFrame` image prompt → Veo generates 2s of smooth continuation. Supports first/last frame conditioning for seamless transitions. |
| **YouTube Data API** | YouTube Data API v3 | Comments, video metadata, catalog scan, clip publishing — official API |
| **Engagement Velocity** | Convex Scheduled Functions + YouTube Data API | Hourly stat snapshots for time-series engagement tracking |
| **Voice Editing** | Gemini Live API + LiveKit Cloud | Real-time speech-to-speech, function calling in live sessions |
| **Thumbnail Generation** | NanoBanana 2 | SOTA image composition with character consistency (creator's likeness preserved across thumbnails), sub-pixel text rendering for perfect text overlays, 4K output with creative controls |
| **Video Synthesis** | Remotion (React + Tailwind CSS) + Lyria + Veo | Code-driven video DOM: STT v2 word timestamps → Remotion `Caption` components for programmatic text rendering; Gemini Vision design tokens injected for brand-matched styling; Lyria audio production layer (music bed + foley + mixing) composed alongside source audio; Veo B-roll overlays and clip extensions composited via Remotion sequences |
| **Brand Identity Extraction** | Gemini 3.1 Pro (Vision) + yt-dlp | Interval-based brand synthesis: downloads creator's 10 most recent Shorts, extracts frames at 2fps, Gemini Vision identifies dynamic caption patterns (active/inactive word colors, outlines, fonts) → JSON payload for `@remotion/captions`. Also processes thumbnails + long-form frames for static brand tokens (hex codes, layout). Combined payload injected into Remotion compositions. |
| **Trend Detection** | Gemini 3.1 Pro + Google Trends API | Trend analysis via structured API access and Gemini reasoning — replaces browser-based scraping with direct API + LLM analysis |
| **Semantic Graph Editor** | React Flow + Tailwind CSS + shadcn/ui + Convex real-time | Force-directed interactive graph visualization of the Multimodal Semantic Graph. Custom node renderers show claim text, node type, audio event badges, and visual annotation tags. Edges color-coded by narrative type. Node/edge CRUD operations write directly to `narrativeNodes` / `narrativeEdges` in Convex. Re-run button triggers ClipScoringAgent on the modified graph, returning updated Content PRs in real-time via Convex subscriptions. |

---

## Technical Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CLYPT ARCHITECTURE (Gemini Live Challenge Build)                       │
└──────────────────────────────────────────────────────────────────────────────┘

  Browser (Next.js on Vercel)
  ├── "Analyze Channel" input        ←── paste any YouTube channel URL
  ├── Content PR dashboard           ←── Convex live subscriptions
  ├── Trend monitor panel            ←── Convex live subscriptions
  ├── Agent activity feed            ←── Convex live subscriptions
  ├── Lyria audio preview            ←── music bed + foley preview before final render
  ├── Remotion Player preview        ←── in-browser video preview (React component)
  ├── Voice editing                  ←── Gemini Live API + LiveKit
  ├── Semantic Graph Editor          ←── React Flow interactive graph (nodes + edges + edit + re-run)
  └── Auth                           ←── Convex auth (Google/YouTube OAuth)
                │
                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  ZERO-TO-ONE ONBOARDING (runs once per channel)                  │
  │                                                                  │
  │  ── Audience-Grounded Catalog Indexing ──                        │
  │  ├── YouTube Data API: Top 10 LF + Top 10 Shorts (last year)     │
  │  ├── For each video: fetch transcript (yt-dlp) + top 15          │
  │  │   comments (Data API)                                         │
  │  ├── Gemini 3.1 Flash: deduce true tone (satire vs. serious)       │
  │  │   from transcript + audience comments → 2-sentence            │
  │  │   tone-aware summary per video                                │
  │  ├── Gemini Embeddings: vectorize summaries → Convex             │
  │  │   (tone-matched catalog for Trend Trim + few-shot RAG)        │
  │  │                                                               │
  │  ── Interval-Based Brand Synthesis ──                            │
  │  ├── yt-dlp: download creator's 10 most recent                   │
  │  │   high-performing Shorts                                      │
  │  ├── Extract frames at 2fps over 5s windows                      │
  │  ├── Gemini 3.1 Pro (Vision): sequential frame analysis →        │
  │  │   dynamic caption patterns (active/inactive word colors,      │
  │  │   outlines, fonts) + static brand tokens (hex, layout)        │
  │  │   → JSON payload for @remotion/captions                       │
  │  │                                                               │
  │                                                               │
  │  └── NOTE: full 3-stream pipeline + Multimodal Semantic Graph     │
  │     (STT v2, SenseVoice, Video Intelligence API →                 │
  │     Interval Overlap Binding) is NOT run here — deferred         │
  └──────────────────────────────────────────────────────────────────┘
                │
                ▼
  Convex Backend (TypeScript)
  ├── Database (all tables — see Data Model)
  ├── Vector search (Gemini Embeddings) — catalog, multimodal narrative graph nodes, comments
  ├── File storage (clips, thumbnails, Lyria audio assets)
  ├── Scheduled functions (trend crons)
  ├── Auth (Google/YouTube OAuth)
  ├── Actions:
  │   ├── Trigger TypeScript agent pipeline via HTTP
  │   ├── Trigger Remotion video synthesis (bundled React compositions)
  │   ├── Trigger Lyria audio production (music bed + foley + mixing)
  │   ├── yt-dlp heatmap + frame extraction
  │   ├── YouTube Data API calls (comments, metadata, publishing)
  │   ├── Gemini File API uploads (raw audio for SED)
  │   ├── Video Intelligence API calls (GCS upload → annotateVideo)
  │   ├── Veo API calls (B-roll generation, clip extension)
  │   └── Engagement velocity polling (cron → stats snapshots)
  └── HTTP endpoints (webhook handlers)
                │
                ▼
  TypeScript Agent Pipeline (Google ADK)
  │
  ├── ORCHESTRATOR (Gemini 3.1 Pro via Google ADK LlmAgent)
  │   ├── get_creator_context      (Convex query)
  │   ├── route_to_agent           (transfer_to_agent)
  │   ├── notify_creator           (email notification)
  │   ├── update_content_pr        (Convex mutation)
  │   ├── trigger_video_synthesis  (Convex action → Remotion render)
  │   └── trigger_audio_production (Convex action → Lyria music bed + foley + mixing)
  │
  │  ──── CROWD CLIP (Feature 2) ────
  │
  ├── [CommentIngestionAgent] — YouTube Data API + Gemini Embeddings + Gemini 3.1 Pro
  │   ├── fetch_comments           (YouTube Data API: commentThreads.list, sorted by relevance/likes)
  │   ├── cluster_comments         (Gemini Embeddings → agglomerative clustering by conceptual proximity)
  │   ├── trace_clusters_to_transcript (Lexical Chains: map comment clusters → transcript segments, correcting for viewer reaction lag)
  │   ├── detect_sound_events      (when cluster flags non-verbal cue → upload raw audio to Gemini File API → Gemini 3.1 Pro returns exact timestamps of acoustic signature)
  │   └── extract_emotional_markers (sentiment on emojis, caps, reactions)
  │
  ├── [HeatmapAgent] — yt-dlp / InnerTube  │   ├── extract_most_replayed    (yt-dlp --dump-json → .heatmap)
  │   └── return 100-segment replay intensity array
  │
  ├── [EngagementVelocityAgent] — Convex cron + YouTube Data API
  │   ├── poll_video_stats         (Convex cron: YouTube Data API videos.list every 30-60 min)
  │   ├── store_snapshot           (save viewCount, likeCount, commentCount with timestamp)
  │   └── compute_deltas           (views gained/hr, like acceleration, comment velocity)
  │
  ├── [EngagementAnalysisAgent] — Gemini 3.1 Pro
  │   ├── fuse_signals             (comment clusters + heatmap + velocity + metadata → unified scoring)
  │   ├── resolve_vague_references (embed "the bread moment" → timestamp via vector search)
  │   ├── rank_clip_candidates     (convergent signal: heatmap peaks + comment clusters + velocity spikes)
  │   └── return top clip windows with reasoning + evidence
  │
  │  ──── TREND TRIM (Feature 3) ────
  │
  ├── [TrendScoutAgent] — Gemini 3.1 Pro + Google Trends API
  │   ├── query_google_trends      (Google Trends API: extract trending topics with spike data)
  │   ├── analyze_youtube_trending (YouTube Data API: trending videos in relevant categories)
  │   └── analyze_social_signals   (Gemini 3.1 Pro: reason over aggregated social trend data)
  │
  ├── [CatalogMatchAgent] — Gemini 3.1 Pro
  │   ├── load_creator_catalog     (Convex query — pre-vectorized during onboarding)
  │   ├── embed_and_search         (Gemini Embeddings + Convex vector search)
  │   └── reason_about_relevance   (long-context reasoning over catalog)
  │
  │  ──── CONTENT CLIP (Feature 1) ────
  │
  ├── [TranscriptAgent] — Vertex AI STT v2 (`latest_long` model, unified single-pass)
  │   └── transcribe_and_diarize   (BatchRecognize with `enableWordTimeOffsets` + `diarizationConfig` → unified `{word, startMs, endMs, speakerTag}` per token → parsed directly into Remotion Caption data structures)
  │
  ├── [AudioEventAgent] — SenseVoice
  │   ├── detect_audio_events      (per-second: speech emotion, laughter, cheering, silence, music, sound effects — native waveform processing, 70ms/10s)
  │   └── build_emotion_timeline   (returns [{t, event, emotion, confidence}] array for full video)
  │
  ├── [VideoAnnotationAgent] — Google Cloud Video Intelligence API
  │   ├── upload_to_gcs            (upload video to Google Cloud Storage)
  │   ├── annotate_video           (single annotateVideo call: SHOT_CHANGE_DETECTION + LABEL_DETECTION + OBJECT_TRACKING + FACE_DETECTION + EXPLICIT_CONTENT_DETECTION)
  │   └── parse_annotations        (structured JSON → temporal bounding boxes for objects, faces, labels, shot changes with exact ms timestamps)
  │
  ├── [NarrativeAnalystAgent] — Gemini 3.1 Pro (Few-Shot CoT NLI) + Gemini Embeddings + Convex
  │   ├── build_semantic_graph     (transcript → semantic nodes stored in Convex; claims, assertions, emotional beats)
  │   ├── bind_multimodal_context  (Interval Overlap Binding: maps AudioEventAgent (SenseVoice) + VideoAnnotationAgent (Video Intelligence API) outputs onto graph nodes by timestamp overlap → each node gets `visualContext[]` + `audioContext[]` arrays bound directly to it)
  │   ├── classify_narrative_edges  (Cross-modal Few-Shot CoT NLI: classifies narrative relationships — contradictions, causal links, thematic continuations, tension/release arcs — using text + bound audio/visual context)
  │   ├── link_cross_segment_pairs (two-stage vector loop: cosine similarity bouncer + edge de-duplication → only surviving pairs sent to LLM for edge classification)
  │   └── surface_nonlinear_highlights (orchestrator can synthesize non-linear clips from linked narrative edge pairs — contradiction compilations, tutorial montages, tension/release supercuts)
  │
  ├── [ClipScoringAgent] — Gemini 3.1 Pro   ← receives Semantic Sub-graphs (clusters of multimodal-bound nodes)
  │   ├── score_candidates         (evaluates Semantic Sub-graphs: Hook/Flow/Emotion/Visual/Speaker coherence + Narrative Edge scoring per node cluster — all multimodal data is bound directly to graph nodes, not resolved by timestamp heuristics)
  │   ├── evaluate_hooks           (score first 3s of each candidate: question → 10pts, surprise/counterintuitive → 9pts, emotional exclamation → 8pts, plain statement → 3pts)
  │   ├── set_clip_boundaries      (deterministic: in-point = `startMs` of first node in sub-graph, out-point = max(`endMs`) of audio/visual reactions bound to last node — no guessing)
  │   ├── suggest_alternative_inpoint (if hook score ≤ 4, find better in-point within ±15s using graph traversal)
  │   ├── synthesize_clip          (Remotion + Lyria: compose video DOM with brand-matched design tokens, Caption components from STT v2 timestamps, Lyria audio production layer)
  │   └── create_content_pr        (saves ContentPR with hookScore, alternativeInPoint, hookReason, narrative edge evidence, bound multimodal evidence per node)
  │
  │  ──── PROMPT-BASED CLIPPING (Cortex) ────
  │
  ├── [PromptClipAgent] — Gemini 3.1 Pro
  │   ├── parse_prompt             (interpret natural language query: topic, person, emotion, action, object, or moment type)
  │   ├── search_narrative_graph   (Gemini Embeddings k-NN search against `narrativeNodes` — nodes contain bound audio/visual context, so "find where he argues about privacy and people laugh" becomes a single semantic query against multimodal-bound nodes)
  │   ├── expand_to_subgraph       (from matched nodes, traverse graph edges to build complete Semantic Sub-graphs spanning setup → reaction)
  │   ├── score_and_rank           (ClipScoringAgent: evaluates Semantic Sub-graphs with deterministic clip boundaries from node timestamps)
  │   └── create_content_pr        (saves ContentPR with promptQuery field, matched multimodal evidence from bound nodes, hookScore)
  │
  │  ──── CREATIVE GENERATION ────
  │
  ├── [BrandIdentityAgent] — Gemini 3.1 Pro (Vision) + yt-dlp
  │   ├── download_creator_shorts  (yt-dlp: 10 most recent high-performing Shorts)
  │   ├── extract_caption_patterns (2fps frame extraction → Gemini Vision sequential analysis: active/inactive word colors, outlines, fonts → dynamic caption JSON for @remotion/captions)
  │   ├── extract_static_tokens    (top 5 thumbnails + long-form frames → hex codes, layout preferences)
  │   └── inject_design_tokens     (combined dynamic + static JSON payload → Remotion composition props)
  │
  ├── [AudioProductionAgent] — Lyria + Gemini 3.1 Pro
  │   ├── classify_clip_mood       (Gemini 3.1 Pro: analyze clip content + predicted_tone → mood/energy profile for Lyria)
  │   ├── generate_music_bed       (Lyria: custom music track matching clip mood, energy, and duration)
  │   ├── generate_foley           (Lyria: transition sounds, impacts, whooshes synced to visual cut points from Remotion composition)
  │   ├── generate_audio_stingers  (Lyria: hook stingers for first 3s, punchline accents, reaction enhancers — timed to graph node boundaries)
  │   ├── mix_audio                (Lyria: intelligent ducking — music dips under speech using STT v2 word timestamps, rises during visual-only moments)
  │   └── store_audio_assets       (Convex file storage: music bed, foley tracks, mixed master → linked to Remotion composition)
  │
  ├── [VeoGenerationAgent] — Veo on Vertex AI (veo-3.1-generate-001) + Gemini 3.1 Pro
  │   ├── detect_broll_opportunities (Gemini 3.1 Pro: scan semantic graph nodes for abstract/conceptual claims that benefit from visual illustration)
  │   ├── generate_broll            (Gemini 3.1 Pro writes Veo prompt from graph node → Veo generates 4–8s of 1080p 9:16 B-roll footage)
  │   ├── extend_clip               (when clip ends abruptly: pass final frame as `firstFrame` image prompt → Veo generates 2s of smooth continuation)
  │   └── store_generated_video     (Convex file storage: B-roll clips, extensions → linked to Remotion composition for overlay/sequencing)
  │
  ├── [ThumbnailAgent] — NanoBanana 2
  │   ├── generate_thumbnail       (clip keyframe + trend context → 4K thumbnail with character consistency + sub-pixel text)
  │   └── store in Convex file storage
  │
  │  ──── PUBLISHING & OPTIMIZATION ────
  │
  ├── [PublishAgent] — YouTube Data API + Gemini 3.1 Pro + Convex vector search
  │   ├── generate_clip_summary    (1-sentence summary of clip content)
  │   ├── retrieve_style_exemplars (Convex vector search: top 3 most semantically similar past videos from creator's catalog)
  │   ├── generate_metadata        (few-shot prompt: inject retrieved titles/descriptions as style exemplars → Gemini 3.1 Pro mimics creator's exact formatting — e.g., all lowercase, specific humor)
  │   ├── upload_short             (YouTube Data API: videos.insert — upload Remotion-rendered clip as YouTube Short)
  │   ├── set_metadata             (YouTube Data API: title, description, tags — creator-style-matched)
  │   ├── set_thumbnail            (YouTube Data API: thumbnails.set — upload NanoBanana 2 thumbnail)
  │   └── return YouTube URL
  │
```

### Key Architectural Decisions

**Parent orchestrator stays lean.** The orchestrator (Gemini 3.1 Pro via Google ADK) routes and coordinates: it loads creator context, decides which mode to trigger, and calls `notify_creator`. Deep work is delegated to subagents. Those tasks would flood the orchestrator's context window. It delegates and waits for clean results.

**Subagents use the ADK Agent-as-a-Tool pattern.** From the orchestrator's perspective, calling `engagement_analysis_agent(video_id, creator_id)` looks like any other tool call, but internally that subagent spins up its own Gemini 3.1 Pro reasoning loop, runs its full pipeline with its own tool set, and returns a structured summary. The orchestrator never sees intermediate steps.

**3-stream pipeline → Multimodal Semantic Graph: specialized models, deterministic binding, not one monolithic call.** TranscriptAgent (Vertex AI STT v2 `latest_long` — unified word timestamps + speaker diarization in a single pass → Remotion Captions), AudioEventAgent (SenseVoice — 70ms latency, per-second emotion + event detection on raw waveform), and VideoAnnotationAgent (Google Cloud Video Intelligence API — structured temporal annotations with exact ms timestamps, zero hallucination) each produce structured, timestamped artifacts. NarrativeAnalystAgent (Gemini 3.1 Pro with Few-Shot CoT NLI prompting) then builds a semantic graph from the transcript and executes **Interval Overlap Binding** — a Convex mutation that deterministically maps audio events and visual descriptions directly onto graph nodes by timestamp overlap. Each node physically contains `visualContext[]` and `audioContext[]` arrays, eliminating the fuzzy timestamp heuristics that plague flat multi-stream systems (where a laughter event at `1:06` might be incorrectly attributed to a joke at `1:02`). This enables **cross-modal narrative edge classification** (contradictions where text claims are contradicted by visual evidence, causal links connecting setup/resolution nodes, thematic continuations tracking entities across tangents, and tension/release arcs spanning challenge/payoff sequences — all within the same bound graph structure) and **Semantic Sub-graph evaluation** (ClipScoringAgent receives clusters of bound nodes instead of arbitrary temporal windows — clip boundaries are deterministic from the `startMs` of the first node to the `endMs` of the last reaction). This surpasses OpusClip's approach: where OpusClip uses LLM frame extraction on flat temporal windows, Clypt uses purpose-built Video Intelligence API for zero-hallucination structured annotations bound to graph nodes — every clip boundary, every reaction attribution, and every narrative edge classification is mathematically exact.

**Onboarding is deep but targeted; full 3-stream processing is deferred.** When a channel URL is first pasted, the Zero-to-One Onboarding phase runs two passes. First, **Audience-Grounded Catalog Indexing**: the system pulls the **Top 10 most viewed long-form videos** and **Top 10 most viewed Shorts** from the last year. For each, it fetches the raw transcript (yt-dlp) and top 15 comments (YouTube Data API), then `Gemini 3.1 Flash` (not Pro — speed over depth here) processes both together to produce a tone-aware 2-sentence summary. These summaries are embedded and stored in Convex — ensuring downstream few-shot RAG retrieves tone-matched exemplars. Second, **Interval-Based Brand Synthesis**: yt-dlp downloads the creator's 10 most recent high-performing Shorts, extracts frames at 2fps, and Gemini Vision identifies dynamic caption patterns for `@remotion/captions`. The heavy 3-stream pipeline (STT v2 word-level timestamps + diarization, SenseVoice audio events, Video Intelligence API annotations) is deferred until a specific video is flagged by a trend, surfaced by engagement signals, or manually requested. Once run, artifacts are cached and reused across all modes.

**Lyria audio production is a first-class pipeline stage, not an afterthought.** After ClipScoringAgent determines clip boundaries and the Remotion visual composition is assembled, AudioProductionAgent (Lyria) generates a complete audio layer: a tone-matched music bed (classified by Gemini 3.1 Pro from the clip's `predicted_tone` + narrative content), foley effects synced to visual cuts from the Remotion timeline, audio stingers timed to graph node boundaries (hook accents, punchline enhancers), and intelligent audio ducking using STT v2 word timestamps to dip music under speech. This produces broadcast-quality Short-form audio that was impossible six months ago — fully automated, tone-aware, and precisely synchronized to the clip's narrative structure.

**Parallel execution for multi-creator trend events.** When a trend fires affecting multiple creators, the orchestrator (via ADK's `ParallelAgent`) spawns concurrent `catalog_match_agent` subagents per creator. This would be impossible with a single-agent tool loop.

**Clip boundaries are deterministic from the Multimodal Semantic Graph, not guessed from timestamp proximity.** After Interval Overlap Binding, every audio/visual event is physically attached to the graph node it belongs to. ClipScoringAgent evaluates Semantic Sub-graphs — clusters of connected nodes spanning from a setup to a reaction. The clip in-point is the `startMs` of the first node in the sub-graph; the out-point is the max `endMs` of the audio/visual reactions bound to the last node. Because reactions (laughter, visual gestures) are already bound to nodes, beat-aware pacing is inherent: the laughter event bound to the punchline node naturally extends the out-point without separate heuristic logic. Word-level timestamps from STT v2 still snap the in-point to the nearest word boundary for clean cuts. These same deterministic boundaries also drive Lyria's audio production: music bed duration, foley sync points, and audio stinger timing are all derived directly from graph node timestamps.

**Dynamic Dimension Weighting makes clip scoring context-aware, not one-size-fits-all.** Before ClipScoringAgent runs, the Orchestrator classifies the video's `predicted_tone` (satire, technical_tutorial, vlog, interview, reaction) by evaluating its metadata against the creator's profile (tone distributions established during onboarding). ClipScoringAgent's system prompt then applies tone-specific mathematical multipliers to its 6 scoring dimensions — e.g., Narrative Edge × 1.2 + Emotional Peak × 1.2 for satire (favoring contradictions and laughter), Narrative Edge × 1.2 + Flow × 1.2 for tutorials (favoring causal setup/resolution chains and step completeness), Narrative Edge × 1.2 + Emotional Peak × 1.2 for vlogs (favoring tension/release arcs and emotional resonance). Raw scores are still computed for all 6 dimensions; the multipliers only adjust the weighted composite ranking. This makes the architecture genuinely personalized: a satirical skit surfaces contradictions, a coding tutorial surfaces causal setup/resolution chains, a challenge vlog surfaces tension/release arcs — without brittle hardcoded rules or manual per-creator configuration.

**Hook scoring is mandatory for every clip regardless of mode.** Whether a clip is found via heatmap (Crowd Clip), trend (Trend Trim), or content analysis (Content Clip), the ClipScoringAgent always evaluates the first 3 seconds and either confirms the hook or suggests a better in-point. This is the single highest-leverage quality improvement over a naive approach.

**Most Replayed heatmap replaces YouTube Analytics API retention curves for the demo.** The heatmap is real per-segment engagement data (replay intensity, 100 segments) extractable from any public video with ~50K+ views. Combined with comment timestamps and the audio emotion timeline from SenseVoice, this provides a robust multi-signal confirmation for clip selection without requiring channel owner authentication.

**Video synthesis uses Remotion (React) + Lyria audio production, not FFmpeg.** Remotion constructs the video DOM programmatically using React + Tailwind CSS. STT v2 word timestamps are parsed into Remotion `Caption` components for dynamic text rendering. Brand identity tokens (dynamic caption patterns + static brand tokens) extracted by Gemini Vision are injected as composition props, ensuring every clip visually matches the creator's brand. The source video's native audio is preserved as the primary track — Lyria generates a complementary audio layer (music bed, foley, audio stingers) that is intelligently mixed underneath using STT v2 word timestamps for precise ducking. This is the opposite of "AI slop" — Lyria's audio production respects the source audio's timing (comedic silences, reaction beats) while adding broadcast-quality production value that creators would otherwise spend hours producing manually. For non-linear narrative edge clips (contradiction compilations, tutorial problem/solution montages, tension/release supercuts), Remotion composites multiple source segments into a single coherent video with Lyria-generated transition audio.

**Convex real-time subscriptions power the dashboard.** Every database write (new Content PR, trend alert, agent status update) instantly pushes to all connected clients via WebSocket. No polling, no manual WebSocket plumbing. This is a demo-quality improvement — judges see data appear live as agents work.


---

