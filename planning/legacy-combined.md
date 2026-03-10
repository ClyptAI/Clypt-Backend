## Multimodal Integration Strategy

**Event Context:** Gemini Live Agent Challenge (Devpost)
**Challenge:** [Gemini Live Agent Challenge: Redefining Interaction: From Static Chatbots to Immersive Experiences](https://geminiliveagentchallenge.devpost.com/)
**Format:** Online challenge submission + judging demo narrative
**Theme:** Full Multimodal Stack — native audio, real-time video, high-fidelity image generation. Build something that wasn't possible six months ago.

**Challenge-Relevant Tech:**
- **Gemini Live API** — mandatory real-time multimodal interaction layer
- **Google ADK (Agent Development Kit)** — mandatory option for agent orchestration
- **Gemini multimodal + interleaved output capabilities** — required across challenge tracks
- **Google Cloud deployment** — required for backend hosting and judging proof

---

### Overview

| # | Feature | What It Does | Technology |
|---|---------|-------------|------------|
| 1 | **Content Clip** | Clips from pure content/metadata analysis + aggregate channel intelligence via Multimodal Semantic Graph | **Vertex AI STT v2 + SenseVoice + Video Intelligence API + Gemini 3.1 Pro + Lyria + Veo + Remotion** |
| 2 | **Crowd Clip** | Surfaces clips from audience signals — comments, Most Replayed heatmaps, engagement velocity over time | **YouTube Data API + yt-dlp + Gemini 3.1 Pro + Lyria** |
| 3 | **Trend Trim** | Detects rising trends, matches to creator's back catalog, generates trend-relevant clips autonomously | **Gemini 3.1 Pro + Gemini 3.1 Flash** |
| — | **Cortex** (cross-cutting) | Visual Semantic Graph Editor — creators see, inspect, and edit the AI's reasoning graph. Includes prompt-based clip search, voice editing, humor decomposition visualization, audience signal overlays, and trend alignment indicators. | **Gemini 3.1 Pro + Gemini Live API + LiveKit + React Flow + Convex** |
| — | **Audio Production** (cross-cutting) | AI-generated music beds, foley/sound design, and audio mixing for every clip | **Lyria + Gemini 3.1 Pro** |

### Models & Services Utilized

```
Gemini 3.1 Pro          →  Orchestrator routing + deep reasoning subagents — Content Clip, Crowd Clip, Trend Trim, Prompt (clip scoring/reasoning, engagement fusion, trend matching, prompt-based search). NarrativeAnalystAgent builds a Multimodal Semantic Graph: transcript → semantic nodes, then Interval Overlap Binding deterministically maps audio events (SenseVoice) and video annotations (Video Intelligence API) onto graph nodes by timestamp overlap → cross-modal Few-Shot CoT NLI narrative edge classification (contradictions, causal links, thematic continuations, tension/release arcs — no fine-tuning, prompt-driven). ClipScoringAgent evaluates Semantic Sub-graphs (clusters of bound nodes) instead of arbitrary temporal windows. Few-shot metadata generation via tone-matched RAG. Also handles open-vocabulary Sound Event Detection (SED) — when comment clusters flag non-verbal cues, raw audio uploaded to Gemini File API → returns exact timestamps of specific acoustic signatures.
Gemini 3.1 Pro (Vision) →  Interval-based brand synthesis — sequential frame analysis of creator's Shorts (2fps) to identify dynamic caption patterns (active/inactive word colors, outlines, fonts) → JSON for @remotion/captions. Also processes thumbnails + long-form frames for static brand tokens.
Gemini 3.1 Flash      →  Audience-grounded summarization during onboarding (transcript + top comments → tone-aware summaries; speed over depth)
Gemini Live API       →  Voice-first editing interface (real-time speech-to-speech via LiveKit)
Gemini Embeddings     →  Semantic vector indexing across narrative graph nodes, audience-grounded summaries, comments (gemini-embedding-001, 1536d via MRL truncation). Powers comment semantic clustering, Multimodal Semantic Graph cross-segment narrative edge classification (cosine similarity bouncer for O(n²) avoidance), prompt-based clip search against bound graph nodes, tone-matched catalog RAG for few-shot prompting.
Vertex AI STT v2       →  Unified transcription + diarization in a single pass (`latest_long` model via BatchRecognize with `enableWordTimeOffsets` + `diarizationConfig`). Returns `{word, startMs, endMs, speakerTag}` per token → parsed directly into Remotion Caption data structures. No multi-model merging.
Video Intelligence API →  Purpose-built temporal video understanding. Single `annotateVideo` call on GCS-uploaded video returns structured JSON with exact millisecond timestamps: SHOT_CHANGE_DETECTION (mathematically exact scene cuts for Remotion editing), LABEL_DETECTION & OBJECT_TRACKING (entities, objects, locations with temporal bounding boxes), FACE_DETECTION (face tracking with bounding boxes), EXPLICIT_CONTENT_DETECTION (safety moderation for outreach/brand deals). Replaces Gemini Flash 1fps frame extraction — eliminates LLM hallucination risk and token cost. Output maps directly into Multimodal Semantic Graph via Interval Overlap Binding.
SenseVoice            →  Audio event + emotion detection — per-second classification of speech emotion (happy/sad/angry/excited/neutral), laughter, cheering, silence, music, sound effects. Events are bound directly onto Multimodal Semantic Graph nodes via Interval Overlap Binding → beat-aware clip pacing is inherent (laughter/silence bound to the punchline node naturally extends clip out-points). Runs natively on raw waveform — 70ms latency per 10s of audio.
Lyria                 →  DeepMind's audio generation model — custom music bed generation (tone-matched to clip energy/mood), foley and sound design (transition sounds, impacts, whooshes synced to visual cuts), audio mixing/ducking (intelligently layers music under speech), transformational audio (reverb on punchlines, bass enhancement on reaction moments, audio stingers for hook points). Every clip gets a complete audio production layer.
Veo (Vertex AI)       →  DeepMind's video generation model (veo-3.1-generate-001). B-roll generation: reads semantic graph nodes, generates 4–8s of 1080p 9:16 footage from text/image prompts to overlay in Remotion (e.g., creator says "imagine a cyberpunk city" → Veo generates matching B-roll). Clip extension: passes the final frame of a clip as `firstFrame` image prompt → Veo generates 2s of smooth continuation for breathing room. Supports first/last frame conditioning for seamless transitions.
Remotion              →  Code-driven video synthesis framework — React + Tailwind CSS video DOM, programmatic text rendering via Caption components, brand-matched styling via injected design tokens, Lyria audio layer integration, Veo B-roll overlay compositing. Replaces FFmpeg for clip rendering.
NanoBanana 2          →  Thumbnail generation for clips — SOTA image composition, character consistency (creator's likeness preserved across compositions), sub-pixel text rendering for perfect text overlays, 4K output with creative controls.
YouTube Data API v3   →  Comments (commentThreads.list), video metadata (videos.list), clip publishing (videos.insert), channel catalog sweep (search.list) — official API, no scraping needed
```

---

## The 3 Core Features

| # | Feature | What It Does |
|---|---------|-------------|
| 1 | **Content Clip** | Clips from pure content/metadata analysis + aggregate channel intelligence (audience preferences, popular topics) |
| 2 | **Crowd Clip** | Surfaces clips from audience signals — comments, Most Replayed heatmaps, engagement velocity snapshots over time |
| 3 | **Trend Trim** | Detects rising trends, matches to creator's back catalog, generates trend-relevant clips autonomously |

**Cross-cutting: Cortex (Visual Semantic Graph Editor + Prompt-Based Clipping + Voice Editing)** — Cortex is the interactive visualization where creators see and edit the AI's reasoning graph (nodes, edges, narrative relationships, audience signals, trend alignment). Available in the web editor regardless of which feature brought the creator there (Content Clip, Crowd Clip, or Trend Trim). Creators can inspect every node (claim text, node type, timestamps, bound `audioContext[]` events from SenseVoice, bound `visualContext[]` labels from Video Intelligence API, humor decomposition context), traverse narrative edges color-coded by type (contradiction → red, causal link → blue, thematic continuation → green, tension/release → orange), and edit the graph before re-running the clipping pipeline. Creators can inject their own nodes (moments the pipeline didn't surface), modify misclassified node types, rewire edges, and then trigger ClipScoringAgent on the modified graph. Includes prompt-based clip search (natural language moment search), voice editing (Gemini Live API), humor decomposition visualization, audience signal overlays, and trend alignment indicators.

**Cross-cutting: Audio Production (Lyria)** — Every clip gets a complete AI audio production layer: custom music beds matched to the clip's tone and energy, foley/sound effects synced to visual transitions, intelligent audio ducking under speech, and transformational audio effects (reverb on punchlines, bass on reactions, audio stingers for hooks). This is the feature that "wasn't possible six months ago" — fully automated, tone-aware audio production for Short-form content.

---

## Product Ethos: Creator-First Model

Clypt's production architecture is **creator-centric**: creators authenticate with their YouTube account, connect their channel, and receive intelligence on *their* content using *their* analytics (OAuth-scoped YouTube Analytics and Data APIs + YouTube Studio Editor access for long-form content optimization).

For the challenge submission demo, since we are not established YouTube creators with meaningful engagement data, we use a **"paste any channel URL"** approach — analyzing real, well-known creators (e.g., Ryan Trahan, MrBeast) to showcase the platform's intelligence capabilities using publicly available data. This is a **challenge submission strategy only**. When we continue building and testing with real users after this challenge cycle, the creator-authenticated model with full Youtube account access is the intended path.

---

## Challenge Strategy: "Analyze Any Creator"

### How It Works
1. User pastes any YouTube channel URL into Clypt
2. **Zero-to-One Onboarding** (deep, high-leverage initialization — see below):
   - YouTube Data API sweeps metadata for the creator's **Top 10 most viewed long-form videos** and **Top 10 most viewed Shorts** from the last year
   - **Audience-Grounded Summarization**: For each video, fetches the transcript (yt-dlp) and top 15 comments (YouTube Data API). `Gemini 3.1 Flash` (not Pro — speed over depth here) processes both together to deduce true tone (satire vs. serious) and generates a tone-aware 2-sentence summary. This prevents misclassifying deadpan satire as literal content (e.g., a skit about using MS Word as an IDE). Summaries are embedded via `Gemini Embeddings` into Convex — ensuring few-shot RAG strictly retrieves tone-matched exemplars (skits with skits, tutorials with tutorials).
   - **Interval-Based Brand Synthesis**: `yt-dlp` downloads the creator's 10 most recent high-performing Shorts. Frames extracted at 2fps over a 5-second window are passed sequentially to `Gemini 3.1 Pro (Vision)`, which identifies dynamic caption patterns: active word color vs. inactive word color, text outlines, font styles. Outputs a dynamic JSON payload (e.g., `{"active_color": "#FF0000", "inactive_color": "#FFFFFF", "outline": "2px black", "font": "Montserrat Bold"}`) injected into `@remotion/captions` to programmatically replicate the creator's exact word-tracking style.
3. yt-dlp extracts "Most Replayed" heatmaps (real per-segment engagement data) for their videos
4. Convex cron begins polling engagement velocity snapshots (views, likes, comments over time) and Most Replayed heatmap snapshots (heatmap evolution as audience broadens)
5. Gemini 3.1 Pro analyzes trends (Google Trends, X, Reddit via web search)
6. Gemini 3.1 Pro surfaces clip opportunities and trend matches
7. Content PRs appear in real-time on the dashboard
8. For publishing demos, clips are synthesized via Remotion (with Lyria audio production) and uploaded to our own test YouTube channel via the YouTube Data API (proves the pipeline end-to-end; may trigger Content ID copyright flag, but the upload flow itself works)

> **Deferred Processing**: The heavy 3-stream pipeline + Multimodal Semantic Graph construction (STT v2, SenseVoice audio events, Video Intelligence API visual annotations → Interval Overlap Binding → graph nodes with bound audio/visual context) is **not** run on the back catalog during onboarding. It is deferred and only runs on-demand when a specific video is flagged by a trend, surfaced by engagement signals, or manually requested by the creator. Onboarding fetches transcripts for audience-grounded summarization (lightweight — just the raw text, no word-level timestamps or diarization), but the full STT v2 processing pass + graph construction is deferred.

### What's Public vs. Private

| Data | Public? | Extraction Method |
|---|---|---|
| Comments (sorted by likes, with timestamps) | Yes | **YouTube Data API** `commentThreads.list(videoId=ID)` — official API, works for any public video |
| Video metadata (title, description, tags) | Yes | **YouTube Data API** `videos.list(id=ID, part=snippet)` |
| View count, like count, comment count | Yes | **YouTube Data API** `videos.list(id=ID, part=statistics)` |
| Channel video list, subscriber count | Yes | **YouTube Data API** `channels.list`, `search.list(channelId=ID)` |
| **"Most Replayed" heatmap** | Yes (~50K+ view videos) | **yt-dlp / InnerTube API** — 100 segments of replay intensity |
| **Engagement velocity (views/likes over time)** | Yes (via polling) | **Convex scheduled function** polls YouTube Data API hourly, stores snapshots |
| **Heatmap evolution over time** | Yes (via polling, ~50K+ views) | **Convex scheduled function** polls yt-dlp heatmap periodically, stores snapshots — heatmap at 1 hour looks different from heatmap at 24 hours |
| Trends (Google, X, Reddit) | Yes | **Gemini 3.1 Pro** with web search tools (Google Trends API, social media APIs) |
| True retention curve (% watching over time) | No (Studio-only) | Most Replayed heatmap serves as proxy |
| Clip publishing | Owner OAuth | **YouTube Data API** `videos.insert` on our test channel |

### The "Most Replayed" Heatmap

YouTube publicly exposes a **"Most Replayed" heatmap** on videos with sufficient views (~50K+). This is a normalized replay intensity graph showing which parts viewers seek back to and rewatch most. It's extractable without authentication via:

- `yt-dlp --dump-json VIDEO_URL | jq '.heatmap'` — returns 100 segments with `start_time`, `end_time`, and `value` (0–1 intensity)
- Direct InnerTube API POST to `youtube.com/youtubei/v1/player`
- Data path: `frameworkUpdates.entityBatchUpdate.mutations[]` where `markerType == 'MARKER_TYPE_HEATMAP'`

Each segment: `{ startMillis, durationMillis, intensityScoreNormalized: 0.0–1.0 }`

This is **real engagement data** — not simulated. Combined with comments from the YouTube Data API and engagement velocity from polling, it provides a robust per-segment signal for clip selection and long-form optimization. All Ryan Trahan and MrBeast videos will have this data.

### Engagement Velocity (Time-Series Polling)

The YouTube Data API and yt-dlp only provide single-point-in-time snapshots of video stats (current view count, current like count). There is no API that provides historical time-series data for videos you don't own.

To build engagement velocity data (views gained per hour, like acceleration, comment growth rate), Clypt uses a **Convex scheduled function** that polls the YouTube Data API at regular intervals and stores each snapshot:

1. **Convex cron** runs every 30-60 minutes
2. Calls YouTube Data API `videos.list(id=VIDEO_IDS, part=statistics)` for all tracked videos (creator's catalog)
3. Stores each snapshot in a `statsSnapshots` table with timestamp
4. Computes deltas: views gained since last snapshot, like acceleration, comment velocity
5. Surfaces trends: "This video gained 50K views in the last 3 hours"

The same Convex cron also polls yt-dlp for **Most Replayed heatmap snapshots**, storing the heatmap evolution over time in a `heatmapSnapshots` table. The heatmap at 1 hour post-upload (early superfans who watch everything) looks different from the heatmap at 24 hours (broader algorithm-pushed audience who skip more). Tracking this evolution reveals which segments retain engagement as the audience broadens vs. which only appealed to the core fanbase.

This works for **any public video**. It's the same approach ContentStats.io uses, but built natively on sponsor tech (Convex crons + database).

### Publishing in the Challenge Demo
- **Clip upload**: Clips are synthesized via Remotion (brand-matched design tokens, STT v2 Captions, Lyria audio production layer) and published to our own test YouTube channel via the **YouTube Data API** `videos.insert`. PublishAgent generates creator-style-matched titles/descriptions via few-shot RAG (top 3 similar past videos as style exemplars). Content ID may flag the upload, but the clip stays up. With a real creator's OAuth, it would post cleanly.

### Challenge Demo Setup
- **Dummy Clypt Gmail account** with the target creator's PFP and channel name — simulates being the creator for each demo case. For each feature demo, we can rename the account to match the creator being analyzed.
- **Demo opens with infographic + business pitch**, clearly noting which elements are pre-computed or simulated for demo purposes. Then each feature is demonstrated in sequence.
- **Target video**: Choose a newly released video from a sizeable creator (released near or during the challenge window) so that engagement velocity + heatmap snapshots have been accruing via Convex cron polling before the demo recording.
- **Audience-resonant content**: Use videos that resonate with the target audience — e.g., Joma Tech for coding interviews / startup mindset (tech crowd will find it engaging), Hot Ones for broad appeal.

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

## Google ADK Agent Design

Clypt uses a **parent orchestrator + specialized subagents** pattern, implemented with Google ADK's agent primitives.

### Orchestrator Agent

**Model:** Gemini 3.1 Pro | **Framework:** Google ADK `LlmAgent` | **Triggered by:** Convex scheduled function (cron), direct API call (manual request), webhook (approval reply)

| Orchestrator Tool | What It Does |
|---|---|
| `get_creator_context` | Loads creator profile and video catalog from Convex |
| `predict_video_tone` | **Context-Aware Heuristic.** Evaluates the new video's metadata (title, description, tags, transcript excerpt) against the creator's profile (`detectedTone` distribution from onboarding). Gemini 3.1 Pro classifies the video's `predicted_tone` (e.g., `"satire"`, `"technical_tutorial"`, `"vlog"`, `"interview"`, `"reaction"`). This `predicted_tone` is explicitly passed downstream to the ClipScoringAgent, which uses it to dynamically weight its scoring dimensions — ensuring the system looks for "funny moments" in a skit and "clear explanations" in a tutorial. |
| `route_to_agent` | Uses `transfer_to_agent` to delegate to the appropriate subagent based on trigger type |
| `notify_creator` | Sends notification email to creator |
| `update_content_pr` | Writes Content PR to Convex database |
| `trigger_video_synthesis` | Invokes Convex action → triggers Remotion video composition (React + Tailwind CSS video DOM with brand design tokens, STT v2 Caption components) + Lyria audio production (music bed + foley + mixing) |

**Routing logic:**
- Channel URL pasted → **Zero-to-One Onboarding** (metadata sweep + embedding + brand token extraction) → `ParallelAgent` (CommentIngestion + HeatmapAgent + **TranscriptAgent + AudioEventAgent + VideoAnnotationAgent** + CatalogMatch) → **`predict_video_tone`** (Gemini 3.1 Pro classifies `predicted_tone` from metadata + creator profile) → `SequentialAgent` (**NarrativeAnalystAgent** (build graph → Interval Overlap Binding → Multimodal Semantic Graph → narrative edge classification → Semantic Sub-graphs) → EngagementAnalysis → **ClipScoringAgent**(predicted_tone, Semantic Sub-graphs, deterministic boundaries) → BrandIdentityAgent + **AudioProductionAgent** (Lyria) + **VeoGenerationAgent** (B-roll + clip extension) + ThumbnailAgent) → Remotion synthesis (video + Lyria audio + Veo B-roll) → notify
- Trend alert fires → TrendScout → CatalogMatch (searches pre-vectorized catalog from onboarding) → [run 3-stream on matched videos if not pre-computed] → **`predict_video_tone`** → **NarrativeAnalystAgent** (Interval Overlap Binding → Multimodal Semantic Graph → Semantic Sub-graphs) → **ClipScoringAgent**(predicted_tone, deterministic boundaries) → **AudioProductionAgent** (Lyria) → Remotion synthesis → notify
- **Prompt submitted** → PromptClipAgent → search `narrativeNodes` (multimodal-bound graph) → expand to Semantic Sub-graphs → ClipScoringAgent(predicted_tone, deterministic boundaries) → **AudioProductionAgent** (Lyria) → Remotion synthesis → notify
- Publish approved → PublishAgent (few-shot RAG: retrieve style exemplars → creator-matched metadata) → YouTube Data API upload

> **Onboarding vs. 3-stream processing**: When a channel URL is first pasted, the **Zero-to-One Onboarding** phase runs two targeted passes. First, **Audience-Grounded Catalog Indexing**: for each of the **Top 10 most viewed long-form videos** and **Top 10 most viewed Shorts** from the last year, raw transcripts + top 15 comments are processed by `Gemini 3.1 Flash` into tone-aware summaries, then embedded into Convex for tone-matched search and few-shot RAG. Second, **Interval-Based Brand Synthesis**: Gemini Vision analyzes frame sequences from the creator's recent Shorts to extract dynamic caption patterns + static brand tokens for `@remotion/captions`. The heavy 3-stream pipeline (TranscriptAgent + AudioEventAgent + VideoAnnotationAgent) is **deferred** — it only runs on-demand when a specific video is flagged by a trend, surfaced by engagement signals, or manually requested. Once run, the 3-stream artifacts are cached in Convex and reused across all modes. The NarrativeAnalystAgent then builds the Multimodal Semantic Graph: transcript → semantic nodes → Interval Overlap Binding (maps audio/visual data onto nodes) → narrative edge classification (contradictions, causal links, thematic continuations, tension/release arcs) → Semantic Sub-graphs for scoring.

### Subagents

Each subagent is a Google ADK `LlmAgent` with its own system prompt, tool set, and model. They run autonomously and return structured results.

---

**Crowd Clip Agent Pipeline (Feature 2)**

**[CommentIngestionAgent]** — YouTube Data API + Gemini Embeddings + Gemini 3.1 Pro
| Tool | What It Does |
|---|---|
| `fetch_comments` | YouTube Data API `commentThreads.list(videoId=ID, order=relevance)` — fetches top/most-relevant comments for any public video |
| `cluster_comments` | Embeds comment corpus via `Gemini Embeddings` → runs agglomerative clustering to group chaotic comments by conceptual proximity rather than exact keywords. Each cluster represents a distinct audience reaction signal. |
| `trace_clusters_to_transcript` | Uses Lexical Chains to map each comment cluster back to specific transcript segments. Corrects for viewer reaction lag (comments about a moment typically reference 5-15s after the actual timestamp). |
| `detect_sound_events` | When a comment cluster flags a non-verbal cue (e.g., "that Microsoft Teams chime", "the fart noise"), uploads the raw audio segment to the `Gemini File API` and prompts `Gemini 3.1 Pro` to return the exact start/end timestamps of that specific acoustic signature. 100% Google stack. |
| `extract_emotional_markers` | Identifies 😂🤣💀🔥, ALL CAPS, "I died when...", high like counts |

**[HeatmapAgent]** — yt-dlp| Tool | What It Does |
|---|---|
| `extract_most_replayed` | Runs `yt-dlp --dump-json` → extracts `.heatmap` array (100 segments, 0–1 intensity) |

**[EngagementVelocityAgent]** — Convex Scheduled Function + YouTube Data API
| Tool | What It Does |
|---|---|
| `poll_video_stats` | Convex cron calls YouTube Data API `videos.list(part=statistics)` every 30-60 minutes for all tracked videos |
| `store_snapshot` | Saves `{ videoId, viewCount, likeCount, commentCount, timestamp }` to Convex `statsSnapshots` table |
| `compute_deltas` | Calculates views gained/hr, like acceleration, comment velocity from successive snapshots |


**[EngagementAnalysisAgent]** — Gemini 3.1 Pro
| Tool | What It Does |
|---|---|
| `fuse_signals` | Combines semantically clustered comments (CommentIngestionAgent) + heatmap (yt-dlp) + engagement velocity (cron snapshots) + **Multimodal Semantic Graph** (nodes with bound `audioContext[]` + `visualContext[]`) + **SED timestamps (SenseVoice continuous + Gemini 3.1 Pro on-demand)** + metadata into unified per-segment scoring. A heatmap peak that aligns with a graph node carrying bound `audioContext: [{event: "laughter"}]` gets a significant confidence boost — no timestamp proximity guessing. Comment clusters traced via Lexical Chains provide lag-corrected anchors mapped to specific graph nodes. |
| `resolve_vague_references` | Embeds vague comments ("the bread moment") → k-NN vector search against `narrativeNodes` (multimodal-bound graph nodes) to find exact timestamps with bound context |
| `rank_clip_candidates` | Ranks segments by convergent signal: heatmap replay peaks that align with graph nodes carrying bound audio/visual context, comment cluster anchors, SED-detected sound events, emotional markers, velocity spikes |
| `identify_subgraphs` | Expands top-ranked graph nodes into Semantic Sub-graphs (setup → reaction clusters) with deterministic boundaries. Passes sub-graphs to ClipScoringAgent for final evaluation. |
| `evaluate_hook` | Scores the candidate clip's first 3 seconds using the transcript + audio emotion data. Suggests a better in-point if the hook scores ≤ 4/10. |
| `synthesize_clip` | Triggers Remotion video composition + Lyria audio production: React + Tailwind CSS video DOM with brand design tokens, STT v2 Caption components, Lyria audio layer (music bed + foley + ducked mix) |
| `create_content_pr` | Saves Content PR with score, evidence, reasoning, hookScore, alternativeInPoint, hookReason |

**Pipeline:** fetch + cluster comments (Gemini Embeddings → agglomerative clustering → Lexical Chain tracing) + SED on flagged non-verbal cues (Gemini 3.1 Pro via File API) + extract heatmap (yt-dlp) + poll velocity (Convex cron) + **run TranscriptAgent + AudioEventAgent + VideoAnnotationAgent in parallel** → NarrativeAnalystAgent (build graph → Interval Overlap Binding → Multimodal Semantic Graph) → fuse engagement signals with graph evidence → resolve vague references (vector search against multimodal nodes) → rank by convergent signal → ClipScoringAgent evaluates Semantic Sub-graphs (deterministic boundaries) + evaluate hooks → **AudioProductionAgent** (Lyria: music bed + foley + mixing) → synthesize clips (Remotion + Lyria audio layer) → save Content PRs → return summary

> **Crowd Clip signal fusion example**: Heatmap peak at 2:53 (intensity 0.91) + comment cluster of 47 semantically grouped comments (traced to t=2:48 via Lexical Chains, lag-corrected) + top comment "the bread moment 🔥". Graph node at 2:48 has bound `audioContext: [{event: "laughter", tStartMs: 2510}, {event: "excited", tStartMs: 2530}]` + bound `visualContext: [{scene: "speaker gesturing animatedly, leaning toward camera"}]` = 99% confidence clip. Comment cluster also flagged "that sound effect at the end" → Gemini 3.1 Pro SED (via File API) confirms chime at t=3:01. Hook evaluation: first 3s opens with a direct rhetorical question → hookScore 9/10. Out-point deterministic from last node's bound `audioContext`: `laughter` ending at t=3:04 — beat preserved inherently from graph structure. AudioProductionAgent (Lyria): generates upbeat music bed matching `predicted_tone: vlog`, foley whoosh on opening cut, audio stinger on hook point, music ducks under speech at 2:48–3:04 using STT v2 timestamps.

---

**Trend Trim Agent Pipeline (Feature 3)**

**[TrendScoutAgent]** — Gemini 3.1 Pro + Google Trends API + YouTube Data API
| Tool | What It Does |
|---|---|
| `query_google_trends` | Google Trends API: extracts trending topics with spike percentages → structured `{topic, spikePct, source}[]` for clean CatalogMatch handoff |
| `analyze_youtube_trending` | YouTube Data API `videos.list(chart=mostPopular)`: fetches trending videos in relevant categories → Gemini 3.1 Pro analyzes relevance to creator's niche |
| `analyze_social_signals` | Gemini 3.1 Pro reasons over aggregated trend data from multiple sources to identify niche trend signals |

**[CatalogMatchAgent]** — Gemini 3.1 Pro
| Tool | What It Does |
|---|---|
| `load_creator_catalog` | Queries Convex for all video metadata — pre-vectorized during Zero-to-One Onboarding (audience-grounded, tone-aware summaries already embedded and indexed) |
| `embed_and_search` | Embeds trend topic via Gemini Embeddings → Convex vector search against pre-indexed catalog embeddings — surfaces semantically relevant videos even without exact keyword match |
| `reason_about_relevance` | Gemini 3.1 Pro reasons over matched videos using long-context window to find the best clip windows |
| `run_3stream_on_candidate` | For each matched video that **lacks** pre-computed stream data (deferred during onboarding), triggers TranscriptAgent + AudioEventAgent + VideoAnnotationAgent in parallel to generate the full signal set before scoring |
| `run_narrative_analysis` | NarrativeAnalystAgent builds Multimodal Semantic Graph from transcript + Interval Overlap Binding (maps audio/visual onto nodes) → narrative edge classification (contradictions, causal links, thematic continuations, tension/release arcs) → extracts Semantic Sub-graphs for scoring |
| `score_with_streams` | ClipScoringAgent evaluates Semantic Sub-graphs (clusters of multimodal-bound nodes) with Dynamic Dimension Weighting (Orchestrator passes `predicted_tone`): narrative edge classification → tone-weighted 6-dimension scoring → deterministic clip boundaries from node timestamps → hook scoring. Trend alignment replaces heatmap as the primary signal, with bound audio/visual context on graph nodes as validators. |
| `produce_audio` | AudioProductionAgent (Lyria): generates tone-matched music bed, foley, audio stingers, and mixed audio layer for the clip |
| `synthesize_clip` | Triggers Remotion video composition with brand-matched design tokens + Lyria audio layer |
| `create_content_pr` | Saves trend-triggered Content PR with trendTopic, hookScore, hookReason, narrative edge evidence, stream evidence |

**Pipeline:** detect trend (Google Trends API + YouTube Data API + Gemini 3.1 Pro) → embed trend topic → search pre-vectorized catalog → for each matched video: run 3-stream pipeline if not pre-computed → NarrativeAnalystAgent (build graph → Interval Overlap Binding → narrative edge classification → Semantic Sub-graphs) → ClipScoringAgent (evaluate Semantic Sub-graphs with deterministic boundaries + hook eval) → AudioProductionAgent (Lyria: music bed + foley + mixing) → synthesize clips (Remotion + Lyria audio layer) → save Content PRs → notify

> **Trend Trim signal fusion example**: "Bad Bunny" trend spike detected → catalog match finds Ryan's 2023 interview video (instant hit from pre-vectorized onboarding catalog) → 3-stream pipeline runs → NarrativeAnalystAgent builds Multimodal Semantic Graph: claim node at 18:22 ("Bad Bunny changed music") gets bound `audioContext: [{event: "excited", tStartMs: 18310}, {event: "laughter", tStartMs: 18310}]` + `visualContext: [{scene: "guest leaning in, direct eye contact, animated hands"}]` via Interval Overlap Binding. Cross-segment narrative edge detected (contradiction): claim at 18:22 contradicts claim at 31:05 ("doesn't listen to reggaeton"). ClipScoringAgent evaluates the Semantic Sub-graph: hook "opens with a surprising claim about Bad Bunny's crossover influence" → hookScore 9/10. Out-point deterministic: max `endMs` of last node's bound `audioContext` = 19:03 (laughter event). All multimodal evidence read directly from graph nodes — no timestamp guessing.

---

**Content Clip Agent Pipeline (Feature 1)**

Three agents run in **parallel** on the source video, then feed into a single scoring agent.

**[TranscriptAgent]** — Vertex AI Speech-to-Text v2 (`latest_long` model, unified single-pass)
| Tool | What It Does |
|---|---|
| `extract_audio` | yt-dlp extracts audio track from video |
| `transcribe_and_diarize` | Single `BatchRecognize` call with `enableWordTimeOffsets: true` + `diarizationConfig` on the `latest_long` model. Returns a unified `[{word, startMs, endMs, speakerTag, confidence}]` array — word-level timestamps and speaker labels in one pass. No multi-model merging required. |
| `build_remotion_captions` | Parses the unified output directly into Remotion `Caption` data structures for dynamic, programmatic React text rendering. Each caption carries word, startMs, endMs, speakerTag — enabling per-word highlight animations synchronized to video playback. |

**[AudioEventAgent]** — SenseVoice
| Tool | What It Does |
|---|---|
| `run_sensevoice` | Runs SenseVoice on full audio: simultaneous ASR + speech emotion (happy/sad/angry/excited/neutral) + audio event detection (laughter, cheering, applause, silence, music, sound effects). 70ms latency per 10s of audio. |
| `build_emotion_timeline` | Returns `[{t_start_ms, t_end_ms, event_type, emotion, confidence}]` — per-segment emotion + event map. SenseVoice runs natively on the raw waveform and outputs continuous, variable-length, un-batched segments with exact millisecond boundaries — preserving true reaction durations for beat-aware pacing. |
| `flag_high_energy_windows` | Pre-filters segments where `emotion=excited` or `event=laughter/cheering` — strong priors for clip-worthiness |

**[VideoAnnotationAgent]** — Google Cloud Video Intelligence API
| Tool | What It Does |
|---|---|
| `upload_to_gcs` | Uploads source video to Google Cloud Storage bucket for Video Intelligence API processing |
| `annotate_video` | Single `annotateVideo` API call with features: `SHOT_CHANGE_DETECTION`, `LABEL_DETECTION`, `OBJECT_TRACKING`, `FACE_DETECTION`, `EXPLICIT_CONTENT_DETECTION`. Returns a structured JSON payload with exact millisecond temporal bounding boxes for everything in the video. |
| `parse_shot_changes` | Extracts mathematically exact scene cut timestamps — fed directly to Remotion for edit point alignment. No LLM guessing. |
| `parse_labels_and_objects` | Extracts entity labels (person, car, food, etc.) and tracked object bounding boxes with temporal segments: `[{entity, category, startMs, endMs, confidence, boundingBox}]`. Maps directly into graph node `visualContext[]` via Interval Overlap Binding. |
| `parse_faces` | Extracts face detection + tracking with temporal bounding boxes — supports speaker identification and subject tracking for reframing. |
| `check_explicit_content` | Returns per-frame explicit content likelihood — used by OutboundAgent for brand-safety scoring before outreach. |
| `build_visual_timeline` | Aggregates all annotations into structured per-segment summaries: `[{t_start_ms, t_end_ms, labels[], objects[], faces[], shotBoundary: boolean, explicitContentLikelihood}]`. Deterministic, structured output — zero hallucination risk. |

> **Why Video Intelligence API over LLM frame extraction**: Passing 1fps frames to an LLM is a hacky workaround that costs massive tokens and risks hallucination (91% hallucination rate on AA-Omniscience Index for Flash — answers when it should say "unclear"). The Video Intelligence API is purpose-built for temporal video understanding. It produces structured JSON with exact millisecond timestamps, temporal bounding boxes, and mathematical shot change detection — all of which map directly into the Multimodal Semantic Graph via Interval Overlap Binding with zero ambiguity.

**[NarrativeAnalystAgent]** — Gemini 3.1 Pro (Few-Shot CoT NLI) + Gemini Embeddings + Convex

Builds the **Multimodal Semantic Graph** — the core data structure that replaces flat, parallel stream arrays. First, the transcript is parsed into semantic nodes (claims, assertions, emotional beats). Then, **Interval Overlap Binding** deterministically maps audio events (SenseVoice) and video annotations (Video Intelligence API) directly onto graph nodes by timestamp overlap — if an audio/visual event's timestamp falls within (or slightly overlaps) a node's `[sourceStartMs, sourceEndMs]`, it is appended to that node's `audioContext[]` or `visualContext[]` array in Convex. This eliminates the fundamental weakness of flat multi-stream architectures: guessing whether a laughter event at `1:06` belongs to a joke at `1:02` or a transition at `1:08`. The punchline node *physically contains* `audioContext: ["laughter"]` and `visualContext: ["speaker gesturing animatedly"]`.

With multimodal context bound, the NLI engine operates on two distinct layers:

**Layer 1: Narrative Edges** — relationships BETWEEN nodes that define the graph's topology. Edges determine what clips are structurally possible — a `CALLBACK` edge between minute 2 and minute 20 creates a sub-graph that ClipScoringAgent can evaluate as a clip candidate. Without that edge, those two moments would never be considered together. The full edge taxonomy:

| Edge Type | What It Connects | Example |
|-----------|-----------------|---------|
| **CONTRADICTION** | Two nodes that logically clash | "I prioritize privacy" ↔ "We sell your data" — comedic juxtaposition, cross-modal clashes (text vs. visual) |
| **CAUSAL_LINK** | Setup → resolution, problem → solution | "Docker keeps running out of memory" → "We added a memory limit flag" |
| **THEMATIC_CONTINUATION** | Same entity/topic recurring across tangents | Creator mentions a concept at minute 3, revisits it at minute 15 after a digression |
| **TENSION_RESOLUTION** | Emotional buildup → payoff | "We have 24 hours to build this entire app" → "And... we shipped it. It's live." |
| **CALLBACK** | Node B deliberately references Node A from earlier | Creator at minute 20: "remember what I said about the API?" → links to claim at minute 3. Different from thematic continuation: callbacks are deliberate authorial references, not organic topic recurrence |
| **ESCALATION** | Each successive node is more extreme/intense | Spice challenge where each round is hotter; coding challenge where each problem is harder. Different from tension/resolution: escalation is a sustained ramp with no release yet |
| **SUBVERSION** | Node B violates the expectation set by Node A | "So I followed the tutorial exactly..." → complete disaster. Different from contradiction: subversion is about audience expectation being broken, not logical incompatibility |
| **ANALOGY** | Node A explains concept X by comparing to Y | "Think of a blockchain like a Google Doc that everyone can edit but nobody can delete" — crucial for educational content |
| **REVELATION** | Node B introduces information that recontextualizes Node A | "Plot twist" moments, surprise reveals — "I haven't mentioned that this was all staged" changes the meaning of every prior node |

**Layer 2: Content Mechanism Decomposition** — properties OF individual nodes that explain WHY a moment is significant. Decompositions don't create new connections — they enrich existing nodes with deeper understanding, which informs clip scoring (boundary decisions, dimension weighting), Lyria audio production (what kind of music/stinger to generate), and what the creator sees in Cortex. Decomposition types:

| Decomposition | Trigger | What It Adds to the Node |
|---------------|---------|-------------------------|
| **Humor** | SenseVoice laughter/excitement + contradiction edges + transcript exaggeration patterns | `humorContext[]`: reference (what's parodied), domain (specialized knowledge needed), mechanism (irony, subversion, callback, etc.), layers (surface→deep), audience prerequisite |
| **Emotional Resonance** | SenseVoice emotion peaks (sadness, excitement, anger) + direct address patterns in transcript | `emotionalContext[]`: type (vulnerability, catharsis, empathy, inspiration, nostalgia, awe, righteous anger), mechanism (personal disclosure, shared experience, underdog narrative, injustice framing), arc position (buildup/peak/aftermath), parasocial intensity (how directly creator addresses audience) |
| **Social Dynamics** | Multiple speaker tags (diarization) + energy/emotion shifts between speakers | `socialContext[]`: dynamic type (status reversal, genuine disagreement, unexpected chemistry, collaborative riffing, confrontation), initiator/recipient speaker tags, energy delta from baseline |
| **Expertise Signal** | Claim nodes with high information density + audience comments confirming "I didn't know that" | `expertiseContext[]`: type (counterintuitive truth, casual flex, elegant simplification, live demonstration), domain, audience reaction validation |

**The two layers work together:** An edge tells you THAT two nodes are connected (e.g., `SUBVERSION` between a setup and its violated expectation). A decomposition tells you WHY the individual node is noteworthy (e.g., humor decomposition explains the subverted moment is funny because it's parodying a specific cultural reference). The edge gives ClipScoringAgent the structure to find clip candidates; the decomposition gives it the depth to score them well and produce appropriate Lyria audio.

NLI evaluation for both layers uses **Few-Shot Chain-of-Thought (CoT) Prompting** via Google ADK — no fine-tuning required. The prompt forces Gemini 3.1 Pro to emit a `reasoning_trace` before its final classification, drastically improving accuracy on subjective narrative analysis compared to zero-shot prompting.

| Tool | What It Does |
|---|---|
| `build_semantic_graph` | Processes transcript into a semantic graph: nodes represent claims, assertions, opinions, and emotional beats. Nodes are stored natively in Convex with their source timestamps and segment context. |
| `bind_multimodal_context` | **Interval Overlap Binding.** Executes a Convex mutation that maps AudioEventAgent (SenseVoice) and VideoAnnotationAgent (Video Intelligence API) outputs directly onto graph nodes. For each audio event or video annotation, if its timestamp falls within (or slightly overlaps with a configurable tolerance) a node's `[sourceStartMs, sourceEndMs]`, the event is appended to that node's `audioContext[]` or `visualContext[]` array. After binding, every node is a self-contained multimodal unit — the ClipScoringAgent never needs to cross-reference separate stream tables. Video Intelligence API annotations provide structured labels, objects, and faces with exact bounding boxes rather than LLM-generated descriptions. |
| `embed_graph_nodes` | Embeds each graph node (including bound multimodal context) via `Gemini Embeddings` and stores vectors in Convex for cross-video vector search. Because nodes now carry audio/visual context, embedding captures multimodal semantics — a prompt like "find where he argues about privacy and people laugh" can match a single node. |
| `classify_narrative_edges` | Uses Gemini 3.1 Pro with **Few-Shot Chain-of-Thought (CoT) Prompting** for cross-modal Natural Language Inference (NLI). Classifies narrative relationships (Layer 1: edges) between graph nodes across the full taxonomy: **Contradiction**, **Causal Link**, **Thematic Continuation**, **Tension/Resolution**, **Callback**, **Escalation**, **Subversion**, **Analogy**, **Revelation** (see edge taxonomy table above). Operates on the Multimodal Semantic Graph, not linear transcript order. The CoT prompt includes curated examples with mandatory `reasoning_trace` output. See system prompt template below. |
| `link_cross_segment_pairs` | Links narratively related nodes across the entire video regardless of temporal distance — a joke setup at minute 2 can be linked to its punchline at minute 45, or a tutorial problem statement at minute 3 can be linked to its resolution at minute 20. Uses a **two-stage engine** to avoid O(n²) LLM calls on large graphs: **(1) Cosine Similarity Bouncer** — each node runs a mathematically cheap Convex vector search against all other nodes, enforcing a strict cosine similarity threshold. Node pairs below the threshold are skipped with zero LLM tokens spent. **(2) Edge De-duplication** — because the search is undirected, once an edge is drawn between Node A and Node B, Node B skips searching Node A later, halving redundant comparisons. **(3) LLM Classification** — only the fraction of candidate pairs that survive the similarity threshold are sent to Gemini 3.1 Pro for the expensive Few-Shot CoT NLI edge classification (`classify_narrative_edges`). This means a 1,000-node graph might produce ~50 candidate pairs instead of 500,000, with only those ~50 requiring LLM reasoning. |
| `surface_nonlinear_highlights` | Returns linked node pairs with timestamps, edge type, confidence, and narrative classification (irony, callback, self-contradiction, escalation, competence_paradox, corporate_hypocrisy, cross_modal_contradiction, cause_effect, problem_solution, topic_recurrence, buildup_payoff, suspense_reveal), along with the bound multimodal evidence for each node. The orchestrator uses these to synthesize non-linear highlights that linear clipping would miss entirely — contradiction-based comedic compilations, tutorial problem/solution montages, or tension/release supercuts. |
| `decompose_content_mechanisms` | **Content Mechanism Decomposition (Layer 2).** After narrative edges are classified, runs a dedicated pass on every node flagged as notable — applying the appropriate decomposition type(s) based on triggers (see decomposition table above). A single node can have multiple decompositions (e.g., a moment can be both funny AND emotionally resonant). Each decomposition is stored as a typed context array on the graph node alongside `audioContext[]` and `visualContext[]`: `humorContext[]`, `emotionalContext[]`, `socialContext[]`, `expertiseContext[]`. When a cultural reference is identified (in any decomposition type, not just humor), a **Reference Node** is created (special node type representing the source material) with a `PARODY_OF` or `REFERENCES` edge — visible in Cortex as ghost nodes with dashed edges. When comment data is available (Crowd Clip), comments validate decompositions (e.g., "the rug pull part killed me" confirms a humor layer; "this hit different" confirms emotional resonance). Validated nodes receive a confidence boost in Cortex; unvalidated ones show lower confidence. Decompositions inform both ClipScoringAgent (boundary decisions — vulnerability moments need wider boundaries; expertise moments need the full explanation) and Lyria audio production (humor mechanism drives stinger/music choices; emotional type determines whether to add music or preserve silence; social dynamics energy delta maps to audio intensity). |

**NLI Few-Shot CoT System Prompt Template:**

```
You are a Universal Narrative Reasoning Engine trained to classify relationships between content segments. You will be provided with a PREMISE (Node A) and a HYPOTHESIS (Node B) from the same video, potentially spoken at different times.

Your task is to classify the relationship as one of: CONTRADICTION, CAUSAL_LINK, THEMATIC_CONTINUATION, TENSION_RESOLUTION, CALLBACK, ESCALATION, SUBVERSION, ANALOGY, REVELATION, or NEUTRAL.

- CONTRADICTION: Logical clash, comedic juxtaposition, self-contradiction, or cross-modal contradiction (text vs. visual/audio).
- CAUSAL_LINK: Setup → resolution, problem → solution, cause → effect. One node establishes context that the other resolves or extends.
- THEMATIC_CONTINUATION: Entity or topic tracking across tangents — the same subject resurfaces after a digression, maintaining narrative continuity.
- TENSION_RESOLUTION: Emotional buildup → payoff. One node escalates tension (challenge, suspense, conflict) and the other resolves it (triumph, reveal, relief).
- CALLBACK: Node B deliberately references Node A from earlier — an intentional authorial callback, not organic topic recurrence.
- ESCALATION: Each successive node is more extreme or intense — a sustained ramp without release (yet).
- SUBVERSION: Node B violates the expectation established by Node A — audience expectation is broken, not just logical incompatibility.
- ANALOGY: One node explains a concept by mapping it to a different domain — X is like Y.
- REVELATION: Node B introduces new information that fundamentally recontextualizes Node A — everything before means something different now.
- NEUTRAL: No meaningful narrative relationship.

--- EXAMPLES ---
Example 1:
PREMISE: "I prioritize user data privacy above all else."
HYPOTHESIS: "We sell our telemetry data to advertisers to stay afloat."
OUTPUT: {
  "reasoning_trace": "The premise explicitly claims privacy is the top priority. The hypothesis admits to selling user data. Selling data directly violates user privacy. Therefore, this is a severe logical clash representing corporate hypocrisy.",
  "classification": "CONTRADICTION",
  "narrative_classification": "corporate_hypocrisy",
  "confidence": 0.95
}

Example 2:
PREMISE: "I'm a 10x Staff Engineer at Google."
HYPOTHESIS: "How do I reverse a linked list in Python?"
OUTPUT: {
  "reasoning_trace": "The premise establishes a high level of elite technical competence. The hypothesis shows the speaker failing at a fundamental, entry-level programming concept. This subverts the established expectation, creating situational irony.",
  "classification": "CONTRADICTION",
  "narrative_classification": "competence_paradox",
  "confidence": 0.98
}

Example 3:
PREMISE: "The problem is that our Docker containers keep running out of memory."
HYPOTHESIS: "So what we did is add a memory limit flag and a health check restart policy."
OUTPUT: {
  "reasoning_trace": "The premise establishes a specific technical problem (OOM in Docker). The hypothesis directly resolves that problem with a concrete solution. This is a clear cause-effect / problem-solution arc.",
  "classification": "CAUSAL_LINK",
  "narrative_classification": "problem_solution",
  "confidence": 0.94
}

Example 4:
PREMISE: "We have 24 hours to build this entire app from scratch."
HYPOTHESIS: "And... we actually shipped it. It's live. People are using it."
OUTPUT: {
  "reasoning_trace": "The premise establishes high-stakes tension (extreme time constraint, ambitious scope). The hypothesis resolves that tension with a triumphant outcome. This is a classic challenge escalation followed by payoff.",
  "classification": "TENSION_RESOLUTION",
  "narrative_classification": "buildup_payoff",
  "confidence": 0.93
}
--- END EXAMPLES ---

Now, evaluate the following:
PREMISE: {node_a_text}
HYPOTHESIS: {node_b_text}

Output ONLY valid JSON matching the structure of the examples. You MUST generate the "reasoning_trace" before the "classification".
```

**[ClipScoringAgent]** — Gemini 3.1 Pro ← receives **Semantic Sub-graphs** (clusters of multimodal-bound nodes) + `predicted_tone` from Orchestrator

The paradigm shift: instead of receiving three flat stream arrays and guessing which events belong together via timestamp proximity, the ClipScoringAgent receives **Semantic Sub-graphs** — clusters of connected graph nodes where each node already contains its bound `audioContext[]` and `visualContext[]` data. A punchline node physically contains `audioContext: [{event: "laughter", tStartMs: 4057}]` and `visualContext: [{scene: "animated facial expression, finger pointing"}]`. No cross-referencing, no timestamp heuristics.

**Clip boundaries are deterministic:** the in-point is the `startMs` of the first node in the sub-graph; the out-point is the max `endMs` across all audio/visual reactions bound to the last node. Because reactions are already bound to nodes, beat-aware pacing is inherent — the laughter event bound to the punchline node naturally extends the out-point without separate heuristic logic.

Uses **Dynamic Dimension Weighting** (context-aware heuristic): the agent's system prompt applies mathematical multipliers to the 6 scoring dimensions based on the `predicted_tone` injected by the Orchestrator.

**Weighting examples:**
- If `predicted_tone == "satire"`: 1.2× multiplier on **Narrative Edge** (favoring contradiction edges — comedic juxtaposition, cross-modal clashes) and **Emotional Peak** (from bound `audioContext` laughter) — naturally favors comedic timing and punchlines without completely ignoring other fundamentals.
- If `predicted_tone == "technical_tutorial"`: 1.2× multiplier on **Narrative Edge** (favoring causal link edges — problem/solution, setup/resolution chains) and **Visual Interest** (from bound `visualContext` screen changes) — surfaces complete explanations with visual step-throughs.
- If `predicted_tone == "vlog"`: 1.2× multiplier on **Narrative Edge** (favoring tension/release edges — challenge escalation → payoff) and **Emotional Peak** — prioritizes dramatic arcs and emotional resonance.
- Default (unclassified): all dimensions weighted equally at 1.0×.

| Tool | What It Does |
|---|---|
| `score_candidates` | For each **Semantic Sub-graph**, scores across 6 dimensions with **Dynamic Dimension Weighting**: Hook (0–10), Flow (0–10), Emotional Peak (0–10, from bound `audioContext`), Visual Interest (0–10, from bound `visualContext`), Speaker Coherence (0–10, from diarization data in nodes), Narrative Edge (0–10, from NarrativeAnalystAgent — scores the strength of narrative relationships across the full edge taxonomy: contradictions for comedy, causal links for tutorials, thematic continuations for narrative depth, tension/release arcs for challenge content). All multimodal evidence is read directly from graph nodes, not cross-referenced from separate stream tables. Raw scores are computed first, then multiplied by tone-specific weights. The weighted composite score determines final clip ranking. |
| `evaluate_hooks` | Scores the first 3 seconds of each candidate: direct question → 10, surprising/counterintuitive statement → 9, strong emotional exclamation → 8, call-to-action → 7, plain statement → 3 |
| `set_clip_boundaries` | **Deterministic from graph structure:** in-point = `startMs` of first node in sub-graph (snapped to nearest word boundary via STT v2 timestamps), out-point = max `endMs` across all audio/visual reactions bound to the last node. Because reactions are already bound, beat-aware pacing is inherent — laughter/silence events extend the out-point naturally. |
| `suggest_alternative_inpoint` | If hookScore ≤ 4, traverses adjacent graph nodes (not arbitrary ±15s windows) for a stronger hook moment using bound multimodal context |
| `synthesize_clip` | Triggers Remotion video composition + Lyria audio production: constructs React + Tailwind CSS video DOM with brand-matched design tokens (from BrandIdentityAgent), STT v2 word timestamps rendered as Remotion `Caption` components. AudioProductionAgent (Lyria) generates tone-matched music bed, foley effects synced to visual cuts, audio stingers on hook/punchline points, and intelligent audio ducking under speech. For non-linear narrative edge clips (contradiction compilations, tutorial montages, tension/release supercuts), Remotion composites multiple source segments with Lyria-generated transition audio. |
| `create_content_pr` | Saves ContentPR with hookScore, alternativeInPoint, hookReason, per-dimension raw scores, dimension weights applied (`predictedTone`, `dimensionWeights`), weighted composite score, sub-graph node IDs, bound multimodal evidence per node (audioContext + visualContext), and narrative edge evidence (edge type, classification, cross-modal evidence) |

**Pipeline:** extract audio (yt-dlp) → [TranscriptAgent + AudioEventAgent + VideoAnnotationAgent] in parallel → NarrativeAnalystAgent (build semantic graph → **Interval Overlap Binding** maps audio/visual onto nodes → narrative edge classification → link cross-segment pairs → extract Semantic Sub-graphs) → Orchestrator: `predict_video_tone` (metadata + creator profile → `predicted_tone`) → ClipScoringAgent receives Semantic Sub-graphs + `predicted_tone`: Dynamic Dimension Weighting (6 dimensions × tone-specific multipliers) → deterministic clip boundaries from node timestamps → hook evaluation → **AudioProductionAgent** (Lyria: classify clip mood → generate music bed + foley + audio stingers + mixed audio layer) → synthesize clips (Remotion + Lyria audio) → save Content PRs → notify

---

**Prompt-Based Clipping (Cortex — cross-cutting, works with Content Clip, Crowd Clip, and Trend Trim)**

**[PromptClipAgent]** — Gemini 3.1 Pro

Prompt-based clipping lets creators (or the voice editing interface) describe what they want in natural language and get back a precise, scored clip. The paradigm shift: instead of searching three separate flat stream tables and fusing results by timestamp overlap, PromptClipAgent queries the **`narrativeNodes` table directly** — where every node already contains bound `audioContext[]` and `visualContext[]` data from Interval Overlap Binding. This means a prompt like *"find where he argues about privacy and people laugh"* becomes a **single semantic vector query** against multimodal-bound nodes, not three parallel searches with post-hoc timestamp fusion.

| Tool | What It Does |
|---|---|
| `parse_prompt` | Gemini 3.1 Pro interprets the natural language query and extracts search dimensions: `topic` ("Bad Bunny"), `person` ("the guest in the red jacket"), `emotion` ("when they laughed"), `action` ("the moment he stands up"), `object` ("the whiteboard"), `moment_type` ("the punchline") |
| `search_narrative_graph` | Embeds the parsed query via Gemini Embeddings → k-NN search against the pre-computed `narrativeNodes` table (multimodal-bound graph nodes). Because nodes carry text + bound `audioContext[]` + `visualContext[]`, a single vector search captures all three modalities simultaneously. "Find where he argues about privacy and people laugh" matches nodes tagged as claims about privacy that have `audioContext: [{event: "laughter"}]` — mathematically exact semantic intent queries rather than keyword matches fused by timestamp proximity. |
| `expand_to_subgraph` | From matched nodes, traverses graph edges to build complete Semantic Sub-graphs spanning from the setup node to the final reaction. This ensures the resulting clip has a complete narrative arc, not a truncated fragment. |
| `score_and_rank` | Passes top-N Semantic Sub-graphs to ClipScoringAgent (with `predicted_tone` for Dynamic Dimension Weighting): deterministic clip boundaries from node timestamps + hook evaluation. Returns ranked clips with hookScore, hookReason, per-dimension raw scores, and tone-specific weights applied. |
| `synthesize_clip` | Triggers Remotion video composition: React + Tailwind CSS video DOM with brand-matched design tokens, STT v2 Caption components |
| `create_content_pr` | Saves ContentPR with `promptQuery` field, matched multimodal evidence from bound nodes (audioContext + visualContext), hookScore, alternativeInPoint |

**Prompt types supported:**
- **Topic/entity**: `"find the part about Bad Bunny"` → semantic search against graph nodes tagged as claims/assertions
- **Visual moment**: `"when he draws on the whiteboard"` → matches nodes with bound `visualContext` containing whiteboard references
- **Emotional moment**: `"the funniest part"` / `"when they both crack up"` → matches nodes with bound `audioContext` containing `laughter` or `excited` events
- **Cross-modal**: `"find where he argues about privacy and people laugh"` → single query matches nodes with claim text about privacy + bound `audioContext: [{event: "laughter"}]`
- **Quote/line**: `"find where he says 'it's not a solid'"` → exact + semantic search against node claim text
- **Vague reference**: `"the bread moment"` → embedding search against graph nodes + comment co-reference resolution

**Pipeline:** receive prompt → parse into search dimensions → search `narrativeNodes` (multimodal-bound graph nodes) → expand matched nodes to Semantic Sub-graphs → ClipScoringAgent (deterministic boundaries + hook eval) → AudioProductionAgent (Lyria: music bed + foley + mixing) → synthesize clip (Remotion + Lyria audio) → save ContentPR → notify

> **Prompt integration with Voice Editing**: When a creator holds the mic button and says "find the part where he laughs at the whiteboard", Gemini Live API captures the speech, calls the `prompt_clip` function tool, which invokes PromptClipAgent to search multimodal-bound graph nodes. The matching node (with bound `audioContext: [{event: "laughter"}]` + `visualContext: [{scene: "whiteboard"}]`) is expanded into a Semantic Sub-graph with deterministic clip boundaries. The resulting clip renders and plays back in the session without leaving the voice interface.

---

**Creative Generation Agents**

**[ThumbnailAgent]** — NanoBanana 2
| Tool | What It Does |
|---|---|
| `generate_thumbnail` | Receives clip keyframe, video title, and trend context → NanoBanana 2 generates a 4K thumbnail with character consistency (creator's likeness preserved across compositions), sub-pixel text rendering for perfect text overlays, and creative composition |
| `store_thumbnail` | Uploads to Convex file storage, links to Content PR |

---

**Publishing Agent**

**[PublishAgent]** — YouTube Data API + Gemini 3.1 Pro + Convex Vector Search (Dynamic Few-Shot RAG)
| Tool | What It Does |
|---|---|
| `generate_clip_summary` | Creates a 1-sentence summary of the clip's content for downstream style matching |
| `retrieve_style_exemplars` | Uses `Convex vector search` against tone-aware summary embeddings to retrieve the top 3 most semantically similar past videos with matching tone (satire retrieves satire, tutorials retrieve tutorials). Returns their original titles, descriptions, and tags as style exemplars. |
| `generate_metadata` | Dynamic few-shot prompting: injects the 3 retrieved exemplar titles/descriptions into the Gemini 3.1 Pro prompt to force it to mimic the creator's exact stylistic formatting (e.g., all lowercase, specific humor patterns, emoji usage, hashtag conventions). This is RAG for style, not RAG for facts. |
| `upload_short` | YouTube Data API `videos.insert` — uploads the Remotion-synthesized clip as a YouTube Short |
| `set_metadata` | YouTube Data API `videos.update` — sets creator-style-matched title, description, tags |
| `set_thumbnail` | YouTube Data API `thumbnails.set` — uploads the NanoBanana 2 generated thumbnail |
| `return_url` | Returns the published YouTube Short URL for the Content PR |

Publishing uses the YouTube Data API for reliable, programmatic upload and metadata operations with proper error handling and quota management. The few-shot RAG approach eliminates zero-shot title generation — every title/description now reads as if the creator wrote it themselves.

---

---

## Clipping Architecture: Model Roles & Division of Labor

The 3-stream pipeline produces raw artifacts that are then deterministically bound into a **Multimodal Semantic Graph** via Interval Overlap Binding. Each model handles the task it's best (and cheapest) at — the graph binding step fuses their outputs into a unified structure:

### Model Division of Labor

| Model | Role in Clipping | Why This Model |
|---|---|---|
| **Vertex AI STT v2** (`latest_long`) | Unified transcription + diarization in a single pass | `BatchRecognize` with `enableWordTimeOffsets` + `diarizationConfig` → returns `{word, startMs, endMs, speakerTag}` per token. Parsed directly into Remotion `Caption` data structures. No multi-model merging. Word timestamps also drive Lyria audio ducking. |
| **SenseVoice** | Emotion + audio event timeline | Per-second classification on raw waveform: speech emotion (happy/sad/angry/excited/neutral), laughter, cheering, silence, music, sound effects. 70ms latency per 10s of audio. Single model covering ASR + 4 emotion classes + event detection. |
| **Gemini 3.1 Pro (SED)** | Open-vocabulary Sound Event Detection | When comment clusters flag non-verbal cues, raw audio is uploaded to Gemini File API → Gemini 3.1 Pro returns exact timestamps of specific acoustic signatures. Targeted, on-demand — not continuous. |
| **Video Intelligence API** | Structured temporal video annotations | Single `annotateVideo` call: SHOT_CHANGE_DETECTION (exact scene cuts), LABEL_DETECTION + OBJECT_TRACKING (entities with temporal bounding boxes), FACE_DETECTION (face tracking), EXPLICIT_CONTENT_DETECTION (brand safety). Structured JSON — zero hallucination, zero token cost. Replaces LLM frame extraction. |
| **Veo (Vertex AI)** | Generative B-roll + clip extension | B-roll generation from semantic graph nodes (text/image → 1080p 9:16 video). Clip extension via first-frame conditioning (final frame → 2s smooth continuation). |
| **Gemini 3.1 Pro (Few-Shot CoT NLI)** | Multimodal Semantic Graph builder + narrative edge classification | Builds semantic graph → Interval Overlap Binding maps audio/visual data onto nodes → cross-modal Few-Shot CoT NLI classifies narrative edges across the full taxonomy (contradictions, causal links, thematic continuations, tension/release arcs) within bound nodes. Two-stage vector loop optimization (cosine similarity bouncer + edge de-duplication) avoids O(n²) LLM calls. Forces `reasoning_trace` before classification. Links cross-segment edges via Gemini Embeddings vector search regardless of temporal distance. Outputs Semantic Sub-graphs for ClipScoringAgent. No fine-tuning — production-grade NLI via prompt engineering. |
| **Gemini 3.1 Pro (Vision)** | Brand identity extraction + native agentic vision | Processes thumbnails + video frames → extracts hex codes, font styles, layout preferences as JSON design tokens → injected into Remotion compositions for brand-matched video rendering. Native agentic vision enables advanced video understanding. |
| **Lyria** | Audio production — music beds, foley, sound design, mixing | Generates tone-matched music beds (mood/energy classified by Gemini 3.1 Pro from `predicted_tone`), foley effects synced to Remotion visual cuts, audio stingers timed to graph node boundaries, and intelligent audio ducking using STT v2 word timestamps. Every clip gets broadcast-quality audio production. |

| **Gemini 3.1 Pro** | Clip scoring (Dynamic Dimension Weighting), hook evaluation, tone prediction, Content PR reasoning, few-shot metadata generation | Receives **Semantic Sub-graphs** (clusters of multimodal-bound nodes) + `predicted_tone` from Orchestrator. All audio/visual evidence is read directly from bound `audioContext[]` + `visualContext[]` on graph nodes — no cross-referencing separate stream tables. Applies tone-specific multipliers to 6 scoring dimensions (e.g., Narrative Edge × 1.2 for satire favoring contradictions, Narrative Edge × 1.2 for tutorials favoring causal links, Narrative Edge × 1.2 for vlogs favoring tension/release arcs). Clip boundaries are deterministic from node timestamps. PublishAgent uses few-shot RAG (Convex vector search for style exemplars) to generate creator-matched titles/descriptions. |
| **Remotion + Lyria** | Code-driven video synthesis + AI audio production | React + Tailwind CSS video DOM: STT v2 Captions, brand design tokens — all composed programmatically. Lyria audio layer (music bed + foley + stingers + ducked mix) integrated alongside source audio. Non-linear narrative edge clips composite multiple source segments with Lyria transition audio. Replaces FFmpeg. |
| **NanoBanana 2** | Thumbnail generation | SOTA image composition with character consistency, sub-pixel text rendering, 4K output — every clip gets a broadcast-quality thumbnail. |

### Content Clip (Feature 1): Multimodal Semantic Graph → Semantic Sub-graph Scoring
Three specialized agents (TranscriptAgent, AudioEventAgent, VideoAnnotationAgent) run in parallel and produce structured, timestamped artifacts. NarrativeAnalystAgent then builds the **Multimodal Semantic Graph**: transcript → semantic nodes, then **Interval Overlap Binding** deterministically maps audio events (SenseVoice) and video annotations (Video Intelligence API) directly onto graph nodes by timestamp overlap — each node physically contains `audioContext[]` and `visualContext[]` arrays. This eliminates the core weakness of flat multi-stream architectures: guessing whether a reaction belongs to a claim. Cross-modal Few-Shot CoT NLI classifies narrative edges across all three modalities — contradictions (including cross-modal clashes like Text: "I am calm" + Visual: "sweating, frantic gestures"), causal links (setup/resolution for tutorials), thematic continuations (entity tracking), and tension/release arcs (challenge/payoff). Before scoring, the Orchestrator runs `predict_video_tone` — classifying the video's tone from its metadata against the creator's profile. ClipScoringAgent (Gemini 3.1 Pro) receives **Semantic Sub-graphs** (clusters of connected, multimodal-bound nodes) plus the `predicted_tone`, and applies **Dynamic Dimension Weighting**: mathematical multipliers adjust how the 6 scoring dimensions are valued based on tone (e.g., Narrative Edge × 1.2 for satire favoring contradictions, Narrative Edge × 1.2 for tutorials favoring causal links). All multimodal evidence is read directly from bound graph nodes — no cross-referencing separate stream tables. Clip boundaries are **deterministic**: in-point = `startMs` of first node in sub-graph, out-point = max `endMs` of audio/visual reactions bound to last node. Because reactions are already bound, beat-aware pacing is inherent — laughter events extend the out-point naturally. After clip scoring, AudioProductionAgent (Lyria) generates a complete audio production layer — music bed, foley, stingers, and mixed audio — using the same deterministic boundaries and `predicted_tone` for tone-matched audio generation. This ensures the architecture truly adapts to each creator's specific content style while making every clip boundary and audio cue mathematically exact.

### Crowd Clip (Feature 2) & Trend Trim (Feature 3): Signal Fusion and Validation
The 3-stream pipeline + Multimodal Semantic Graph runs alongside (Crowd Clip) or on-demand for (Trend Trim) the engagement/trend pipeline:
- **Crowd Clip**: CommentIngestionAgent clusters comments semantically (Gemini Embeddings → agglomerative clustering) and traces clusters to transcript segments via Lexical Chains (correcting for viewer reaction lag). When clusters flag non-verbal cues, Gemini 3.1 Pro SED (via File API) returns exact timestamps. TranscriptAgent + AudioEventAgent run in parallel with CommentIngestion + HeatmapAgent. NarrativeAnalystAgent builds the Multimodal Semantic Graph (Interval Overlap Binding maps audio/visual onto nodes). EngagementAnalysisAgent fuses engagement signals with graph evidence — a heatmap peak that also has bound `audioContext: [{event: "laughter"}]` on a graph node + a semantically clustered comment anchor gets a major confidence boost. ClipScoringAgent evaluates Semantic Sub-graphs with deterministic boundaries. AudioProductionAgent (Lyria) then generates tone-matched audio for each approved clip.
- **Trend Trim**: TrendScoutAgent detects trends via Google Trends API + YouTube Data API + Gemini 3.1 Pro reasoning. CatalogMatchAgent searches the pre-vectorized catalog (from onboarding) for trend-relevant videos. 3-stream pipeline + NarrativeAnalystAgent (Interval Overlap Binding → Multimodal Semantic Graph) run on-demand for matched videos that lack pre-computed data. Trend alignment replaces heatmap as the primary signal; the bound graph nodes carry all audio/visual evidence for validation and scoring via Semantic Sub-graphs. AudioProductionAgent (Lyria) generates audio production for each clip.

### Cortex: Prompt-Based Clipping
The **Multimodal Semantic Graph** (`narrativeNodes` with bound `audioContext[]` + `visualContext[]`) is pre-computed and indexed in Convex — making the entire multimodal signal set queryable by natural language intent via a **single vector search** against graph nodes, not three parallel searches across separate stream tables. PromptClipAgent parses a prompt into search dimensions (topic, person, emotion, action, visual object) and queries `narrativeNodes` directly. Because nodes carry bound multimodal context, a prompt like *"find where he argues about privacy and people laugh"* matches nodes with claim text about privacy + bound `audioContext: [{event: "laughter"}]` — a mathematically exact semantic intent query, not a fuzzy timestamp fusion. Matched nodes are expanded into complete Semantic Sub-graphs (setup → reaction), then passed to ClipScoringAgent for deterministic boundary setting + hook scoring, then AudioProductionAgent (Lyria) for tone-matched audio generation, then synthesized via Remotion with brand-matched design tokens + Lyria audio layer. Prompt-based clipping is also the primary mechanism for the Gemini Live voice editing interface — `"find the funny moment"` during a voice session invokes PromptClipAgent directly via function calling.

### Content PR Reasoning
Every Content PR surfaces grounded, multi-stream evidence the creator can inspect:

> **Crowd Clip**: "At 2:53, heatmap intensity 0.91 + semantic comment cluster (47 comments, traced to t=2:48 via Lexical Chains, lag-corrected) + top comment 'the bread moment 🔥', 5.2K likes + cluster flagged 'that sound effect' → Gemini 3.1 Pro SED (via File API) confirms chime at t=3:01. Graph node at 2:48 bound with `audioContext: [{event: 'laughter', tStartMs: 2510}, {event: 'excited', tStartMs: 2530}]` + `visualContext: [{scene: 'speaker gesturing animatedly, leaning toward camera'}]`. Hook: opens with a direct rhetorical question → hookScore 9/10. Out-point deterministic: last node's bound `audioContext` laughter ends at 3:04. All multimodal evidence from bound graph nodes. Rendered via Remotion with creator's brand tokens (hex #FF6B35, Inter font)."

> **Trend Trim**: "Bad Bunny +340% search spike. Catalog match from pre-vectorized onboarding index (instant). Semantic Sub-graph: 4 nodes spanning 18:22–19:05. Setup node at 18:22 ('Bad Bunny changed music') bound with `audioContext: [{event: 'excited'}, {event: 'laughter'}]` + `visualContext: [{scene: 'guest leaning in, direct eye contact, animated hands'}]`. NarrativeAnalystAgent: cross-segment narrative edge (contradiction) — claim at 18:22 vs. claim at 31:05 ('doesn't listen to reggaeton') → narrative edge score 8/10. Hook: 'opens with a surprising claim about Bad Bunny's crossover influence' → hookScore 9/10. Out-point deterministic: last node's bound `audioContext` laughter ends at 19:03. All evidence from bound graph nodes. PublishAgent: few-shot RAG retrieves 3 similar past titles → generated title mimics creator's all-lowercase style."

> **Content Clip**: "Semantic Sub-graph: 5 nodes spanning 4:10–5:02. `predicted_tone: satire` → Dynamic Dimension Weighting applied: Narrative Edge × 1.2, Emotional Peak × 1.2. hookScore 8/10 — opens with a rhetorical question. Flow: complete narrative arc across bound nodes (setup node at 4:10, payoff node at 4:52, reaction node at 4:58). NarrativeAnalystAgent: ironic callback to claim node at 1:15 (edge type: contradiction) → narrative edge raw score 7/10 (weighted: 8.4). Punchline node's bound `audioContext: [{event: 'excited', tStartMs: 44500}, {event: 'laughter', tStartMs: 45700}]` → emotional peak raw 8/10 (weighted: 9.6). Bound `visualContext: [{scene: 'animated facial expression, finger pointing for emphasis'}]`. Speaker coherence: single speakerTag across all sub-graph nodes. Out-point deterministic: max `endMs` of last node's bound `audioContext` = 5:01.4s (laughter event). In-point: `startMs` of first node = 4:10.7s (snapped to word boundary). All multimodal evidence read directly from graph nodes — no cross-referencing separate stream tables. Remotion composition: brand-matched design tokens, Caption components synced to STT v2 word timestamps."

---

## Voice Editing: Gemini Live API + LiveKit

After receiving a notification and opening a Content PR, the creator can refine the clip using voice.

### Architecture
- **Model**: Gemini Flash (Live API, native audio)
- **Transport**: LiveKit Cloud free tier
- **Plugin**: `livekit-plugins-google` → `RealtimeModel` class
- **Capabilities**: Native audio processing (no STT→LLM→TTS pipeline), built-in VAD, function calling during live audio sessions, 30 HD voices

### Capabilities
- **Prompt-based clip search**: describe a moment in natural language → PromptClipAgent finds and renders it (`"find the part where they laugh at the whiteboard"`)
- Extend or trim clip start/end points conversationally ("include the part where he says gas")
- Adjust caption timing or regenerate
- Confirm and trigger re-render via Remotion
- Final approval before posting ("looks good, post it") which triggers PublishAgent

### Function Tools (called during live session)
| Tool | Triggered By | What It Does |
|---|---|---|
| `prompt_clip` | "Find the part where..." / "Clip the moment when..." | Invokes PromptClipAgent → searches multimodal graph nodes → expands to Semantic Sub-graph → renders clip with deterministic boundaries → returns preview URL |
| `extend_clip` | "Include 3 more seconds" / "Start earlier" | Adjusts startMs/endMs using word-level timestamps → Remotion re-render |
| `trim_clip` | "Cut the last 2 seconds" / "Start after the intro" | Same as extend_clip, opposite direction |
| `regenerate_caption` | "Change the caption" / "Make it punchier" | Gemini 3.1 Pro rewrites caption → updates ContentPR |
| `approve_and_publish` | "Post it" / "Looks good" | Triggers PublishAgent → YouTube Data API upload |

### UX Flow

```
Creator receives notification → opens Content PR on desktop/phone
→ previews clip → holds mic button

→ "Find the part where he laughs at the whiteboard"
→ Gemini Live API calls prompt_clip("laughs at whiteboard")
→ PromptClipAgent: searches `narrativeNodes` → finds graph node with bound `audioContext: [{event: "laughter"}]` + `visualContext: [{scene: "whiteboard"}]` at 14:22 → expands to Semantic Sub-graph
→ ClipScoringAgent: hookScore 8/10, deterministic boundaries from sub-graph nodes → Remotion renders → new preview plays

→ "Actually extend it by 2 seconds at the end"
→ Gemini Live API calls extend_clip(endMs + 2000) → Remotion re-renders → preview updates

→ "Perfect, post it."
→ PublishAgent uploads and publishes the Short via YouTube Data API
```

---

## Semantic Graph Editor: Visual Multimodal Graph Inspection & Editing

After the 3-stream pipeline runs and NarrativeAnalystAgent constructs the Multimodal Semantic Graph, creators can open the **Graph Editor** — an interactive visualization of every node and edge in the graph before (and after) the clipping pipeline scores clips.

### Why It Exists
The Multimodal Semantic Graph is the core artifact that drives all clip boundaries, narrative edge classifications, and scoring. Surfacing it visually gives creators:
- **Transparency**: inspect exactly which moments were identified, what multimodal context (SenseVoice audio events, Video Intelligence API annotations) is bound to each node, and why edges were classified as contradictions vs. causal links
- **Control**: correct misclassifications, inject moments the pipeline didn't surface, rewire narrative relationships
- **Iteration**: after editing the graph, re-run ClipScoringAgent on the modified structure to get updated clip recommendations that reflect the creator's intent

### Architecture
- **Visualization**: React Flow — force-directed interactive graph, custom node/edge renderers, full drag/zoom/pan
- **Styling**: Tailwind CSS + shadcn/ui for node cards, edge labels, and the edit sidebar
- **Real-time sync**: Convex subscriptions — graph mutations (add/edit/delete node or edge) write directly to `narrativeNodes` / `narrativeEdges` tables; UI updates instantly across open sessions
- **Pipeline re-run**: single button triggers `ClipScoringAgent` on the modified `narrativeNodes` snapshot in Convex, returns updated Content PRs; dashboard updates in real-time via Convex subscriptions

### Node Visualization
Each node renders as a card in the graph showing:
- **Claim text** — the transcript phrase/sentence the node represents
- **Node type badge** — `claim` / `assertion` / `opinion` / `emotional_beat`, color-coded
- **Timestamp chip** — `mm:ss.S → mm:ss.S` — clicking scrubs the video preview to that moment
- **Audio context badges** — SenseVoice events bound to this node: `laughter`, `excited`, `cheering`, `silence`, `music` — each as a colored pill with confidence score
- **Visual context tags** — Video Intelligence API annotations bound to this node: entity labels, object tracks, face detections, shot boundaries — listed with temporal overlap range
- **Speaker tag** — speaker identity from STT v2 diarization (relevant for speaker coherence scoring)

### Edge Visualization
Edges between nodes are color-coded by narrative type:

| Edge Type | Color | Meaning |
|---|---|---|
| `contradiction` | Red | Text claim contradicted by visual/audio evidence or another claim |
| `causal_link` | Blue | Setup → resolution (tutorial problem/solution chains) |
| `thematic_continuation` | Green | Entity or topic tracking across tangents |
| `tension_resolution` | Orange | Challenge → payoff (vlog arc, buildup/reveal) |

Edge label shows `confidence` (0–1) and `narrativeClassification` (e.g., `ironic_callback`, `cross_modal_contradiction`, `cause_effect`). Hovering an edge shows `crossModalEvidence` if present (e.g., Text: "I am calm" + Visual: "sweating, frantic gestures").

### Editing Operations
| Action | UI | Backend |
|---|---|---|
| **Edit node** | Click node → sidebar opens with editable fields (claim text, node type, startMs, endMs) → Save | `updateNarrativeNode` Convex mutation |
| **Add node** | "+ Node" button → fill claim text, timestamp range, node type → Create | `createNarrativeNode` mutation; auto-runs Interval Overlap Binding to attach audio/visual context from existing `audioEventTimelines` / `videoAnnotations` |
| **Delete node** | Right-click → Delete (cascades to connected edges) | `deleteNarrativeNode` mutation |
| **Add edge** | Click source node → drag to target node → edge type selector → Create | `createNarrativeEdge` mutation |
| **Edit edge** | Click edge → change edge type, confidence → Save | `updateNarrativeEdge` mutation |
| **Delete edge** | Right-click edge → Delete | `deleteNarrativeEdge` mutation |

### Re-Running the Pipeline
After editing, the creator clicks **"Re-run Clipping Pipeline"** — this:
1. Reads the current `narrativeNodes` + `narrativeEdges` snapshot from Convex (including creator edits)
2. Re-extracts Semantic Sub-graphs from the modified graph
3. Passes sub-graphs to `ClipScoringAgent` with the stored `predictedTone`
4. Returns updated Content PRs that reflect the creator's edits
5. Dashboard updates in real-time via Convex subscriptions

Nodes the creator injected become part of the scoring pass — giving creators direct input into what gets clipped, not just what gets approved.

### Filters & Views
- **Filter by edge type** — show only contradictions, only causal links, etc.
- **Filter by audio event** — highlight all nodes with bound `laughter`, `excited`, or specific event types
- **Filter by confidence** — hide nodes or edges below a threshold to reduce visual noise
- **Timeline mode** — arrange nodes chronologically on a horizontal scrubber instead of force-directed layout, useful for understanding clip boundary placement relative to video progression
- **Sub-graph highlight** — select a Content PR → the corresponding Semantic Sub-graph highlights in the graph (nodes + edges that comprise that clip), making it easy to trace why specific boundaries were chosen

---

## Video Synthesis + Creative Generation

### Video Synthesis (Remotion + Lyria)
Remotion replaces FFmpeg as the core video rendering framework. Instead of shell-based transcoding, clips are constructed as React components with Tailwind CSS styling — a code-driven video DOM. Lyria adds a complete AI audio production layer to every clip.

1. Agent returns clip window (startMs, endMs, videoId) + narrative context (narrative edge pairs, hook data, predicted_tone)
2. Convex action assembles the Remotion composition:
   - **Source video**: yt-dlp downloads the source segment
   - **Captions**: STT v2 word-level timestamps are parsed into Remotion `Caption` data structures via `@remotion/captions`. Each word is a React component with precise timing. Dynamic caption tokens from onboarding (active/inactive word colors, outlines, fonts) are injected as `@remotion/captions` props — replicating the creator's exact word-tracking style programmatically.
   - **Brand styling**: Combined JSON design tokens (dynamic caption patterns + static brand tokens) extracted by Gemini 3.1 Pro (Vision) during onboarding are injected as composition props. Every clip visually matches the creator's brand identity — from word-highlight colors to thumbnail-level color palette.
   - **Audio Production (Lyria)**: AudioProductionAgent generates a complete audio layer:
     - **Music bed**: Lyria generates a custom music track matched to the clip's `predicted_tone` and energy profile (classified by Gemini 3.1 Pro from narrative content + audio events). Duration precisely matches clip length from graph node timestamps.
     - **Foley / Sound design**: Lyria generates transition sounds (whooshes, impacts) synced to visual cut points from the Remotion composition timeline, plus ambient audio enhancement.
     - **Audio stingers**: Hook accents for the first 3 seconds (timed to hookScore evaluation), punchline enhancers timed to graph nodes with bound `audioContext: [{event: "laughter"}]`, and reaction amplifiers.
     - **Intelligent mixing**: Lyria produces a final mix where the music bed ducks under speech (using STT v2 word timestamps for precise ducking boundaries) and rises during visual-only moments or transitions. Source audio is preserved as the primary track — Lyria's layer is complementary, not replacement.
   - **Non-linear clips**: For narrative-edge-based clips (NarrativeAnalystAgent output — contradiction compilations, tutorial problem/solution montages, tension/release supercuts), Remotion composites multiple source segments with Lyria-generated transition audio between segments.
3. Remotion renders the composition to video (visual + Lyria audio layer)
4. Result uploaded to Convex file storage; dashboard shows the rendered clip preview in real-time (Convex subscription)
5. Remotion Player component provides in-browser preview before final render — creators can preview the clip with full brand styling, captions, and Lyria audio without waiting for a full render

### Brand Identity Extraction (Interval-Based Brand Synthesis)
Static thumbnails alone are insufficient for capturing dynamic caption mechanics like word-level color tracking. The BrandIdentityAgent uses a two-layer extraction approach:

**Layer 1 — Dynamic Caption Patterns (from Shorts):**
1. During Zero-to-One Onboarding, `yt-dlp` downloads the creator's 10 most recent high-performing YouTube Shorts
2. Frames are extracted at a moderate interval (2 frames per second over a 5-second window) to capture caption animation cycles
3. This frame sequence is passed to `Gemini 3.1 Pro (Vision)` with a prompt to identify dynamic patterns: the color of the "active" (currently spoken) word vs. "inactive" (upcoming/past) words, text outline styles, font families, shadow/glow effects, text positioning
4. Output is a **dynamic caption JSON payload** — e.g., `{"active_color": "#FF0000", "inactive_color": "#FFFFFF", "outline": "2px solid #000000", "font": "Montserrat Bold", "position": "center-bottom"}` — stored in Convex on the creator profile
5. This payload is passed natively into `@remotion/captions` to programmatically replicate the creator's exact word-tracking style, eliminating generic neon-green defaults

**Layer 2 — Static Brand Tokens (from Thumbnails + Long-Form):**
1. Gemini 3.1 Pro (Vision) also processes the top 5 thumbnails and sample long-form video frames
2. Extracts: primary hex codes, secondary colors, logo positioning, layout preferences, aspect ratio tendencies
3. These static tokens complement the dynamic caption patterns — together they define the full visual brand identity

**Combined output**: A unified JSON design token payload covering both dynamic captions (`@remotion/captions` props) and static brand styling (Remotion composition props). Stored in Convex, linked to the creator profile. Every Remotion composition automatically inherits both layers. Design tokens are refreshed when new Shorts or thumbnails are analyzed, allowing the brand profile to evolve with the creator's visual identity.

### Thumbnail Generation (NanoBanana 2)
1. After clip is synthesized, ThumbnailAgent receives: clip keyframe screenshot, video title, trend context
2. NanoBanana 2 generates a 4K thumbnail with character consistency (creator's likeness preserved across compositions), sub-pixel text rendering for perfect text overlays, and creative composition with trend-relevant imagery
3. Stored in Convex file storage, linked to Content PR
4. PublishAgent uses this thumbnail when uploading the Short

### AI-Driven Video Reframing and Subject Tracking

Currently, Clypt executes temporal segment cuts without applying spatial cropping or dynamic aspect ratio conversions — clips inherit the original frame dimensions of the long-form source. This pipeline adds automated aspect ratio conversion (e.g., 16:9 → 9:16) coupled with AI-driven subject tracking to keep the primary subject centered throughout every generated clip.

**Step 1 — Spatial Coordinate Extraction:**
1. Transcode source videos to exactly 1fps prior to uploading to the Gemini File API to avoid token saturation and minimize computational cost
2. Use the Gemini Vision API at this normalized 1fps rate to output precise spatial bounding box coordinates alongside standard scene descriptions. Output is a JSON array of `[ymin, xmin, ymax, xmax]` values normalized to a scale of **0–1000**
3. Configure inference with `temperature: 0.0` to enforce deterministic, structured outputs and minimize trajectory jitter
4. Manual tracking overrides: inject user-defined spatial constraints (initial coordinate data) directly into the LLM context window to override default saliency detection when the subject of interest is not the most visually salient element

**Step 2 — Trajectory Smoothing and Interpolation:**
Raw 1fps coordinates produce severe visual jitter when applied to a 30fps or 60fps playback timeline. Mathematical smoothing algorithms up-sample the sparse data to match the target framerate:
1. **Exponential Moving Average (EMA)** — Digital low-pass filter for low-latency, synchronous processing environments. Fast, stable smoothing with minimal computational overhead
2. **Kalman Filter** — Handles missing data, occlusion, and rapid subject movement gracefully. Continuously operates in a **Prediction → Update** cycle, estimating the true state of the internal dynamic system even when observations are noisy or absent
3. *(Optional — Cinematic Tier)* **Euclidean-Norm Minimization / Polynomial Spline** — For the highest tier of cinematic output, calculate a low-degree polynomial spline that allows the subject to drift naturally within the "safe zones" of the 9:16 frame rather than being rigidly center-locked, producing a more organic pan-and-scan feel

**Step 3 — Spatial Geometry and Boundary Enforcement:**
1. **Coordinate Denormalization** — Convert the Gemini model's 0–1000 coordinate space into absolute pixel dimensions of the source video
2. **Centroid Extraction** — Extract the exact centroid of the subject's bounding box as the primary anchor point for the virtual camera
3. **Viewport Calculation** — Calculate the exact pixel dimensions of the new 9:16 viewport and horizontally center it on the subject's centroid
4. **Boundary Clamp Enforcement** — Strict left and right boundary clamps applied before passing spatial data to the Remotion layer. Guarantees the virtual camera never pans off the edge of the source media, preventing black bars or transparent backgrounds

**Step 4 — Remotion Composition Integration:**
The reframing pipeline feeds directly into the existing Remotion rendering layer:
1. Drive standard CSS transforms parametrically via Remotion's frame-based React hooks
2. Use `<OffthreadVideo>` rather than standard HTML5 `<video>` tags to guarantee frame-accurate seeking during headless browser evaluation
3. Map the smoothed spatial arrays to `useCurrentFrame()` using Remotion's native `interpolate()` function with `extrapolateLeft: 'clamp'` and `extrapolateRight: 'clamp'` to prevent the video layer from sliding off-screen at sequence boundaries
4. Apply `translateX` and `translateY` CSS properties to enact the dynamic pan-and-scan reframing effect
5. **Dynamic Zoom** — Interpolate a CSS `scale` factor that increases as the calculated bounding box area shrinks (subject moves further from camera). Source ingestion should utilize 4K or exceedingly high-resolution media to prevent pixelation during scaling operations

---

## Gemini Embeddings: Signal Resolution

Gemini Embeddings (`gemini-embedding-001`, 1536 dimensions via MRL truncation) power the semantic layer connecting audience signals to exact video moments. The `gemini-embedding-001` model natively supports Matryoshka Representation Learning (MRL), which encodes the most semantically significant information into the earliest dimensions of the vector. Truncating from the default 3072d to 1536d yields a 50% reduction in Convex vector index storage and proportional search latency improvement with negligible precision loss on MTEB benchmarks — this is Google's officially recommended optimization.

### What Gets Embedded
- **Comment corpus** — entire comment set embedded for agglomerative semantic clustering (conceptual grouping rather than keyword matching)
- **Multimodal graph nodes** — claims, assertions, opinions extracted by NarrativeAnalystAgent, each carrying bound `audioContext[]` + `visualContext[]` from Interval Overlap Binding. Because embeddings capture the full multimodal context, vector search returns nodes matching across text, audio, and visual modalities simultaneously.
- **Audience-grounded video summaries** — tone-aware 2-sentence summaries (generated from transcript + top 15 comments by Gemini 3.1 Flash during onboarding) for catalog-level indexing and tone-matched few-shot RAG

All embeddings live in Convex's built-in vector search, indexed per creator. When a comment says "the bread moment," the agent embeds the phrase and runs a vector search against `narrativeNodes` — matching the graph node whose claim text + bound audio/visual context best fits the query. When NarrativeAnalystAgent detects a claim at minute 2, it runs a vector search across all other graph nodes to find contradictory claims anywhere in the video. When PromptClipAgent receives "find where he argues about privacy and people laugh," a single vector search against multimodal-bound nodes returns exact matches — no post-hoc timestamp fusion required.

### Catalog-Level Embeddings (Onboarding + Trend Trim + Few-Shot RAG)
During Zero-to-One Onboarding, the agent produces **audience-grounded summaries** for each of the creator's **Top 10 most viewed long-form videos** and **Top 10 most viewed Shorts** from the last year: `Gemini 3.1 Flash` processes the raw transcript alongside the top 15 YouTube comments to deduce the video's true tone (satire vs. serious) and generates a tone-aware 2-sentence summary. These summaries are embedded via Gemini Embeddings and stored in Convex. This approach is deeper than vectorizing raw metadata (titles/tags) because it captures semantic content and tonal intent — preventing the system from misclassifying deadpan satire as literal content (e.g., a skit about using MS Word as an IDE would be correctly tagged as comedy, not a tutorial).

This pre-vectorized catalog serves three purposes:
1. **Trend Trim**: When a trend fires, a vector search surfaces relevant videos even when exact keywords don't match (e.g., a trend around "street food in Tokyo" matches a travel vlog that never uses that exact phrase). Tone-aware summaries ensure satirical videos aren't falsely matched to serious trends.
2. **Tone-Matched Few-Shot RAG for Metadata**: When PublishAgent generates titles/descriptions for a new clip, it creates a 1-sentence summary, runs a Convex vector search to retrieve the top 3 most semantically similar past videos, and injects their original titles/descriptions as style exemplars into the Gemini prompt. Because the catalog is tone-indexed, a comedic clip strictly retrieves comedic exemplars — the model mimics the creator's humor formatting, not their tutorial formatting.
3. **Onboarding depth without 3-stream cost**: The audience-grounded summarization fetches raw transcripts (fast, text-only via yt-dlp) and top comments (YouTube Data API), not the heavy STT v2 word-level timestamps or SenseVoice emotion analysis or Video Intelligence API annotations. This gives rich semantic indexing at a fraction of the compute cost of the full 3-stream pipeline.

---

## YouTube Integration

Clypt uses the **YouTube Data API v3** as the primary data source for structured YouTube data (comments, video metadata, channel catalog, publishing). **yt-dlp** extracts Most Replayed heatmaps via InnerTube.

| Scope | Used For |
|---|---|
| `youtube.readonly` | Fetching video metadata, catalog scan, channel info |
| `youtube.upload` | Posting clips (Shorts) via `videos.insert` |
| `youtube.force-ssl` | Comment ingestion via `commentThreads.list` |

For the demo, all three scopes are used: `youtube.readonly` for reading any public video's metadata and comments, `youtube.upload` for publishing clips to our test channel, and `youtube.force-ssl` for comment thread access. The creator's OAuth token is stored in Convex after account connection.

---

## Data Model (Convex)

```typescript
// convex/schema.ts

creators          // channelId, isAuthenticated, OAuth tokens, brandDesignTokens (JSON: dynamic caption patterns {active_color, inactive_color, outline, font} + static brand tokens {hex codes, layout} from Gemini Vision interval-based synthesis)
videos            // creatorId, youtubeVideoId, title, description, tags, chapters, viewCount, likeCount, commentCount, toneAwareSummary (2-sentence, audience-grounded), detectedTone ("satire"|"serious"|"educational"|"entertainment"|"technical_tutorial"|"vlog"|"interview"|"reaction")
videoEmbeddings   // videoId, creatorId, embedding[1536], text (tone-aware summary, not raw metadata) — vectorIndex by_embedding (MRL-truncated from 3072d); pre-vectorized during onboarding for tone-matched catalog search + few-shot RAG
scrapedComments   // videoId, text, likeCount, authorName, emotionalMarkers[]
commentClusters   // videoId, clusterId, commentIds[], centroidEmbedding[1536], conceptLabel, tracedTranscriptSegment: {startMs, endMs}, lagCorrectionMs — agglomerative clusters via Gemini Embeddings, traced to transcript via Lexical Chains
sedEvents         // videoId, clusterId, acousticSignature (description), startMs, endMs, confidence — Gemini 3.1 Pro SED timestamps (via File API) for non-verbal cues flagged by comment clusters
replayHeatmaps    // videoId, segments[{startMs, durationMs, intensity}] — 100 segments from yt-dlp
heatmapSnapshots  // videoId, segments[{startMs, durationMs, intensity}], timestamp — tracks heatmap evolution over time (heatmap at 1hr vs 24hr as audience broadens)

// ── 3-Stream Pipeline Artifacts (deferred — run on-demand per video, not during onboarding) ──
transcripts       // videoId, words[{word, startMs, endMs, speaker, confidence}] — Vertex AI STT v2 unified output (word timestamps + speaker tags in single pass); parsed into Remotion Caption data structures
audioEventTimelines // videoId, events[{tStartMs, tEndMs, eventType, emotion, confidence}] — SenseVoice per-segment emotion + event map; highEnergyWindows[] pre-filtered
videoAnnotations  // videoId, shotChanges[{timeMs}], labels[{entity, category, startMs, endMs, confidence}], objects[{entity, startMs, endMs, boundingBox, confidence}], faces[{startMs, endMs, boundingBox}], explicitContent[{timeMs, likelihood}] — Video Intelligence API structured output

// ── Multimodal Semantic Graph (built by NarrativeAnalystAgent after 3-stream processing + Interval Overlap Binding) ──
narrativeNodes    // videoId, nodeId, claim (text), nodeType ("claim"|"assertion"|"opinion"|"emotional_beat"), sourceStartMs, sourceEndMs, embedding[1536],
                  //   audioContext[]: [{tStartMs, tEndMs, eventType ("laughter"|"cheering"|"silence"|"music"|"sound_effect"), emotion ("happy"|"sad"|"angry"|"excited"|"neutral"), confidence}] — bound from AudioEventAgent via Interval Overlap Binding,
                  
//   visualContext[]: [{tStartMs, tEndMs, entity, category, boundingBox, facesVisible, confidence}] — bound from VideoAnnotationAgent via Interval Overlap Binding,
                  //   humorContext[]: [{reference, domain, mechanism, layers[], audiencePrerequisite, confidence}] — from Content Mechanism Decomposition; populated for nodes flagged as potentially comedic,
                  //   emotionalContext[]: [{type, mechanism, arcPosition, parasocialIntensity, confidence}] — from Content Mechanism Decomposition; populated for nodes with emotion peaks,
                  //   socialContext[]: [{dynamicType, initiatorSpeaker, recipientSpeaker, energyDelta, confidence}] — from Content Mechanism Decomposition; populated for multi-speaker dynamic shifts,
                  //   expertiseContext[]: [{type, domain, audienceReactionValidation, confidence}] — from Content Mechanism Decomposition; populated for expertise signaling nodes,
                  //   speakerTag (string, from STT v2 diarization) — carries speaker identity for speaker coherence scoring
narrativeEdges    // videoId, sourceNodeId, targetNodeId, edgeType ("contradiction"|"causal_link"|"thematic_continuation"|"tension_resolution"|"callback"|"escalation"|"subversion"|"analogy"|"revelation"), confidence, narrativeClassification (irony, callback, self_contradiction, escalation, competence_paradox, corporate_hypocrisy, cross_modal_contradiction, cause_effect, problem_solution, topic_recurrence, buildup_payoff, suspense_reveal, expectation_violation, conceptual_mapping, recontextualization),
                  //   crossModalEvidence: {textClaim, contradictingModality ("visual"|"audio"), contradictingData} (optional — populated for cross-modal contradictions, e.g., Text: "I am calm" + Visual: "sweating")

contentPRs        // creatorId, videoId, mode ("engagement"|"trend"|"content"|"prompt"), trendTopic, promptQuery,
                  //   startMs, endMs, score (weighted composite),
                  //   subgraphNodeIds: [nodeId] (ordered list of graph nodes comprising the scored Semantic Sub-graph — clip boundaries are deterministic from first/last node timestamps + bound reactions),
                  //   predictedTone ("satire"|"technical_tutorial"|"vlog"|"interview"|"reaction"),
                  //   dimensionWeights: {hook, flow, emotionalPeak, visualInterest, speakerCoherence, narrativeEdge} (tone-specific multipliers, e.g., {narrativeEdge: 1.2, emotionalPeak: 1.2, ...} for satire),
                  //   hookScore (0–10), hookReason (string), alternativeInPoint (ms, optional),
                  //   dimensionScores: {hook, flow, emotionalPeak, visualInterest, speakerCoherence, narrativeEdge} (raw, pre-weight),
                  //   narrativeEdgeEvidence: {sourceNodeId, targetNodeId, edgeType, narrativeClassification, crossModalEvidence (optional)},
                  //   status, reasoning, evidence: {boundNodeEvidence[] (audioContext + visualContext per node), heatmapIntensity, commentCluster, sedEvent},
                  //   thumbnailStorageId,
                  //   remotionCompositionId, brandDesignTokensUsed

audioProductions  // contentPRId, videoId, musicBedStorageId, foleyStorageId, mixedMasterStorageId, moodProfile (from Gemini 3.1 Pro), predictedTone, duckingTimestamps[{startMs, endMs}] (from STT v2 word timestamps), stingerTimestamps[{ms, type}], renderStatus
veoGenerations    // contentPRId, videoId, type ("broll"|"extension"), sourceNodeId (graph node that triggered generation), veoPrompt (text), sourceFrameStorageId (for extensions), generatedVideoStorageId, durationMs, renderStatus
clips             // contentPRId, videoId, startMs, endMs, storageId, audioProductionId, veoGenerationIds[], renderStatus, score, isNonLinear (boolean — true for multi-segment narrative edge clips)
trendAlerts       // topic, spikePct, tier, source, status, matchedCreatorIds[]
agentRuns         // creatorId, agentType, status, summary, laminarTraceId
```

Key design notes:
- `creators.isAuthenticated` distinguishes between "we're analyzing this channel" (demo mode) and "this creator can publish via Clypt" (production mode)
- `creators.brandDesignTokens` stores the combined JSON payload from interval-based brand synthesis: dynamic caption patterns (`active_color`, `inactive_color`, `outline`, `font` → injected into `@remotion/captions`) + static brand tokens (hex codes, layout → Remotion composition props)
- `videos.toneAwareSummary` + `detectedTone` store the audience-grounded summary generated during onboarding (Gemini 3.1 Pro processes transcript + top 15 comments to deduce satire vs. serious tone)
- `videoEmbeddings` are pre-populated during Zero-to-One Onboarding from tone-aware summaries (not raw metadata) — ensures tone-matched retrieval for few-shot RAG; 3-stream artifacts are deferred
- `commentClusters` stores semantically grouped comments (agglomerative clustering via Gemini Embeddings) with Lexical Chain traces to transcript segments and lag correction — replaces rigid regex timestamp extraction
- `sedEvents` stores Gemini 3.1 Pro SED timestamps (via File API) for non-verbal cues flagged by comment clusters — 100% Google stack
- `narrativeNodes` stores the **Multimodal Semantic Graph**: each node contains the transcript claim text + deterministically bound `audioContext[]` (SenseVoice events) and `visualContext[]` (Video Intelligence API structured annotations: labels, objects, faces, shot changes) mapped via Interval Overlap Binding. This makes every node a self-contained multimodal unit — the ClipScoringAgent reads all evidence from graph nodes, never cross-referencing separate stream tables. `narrativeEdges` stores classified narrative relationships across the full taxonomy: `contradiction` (including `cross_modal_contradiction` — text claims contradicted by bound visual/audio evidence), `causal_link` (setup/resolution chains), `thematic_continuation` (entity tracking across tangents), and `tension_resolution` (challenge/payoff arcs)
- `replayHeatmaps` stores real YouTube Most Replayed data extracted via yt-dlp — not simulated
- `transcripts` stores unified word-level timestamps + speaker tags from Vertex AI STT v2 (`latest_long`, single pass) — parsed into Remotion Caption data structures for programmatic text rendering
- `audioEventTimelines` stores the SenseVoice per-segment output — consumed by NarrativeAnalystAgent's Interval Overlap Binding step (mapped onto graph nodes as `audioContext[]`) and by EngagementAnalysisAgent (Crowd Clip) for signal fusion
- `videoAnnotations` stores the Video Intelligence API structured output (shot changes, labels, objects, faces, explicit content) — consumed by NarrativeAnalystAgent's Interval Overlap Binding step (mapped onto graph nodes as `visualContext[]`). Structured JSON with exact ms timestamps — zero hallucination, deterministic
- `veoGenerations` stores Veo-generated B-roll clips and clip extensions — linked to the semantic graph nodes that triggered generation and to Remotion compositions for overlay/sequencing
- `audioProductions` stores Lyria-generated audio assets per clip — music bed, foley tracks, mixed master, mood profile, ducking timestamps derived from STT v2 word boundaries, and audio stinger timing from graph node boundaries
- `contentPRs.predictedTone` stores the Orchestrator's tone classification for the source video — drives Dynamic Dimension Weighting in ClipScoringAgent
- `contentPRs.dimensionWeights` stores the tone-specific multipliers applied (e.g., `{narrativeEdge: 1.2, emotionalPeak: 1.2}` for satire) — enables full auditability of why a clip ranked the way it did
- `contentPRs.dimensionScores` stores raw (pre-weight) scores across all 6 dimensions including `narrativeEdge` (from NarrativeAnalystAgent — scores the strongest edge type for the sub-graph: contradiction for comedy, causal link for tutorials, tension/release for vlogs)
- `contentPRs.narrativeEdgeEvidence` links to specific narrative graph nodes — the source and target of the detected narrative edge, with edge type, narrative classification, and optional cross-modal evidence
- `contentPRs.subgraphNodeIds` stores the ordered list of graph node IDs comprising the Semantic Sub-graph that was scored — enables full traceability from clip boundaries to the exact multimodal evidence that justified them
- `contentPRs.remotionCompositionId` references the Remotion composition used to render the clip
- `clips.isNonLinear` flags clips that composite multiple source segments (narrative-edge-based clips — contradiction compilations, tutorial montages, tension/release supercuts)
- `outboundDrafts` stores Gemini-drafted brand collaboration emails
- All tables support Convex real-time subscriptions — the dashboard updates live as agents work

---

## Challenge Demo Flow

Each feature has its own demo case. The challenge demo opens with the infographic and business pitch, clearly noting which elements are pre-computed or simulated for demo purposes. Then each feature is demonstrated in sequence.

### Feature 1 Challenge Demo: Content Clip
- **Setup**: Easiest demo case. Put in a video URL → Clypt clips based on pure content/metadata analysis AND aggregate channel intelligence (what kind of audiences watch, popular topics, etc.).
- **Show**: 3-stream pipeline running (transcript + audio events + visual descriptions) → Multimodal Semantic Graph construction (Interval Overlap Binding) → ClipScoringAgent evaluating Semantic Sub-graphs, producing Content PRs with hook scores and bound multimodal evidence per node. **Lyria audio production**: show the music bed generation, foley sync, and audio ducking happening in real-time — this is the multimodal showcase moment.
- **Also demo Cortex here** (prompt-based clipping + voice editing): Since this mode is the simplest and fastest, it gives the most time to showcase the editor experience:
  - Open a Content PR → hold mic button → "Find the part where they laugh at the whiteboard"
  - Gemini Live API → PromptClipAgent → clip renders → preview plays
  - "Extend by 2 seconds" → re-render → "Post it" → PublishAgent fires
- **Note**: Prompt-based clipping and voice editing are cross-cutting editor capabilities — they work in Cortex regardless of how the creator got there (Content Clip, Crowd Clip, or Trend Trim). Demoed here because Content Clip is the simplest and gives us the most room.
- **Emphasize**: This is the mode where creators actively direct the clipping, vs. Crowd Clip and Trend Trim which are autonomous.

### Feature 2 Challenge Demo: Crowd Clip
- **Setup**: Convex cron has been polling a newly released video from a major creator (chosen before the challenge demo) — engagement velocity snapshots (views, likes, comments) and Most Replayed heatmap snapshots have been accruing over time.
- **Entry point**: Notification: "Your new video '[title]' is trending with your food audience — here are some clips." Open notification, click through to Cortex.
- **Show**: Comments (already fetched via Data API), heatmap overlay, engagement velocity chart, metadata, actual video frame. Content PRs with multi-signal evidence (heatmap peak + comment cluster + SenseVoice laughter + Video Intelligence API labels/objects). Thumbnail generated by NanoBanana 2. **Lyria audio**: play the clip with its AI-generated music bed — demonstrate audio ducking under speech and foley on transitions.
- **Emphasize**: This happened automatically — the Crowd Clip pipeline surfaced these clips without the creator doing anything.

### Feature 3 Challenge Demo: Trend Trim
- **Entry point**: Trend notification (simulated trigger): "Bad Bunny is trending +340% — we found clips in your catalog."
- **Show**: TrendScout analyzing Google Trends + YouTube trending data via Gemini 3.1 Pro, then CatalogMatch finding relevant videos from the creator's back catalog via Gemini Embeddings vector search. Content PRs with trend-matched clips + Lyria audio production.
- **Demo cases**:
  - **Tier 1 (macro trend)**: e.g., Bad Bunny trending → Hot Ones channel has interview clips mentioning Bad Bunny (large, obvious match)
  - **Tier 2 (niche trend)**: Find a niche trend during the challenge window → match to a tech channel like Joma Tech (resonates with the target audience)
- **Back-catalog resurfacing**: Explain verbally that YouTube periodically resurfaces old content — Clypt detects when old videos start gaining velocity and generates fresh clips. (Can't easily trigger YouTube's resurfacing algorithm live, so explain the mechanism rather than demoing it.)
- **Emphasize**: Fully autonomous — trend fires, catalog matched, clips generated, creator notified. No creator action required.

---

## Model Summary

| Model | Role |
|---|---|
| **Gemini 3.1 Pro** | Orchestrator — routing, coordination. Deep reasoning subagents — Crowd Clip signal fusion (EngagementAnalysisAgent), Trend Trim catalog matching (CatalogMatchAgent), Content Clip scoring via Semantic Sub-graphs (ClipScoringAgent), prompt-based search against multimodal graph nodes (PromptClipAgent). NarrativeAnalystAgent builds Multimodal Semantic Graph (Interval Overlap Binding → cross-modal Few-Shot CoT NLI narrative edge classification across full taxonomy: contradictions, causal links, thematic continuations, tension/release arcs — two-stage vector loop optimization avoids O(n²) LLM calls — no fine-tuning, prompt-driven `reasoning_trace` before classification). Humor Decomposition Layer decomposes comedic nodes into reference, domain, mechanism, layers, and audience prerequisite dimensions. Tone-matched few-shot metadata generation via RAG (PublishAgent). AudioProductionAgent mood/energy classification for Lyria. |
| **Gemini 3.1 Pro (Vision)** | BrandIdentityAgent — interval-based brand synthesis: sequential frame analysis of creator's Shorts (2fps) to identify dynamic caption patterns (active/inactive word colors, outlines, fonts) → JSON for `@remotion/captions`. Also processes thumbnails + long-form frames for static brand tokens. Native agentic vision for advanced video understanding. Runs during onboarding and periodically refreshed. |
| **SenseVoice** | AudioEventAgent — per-second emotion + audio event detection (speech emotion, laughter, cheering, silence, music). 70ms latency, native waveform processing. Integrated into Crowd Clip signal fusion, Trend Trim candidate validation, Content Clip scoring. |
| **Video Intelligence API** | VideoAnnotationAgent — structured temporal video understanding: SHOT_CHANGE_DETECTION (exact scene cuts), LABEL_DETECTION + OBJECT_TRACKING (entities/objects with temporal bounding boxes), FACE_DETECTION (face tracking), EXPLICIT_CONTENT_DETECTION (brand safety). Single API call, structured JSON output, zero hallucination — replaces LLM frame extraction. |
| **Veo (Vertex AI)** | VeoGenerationAgent — B-roll generation from semantic graph nodes (text/image prompt → 1080p 9:16 video), clip extension via first-frame conditioning (final frame → 2s smooth continuation). Enables generative video that the current stack can't cut from source footage. |
| **Gemini 3.1 Flash** | Audience-grounded summarization during onboarding (transcript + top 15 comments → tone-aware summaries embedded for few-shot RAG; speed over depth). |
| **Vertex AI STT v2** (`latest_long`) | TranscriptAgent — unified transcription + diarization in a single `BatchRecognize` call with `enableWordTimeOffsets` + `diarizationConfig`. Returns `{word, startMs, endMs, speakerTag}` per token → parsed directly into Remotion `Caption` data structures. No multi-model merging. Enables speaker coherence scoring, clean single-speaker clip selection, and precise Lyria audio ducking boundaries. |
| **Lyria** | AudioProductionAgent — DeepMind's audio generation model for complete clip audio production: custom music beds (tone-matched via Gemini 3.1 Pro mood classification), foley/sound design (transitions, impacts synced to Remotion visual cuts), audio stingers (hook accents, punchline enhancers timed to graph node boundaries), intelligent audio mixing/ducking (music dips under speech using STT v2 word timestamps). Every clip gets broadcast-quality audio. |

| **Gemini Live API** | Voice-first editing interface via real-time speech-to-speech with LiveKit |
| **Gemini Embeddings** | Semantic vector indexing — audience-grounded summary embedding (tone-matched catalog), comment semantic clustering, multimodal graph node embedding (captures text + bound audio/visual context for cross-modal search), catalog search (onboarding + tone-matched few-shot RAG), vague reference resolution, prompt-based graph search (gemini-embedding-001, 1536d via MRL truncation) |
| **Remotion + Lyria + Veo** | Code-driven video synthesis + AI audio production + generative video — React + Tailwind CSS video DOM, STT v2 Caption components, brand design tokens, Lyria audio layer (music bed + foley + stingers + ducked mix), Veo B-roll overlays and clip extensions. Non-linear narrative edge clips composite multiple source segments with Lyria transition audio and Veo-generated footage. Replaces FFmpeg. |
| **NanoBanana 2** | Thumbnail generation — SOTA image composition with character consistency (creator's likeness preserved across thumbnails), sub-pixel text rendering, 4K output with creative controls |


---

## Platform Tech Coverage

| Tech | How Clypt Uses It | Prominence |
|---|---|---|
| **Gemini 3.1 (provided)** | Gemini 3.1 Pro (orchestrator + reasoning + Multimodal Semantic Graph builder + cross-modal Few-Shot CoT NLI + Humor Decomposition Layer + tone-matched few-shot RAG + AudioProductionAgent mood classification + open-vocabulary SED via File API), Gemini 3.1 Pro (Vision) (interval-based brand synthesis + dynamic caption extraction), Gemini 3.1 Flash (audience-grounded summarization), Gemini Live API (voice editing), Gemini Embeddings (multimodal graph node embedding + tone-aware catalog indexing + semantic clustering + few-shot RAG), Vertex AI STT v2 | Central — orchestration, reasoning, embeddings |
| **Veo** (provided) | VeoGenerationAgent — B-roll generation from semantic graph nodes (Gemini writes Veo prompt → 4–8s 1080p 9:16 footage for Remotion overlay). Clip extension via first-frame conditioning (final frame → 2s smooth continuation). Enables generative video the source footage can't provide. | Central — generative video, new capability |
| **Video Intelligence API** | VideoAnnotationAgent — purpose-built temporal video understanding. Single `annotateVideo` call: SHOT_CHANGE_DETECTION (exact scene cuts), LABEL_DETECTION + OBJECT_TRACKING (temporal bounding boxes), FACE_DETECTION, EXPLICIT_CONTENT_DETECTION (brand safety). Structured JSON output — replaces LLM frame extraction, eliminates hallucination. | Central — video understanding backbone |
| **SenseVoice** | AudioEventAgent — per-second emotion + audio event detection on raw waveform (70ms latency). Speech emotion, laughter, cheering, silence, music, sound effects — all with exact ms timestamps for Interval Overlap Binding. | Central — audio understanding backbone |
| **Lyria** (provided) | AudioProductionAgent — complete AI audio production for every clip: custom music beds (tone-matched via Gemini 3.1 Pro mood/energy classification), foley/sound design (transitions, impacts synced to Remotion visual cuts), audio stingers (hook accents, punchline enhancers timed to graph node boundaries), intelligent audio mixing/ducking (music dips under speech using STT v2 word timestamps). This is the "wasn't possible six months ago" feature. | Central — every clip gets broadcast-quality audio |
| **NanoBanana 2** (provided) | ThumbnailAgent — SOTA thumbnail generation with character consistency (creator's likeness preserved across compositions), sub-pixel text rendering for perfect text overlays, 4K creative composition. | Central — every clip gets a SOTA thumbnail |
| **Convex** | Full backend: database, vector search, file storage, crons, auth, actions | Central — entire data layer |
| **Vercel** | Frontend hosting, CI/CD, preview deployments | Supporting — deployment |

---

## Appendix: Service Reference

**Challenge-Relevant Tech:**
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Bidi-Streaming Dev Guide](https://google.github.io/adk-docs/streaming/dev-guide/part1/)
- [Live API Notebooks/Apps (Google Cloud)](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api)
- [Gemini API Docs (Multimodal + Interleaved Output)](https://ai.google.dev/gemini-api/docs)
- [Challenge FAQ: Google Cloud Deployment Proof](https://geminiliveagentchallenge.devpost.com/details/faqs)

**Google Cloud:**
- [Vertex AI Speech-to-Text v2 — `latest_long` model](https://cloud.google.com/speech-to-text/v2/docs/chirp-model)
- [Vertex AI STT v2 — Speaker Diarization Config](https://cloud.google.com/speech-to-text/v2/docs/multiple-voices)
- [Google Cloud Video Intelligence API](https://cloud.google.com/video-intelligence/docs)
- [Veo on Vertex AI — Video Generation](https://cloud.google.com/vertex-ai/generative-ai/docs/video/overview)
- [Google ADK (Agent Development Kit)](https://github.com/google/adk-node)

**Infrastructure:**
- [Remotion — Programmatic Video Synthesis (React)](https://www.remotion.dev/docs)
- [Remotion Captions — Word-Level Caption Components](https://www.remotion.dev/docs/captions)
- [Convex](https://docs.convex.dev/)
- [Vercel](https://vercel.com/docs)
- [LiveKit](https://docs.livekit.io/)
- [yt-dlp — Heatmap + Frame Extraction](https://github.com/yt-dlp/yt-dlp)
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
