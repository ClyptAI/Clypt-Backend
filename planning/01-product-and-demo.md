# Product and Demo

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md)

---
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
