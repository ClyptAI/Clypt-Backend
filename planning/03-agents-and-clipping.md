# Agents and Clipping

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Data/Integrations](./04-data-integrations-and-reference.md)

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

