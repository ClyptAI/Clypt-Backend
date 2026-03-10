# Clypt Semantic Graph Pipeline

> [!WARNING]
> This document is legacy and does not reflect the current Phase 1A migration plan (Modal GPU extraction, no Phase 1A-R reconciliation stage).
> Use `planning/01-product-and-demo.md`, `planning/02-system-architecture.md`, `planning/03-agents-and-clipping.md`, and `planning/04-data-integrations-and-reference.md` as the active planning source.

Complete documentation for the Clypt video analysis and clip generation pipeline. This system takes a YouTube URL and produces rendered 9:16 short-form video clips by building a semantic graph of the video's content, storing it in Google Cloud Spanner, and using Gemini to identify viral moments.

## Architecture Overview

The pipeline runs sequentially through seven stages:

```
YouTube URL
    │
    ▼
Phase 1A: Deterministic Extraction ──→ phase_1a_visual.json, phase_1a_audio.json
    │
    ▼
Phase 1A-R: Speaker Reconciliation ──→ phase_1a_speaker_map.json
    │
    ▼
FFmpeg Re-encode ──→ clean H.264 video for Remotion
    │
    ▼
Phase 1B: Content Mechanism Decomposition ──→ phase_1b_nodes.json
    │
    ▼
Phase 1C: Narrative Edge Mapping ──→ phase_1c_narrative_edges.json
    │
    ▼
Phase 2: Multimodal Embedding ──→ phase_2_embeddings.json
    │
    ▼
Phase 3: Storage & Graph Binding ──→ Spanner (nodes + edges) + GCS (speaker-tagged tracking)
    │
    ▼
Phase 4: Auto-Curate ──→ remotion_payloads_array.json (with active_speaker_timeline)
    │
    ▼
Remotion Render ──→ out/clip-1.mp4, clip-2.mp4, ...
```

### Design Principles

1. **Math First, Reasoning Second.** Phase 1A establishes mathematically precise ground truth (timestamps, bounding boxes, word boundaries) before any AI reasoning occurs. Gemini never has to guess at timestamps — it anchors every semantic decision to deterministic API data.

2. **Two-Call Graph Generation.** Semantic extraction is split across two Gemini calls to stay within the 65,536 output token limit. Phase 1B is the multimodal "heavy" call (video + ledgers). Phase 1C is a text-only "light" call that only processes Phase 1B's output. Both enforce structured JSON output via `response_mime_type` and `response_schema`.

3. **Decoupled Spatial Tracking.** Graph databases are optimized for relationship traversal, not storing 30fps time-series arrays. Spanner nodes store only a GCS URI pointer to their tracking data. The heavy bounding box coordinates live in GCS, fetched at render time.

4. **Spanner Multi-Model Architecture.** Google Cloud Spanner serves triple duty: relational storage for node metadata, ScaNN vector index for embedding similarity search, and Spanner Graph for narrative edge traversal — all in a single database with ACID transactions.

5. **Late Fusion Embeddings.** Each node's embedding fuses text semantics (transcript + vocal delivery + content mechanisms) with actual video frame embeddings from the node's exact time range, producing a single 1408-dimensional vector that captures both modalities.

---

## Project Structure

```
Clypt-PreYC/
├── pipeline/                          # All Python pipeline scripts
│   ├── __init__.py
│   ├── run_pipeline.py                # Orchestrator (entry point)
│   ├── phase_1a_extract.py            # Deterministic extraction
│   ├── phase_1a_reconcile.py          # Speaker-to-face reconciliation
│   ├── phase_1b_decompose.py          # Content mechanism decomposition
│   ├── phase_1c_edges.py              # Narrative edge mapping
│   ├── phase_2_embed.py               # Multimodal embedding
│   ├── phase_3_store.py               # Spanner + GCS storage
│   ├── phase_4_auto_curate.py         # Auto-curator (full-graph sweep)
│   └── phase_4_retrieve.py            # Query-based retrieval (standalone)
├── clypt-render-engine/               # Remotion rendering project
│   ├── src/
│   │   ├── Root.tsx                   # Composition registration
│   │   ├── ClyptViralShort.tsx        # Video component with dynamic cropping
│   │   └── index.ts
│   ├── scripts/
│   │   └── fetch_tracking.js          # GCS tracking data fetcher
│   ├── public/                        # Symlinked video + tracking at render time
│   └── out/                           # Rendered MP4 clips
├── outputs/                           # All intermediate JSON files (gitignored)
├── downloads/                         # Downloaded media files (gitignored)
├── planning/
│   └── Semantic_Graph_Architecture.md # This file
├── spannerSchema.sql                  # Spanner DDL
├── requirements.txt                   # Python dependencies
└── .gitignore
```

---

## Running the Pipeline

### Prerequisites

- Python 3.12+
- Node.js 18+ and npm
- FFmpeg (`brew install ffmpeg`)
- Google Cloud SDK authenticated (`gcloud auth application-default login`)
- GCP project `clypt-preyc` with:
  - Cloud Storage bucket `clypt-test-bucket`
  - Video Intelligence API enabled
  - Speech-to-Text v2 API enabled
  - Vertex AI API enabled
  - Spanner instance `clypt-preyc-db` with database `clypt-db` (schema from `spannerSchema.sql`)

### Install Dependencies

```bash
pip install -r requirements.txt
cd clypt-render-engine && npm install && cd ..
```

### Run

```bash
python3 pipeline/run_pipeline.py
```

The orchestrator prompts for a YouTube URL, then runs every phase sequentially through rendering. Output clips land in `clypt-render-engine/out/`.

---

## Pipeline Stages

### Orchestrator — `pipeline/run_pipeline.py`

The single entry point for the entire pipeline. It:

1. Prompts the user for a YouTube URL via stdin.
2. Calls each phase's `main()` function in sequence.
3. After Phase 1A, runs **Phase 1A-R** (speaker-to-face reconciliation) to correlate STT speaker tags with face tracks.
4. Runs an FFmpeg re-encode pass on the downloaded video to produce a clean H.264 file that Chrome Headless Shell (used by Remotion) can decode without glitches. The original download is preserved as `video_original.mp4`.
5. After Phase 4, symlinks the pipeline outputs (`downloads/video.mp4` and `outputs/remotion_payloads_array.json`) into the Remotion project directory — no file copying.
6. Runs the Node.js tracking fetcher script.
7. Renders each Remotion composition individually to `clypt-render-engine/out/`.

The orchestrator excludes `phase_4_retrieve.py` since that script is query-based and designed for standalone use.

---

### Phase 1A: Deterministic Extraction — `pipeline/phase_1a_extract.py`

**Purpose:** Establish the mathematical ground truth of the video before any AI reasoning happens.

**GCP Services:** Video Intelligence API v1, Speech-to-Text v2 (Chirp 3), Cloud Storage

**Process:**

1. **Media Acquisition** — Downloads two files from YouTube using `yt-dlp`:
   - A muxed video+audio stream (`video.mp4`), forcing H.264 codec (`vcodec^=avc1`) for compatibility with both the Video Intelligence API and Remotion. The audio track is preserved in this file because Phase 1B's Gemini call needs to hear vocal prosody.
   - A separate audio-only stream (`audio.m4a`) for the Speech-to-Text engine. If no standalone audio stream is available on YouTube, the script falls back to extracting audio from the muxed video via FFmpeg.

2. **GCS Upload** — Both files are uploaded to `gs://clypt-test-bucket/phase_1a/` so the cloud APIs can access them.

3. **Parallel Engine Execution** — Two analysis engines run concurrently via `asyncio.gather(return_exceptions=True)`:

   **Visual Engine** submits five separate Video Intelligence API long-running operations (split for reliability):
   - `SHOT_CHANGE_DETECTION` — scene cut boundaries with start/end milliseconds.
   - `FACE_DETECTION` — per-frame face bounding boxes with attributes, enabling spatial tracking.
   - `PERSON_DETECTION` — per-frame person bounding boxes with pose landmarks.
   - `OBJECT_TRACKING` — tracked objects with entity IDs and per-frame bounding boxes.
   - `LABEL_DETECTION` — shot-level, segment-level, and frame-level semantic labels.

   All five LROs execute server-side in parallel and are polled sequentially. Results are merged across all `annotation_results` entries (the API may split results across multiple entries).

   **Audio Engine** submits a Speech-to-Text v2 batch request using the Chirp 3 model with:
   - Word-level time offsets (`enable_word_time_offsets`).
   - Speaker diarization (assigns a `speaker_tag` to each word).
   - The `us` regional endpoint for Chirp 3 availability.

4. **Output** — Two JSON ledgers saved to `outputs/`:
   - `phase_1a_visual.json` — shot changes, person detections, face detections, object tracking, and label detections, all with millisecond timestamps and normalized bounding boxes.
   - `phase_1a_audio.json` — a flat array of words (each with `start_time_ms`, `end_time_ms`, `speaker_tag`, `confidence`) and full transcript segments.

**Known Behavior:** `PERSON_DETECTION` and `OBJECT_TRACKING` may return "Calculator failure" errors from the Video Intelligence API for certain videos. The script logs these errors but continues — the pipeline handles missing data gracefully.

---

### Phase 1A-R: Speaker Reconciliation — `pipeline/phase_1a_reconcile.py`

**Purpose:** Correlate STT speaker tags (acoustic identity, no spatial info) with Video Intelligence face tracks (spatial bounding boxes, no speaker identity) to produce a unified speaker→face mapping.

**Inputs:** `phase_1a_visual.json`, `phase_1a_audio.json`

**Algorithm:**

1. **Spatial Identity Assignment** — Computes the median `center_x` of each face track's bounding boxes to assign a stable horizontal position (e.g., "left speaker" vs "right speaker").

2. **Voting Matrix** — For each word with a `speaker_tag` in the audio ledger:
   - Finds all face tracks active at that word's timestamp.
   - Computes a lip-activity proxy using bounding box height variance (speakers exhibit more vertical bbox fluctuation due to mouth movement).
   - Accumulates weighted votes into a `speaker_tag × face_track_index` matrix.

3. **Greedy Argmax Assignment** — Walks through speakers by vote count descending. Each speaker is assigned to its highest-voted face track. Already-assigned tracks are excluded to enforce 1:1 mapping. A spatial fallback (left-to-right ordering) handles speakers with no clear winner.

**Output:** `outputs/phase_1a_speaker_map.json` — a JSON object with:
- `speaker_to_track`: mapping of `speaker_tag` → `face_track_index`
- `track_positions`: metadata with median `center_x` per face track for debugging

---

### Phase 1B: Content Mechanism Decomposition — `pipeline/phase_1b_decompose.py`

**Purpose:** Use Gemini 3.1 Pro as a multimodal reasoning engine to decompose the video into semantically meaningful nodes with rich metadata about *why* each moment works.

**GCP Services:** Vertex AI (Gemini 3.1 Pro), via `google-genai` SDK

**The Token Limit Problem:** The Phase 1A visual ledger can exceed 4 million characters for a long video, far beyond Gemini's 1,048,576-token input limit. This script solves it with adaptive chunked processing.

**Process:**

1. **Load Phase 1A Ledgers** — Reads `phase_1a_visual.json` and `phase_1a_audio.json` from disk.

2. **Chunk Planning** — Groups consecutive shots into chunks that fit within a 200,000-token budget. Token costs are estimated per-shot based on the density of face frames, person frames, object frames, word count, and label segments in that shot's time range. Each chunk's boundaries are aligned to shot changes to avoid splitting a shot across chunks.

3. **Per-Chunk Processing** — For each chunk, the script:
   - Slices the visual and audio ledgers to only include data within the chunk's time range.
   - Constructs a user prompt specifying the time window and including the sliced ledger data.
   - Sends the prompt along with the GCS video file (as a `Part.from_uri`) to Gemini, with the system instruction and a Pydantic-enforced response schema.
   - Gemini returns a JSON array of `SemanticNode` objects for that time range.

4. **Merge & Deduplication** — Nodes from all chunks are sorted by `start_time` and deduplicated at boundaries (nodes within 1 second of each other are merged, keeping the higher-confidence version).

**Gemini System Instruction:** Instructs the model to:
- Assemble `transcript_segment` strictly from the audio ledger's word timestamps, filtering out STT noise.
- Cross-reference visual actions, facial expressions, and vocal prosody from the video with the transcript.
- Describe non-verbal audio cues (laughter, sarcastic tones, shouting) in the `vocal_delivery` field, keeping the transcript clean.
- Evaluate four content mechanism dimensions: Humor, Emotion, Social Dynamics, and Expertise.

**Schema (Pydantic-enforced):** Each `SemanticNode` contains:
| Field | Type | Description |
|---|---|---|
| `start_time` | float | Start time in seconds, matched to audio ledger word boundaries |
| `end_time` | float | End time in seconds |
| `transcript_segment` | string | Clean transcript assembled from Phase 1A word timestamps |
| `visual_description` | string | Brief description of visual actions, referencing shot changes and objects |
| `vocal_delivery` | string | Non-verbal audio cues (laughter, tone, background sounds) |
| `confidence_score` | float | 0.0–1.0 confidence in the node's accuracy |
| `content_mechanisms` | object | Four sub-objects (`humor`, `emotion`, `social`, `expertise`), each with `present` (bool), `type` (string), `intensity` (float 0.0–1.0) |

**Gemini Configuration:** Uses `thinking_config=ThinkingConfig(thinking_level="HIGH")` for maximum reasoning depth, and `temperature=0.2` with Pydantic-enforced structured output.

**Output:** `outputs/phase_1b_nodes.json` — a deduplicated array of Semantic Nodes ordered chronologically.

---

### Phase 1C: Narrative Edge Mapping — `pipeline/phase_1c_edges.py`

**Purpose:** Map the structural narrative relationships between Semantic Nodes by drawing directional edges — the "graph topology" pass.

**GCP Services:** Vertex AI (Gemini 3.1 Pro, text-only), via `google-genai` SDK

**Process:**

1. **Load Nodes** — Reads `phase_1b_nodes.json`.
2. **Text-Only Gemini Call** — Sends the entire node array as text to Gemini (no video attachment). The prompt places the node JSON data first with instructions after it (data-first ordering anchors Gemini's reasoning on the data before generation). Uses `thinking_config=ThinkingConfig(thinking_level="HIGH")` for deeper reasoning — this is the cheapest phase to enable thinking on since it's text-only.
3. **Pydantic Validation** — Parses the response and validates edge references match actual node `start_time` values.

**Edge Taxonomy** (enum-constrained):
| Edge Type | Description | Example |
|---|---|---|
| Contradiction | Logically clashing claims, or text contradicted by evidence | "I prioritize privacy" ↔ "We sell your data" |
| Causal Link | One node establishes a problem; another resolves or extends it | "Docker keeps crashing" → "Added memory limits" |
| Thematic | Same topic resurfaces after a digression | Mentions React at min 3, tangent, React again at min 12 |
| Tension / Release | Emotional stakes escalate, then resolve | "24 hours to ship" → "We actually did it" |
| Callback | Deliberate authorial reference to an earlier moment | "Remember the API I showed you?" → links to min 3 |
| Escalation | Sustained intensity ramp with no release yet | Spice challenge: mild → hot → Carolina Reaper |
| Subversion | Expectation established, then broken | "I followed the tutorial exactly" → total disaster |
| Analogy | Concept explained by mapping onto a familiar domain | "A blockchain is like a Google Doc everyone edits" |
| Revelation | New information recontextualizes prior nodes | "What I didn't mention is... this was all staged" |

**Schema:** Each `NarrativeEdge` contains:
| Field | Type | Description |
|---|---|---|
| `from_node_start_time` | float | `start_time` of the origin node |
| `to_node_start_time` | float | `start_time` of the destination node |
| `edge_type` | enum | One of the nine edge types above |
| `narrative_classification` | string | 1-sentence explanation of why these nodes connect |
| `confidence_score` | float | 0.0–1.0 |

**Output:** `outputs/phase_1c_narrative_edges.json`

---

### Phase 2: Multimodal Embedding — `pipeline/phase_2_embed.py`

**Purpose:** Project each Semantic Node into a shared 1408-dimensional multimodal vector space, fusing text semantics with actual video frame embeddings.

**GCP Services:** Vertex AI Multimodal Embeddings (`multimodalembedding@001`)

**Process:**

For each node in `phase_1b_nodes.json`:

1. **Text Payload Construction** — Concatenates the node's `transcript_segment`, `vocal_delivery`, and a stringified summary of active `content_mechanisms` (e.g., `"Humor: subversion (0.8), Social: status reversal (0.9)"`) into a single rich text string.

2. **Dual Embedding Request** — Calls `MultiModalEmbeddingModel.get_embeddings()` with:
   - `contextual_text`: the rich text payload → produces a 1408-d text vector.
   - `video`: the GCS video file, with a `VideoSegmentConfig` constraining to the node's exact `start_time`–`end_time` range → produces one or more 1408-d video vectors.

3. **Late Fusion (Mean Pooling)** — If multiple video embeddings are returned for the segment, they're averaged into a single visual vector. The final embedding is the element-wise mean of the text vector and the visual vector.

4. **Minimum Segment Enforcement** — The embedding API requires segments of at least 4 seconds. Segments shorter than this are padded.

**Output:** `outputs/phase_2_embeddings.json` — mirrors the input nodes with an appended `multimodal_embedding` key containing the fused 1408-float array.

---

### Phase 3: Storage & Graph Binding — `pipeline/phase_3_store.py`

**Purpose:** Ingest all computed data into Google Cloud Spanner and upload spatial tracking data to GCS, establishing the persistent semantic graph.

**GCP Services:** Cloud Spanner, Cloud Storage

**Process:**

1. **Data Loading** — Reads `phase_2_embeddings.json`, `phase_1c_narrative_edges.json`, `phase_1a_visual.json`, `phase_1a_audio.json`, and (optionally) `phase_1a_speaker_map.json` from Phase 1A-R.

2. **UUID Assignment** — Generates a UUID for every node and builds a `start_time → node_id` mapping dictionary used to resolve edge references.

3. **Spanner Node Ingestion** (`SemanticClipNode` table) — For each node:
   - Converts float seconds to integer milliseconds for `start_time_ms` and `end_time_ms`.
   - Cross-references the audio ledger to extract unique `speakers` (by matching word timestamps to the node's time range).
   - Cross-references the visual ledger to extract `objects_present` (from object tracking) and `visual_labels` (from label detections).
   - Serializes `speakers`, `objects_present`, `visual_labels`, and `content_mechanisms` as JSON strings.
   - Stores the 1408-d `embedding` vector directly in the `ARRAY<FLOAT32>` column.
   - Sets `spatial_tracking_uri` to `gs://clypt-test-bucket/tracking/[node_id].json`.
   - Writes via `insert_or_update` mutations in batches of 100.

4. **Spanner Edge Ingestion** (`NarrativeEdge` table) — For each edge:
   - Resolves `from_node_id` and `to_node_id` using the start_time mapping.
   - Converts the edge type to `UPPERCASE_SNAKE_CASE` (e.g., `"Causal Link"` → `"CAUSAL_LINK"`).
   - Generates a UUID for `edge_id`.
   - Writes via batched mutations.

5. **Spatial Tracking Upload** (GCS) — For each node:
   - Slices `person_detections` and `face_detections` from the visual ledger based on the node's millisecond time range.
   - If the speaker map is present, tags each face detection entry with its corresponding `speaker_tag` (via the reverse `track_index → speaker_tag` mapping). This propagates speaker identity into the spatial tracking data for use by the render engine.
   - Uploads the speaker-tagged tracking data as `gs://clypt-test-bucket/tracking/[node_id].json`.

**Spanner Schema:**

The database uses four DDL statements (see `spannerSchema.sql`):
- `SemanticClipNode` table with a `ARRAY<FLOAT32>(vector_length=>1408)` embedding column.
- `NarrativeEdge` table with `from_node_id` and `to_node_id` foreign references.
- `ClyptSemanticIndex` — a ScaNN vector index on the embedding column with cosine distance, `tree_depth=2`, `num_leaves=1000`.
- `ClyptGraph` — a Spanner property graph binding both tables, with `NarrativeEdge` sourcing from and destination to `SemanticClipNode`.

---

### Phase 4 (Auto-Curator): Full-Graph Sweep — `pipeline/phase_4_auto_curate.py`

**Purpose:** Sweep the entire semantic graph to identify the best viral clip candidates without relying on a user query. This mimics OpusClip-style automatic clip detection.

**GCP Services:** Cloud Spanner, Vertex AI (Gemini 3.1 Pro)

**Process:**

1. **Full-Graph Sweep** — Pulls every `SemanticClipNode` and `NarrativeEdge` from Spanner, ordered chronologically. No vector search involved.

2. **Narrative Chunking** — Groups sequential nodes into "Narrative Chapters" using edge connectivity. Walking chronologically, nodes belong to the same chapter if any `NarrativeEdge` connects them (directly or transitively within the chapter). When a node has no edge connection to any node in the current chapter, a new chapter begins.

3. **AI Batch Evaluator** — Iterates through each chapter and sends its nodes to the Gemini 3.1 Pro `ClipScoringAgent` with `thinking_config=ThinkingConfig(thinking_level="HIGH")` for maximum reasoning depth. The prompt places the chapter data JSON first with scoring instructions after (data-first ordering). The system instruction directs Gemini to evaluate:
   - **The Hook** — immediate high-retention opening (controversial statement, high-energy vocal delivery, arresting visual).
   - **The Payoff** — logical flow to a punchline, profound statement, or emotional spike, identified via `content_mechanisms`.
   - **Pacing & Length** — 15–60 seconds, trimming filler nodes.
   - **Scoring** — 85–100 for perfect viral moments, 50–84 for mildly interesting, below 50 for boring.

   Gemini returns a `ClipScore` (Pydantic-enforced) with `final_score`, `justification`, `recommended_start_ms`, `recommended_end_ms`, and `included_node_ids`.

4. **Global Ranking & Filtering** — Sorts all scored clips by score descending. Keeps clips scoring 85+. Failsafe: if fewer than 3 clips meet the threshold, returns the Top 3 regardless of score.

5. **Remotion Output** — Constructs `remotion_payloads_array.json` — a JSON array where each clip contains:
   - `clip_start_ms` and `clip_end_ms`
   - `final_score` and `justification`
   - `combined_transcript` (concatenated text of included nodes)
   - `tracking_uris` (GCS URIs for the spatial tracking files of included nodes)
   - `included_node_ids`
   - `active_speaker_timeline` — an array of `{ start_ms, end_ms, speaker_tag }` segments derived from the `speakers` JSON column in Spanner. The render engine uses this to filter tracking frames by the active speaker.

**Output:** `outputs/remotion_payloads_array.json`

---

### Phase 4 (Retrieve): Query-Based Retrieval — `pipeline/phase_4_retrieve.py`

**Purpose:** Find a specific clip matching a natural language query using hybrid vector search + graph traversal. This is a standalone script, not included in the orchestrator.

**GCP Services:** Vertex AI (Multimodal Embeddings + Gemini 3.1 Pro), Cloud Spanner

**Process:**

1. **Embed the Query** — Converts a text query (e.g., `"Find a cynical or sarcastic moment about crypto."`) into a 1408-d vector via `multimodalembedding@001`.

2. **Anchor Search** — Uses `APPROX_COSINE_DISTANCE` against Spanner's ScaNN vector index to find the single closest `SemanticClipNode`.

3. **1-Hop Graph Traversal** — Queries `ClyptGraph` via `GRAPH_TABLE` to find all nodes exactly one edge hop from the anchor, retrieving their metadata and the connecting edge labels.

4. **Sub-Graph Assembly** — Combines the anchor and context nodes into a chronologically ordered sub-graph dictionary.

5. **ClipScoringAgent** — Sends the sub-graph to Gemini for evaluation with `thinking_config=ThinkingConfig(thinking_level="HIGH")` and data-first prompt ordering. Returns optimal clip boundaries and a score.

6. **Remotion Payload** — Outputs a single `remotion_payload.json` with the same structure as the auto-curate payloads, including `active_speaker_timeline`.

**Usage:** `python3 pipeline/phase_4_retrieve.py` (edit `USER_QUERY` in the script).

---

## Remotion Render Engine — `clypt-render-engine/`

A Remotion project that renders the identified clips into 9:16 (1080x1920) short-form videos with speaker-aware, spring-animated camera tracking.

### `scripts/fetch_tracking.js`

A Node.js script that bridges the pipeline output to the Remotion project:

1. Reads `src/remotion_payloads_array.json` (or `src/remotion_payload.json` as fallback).
2. For each clip, iterates through its `tracking_uris` and downloads the corresponding JSON files from GCS.
3. Parses tracking data per detection track (face and person), iterating through each track's `timestamped_objects` to extract `time_ms`, `center_x`/`center_y` (computed as the midpoint of the bounding box), and `speaker_tag` (propagated from the detection level, where Phase 3 tagged it via the speaker map).
4. Concatenates and sorts all tracking frames chronologically per clip.
5. Saves `public/merged_tracking.json` — a keyed object where keys are clip indices (`"0"`, `"1"`, etc.) and values are sorted arrays of tracking frames, each with `{ time_ms, center_x, center_y, speaker_tag }`.

### `src/Root.tsx`

The Remotion root component:

1. Imports `remotion_payloads_array.json` (with fallback to `remotion_payload.json`) and `merged_tracking.json`.
2. For each clip payload, registers a `<Composition>` with:
   - A unique ID: `ClyptViralShort` for a single clip, or `ClyptViralShort-1`, `ClyptViralShort-2`, etc. for multiple clips.
   - Duration calculated from `clip_end_ms - clip_start_ms` at 30fps.
   - Output dimensions of 1080x1920 (9:16).
   - Props: `clipStartMs`, `clipEndMs`, `videoSrc` (pointing to `public/video.mp4` via `staticFile`), per-clip `tracking` array, and `speakerTimeline` (from the payload's `active_speaker_timeline`).

### `src/ClyptViralShort.tsx`

The video rendering component with spring-eased camera transitions and speaker-aware tracking:

1. **Speaker-Aware Filtering** — If `speakerTimeline` is provided and tracking frames contain `speaker_tag` data, the component determines the active speaker at each timestamp and filters tracking frames to only follow that speaker. This prevents the camera from jumping between multiple people in multi-speaker content.

2. **Spring-Based Camera Transitions** — Instead of snapping the camera to new positions, the component uses Remotion's `spring()` function (mass=1, stiffness=80, damping=15) to produce smooth 18-frame eased transitions. A jump is detected when the target position shifts by more than 10% (`JUMP_THRESHOLD=0.1`). The spring scan runs forward from frame 0 on each render to maintain Remotion's pure/deterministic rendering guarantee.

3. **Transform Pipeline** — Uses `makeTransform()` from `@remotion/animation-utils` to safely compose CSS transforms:
   - `translate(-50%, -50%)` — centers the video element.
   - `scale(1.7778)` — scales the 1920×1080 source by 16/9 to fill the 1080-wide canvas.
   - `translate(panX%, panY%)` — shifts horizontally and vertically based on the tracked face/person position.

4. **Pan Clamping** — Horizontal pan is clamped to ±28% (`MAX_PAN_X`) and vertical pan to ±10% (`MAX_PAN_Y`) via `interpolate()` with `extrapolateLeft/Right: "clamp"` to prevent the camera from panning beyond video frame edges.

5. Uses Remotion's `<Video>` component with `startFrom` and `endAt` props to play only the clip's segment.

---

## Spanner Schema

Defined in `spannerSchema.sql`:

```sql
CREATE TABLE SemanticClipNode (
    node_id STRING(MAX) NOT NULL,
    video_uri STRING(MAX),
    start_time_ms INT64,
    end_time_ms INT64,
    transcript_text STRING(MAX),
    vocal_delivery STRING(MAX),
    speakers JSON,
    objects_present JSON,
    visual_labels JSON,
    content_mechanisms JSON,
    embedding ARRAY<FLOAT32>(vector_length=>1408),
    spatial_tracking_uri STRING(MAX)
) PRIMARY KEY (node_id);

CREATE TABLE NarrativeEdge (
    edge_id STRING(MAX) NOT NULL,
    from_node_id STRING(MAX) NOT NULL,
    to_node_id STRING(MAX) NOT NULL,
    label STRING(MAX),
    narrative_classification STRING(MAX),
    confidence_score FLOAT64
) PRIMARY KEY (edge_id);

CREATE VECTOR INDEX ClyptSemanticIndex
ON SemanticClipNode(embedding)
WHERE embedding IS NOT NULL
OPTIONS (distance_type = 'COSINE', tree_depth = 2, num_leaves = 1000);

CREATE PROPERTY GRAPH ClyptGraph
  NODE TABLES (SemanticClipNode)
  EDGE TABLES (NarrativeEdge
    SOURCE KEY (from_node_id) REFERENCES SemanticClipNode(node_id)
    DESTINATION KEY (to_node_id) REFERENCES SemanticClipNode(node_id)
  );
```

---

## Intermediate File Reference

All intermediate files are written to `outputs/` (gitignored) and `downloads/` (gitignored).

| File | Produced By | Consumed By |
|---|---|---|
| `downloads/video.mp4` | Phase 1A | GCS upload, FFmpeg re-encode, Remotion |
| `downloads/audio.m4a` | Phase 1A | GCS upload |
| `outputs/phase_1a_visual.json` | Phase 1A | Phase 1A-R, Phase 1B, Phase 3 |
| `outputs/phase_1a_audio.json` | Phase 1A | Phase 1A-R, Phase 1B, Phase 3 |
| `outputs/phase_1a_speaker_map.json` | Phase 1A-R | Phase 3 |
| `outputs/phase_1b_nodes.json` | Phase 1B | Phase 1C, Phase 2 |
| `outputs/phase_1c_narrative_edges.json` | Phase 1C | Phase 3 |
| `outputs/phase_2_embeddings.json` | Phase 2 | Phase 3 |
| `outputs/remotion_payloads_array.json` | Phase 4 Auto-Curate | Remotion |
| `outputs/remotion_payload.json` | Phase 4 Retrieve | Remotion (standalone) |

---

## GCP Resource Summary

| Resource | Service | Purpose |
|---|---|---|
| `clypt-preyc` | GCP Project | All resources live here |
| `clypt-test-bucket` | Cloud Storage | Video/audio upload, spatial tracking files |
| `clypt-preyc-db` | Spanner Instance | Database host |
| `clypt-db` | Spanner Database | Nodes, edges, vector index, property graph |
| `gemini-3.1-pro-preview` | Vertex AI | Multimodal reasoning (1B, 1C, 4) |
| `multimodalembedding@001` | Vertex AI | 1408-d text+video embeddings (Phase 2, 4-retrieve) |
| `chirp_3` | Speech-to-Text v2 | Word-level transcription with diarization |
| Video Intelligence API v1 | Video Intelligence | Shot detection, face/person/object/label analysis |

---

## Python Dependencies

```
google-cloud-videointelligence>=2.13.0
google-cloud-speech>=2.27.0
google-cloud-storage>=2.18.0
google-genai>=1.66.0
google-cloud-aiplatform>=1.82.0
google-cloud-spanner>=3.49.0
yt-dlp>=2024.12.0
```

Install with `pip install -r requirements.txt`. The Remotion project's Node.js dependencies are managed separately via `cd clypt-render-engine && npm install`.
