export interface PipelinePhase {
  id: string;
  name: string;
  shortName: string;
  description: string;
  status: "pending" | "running" | "completed" | "failed";
  artifacts: string[];
  metrics: Record<string, string | number>;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
}

export interface ActivityLogEntry {
  id: string;
  timestamp: string;
  phase: string;
  message: string;
  level: "info" | "success" | "warn" | "error";
}

export const mockPipeline: PipelinePhase[] = [
  {
    id: "phase-1",
    name: "Deterministic Grounding",
    shortName: "Ground",
    description: "GPU-accelerated video extraction: visual tracks, face detection, audio segmentation, word-level timings",
    status: "completed",
    artifacts: ["phase_1_visual.json", "phase_1_audio.json"],
    metrics: { tracks: 4, personDetections: 287, faceDetections: 312, speakerBoundWords: 3842, transcriptCoverage: "98.2%" },
    startedAt: "2024-12-10T14:32:00Z",
    completedAt: "2024-12-10T14:34:12Z",
    durationMs: 132000,
  },
  {
    id: "phase-2a",
    name: "Semantic Nodes",
    shortName: "Nodes",
    description: "Gemini-powered semantic moment extraction: hooks, conflicts, insights, payoffs, punchlines",
    status: "completed",
    artifacts: ["phase_2a_nodes.json"],
    metrics: { nodesExtracted: 15, avgConfidence: "0.87", speakers: 2, clipCandidates: 11 },
    startedAt: "2024-12-10T14:34:12Z",
    completedAt: "2024-12-10T14:35:45Z",
    durationMs: 93000,
  },
  {
    id: "phase-2b",
    name: "Narrative Edges",
    shortName: "Edges",
    description: "Gemini narrative reasoning: setup→payoff arcs, contrasts, escalations, callbacks between semantic moments",
    status: "completed",
    artifacts: ["phase_2b_narrative_edges.json"],
    metrics: { edgesCreated: 20, avgStrength: "0.78", narrativePaths: 3, strongArcs: 6 },
    startedAt: "2024-12-10T14:35:45Z",
    completedAt: "2024-12-10T14:36:58Z",
    durationMs: 73000,
  },
  {
    id: "phase-3",
    name: "Embeddings",
    shortName: "Embed",
    description: "Multimodal embedding generation for semantic search, similarity scoring, and clip boundary refinement",
    status: "completed",
    artifacts: ["phase_3_embeddings.json"],
    metrics: { embeddingDims: 768, segmentsEmbedded: 42, modelVersion: "gemini-embed-v1" },
    startedAt: "2024-12-10T14:36:58Z",
    completedAt: "2024-12-10T14:37:30Z",
    durationMs: 32000,
  },
  {
    id: "phase-5",
    name: "Clip Scoring",
    shortName: "Score",
    description: "Graph-aware clip ranking: hook strength, payoff density, pacing quality, narrative arc completeness",
    status: "completed",
    artifacts: ["clip_payloads.json", "remotion_configs/"],
    metrics: { clipsScored: 6, avgScore: "0.90", bestScore: "0.95", framingTypes: 3 },
    startedAt: "2024-12-10T14:37:30Z",
    completedAt: "2024-12-10T14:38:22Z",
    durationMs: 52000,
  },
];

export const mockActivityLog: ActivityLogEntry[] = [
  { id: "log-01", timestamp: "14:32:00", phase: "phase-1", message: "Starting deterministic grounding pipeline", level: "info" },
  { id: "log-02", timestamp: "14:32:03", phase: "phase-1", message: "GPU extraction service connected (DigitalOcean H100)", level: "info" },
  { id: "log-03", timestamp: "14:32:15", phase: "phase-1", message: "Video downloaded: 1080p, 18:42 duration", level: "info" },
  { id: "log-04", timestamp: "14:32:45", phase: "phase-1", message: "Visual track extraction: 4 tracks detected", level: "success" },
  { id: "log-05", timestamp: "14:33:10", phase: "phase-1", message: "Face detection complete: 312 detections across 2 identities", level: "success" },
  { id: "log-06", timestamp: "14:33:30", phase: "phase-1", message: "Audio segmentation: 2 speakers isolated", level: "success" },
  { id: "log-07", timestamp: "14:33:55", phase: "phase-1", message: "Word-level timing alignment: 3,842 words bound", level: "success" },
  { id: "log-08", timestamp: "14:34:12", phase: "phase-1", message: "Phase 1 complete — 2 artifacts written", level: "success" },
  { id: "log-09", timestamp: "14:34:12", phase: "phase-2a", message: "Starting semantic node extraction (Gemini 1.5 Pro)", level: "info" },
  { id: "log-10", timestamp: "14:34:45", phase: "phase-2a", message: "Transcript context window loaded (12,480 tokens)", level: "info" },
  { id: "log-11", timestamp: "14:35:20", phase: "phase-2a", message: "Extracted 15 semantic nodes across 7 categories", level: "success" },
  { id: "log-12", timestamp: "14:35:45", phase: "phase-2a", message: "Node confidence scores computed — avg 0.87", level: "success" },
  { id: "log-13", timestamp: "14:35:45", phase: "phase-2b", message: "Starting narrative edge reasoning", level: "info" },
  { id: "log-14", timestamp: "14:36:30", phase: "phase-2b", message: "20 narrative edges identified, 6 strong arcs", level: "success" },
  { id: "log-15", timestamp: "14:36:58", phase: "phase-2b", message: "Phase 2B complete — story graph constructed", level: "success" },
  { id: "log-16", timestamp: "14:36:58", phase: "phase-3", message: "Generating multimodal embeddings (768-dim)", level: "info" },
  { id: "log-17", timestamp: "14:37:30", phase: "phase-3", message: "42 segments embedded — similarity matrix ready", level: "success" },
  { id: "log-18", timestamp: "14:37:30", phase: "phase-5", message: "Starting graph-aware clip scoring", level: "info" },
  { id: "log-19", timestamp: "14:37:55", phase: "phase-5", message: "6 clips scored — best: 0.95 (The Authenticity Paradox)", level: "success" },
  { id: "log-20", timestamp: "14:38:10", phase: "phase-5", message: "Remotion render configs generated for 3 framing types", level: "success" },
  { id: "log-21", timestamp: "14:38:22", phase: "phase-5", message: "Pipeline complete — all artifacts ready", level: "success" },
];
