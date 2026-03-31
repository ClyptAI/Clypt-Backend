export interface ArtifactFile {
  id: string;
  name: string;
  phase: string;
  description: string;
  size: string;
  itemCount: number;
  preview: Record<string, unknown>;
}

export const mockArtifacts: ArtifactFile[] = [
  {
    id: "art-1",
    name: "phase_1_visual.json",
    phase: "Phase 1 — Deterministic Grounding",
    description: "Visual extraction results: person tracks, face detections, scene boundaries, and tracking metrics",
    size: "2.4 MB",
    itemCount: 4,
    preview: {
      tracks: [
        { trackId: "T001", label: "Person A (Colin)", startFrame: 0, endFrame: 33660, confidence: 0.96, boundingBoxes: 33660 },
        { trackId: "T002", label: "Person B (Samir)", startFrame: 0, endFrame: 33660, confidence: 0.94, boundingBoxes: 33660 },
        { trackId: "T003", label: "B-Roll / Insert", startFrame: 4200, endFrame: 4800, confidence: 0.82, boundingBoxes: 600 },
        { trackId: "T004", label: "Screen Share", startFrame: 15000, endFrame: 15900, confidence: 0.91, boundingBoxes: 900 },
      ],
      personDetections: {
        total: 287,
        uniqueIdentities: 2,
        avgConfidence: 0.95,
        detectionsByIdentity: { Colin: 148, Samir: 139 },
      },
      faceDetections: {
        total: 312,
        avgFaceSize: "18.4%",
        frontFacingRatio: 0.89,
        detectionsByIdentity: { Colin: 161, Samir: 151 },
      },
      trackingMetrics: {
        fps: 30,
        totalFrames: 33660,
        processedFrames: 33660,
        trackSwitches: 12,
        idConsistency: 0.97,
      },
    },
  },
  {
    id: "art-2",
    name: "phase_1_audio.json",
    phase: "Phase 1 — Deterministic Grounding",
    description: "Audio extraction: word-level timings, speaker diarization, and speaker-to-track bindings",
    size: "1.8 MB",
    itemCount: 3842,
    preview: {
      wordTimings: [
        { word: "Here's", start: 0.12, end: 0.38, speaker: "Colin", confidence: 0.98 },
        { word: "the", start: 0.38, end: 0.45, speaker: "Colin", confidence: 0.99 },
        { word: "thing", start: 0.45, end: 0.72, speaker: "Colin", confidence: 0.97 },
        { word: "nobody", start: 0.75, end: 1.12, speaker: "Colin", confidence: 0.96 },
        { word: "wants", start: 1.15, end: 1.42, speaker: "Colin", confidence: 0.98 },
        { word: "to", start: 1.42, end: 1.52, speaker: "Colin", confidence: 0.99 },
        { word: "say", start: 1.55, end: 1.78, speaker: "Colin", confidence: 0.97 },
      ],
      speakerBindings: {
        speakers: [
          { id: "SPK_01", label: "Colin", trackRef: "T001", totalWords: 2105, talkTimeSeconds: 542 },
          { id: "SPK_02", label: "Samir", trackRef: "T002", totalWords: 1737, talkTimeSeconds: 448 },
        ],
        overlapSegments: 14,
        silenceSegments: 28,
      },
      transcriptCoverage: {
        totalDuration: 1122,
        coveredDuration: 1102,
        coveragePercent: 98.2,
        gapCount: 28,
        avgGapDuration: 0.71,
      },
    },
  },
  {
    id: "art-3",
    name: "phase_2a_nodes.json",
    phase: "Phase 2A — Semantic Nodes",
    description: "Semantic moment extraction: 15 nodes across 7 narrative categories with confidence scores",
    size: "48 KB",
    itemCount: 15,
    preview: {
      summary: {
        totalNodes: 15,
        byType: { hook: 2, conflict: 3, punchline: 1, payoff: 2, insight: 3, topic_shift: 1, speaker_beat: 2 },
        avgConfidence: 0.87,
        clipCandidates: 11,
        speakers: ["Colin", "Samir"],
      },
    },
  },
  {
    id: "art-4",
    name: "phase_2b_narrative_edges.json",
    phase: "Phase 2B — Narrative Edges",
    description: "Narrative relationship graph: 20 edges connecting semantic nodes with typed relationships",
    size: "32 KB",
    itemCount: 20,
    preview: {
      summary: {
        totalEdges: 20,
        byRelation: { setup_payoff: 5, contrast: 3, continuation: 6, escalation: 2, callback: 2, pivot: 1, response: 1 },
        avgStrength: 0.78,
        strongArcs: 6,
        narrativePaths: 3,
      },
    },
  },
  {
    id: "art-5",
    name: "phase_3_embeddings.json",
    phase: "Phase 3 — Embeddings",
    description: "Multimodal embeddings for semantic similarity scoring and clip boundary refinement",
    size: "1.2 MB",
    itemCount: 42,
    preview: {
      config: { dimensions: 768, model: "gemini-embed-v1", segments: 42 },
      sampleVector: "[0.0234, -0.1456, 0.0891, 0.2103, -0.0567, ... (768 dims)]",
      similarityMatrix: { topPairs: [{ a: "n10", b: "n14", similarity: 0.94 }, { a: "n1", b: "n15", similarity: 0.91 }, { a: "n5", b: "n6", similarity: 0.89 }] },
    },
  },
  {
    id: "art-6",
    name: "remotion_configs/",
    phase: "Phase 5 — Clip Scoring",
    description: "Render-ready Remotion payloads for each scored clip with framing and timing configuration",
    size: "18 KB",
    itemCount: 6,
    preview: {
      samplePayload: {
        clipId: "clip-1",
        composition: "CliptVerticalShort",
        fps: 30,
        durationInFrames: 900,
        props: {
          sourceVideo: "source.mp4",
          startFrame: 11100,
          endFrame: 12000,
          framingMode: "single_person",
          targetSpeaker: "Colin",
          captionStyle: "dynamic_word",
        },
      },
    },
  },
];
