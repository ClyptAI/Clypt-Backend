export const mockVideo = {
  id: "dQw4w9WgXcQ",
  url: "https://www.youtube.com/watch?v=xR4FC5jEMtQ",
  title: "The Problem With Being Too Honest — Colin & Samir",
  channel: "Colin and Samir",
  duration: "18:42",
  durationSeconds: 1122,
  thumbnail: "https://img.youtube.com/vi/xR4FC5jEMtQ/maxresdefault.jpg",
  publishedAt: "2024-11-15",
  description: "We talk about the creator economy, honesty in content creation, and what it means to build something real.",
};

export const sampleRuns = [
  {
    id: "run-001",
    video: mockVideo,
    status: "completed" as const,
    startedAt: "2024-12-10T14:32:00Z",
    completedAt: "2024-12-10T14:38:22Z",
    metrics: { nodes: 15, edges: 22, clips: 6 },
  },
  {
    id: "run-002",
    video: {
      ...mockVideo,
      title: "Why Every Creator Needs a Second Brain — Ali Abdaal",
      channel: "Ali Abdaal",
      duration: "24:18",
    },
    status: "completed" as const,
    startedAt: "2024-12-09T09:11:00Z",
    completedAt: "2024-12-09T09:19:45Z",
    metrics: { nodes: 21, edges: 34, clips: 8 },
  },
  {
    id: "run-003",
    video: {
      ...mockVideo,
      title: "I Spent 30 Days Building an AI App — Fireship",
      channel: "Fireship",
      duration: "12:07",
    },
    status: "processing" as const,
    startedAt: "2024-12-10T16:01:00Z",
    completedAt: null,
    metrics: { nodes: 8, edges: 11, clips: 0 },
  },
];
