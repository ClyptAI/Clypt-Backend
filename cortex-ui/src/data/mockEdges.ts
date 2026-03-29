import type { Edge } from "@xyflow/react";

export type NarrativeRelation =
  | "setup_payoff"
  | "contrast"
  | "continuation"
  | "escalation"
  | "callback"
  | "pivot"
  | "response";

export interface NarrativeEdgeData {
  relation: NarrativeRelation;
  description: string;
  strength: number;
  [key: string]: unknown;
}

export const relationConfig: Record<NarrativeRelation, { color: string; label: string; style: string }> = {
  setup_payoff: { color: "hsl(168, 64%, 50%)", label: "Setup → Payoff", style: "solid" },
  contrast: { color: "hsl(4, 80%, 63%)", label: "Contrast", style: "dashed" },
  continuation: { color: "hsl(215, 17%, 55%)", label: "Continuation", style: "solid" },
  escalation: { color: "hsl(38, 92%, 50%)", label: "Escalation", style: "solid" },
  callback: { color: "hsl(168, 64%, 60%)", label: "Callback", style: "dotted" },
  pivot: { color: "hsl(215, 17%, 45%)", label: "Pivot", style: "dashed" },
  response: { color: "hsl(215, 17%, 65%)", label: "Response", style: "solid" },
};

export const mockEdges: Edge<NarrativeEdgeData>[] = [
  { id: "e1-2", source: "n1", target: "n2", type: "narrative", data: { relation: "contrast", description: "Samir directly challenges Colin's opening provocation", strength: 0.9 } },
  { id: "e2-3", source: "n2", target: "n3", type: "narrative", data: { relation: "continuation", description: "The disagreement leads into algorithm discussion", strength: 0.7 } },
  { id: "e3-4", source: "n3", target: "n4", type: "narrative", data: { relation: "continuation", description: "Algorithm point illustrated with MrBeast example", strength: 0.6 } },
  { id: "e1-5", source: "n1", target: "n5", type: "narrative", data: { relation: "setup_payoff", description: "Opening question about honesty finds its answer in vulnerability", strength: 0.95 } },
  { id: "e5-6", source: "n5", target: "n6", type: "narrative", data: { relation: "escalation", description: "Vulnerability concept escalates into personal burnout admission", strength: 0.88 } },
  { id: "e6-7", source: "n6", target: "n7", type: "narrative", data: { relation: "pivot", description: "Burnout pivots the conversation from audience-facing to self-facing honesty", strength: 0.75 } },
  { id: "e7-8", source: "n7", target: "n8", type: "narrative", data: { relation: "continuation", description: "Self-honesty leads to money/passion tension", strength: 0.7 } },
  { id: "e8-9", source: "n8", target: "n9", type: "narrative", data: { relation: "contrast", description: "Revenue corruption vs creator middle class optimism", strength: 0.85 } },
  { id: "e9-10", source: "n9", target: "n10", type: "narrative", data: { relation: "setup_payoff", description: "Middle class insight sets up the paradox punchline", strength: 0.92 } },
  { id: "e10-11", source: "n10", target: "n11", type: "narrative", data: { relation: "continuation", description: "Punchline flows into trust model framework", strength: 0.8 } },
  { id: "e11-12", source: "n11", target: "n12", type: "narrative", data: { relation: "continuation", description: "Trust framework illustrated with personal history", strength: 0.65 } },
  { id: "e12-13", source: "n12", target: "n13", type: "narrative", data: { relation: "continuation", description: "Personal story leads to format discussion", strength: 0.55 } },
  { id: "e13-14", source: "n13", target: "n14", type: "narrative", data: { relation: "setup_payoff", description: "Format honesty feeds into final synthesis", strength: 0.78 } },
  { id: "e10-14", source: "n10", target: "n14", type: "narrative", data: { relation: "callback", description: "Final synthesis references the paradox punchline", strength: 0.85 } },
  { id: "e14-15", source: "n14", target: "n15", type: "narrative", data: { relation: "continuation", description: "Synthesis wraps into closing hook for next episode", strength: 0.6 } },
  { id: "e1-15", source: "n1", target: "n15", type: "narrative", data: { relation: "callback", description: "Closing hook mirrors the opening provocation structure", strength: 0.82 } },
  { id: "e2-8", source: "n2", target: "n8", type: "narrative", data: { relation: "escalation", description: "Samir's initial pushback escalates into deeper revenue tension", strength: 0.7 } },
  { id: "e4-9", source: "n4", target: "n9", type: "narrative", data: { relation: "contrast", description: "MrBeast scale vs creator middle class counter-model", strength: 0.72 } },
  { id: "e3-11", source: "n3", target: "n11", type: "narrative", data: { relation: "setup_payoff", description: "Algorithm tension resolved by trust equity framework", strength: 0.83 } },
  { id: "e6-14", source: "n6", target: "n14", type: "narrative", data: { relation: "setup_payoff", description: "Burnout vulnerability resolves in final optimistic synthesis", strength: 0.88 } },
];
