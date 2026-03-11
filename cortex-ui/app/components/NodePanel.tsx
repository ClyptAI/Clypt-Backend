"use client";

interface Mechanism {
  present: boolean;
  type: string;
  intensity: number;
}

interface NodeData {
  start_time: number;
  end_time: number;
  transcript_segment: string;
  visual_description: string;
  vocal_delivery: string;
  confidence_score: number;
  content_mechanisms: {
    humor?: Mechanism;
    emotion?: Mechanism;
    social?: Mechanism;
    expertise?: Mechanism;
  };
}

interface NodePanelProps {
  node: NodeData | null;
  onClose: () => void;
  onChange: (updated: NodeData) => void;
}

export const EDGE_COLORS: Record<string, string> = {
  "Causal Link": "#0ea5e9",
  Callback: "#8b5cf6",
  Escalation: "#f59e0b",
  Subversion: "#ef4444",
  Revelation: "#10b981",
  Thematic: "#3b82f6",
  Elaboration: "#64748b",
  Transition: "#4b5563",
  "Tension / Release": "#ec4899",
  Contradiction: "#dc2626",
  Analogy: "#22d3ee",
};

const MECHANISM_CONFIG: Record<string, { color: string; label: string }> = {
  humor: { color: "#f59e0b", label: "Humor" },
  emotion: { color: "#ef4444", label: "Emotion" },
  social: { color: "#3b82f6", label: "Social" },
  expertise: { color: "#14b8a6", label: "Expertise" },
};

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(2);
  return `${m}:${sec.padStart(5, "0")}`;
}

export default function NodePanel({ node, onClose, onChange }: NodePanelProps) {
  if (!node) return null;

  const duration = (node.end_time - node.start_time).toFixed(2);

  return (
    <div className="fixed right-0 top-0 z-50 h-full w-96 overflow-y-auto border-l border-[#1f2937] bg-[#0b1220] shadow-2xl">
      <div className="flex items-center justify-between border-b border-[#1f2937] p-4">
        <div>
          <span className="font-mono text-xs uppercase tracking-wider text-[#94a3b8]">Node</span>
          <p className="mt-0.5 font-mono text-sm font-semibold text-white">
            {fmt(node.start_time)} -&gt; {fmt(node.end_time)}
          </p>
          <p className="font-mono text-xs text-[#94a3b8]">{duration}s duration</p>
        </div>
        <button
          onClick={onClose}
          className="flex h-8 w-8 items-center justify-center text-2xl leading-none text-[#94a3b8] transition-colors hover:text-white"
        >
          x
        </button>
      </div>

      <div className="space-y-5 p-4">
        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Content Mechanisms</label>
          <div className="mt-2 space-y-2">
            {Object.entries(MECHANISM_CONFIG).map(([key, cfg]) => {
              const mech = node.content_mechanisms[key as keyof typeof node.content_mechanisms];
              const active = mech?.present ?? false;
              const intensity = mech?.intensity ?? 0;
              const type = mech?.type || "none";

              return (
                <div
                  key={key}
                  className={`rounded border p-2.5 ${
                    active
                      ? "border-[#334155] bg-[#111827]"
                      : "border-[#1f2937] bg-[#0b1220] opacity-45"
                  }`}
                >
                  <div className="mb-1.5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-2 w-2 flex-shrink-0 rounded-full"
                        style={{ background: active ? cfg.color : "#475569" }}
                      />
                      <span className="text-xs font-medium text-[#e2e8f0]">{cfg.label}</span>
                    </div>
                    <span
                      className="rounded px-1.5 py-0.5 font-mono text-xs"
                      style={{
                        color: active ? cfg.color : "#64748b",
                        background: active ? `${cfg.color}18` : "transparent",
                      }}
                    >
                      {active ? type : "-"}
                    </span>
                  </div>

                  <div className="h-1.5 overflow-hidden rounded-full bg-[#0b1220]">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: active ? `${intensity * 100}%` : "0%",
                        background: active
                          ? `linear-gradient(90deg, ${cfg.color}99, ${cfg.color})`
                          : "transparent",
                      }}
                    />
                  </div>

                  {active && (
                    <p className="mt-0.5 text-right font-mono text-xs" style={{ color: cfg.color }}>
                      {(intensity * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Node Confidence</label>
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#1f2937]">
              <div
                className="h-full rounded-full bg-[#14b8a6]"
                style={{ width: `${node.confidence_score * 100}%` }}
              />
            </div>
            <span className="w-8 text-right font-mono text-xs text-white">
              {(node.confidence_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Transcript</label>
          <textarea
            className="mt-1 w-full resize-none rounded border border-[#334155] bg-[#111827] p-2 text-sm text-white focus:border-[#14b8a6] focus:outline-none"
            rows={3}
            value={node.transcript_segment}
            onChange={(e) => onChange({ ...node, transcript_segment: e.target.value })}
          />
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Visual Description</label>
          <textarea
            className="mt-1 w-full resize-none rounded border border-[#334155] bg-[#111827] p-2 text-sm text-white focus:border-[#14b8a6] focus:outline-none"
            rows={3}
            value={node.visual_description}
            onChange={(e) => onChange({ ...node, visual_description: e.target.value })}
          />
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Vocal Delivery</label>
          <textarea
            className="mt-1 w-full resize-none rounded border border-[#334155] bg-[#111827] p-2 text-sm text-white focus:border-[#14b8a6] focus:outline-none"
            rows={2}
            value={node.vocal_delivery}
            onChange={(e) => onChange({ ...node, vocal_delivery: e.target.value })}
          />
        </div>
      </div>
    </div>
  );
}
