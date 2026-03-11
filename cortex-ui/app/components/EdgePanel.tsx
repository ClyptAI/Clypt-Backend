"use client";

import { EDGE_COLORS } from "./NodePanel";

interface EdgeData {
  id: string;
  edge_type: string;
  narrative_classification: string;
  confidence_score: number;
  from_node_start_time: number;
  to_node_start_time: number;
}

interface EdgePanelProps {
  edge: EdgeData | null;
  onClose: () => void;
}

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(2);
  return `${m}:${sec.padStart(5, "0")}`;
}

export default function EdgePanel({ edge, onClose }: EdgePanelProps) {
  if (!edge) return null;

  const color = EDGE_COLORS[edge.edge_type] || "#64748b";

  return (
    <div className="fixed right-0 top-0 z-50 h-full w-96 overflow-y-auto border-l border-[#1f2937] bg-[#0b1220] shadow-2xl">
      <div className="flex items-center justify-between border-b border-[#1f2937] p-4">
        <div>
          <span className="font-mono text-xs uppercase tracking-wider text-[#94a3b8]">Edge</span>
          <p className="mt-0.5 font-mono text-sm font-semibold text-white">
            {fmt(edge.from_node_start_time)} -&gt; {fmt(edge.to_node_start_time)}
          </p>
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
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Type</label>
          <div
            className="mt-2 flex items-center gap-2 rounded border px-3 py-2"
            style={{ borderColor: `${color}50`, background: `${color}14` }}
          >
            <div className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ background: color }} />
            <span className="text-sm font-semibold" style={{ color }}>
              {edge.edge_type}
            </span>
          </div>
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Classification</label>
          <p className="mt-1.5 rounded border border-[#334155] bg-[#111827] px-3 py-2 text-sm text-[#e2e8f0]">
            {edge.narrative_classification || "-"}
          </p>
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Confidence</label>
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#1f2937]">
              <div
                className="h-full rounded-full"
                style={{ width: `${edge.confidence_score * 100}%`, background: color }}
              />
            </div>
            <span className="w-8 text-right font-mono text-xs text-white">
              {(edge.confidence_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <label className="text-xs uppercase tracking-wider text-[#94a3b8]">Direction</label>
          <div className="mt-2 flex items-center gap-2 font-mono text-xs">
            <span className="rounded border border-[#334155] bg-[#111827] px-2 py-1 text-[#e2e8f0]">
              {fmt(edge.from_node_start_time)}
            </span>
            <span style={{ color }}>-&gt;</span>
            <span className="rounded border border-[#334155] bg-[#111827] px-2 py-1 text-[#e2e8f0]">
              {fmt(edge.to_node_start_time)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
