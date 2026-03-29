import type { Node } from "@xyflow/react";
import { type SemanticNodeData } from "@/data/mockNodes";

interface TimelineStripProps {
  nodes: Node<SemanticNodeData>[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
}

export function TimelineStrip({ nodes, selectedNodeId, onSelectNode }: TimelineStripProps) {
  const sorted = [...nodes].sort((a, b) => (a.data as unknown as SemanticNodeData).startTime - (b.data as unknown as SemanticNodeData).startTime);
  const maxTime = Math.max(...sorted.map((n) => (n.data as unknown as SemanticNodeData).endTime));

  return (
    <div className="h-14 clypt-surface border-t border-border px-4 py-1.5 mb-2 mx-1 rounded-b-lg shrink-0">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">Timeline</span>
        <span className="text-[10px] font-mono text-muted-foreground">
          {Math.floor(maxTime / 60)}:{String(maxTime % 60).padStart(2, "0")}
        </span>
      </div>
      <div className="relative h-8 rounded bg-secondary/50 overflow-hidden">
        {/* Time axis markers */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
          <div
            key={pct}
            className="absolute top-0 h-full border-l border-border/50"
            style={{ left: `${pct * 100}%` }}
          >
            <span className="absolute -bottom-4 left-0.5 text-[8px] font-mono text-muted-foreground">
              {Math.floor((pct * maxTime) / 60)}:{String(Math.floor((pct * maxTime) % 60)).padStart(2, "0")}
            </span>
          </div>
        ))}

        {/* Node markers */}
        {sorted.map((node) => {
          const d = node.data as unknown as SemanticNodeData;
          const left = (d.startTime / maxTime) * 100;
          const width = Math.max(((d.endTime - d.startTime) / maxTime) * 100, 0.8);
          const isSelected = node.id === selectedNodeId;

          const bgColors: Record<string, string> = {
            hook: "bg-clypt-teal",
            conflict: "bg-primary",
            punchline: "bg-clypt-amber",
            payoff: "bg-clypt-green",
            insight: "bg-[hsl(270,70%,60%)]",
            topic_shift: "bg-clypt-slate",
            speaker_beat: "bg-clypt-mist",
          };

          return (
            <button
              key={node.id}
              onClick={() => onSelectNode(node.id)}
              className={`
                absolute top-1 h-6 rounded-sm transition-all cursor-pointer
                ${bgColors[d.type] || "bg-primary"}
                ${isSelected ? "opacity-100 ring-1 ring-foreground z-10" : "opacity-40 hover:opacity-70"}
              `}
              style={{ left: `${left}%`, width: `${width}%`, minWidth: "6px" }}
              title={d.label}
            />
          );
        })}
      </div>
    </div>
  );
}
