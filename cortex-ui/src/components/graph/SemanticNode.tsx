import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { type SemanticNodeData, nodeTypeConfig } from "@/data/mockNodes";

const typeColorMap: Record<string, string> = {
  hook: "border-clypt-teal/60 bg-clypt-teal/10",
  conflict: "border-primary/60 bg-primary/10",
  punchline: "border-clypt-amber/60 bg-clypt-amber/10",
  payoff: "border-clypt-green/60 bg-clypt-green/10",
  insight: "border-[hsl(270,70%,60%)]/60 bg-[hsl(270,70%,60%)]/10",
  topic_shift: "border-clypt-slate/60 bg-clypt-slate/10",
  speaker_beat: "border-clypt-mist/60 bg-clypt-mist/10",
};

const textColorMap: Record<string, string> = {
  hook: "text-clypt-teal",
  conflict: "text-primary",
  punchline: "text-clypt-amber",
  payoff: "text-clypt-green",
  insight: "text-[hsl(270,70%,65%)]",
  topic_shift: "text-clypt-slate",
  speaker_beat: "text-clypt-mist",
};

const ringColorMap: Record<string, string> = {
  hook: "ring-clypt-teal shadow-clypt-teal/15",
  conflict: "ring-primary shadow-primary/15",
  punchline: "ring-clypt-amber shadow-clypt-amber/15",
  payoff: "ring-clypt-green shadow-clypt-green/15",
  insight: "ring-[hsl(270,70%,60%)] shadow-[hsl(270,70%,60%)]/15",
  topic_shift: "ring-clypt-slate shadow-clypt-slate/15",
  speaker_beat: "ring-clypt-mist shadow-clypt-mist/15",
};

const glowShadowMap: Record<string, string> = {
  hook: "0 0 16px hsl(168, 64%, 45%, 0.5)",
  conflict: "0 0 16px hsl(0, 90%, 45%, 0.5)",
  punchline: "0 0 16px hsl(38, 92%, 50%, 0.5)",
  payoff: "0 0 16px hsl(142, 60%, 45%, 0.5)",
  insight: "0 0 16px hsl(270, 70%, 60%, 0.5)",
  topic_shift: "0 0 16px hsl(210, 60%, 55%, 0.5)",
  speaker_beat: "0 0 16px hsl(320, 65%, 58%, 0.5)",
};

function SemanticNodeComponent({ data, selected }: NodeProps) {
  const nodeData = data as unknown as SemanticNodeData & { _isHoverTarget?: boolean; _isHoverConnected?: boolean; _hasHover?: boolean };
  const config = nodeTypeConfig[nodeData.type];
  const colors = typeColorMap[nodeData.type] || "border-border bg-secondary";
  const textColor = textColorMap[nodeData.type] || "text-foreground";
  const ringColor = ringColorMap[nodeData.type] || "ring-primary shadow-primary/15";

  const isGlowing = nodeData._isHoverTarget || nodeData._isHoverConnected;
  const isDimmed = nodeData._hasHover && !isGlowing;

  return (
    <div
      className={`
        rounded-xl border px-3.5 py-2.5 min-w-[150px] max-w-[210px] cursor-pointer
        transition-all duration-150 backdrop-blur-sm
        ${colors}
        ${selected ? `ring-2 shadow-lg scale-105 ${ringColor}` : "hover:scale-[1.02]"}
        ${isDimmed ? "opacity-25" : ""}
        ${isGlowing && !selected ? "scale-[1.03]" : ""}
      `}
      style={isGlowing ? { boxShadow: glowShadowMap[nodeData.type] || "0 0 16px hsl(0,0%,50%,0.4)" } : undefined}
    >
      <Handle type="target" position={Position.Left} className="!bg-primary !border-0 !w-2 !h-2" />
      <Handle type="source" position={Position.Right} className="!bg-primary !border-0 !w-2 !h-2" />

      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="text-sm">{config.icon}</span>
        <span className={`text-[10px] font-mono uppercase tracking-wider font-medium ${textColor}`}>
          {config.label}
        </span>
        {nodeData.clipWorthy && (
          <span className="ml-auto w-1.5 h-1.5 rounded-full bg-primary animate-pulse-glow" />
        )}
      </div>
      <p className="text-xs font-medium text-foreground leading-snug line-clamp-2">{nodeData.label}</p>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-data text-[10px] text-muted-foreground">{nodeData.speaker}</span>
        <span className="text-data text-[10px] text-muted-foreground">
          {Math.floor(nodeData.startTime / 60)}:{String(nodeData.startTime % 60).padStart(2, "0")}
        </span>
        <span className={`text-data text-[10px] ml-auto font-bold ${textColor}`}>
          {(nodeData.score * 100).toFixed(0)}
        </span>
      </div>
    </div>
  );
}

export const SemanticNode = memo(SemanticNodeComponent);