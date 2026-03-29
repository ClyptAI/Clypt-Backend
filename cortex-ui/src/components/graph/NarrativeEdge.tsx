import { memo } from "react";
import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";
import { type NarrativeEdgeData, relationConfig } from "@/data/mockEdges";

function NarrativeEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}: EdgeProps) {
  const edgeData = data as unknown as NarrativeEdgeData & { _isHoverHighlighted?: boolean; _hasHover?: boolean; _isEdgeHovered?: boolean };
  const config = relationConfig[edgeData?.relation ?? "continuation"];

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const isHighlighted = edgeData?._isHoverHighlighted;
  const isEdgeHovered = edgeData?._isEdgeHovered;
  const isDimmed = edgeData?._hasHover && !isHighlighted;

  return (
    <>
      {/* Glow layer for highlighted/hovered edges */}
      {(isHighlighted || isEdgeHovered) && (
        <BaseEdge
          id={`${id}-glow`}
          path={edgePath}
          style={{
            stroke: config.color,
            strokeWidth: isEdgeHovered ? 8 : 6,
            strokeDasharray: undefined,
            opacity: isEdgeHovered ? 0.4 : 0.3,
            filter: `drop-shadow(0 0 ${isEdgeHovered ? '6px' : '4px'} ${config.color})`,
          }}
        />
      )}
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: config.color,
          strokeWidth: isHighlighted || isEdgeHovered ? 2.5 : selected ? 2.5 : 1.5,
          strokeDasharray: config.style === "dashed" ? "6 4" : config.style === "dotted" ? "2 4" : undefined,
          opacity: isDimmed ? 0.1 : (isHighlighted || isEdgeHovered) ? 1 : selected ? 1 : 0.5,
          transition: "all 0.15s",
        }}
      />
      {/* Animated streaming dot for highlighted edges */}
      {isHighlighted && (
        <circle r="3" fill={config.color} filter={`drop-shadow(0 0 3px ${config.color})`}>
          <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}
      {selected && (
        <foreignObject
          x={labelX - 60}
          y={labelY - 12}
          width={120}
          height={24}
          className="pointer-events-none"
        >
          <div className="flex items-center justify-center">
            <span
              className="px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap border"
              style={{
                backgroundColor: "hsl(225, 20%, 8%)",
                borderColor: config.color,
                color: config.color,
              }}
            >
              {config.label}
            </span>
          </div>
        </foreignObject>
      )}
    </>
  );
}

export const NarrativeEdge = memo(NarrativeEdgeComponent);