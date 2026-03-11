"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

const MECHANISM_COLORS: Record<string, string> = {
  humor: "#f59e0b",
  emotion: "#ef4444",
  social: "#3b82f6",
  expertise: "#14b8a6",
};

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

export default memo(function SegmentNode({ data, selected }: NodeProps) {
  const d = data as {
    start_time: number;
    end_time: number;
    transcript_segment?: string;
    content_mechanisms?: Record<string, { present: boolean; intensity: number }>;
    clipRank: number | null;
    clipScore: number | null;
    topColor: string;
    borderColor: string;
    bg: string;
    shadow: string;
  };

  const mechs = Object.entries(d.content_mechanisms ?? {}).filter(([, m]) => m?.present);

  const isTopClip = d.clipRank === 0;
  const isRecommended = d.clipRank !== null;
  const label = d.transcript_segment?.slice(0, 56) || `${fmt(d.start_time)}`;

  return (
    <div
      style={{
        background: d.bg,
        border: `${isTopClip ? 2 : 1}px solid ${selected ? "#67e8f9" : d.borderColor}`,
        borderRadius: "12px",
        width: 186,
        minHeight: 108,
        overflow: "hidden",
        boxShadow: selected
          ? `0 0 0 2px #06b6d440, 0 10px 24px #02061799, ${d.shadow}`
          : `0 8px 18px #02061780, ${d.shadow}`,
        transform: isTopClip ? "translateY(-1px) scale(1.03)" : "none",
        fontFamily: "var(--font-sans, 'Space Grotesk', sans-serif)",
        cursor: "pointer",
      }}
    >
      <div
        style={{
          height: 4,
          background: isTopClip
            ? "linear-gradient(90deg, #f59e0b, #f97316)"
            : isRecommended
            ? "linear-gradient(90deg, #fb7185, #f43f5e)"
            : `linear-gradient(90deg, ${d.topColor}99, ${d.topColor})`,
        }}
      />

      <div style={{ padding: "8px 10px 10px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 6,
          }}
        >
          <span
            style={{
              fontSize: 10,
              fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
              color: isTopClip ? "#f59e0b" : isRecommended ? "#fb7185" : "#94a3b8",
              letterSpacing: "0.02em",
              lineHeight: 1,
            }}
          >
            {fmt(d.start_time)}
          </span>

          {(isTopClip || isRecommended) && (
            <span
              style={{
                fontSize: 9,
                fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
                color: isTopClip ? "#f59e0b" : "#fb7185",
                background: isTopClip ? "#f59e0b2b" : "#fb718518",
                border: `1px solid ${isTopClip ? "#f59e0b88" : "#fb718540"}`,
                borderRadius: 999,
                padding: "2px 6px",
                lineHeight: 1.4,
                boxShadow: isTopClip ? "0 0 10px #f59e0b55" : "none",
              }}
            >
              {isTopClip ? "TOP PICK" : `#${(d.clipRank ?? 0) + 1}`}
            </span>
          )}
        </div>

        <p
          style={{
            fontSize: 11,
            color: "#cbd5e1",
            margin: 0,
            lineHeight: 1.45,
            overflow: "hidden",
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical" as const,
            wordBreak: "break-word",
          }}
        >
          {label}
        </p>

        {mechs.length > 0 && (
          <div style={{ display: "flex", gap: 4, marginTop: 7 }}>
            {mechs.map(([key, m]) => (
              <div
                key={key}
                title={`${key}: ${(m.intensity * 100).toFixed(0)}%`}
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: MECHANISM_COLORS[key] || "#0ea5e9",
                  opacity: 0.4 + m.intensity * 0.6,
                  flexShrink: 0,
                }}
              />
            ))}
          </div>
        )}
      </div>

      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#64748b", width: 7, height: 7, border: "none" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#64748b", width: 7, height: 7, border: "none" }}
      />
    </div>
  );
});
