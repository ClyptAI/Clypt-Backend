"use client";

interface Payload {
  clip_start_ms: number;
  clip_end_ms: number;
  final_score: number;
  justification: string;
  combined_transcript?: string;
}

interface ClipsPanelProps {
  payloads: Payload[];
  onClose: () => void;
}

function fmtMs(ms: number) {
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = (total % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

function fmtDuration(startMs: number, endMs: number) {
  return ((endMs - startMs) / 1000).toFixed(1) + "s";
}

function scoreColor(score: number) {
  if (score >= 90) return "#10b981";
  if (score >= 75) return "#f59e0b";
  return "#6b7280";
}

export default function ClipsPanel({ payloads, onClose }: ClipsPanelProps) {
  const sorted = [...payloads].sort((a, b) => b.final_score - a.final_score);

  return (
    <div className="fixed inset-0 z-50 flex items-stretch">
      {/* Backdrop */}
      <div
        className="flex-1 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="w-[480px] bg-[#0b0f19] border-l border-[#1e2433] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#1e2433]">
          <div>
            <h2 className="text-white font-semibold text-sm">Recommended Clips</h2>
            <p className="text-[#6b7280] text-xs mt-0.5 font-mono">{sorted.length} clips · ranked by score</p>
          </div>
          <button
            onClick={onClose}
            className="text-[#6b7280] hover:text-white transition-colors text-2xl leading-none w-8 h-8 flex items-center justify-center"
          >
            ×
          </button>
        </div>

        {/* Clips list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {sorted.map((clip, i) => {
            const color = scoreColor(clip.final_score);
            const isTop = i === 0;

            return (
              <div
                key={i}
                className="rounded-lg border overflow-hidden"
                style={{
                  borderColor: isTop ? "#fbbf2430" : "#1e2433",
                  background: isTop ? "#fbbf2408" : "#0f1117",
                }}
              >
                {/* Clip header row */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-[#1e2433]">
                  {/* Rank badge */}
                  <div
                    className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                    style={{
                      background: isTop ? "#fbbf2418" : "#1a1f2e",
                      color: isTop ? "#fbbf24" : "#6b7280",
                      border: `1px solid ${isTop ? "#fbbf2440" : "#2a3147"}`,
                    }}
                  >
                    {isTop ? "★" : i + 1}
                  </div>

                  {/* Time range */}
                  <div className="flex-1">
                    <span className="font-mono text-xs text-[#e2e8f0]">
                      {fmtMs(clip.clip_start_ms)} → {fmtMs(clip.clip_end_ms)}
                    </span>
                    <span className="text-[#6b7280] font-mono text-xs ml-2">
                      {fmtDuration(clip.clip_start_ms, clip.clip_end_ms)}
                    </span>
                  </div>

                  {/* Score */}
                  <div
                    className="px-2.5 py-1 rounded text-xs font-bold font-mono"
                    style={{
                      color,
                      background: `${color}18`,
                      border: `1px solid ${color}40`,
                    }}
                  >
                    {clip.final_score.toFixed(0)}
                  </div>
                </div>

                {/* Score bar */}
                <div className="h-1 bg-[#1a1f2e]">
                  <div
                    className="h-full"
                    style={{
                      width: `${clip.final_score}%`,
                      background: `linear-gradient(90deg, ${color}80, ${color})`,
                    }}
                  />
                </div>

                <div className="px-4 py-3 space-y-3">
                  {/* Transcript */}
                  {clip.combined_transcript && (
                    <div>
                      <p className="text-[#6b7280] text-xs uppercase tracking-wider mb-1.5">Transcript</p>
                      <p className="text-[#cbd5e1] text-xs leading-relaxed italic">
                        &ldquo;{clip.combined_transcript}&rdquo;
                      </p>
                    </div>
                  )}

                  {/* Justification */}
                  <div>
                    <p className="text-[#6b7280] text-xs uppercase tracking-wider mb-1.5">Why this clip</p>
                    <p className="text-[#94a3b8] text-xs leading-relaxed">{clip.justification}</p>
                  </div>

                  {/* Video placeholder */}
                  <div
                    className="rounded border border-dashed border-[#2a3147] bg-[#1a1f2e] flex items-center justify-center gap-2 py-3"
                  >
                    <div className="w-1.5 h-1.5 rounded-full bg-[#4b5563]" />
                    <span className="text-[#4b5563] text-xs font-mono">video preview · coming soon</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
