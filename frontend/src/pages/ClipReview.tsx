import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import { Pin, Star, Clock, User, Film, Loader2, AlertCircle } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/badge";
import { framingLabels, type ClipCandidate, type FramingType } from "@/data/mockClips";
import { pipelineApi } from "@/lib/api";

function mapCompositionMode(mode: string): FramingType {
  if (mode === "single_speaker_follow") return "single_person";
  if (mode === "two_speaker_split") return "split_layout";
  return "shared_two_shot";
}

function mapBackendClips(clips: any[]): ClipCandidate[] {
  return clips.map((c: any, index: number) => {
    const score = c.overall_score ?? 0;
    const scores = c.scores ?? {};
    return {
      id: `clip-${index}`,
      rank: c.rank ?? index + 1,
      title: c.title ?? `Clip ${index + 1}`,
      startTime: c.start_time_s ?? 0,
      endTime: c.end_time_s ?? 0,
      duration: c.duration_s ?? 0,
      score,
      transcript: c.transcript_snippet ?? "",
      justification: c.justification ?? "",
      framingType: mapCompositionMode(c.composition_mode ?? ""),
      speaker: c.primary_speaker ?? "Unknown",
      scores: {
        hook: scores.hook_strength ?? score,
        payoff: scores.payoff_density ?? score,
        pacing: scores.pacing_quality ?? score,
        narrativeArc: scores.narrative_arc ?? score,
        clipWorthiness: scores.clip_worthiness ?? score,
      },
      nodeIds: c.node_ids ?? [],
      pinned: false,
      bestCut: index === 0,
      renderedVideoUrl: c.rendered_video_url ?? null,
    };
  });
}

const framingColors: Record<string, string> = {
  single_person: "bg-clypt-teal/15 text-clypt-teal border-clypt-teal/30",
  shared_two_shot: "bg-clypt-amber/15 text-clypt-amber border-clypt-amber/30",
  split_layout: "bg-primary/10 text-primary border-primary/30",
};

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function ScoreRadar({ scores }: { scores: ClipCandidate["scores"] }) {
  const data = [
    { axis: "Hook", value: scores.hook * 100 },
    { axis: "Payoff", value: scores.payoff * 100 },
    { axis: "Pacing", value: scores.pacing * 100 },
    { axis: "Arc", value: scores.narrativeArc * 100 },
    { axis: "Clip", value: scores.clipWorthiness * 100 },
  ];

  return (
    <ResponsiveContainer width="100%" height={150}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="68%">
        <PolarGrid stroke="hsl(0, 0%, 15%)" strokeDasharray="3 3" />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 9, fill: "hsl(0, 0%, 45%)" }} />
        <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
        <Radar
          dataKey="value"
          stroke="hsl(0, 90%, 45%)"
          fill="hsl(0, 90%, 45%)"
          fillOpacity={0.15}
          strokeWidth={1.5}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function ClipCard({ clip, onPin, onBestCut }: { clip: ClipCandidate & { renderedVideoUrl?: string | null }; onPin: () => void; onBestCut: () => void }) {
  return (
    <div className={`rounded-xl clypt-glass p-5 transition-all duration-300 h-full flex flex-col ${clip.bestCut ? "clypt-glow-ring" : "hover:border-primary/15"}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <span className="text-data text-lg font-bold text-primary">#{clip.rank}</span>
          <div>
            <h3 className="font-display font-bold text-sm text-foreground">{clip.title}</h3>
            <Badge variant="outline" className={`text-[10px] font-mono border mt-1 ${framingColors[clip.framingType]}`}>
              {framingLabels[clip.framingType]}
            </Badge>
          </div>
        </div>
        <div className="flex items-center gap-0.5">
          <button onClick={onPin} className={`p-1.5 rounded-md transition-colors ${clip.pinned ? "text-clypt-amber bg-clypt-amber/10" : "text-muted-foreground hover:text-foreground"}`}>
            <Pin className="w-3.5 h-3.5" />
          </button>
          <button onClick={onBestCut} className={`p-1.5 rounded-md transition-colors ${clip.bestCut ? "text-primary bg-primary/10" : "text-muted-foreground hover:text-foreground"}`}>
            <Star className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 text-[11px] text-muted-foreground mb-3 font-mono">
        <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{formatTime(clip.startTime)} → {formatTime(clip.endTime)}</span>
        <span>{clip.duration}s</span>
        <span className="flex items-center gap-1"><User className="w-3 h-3" />{clip.speaker}</span>
        <span className="ml-auto text-sm font-bold text-primary">{(clip.score * 100).toFixed(0)}</span>
      </div>

      {clip.renderedVideoUrl ? (
        <div className="rounded-lg overflow-hidden mb-3 border border-border/40 bg-secondary/30">
          <video
            src={clip.renderedVideoUrl}
            controls
            preload="metadata"
            className="w-full h-28 object-cover"
          />
        </div>
      ) : (
        <div className="rounded-lg bg-secondary/30 h-28 flex items-center justify-center mb-3 border border-border/40">
          <div className="text-center">
            <Film className="w-5 h-5 text-muted-foreground/40 mx-auto mb-1" />
            <span className="text-[10px] text-muted-foreground/40 font-mono">9:16 preview</span>
          </div>
        </div>
      )}

      <div className="rounded-lg bg-secondary/20 p-3 mb-3 border border-border/30">
        <p className="text-xs text-foreground/80 italic leading-relaxed line-clamp-3">{clip.transcript}</p>
      </div>

      <p className="text-[11px] text-muted-foreground leading-relaxed mb-3 line-clamp-2 flex-1">{clip.justification}</p>

      <ScoreRadar scores={clip.scores} />
    </div>
  );
}

export default function ClipReview() {
  const { id: runId } = useParams();
  const [clips, setClips] = useState<(ClipCandidate & { renderedVideoUrl?: string | null })[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    pipelineApi.getClips(runId)
      .then((data: any) => {
        if (cancelled) return;
        const mapped = mapBackendClips(data.clips ?? []);
        setClips(mapped);
      })
      .catch((err: any) => {
        if (cancelled) return;
        setError(err.message || "Failed to load clips");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [runId]);

  const togglePin = (clipId: string) => {
    setClips((prev) => prev.map((c) => (c.id === clipId ? { ...c, pinned: !c.pinned } : c)));
  };

  const toggleBestCut = (clipId: string) => {
    setClips((prev) =>
      prev.map((c) => (c.id === clipId ? { ...c, bestCut: !c.bestCut } : { ...c, bestCut: false }))
    );
  };

  const pinnedClips = clips.filter((c) => c.pinned);
  const otherClips = clips.filter((c) => !c.pinned);

  return (
    <AppShell runId={runId}>
      <div className="h-full overflow-y-auto">
        <div className="max-w-6xl mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="font-display text-xl font-bold text-foreground">Clip Shortlist</h1>
              {!loading && !error && clips.length > 0 && (
                <p className="text-xs text-muted-foreground mt-1 font-mono">
                  {clips.length} clips scored · best: {(Math.max(...clips.map((c) => c.score)) * 100).toFixed(0)}%
                </p>
              )}
            </div>
          </div>

          {loading && (
            <div className="flex flex-col items-center justify-center py-24 gap-3">
              <Loader2 className="w-6 h-6 text-primary animate-spin" />
              <p className="text-sm text-muted-foreground font-mono">Loading clips…</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-24 gap-3">
              <AlertCircle className="w-6 h-6 text-destructive" />
              <p className="text-sm text-destructive font-mono">{error}</p>
            </div>
          )}

          {!loading && !error && clips.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 gap-3">
              <Film className="w-6 h-6 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground font-mono">No clips found for this run.</p>
            </div>
          )}

          {!loading && !error && pinnedClips.length > 0 && (
            <div className="mb-8">
              <h2 className="font-display text-xs font-bold text-clypt-amber uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
                <Pin className="w-3.5 h-3.5" /> Pinned
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {pinnedClips.map((clip) => (
                  <ClipCard key={clip.id} clip={clip} onPin={() => togglePin(clip.id)} onBestCut={() => toggleBestCut(clip.id)} />
                ))}
              </div>
            </div>
          )}

          {!loading && !error && otherClips.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {otherClips.map((clip, i) => (
                <motion.div key={clip.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                  <ClipCard clip={clip} onPin={() => togglePin(clip.id)} onBestCut={() => toggleBestCut(clip.id)} />
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
