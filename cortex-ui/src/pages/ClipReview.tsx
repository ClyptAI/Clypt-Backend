import { useState } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import { Pin, Star, Clock, User, Film } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/badge";
import { mockClips, framingLabels, type ClipCandidate } from "@/data/mockClips";

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

function ClipCard({ clip, onPin, onBestCut }: { clip: ClipCandidate; onPin: () => void; onBestCut: () => void }) {
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

      <div className="rounded-lg bg-secondary/30 h-28 flex items-center justify-center mb-3 border border-border/40">
        <div className="text-center">
          <Film className="w-5 h-5 text-muted-foreground/40 mx-auto mb-1" />
          <span className="text-[10px] text-muted-foreground/40 font-mono">9:16 preview</span>
        </div>
      </div>

      <div className="rounded-lg bg-secondary/20 p-3 mb-3 border border-border/30">
        <p className="text-xs text-foreground/80 italic leading-relaxed line-clamp-3">{clip.transcript}</p>
      </div>

      <p className="text-[11px] text-muted-foreground leading-relaxed mb-3 line-clamp-2 flex-1">{clip.justification}</p>

      <ScoreRadar scores={clip.scores} />
    </div>
  );
}

export default function ClipReview() {
  const { id } = useParams();
  const [clips, setClips] = useState(mockClips);

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
    <AppShell runId={id}>
      <div className="h-full overflow-y-auto">
        <div className="max-w-6xl mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="font-display text-xl font-bold text-foreground">Clip Shortlist</h1>
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                {clips.length} clips scored · best: {(Math.max(...clips.map((c) => c.score)) * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          {pinnedClips.length > 0 && (
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {otherClips.map((clip, i) => (
              <motion.div key={clip.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <ClipCard clip={clip} onPin={() => togglePin(clip.id)} onBestCut={() => toggleBestCut(clip.id)} />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
