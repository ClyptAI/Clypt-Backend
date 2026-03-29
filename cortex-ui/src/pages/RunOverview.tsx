import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ExternalLink, Clock, CheckCircle2, Loader2, AlertCircle, GitBranch, Film, Database, ChevronRight, ArrowUpRight } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { mockVideo } from "@/data/mockVideo";
import { mockPipeline, mockActivityLog } from "@/data/mockPipeline";

function formatDuration(ms: number) {
  const s = Math.floor(ms / 1000);
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

const statusIcons = {
  completed: <CheckCircle2 className="w-4 h-4 text-clypt-green" />,
  running: <Loader2 className="w-4 h-4 text-clypt-amber animate-spin" />,
  failed: <AlertCircle className="w-4 h-4 text-destructive" />,
  pending: <Clock className="w-4 h-4 text-muted-foreground" />,
};

const logLevelColors = {
  info: "text-muted-foreground",
  success: "text-clypt-green",
  warn: "text-clypt-amber",
  error: "text-destructive",
};

export default function RunOverview() {
  const { id } = useParams();
  const navigate = useNavigate();
  const totalDuration = mockPipeline.reduce((sum, p) => sum + (p.durationMs ?? 0), 0);

  return (
    <AppShell runId={id}>
      <div className="h-full overflow-y-auto">
        <div className="max-w-6xl mx-auto p-6 space-y-5">
          {/* Job metadata */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl clypt-glass p-5"
          >
            <div className="flex flex-wrap items-start gap-4">
              <div className="flex-1 min-w-0">
                <h1 className="font-display text-xl font-bold text-foreground truncate">
                  {mockVideo.title}
                </h1>
                <div className="flex items-center gap-2.5 mt-2">
                  <span className="text-xs text-muted-foreground font-mono">{mockVideo.channel}</span>
                  <span className="w-1 h-1 rounded-full bg-border" />
                  <span className="text-xs text-muted-foreground font-mono">{mockVideo.duration}</span>
                  <a
                    href={mockVideo.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary flex items-center gap-1 hover:underline"
                  >
                    <ExternalLink className="w-3 h-3" /> source
                  </a>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline" className="font-mono text-[10px] gap-1 border-clypt-green/30 text-clypt-green">
                  <CheckCircle2 className="w-3 h-3" /> Completed
                </Badge>
                <Badge variant="outline" className="font-mono text-[10px]">
                  <Clock className="w-3 h-3 mr-1" /> {formatDuration(totalDuration)}
                </Badge>
                <Badge variant="outline" className="font-mono text-[10px]">GPU · H100</Badge>
                <Badge variant="outline" className="font-mono text-[10px]">Gemini 1.5 Pro</Badge>
              </div>
            </div>
          </motion.div>

          {/* Quick actions */}
          <div className="flex gap-2 flex-wrap">
            {[
              { label: "Open Graph", icon: GitBranch, path: "graph" },
              { label: "Review Clips", icon: Film, path: "clips" },
              { label: "Artifacts", icon: Database, path: "artifacts" },
            ].map((action) => (
              <Button
                key={action.path}
                variant="outline"
                size="sm"
                className="gap-1.5 text-xs font-display"
                onClick={() => navigate(`/run/${id}/${action.path}`)}
              >
                <action.icon className="w-3.5 h-3.5" />
                {action.label}
                <ArrowUpRight className="w-3 h-3 text-muted-foreground" />
              </Button>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Pipeline */}
            <div className="lg:col-span-2 space-y-3">
              <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em]">Pipeline Phases</h2>
              <div className="space-y-2">
                {mockPipeline.map((phase, i) => (
                  <motion.div
                    key={phase.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.06 }}
                    className="group rounded-xl clypt-glass p-4 hover:border-primary/15 transition-all duration-300"
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">{statusIcons[phase.status]}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-display font-semibold text-sm text-foreground">{phase.name}</h3>
                          {phase.durationMs && (
                            <span className="text-data text-[10px] text-muted-foreground">{formatDuration(phase.durationMs)}</span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mb-2.5">{phase.description}</p>

                        <div className="flex flex-wrap gap-1.5 mb-2">
                          {phase.artifacts.map((a) => (
                            <span key={a} className="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-mono bg-secondary/80 text-secondary-foreground border border-border/40">
                              {a}
                            </span>
                          ))}
                        </div>

                        <div className="flex flex-wrap gap-x-4 gap-y-1">
                          {Object.entries(phase.metrics).map(([key, val]) => (
                            <span key={key} className="text-[11px]">
                              <span className="text-muted-foreground">{key.replace(/([A-Z])/g, " $1").toLowerCase()}: </span>
                              <span className="text-data text-foreground">{String(val)}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground/30 group-hover:text-muted-foreground transition-colors" />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Activity feed */}
            <div className="space-y-3">
              <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em]">Activity Feed</h2>
              <div className="rounded-xl clypt-glass p-3 max-h-[600px] overflow-y-auto space-y-1">
                {mockActivityLog.map((entry, i) => {
                  const levelDot: Record<string, string> = {
                    info: "bg-muted-foreground/40",
                    success: "bg-clypt-green",
                    warn: "bg-clypt-amber",
                    error: "bg-destructive",
                  };
                  return (
                    <motion.div
                      key={entry.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className="group flex items-start gap-3 py-2 px-2 rounded-lg hover:bg-secondary/30 transition-colors"
                    >
                      <div className="flex items-center gap-2 shrink-0 mt-0.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${levelDot[entry.level]}`} />
                        <span className="text-data text-[10px] text-muted-foreground/50 font-mono w-12">{entry.timestamp}</span>
                      </div>
                      <span className={`text-[11px] leading-relaxed flex-1 ${logLevelColors[entry.level]}`}>
                        {entry.message}
                      </span>
                      <Badge variant="outline" className="text-[8px] font-mono opacity-0 group-hover:opacity-100 transition-opacity shrink-0 border-border/30 text-muted-foreground/50">
                        {entry.phase}
                      </Badge>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
