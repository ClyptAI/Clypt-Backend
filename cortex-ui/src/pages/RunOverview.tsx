import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ExternalLink, Clock, CheckCircle2, Loader2, AlertCircle, GitBranch, Film, Database, ArrowUpRight, Network, Scissors, BarChart3 } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { pipelineApi } from "@/lib/api";

interface RunData {
  run_id: string;
  status: "pending" | "running" | "succeeded" | "failed";
  video_url: string;
  creator_id: string;
  created_at: string;
  updated_at: string;
  phase: string | null;
  progress_pct: number;
  detail: string | null;
  summary: {
    video_url: string;
    node_count: number;
    edge_count: number;
    clip_candidate_count: number;
    rendered_clip_count: number;
  } | null;
}

const statusConfig: Record<string, { icon: React.ReactNode; label: string; badgeClass: string }> = {
  succeeded: {
    icon: <CheckCircle2 className="w-3 h-3" />,
    label: "Succeeded",
    badgeClass: "border-clypt-green/30 text-clypt-green",
  },
  running: {
    icon: <Loader2 className="w-3 h-3 animate-spin" />,
    label: "Running",
    badgeClass: "border-clypt-amber/30 text-clypt-amber",
  },
  failed: {
    icon: <AlertCircle className="w-3 h-3" />,
    label: "Failed",
    badgeClass: "border-destructive/30 text-destructive",
  },
  pending: {
    icon: <Clock className="w-3 h-3" />,
    label: "Pending",
    badgeClass: "border-muted-foreground/30 text-muted-foreground",
  },
};

function extractVideoTitle(url: string): string {
  try {
    const u = new URL(url);
    // Try to get a readable name from the URL
    if (u.hostname.includes("youtube.com") || u.hostname.includes("youtu.be")) {
      return url;
    }
    return url;
  } catch {
    return url;
  }
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function RunOverview() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [run, setRun] = useState<RunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    async function fetchRun() {
      try {
        setLoading(true);
        setError(null);
        const data = await pipelineApi.getRun(id!);
        if (!cancelled) setRun(data as RunData);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to load run");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchRun();
    return () => { cancelled = true; };
  }, [id]);

  if (loading) {
    return (
      <AppShell runId={id}>
        <div className="h-full flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <span className="text-sm text-muted-foreground font-mono">Loading run…</span>
          </div>
        </div>
      </AppShell>
    );
  }

  if (error || !run) {
    return (
      <AppShell runId={id}>
        <div className="h-full flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl clypt-glass p-6 max-w-md text-center space-y-3"
          >
            <AlertCircle className="w-8 h-8 text-destructive mx-auto" />
            <h2 className="font-display text-lg font-bold text-foreground">Failed to load run</h2>
            <p className="text-sm text-muted-foreground">{error || "Run not found"}</p>
            <Button variant="outline" size="sm" onClick={() => navigate("/")}>
              Back to Dashboard
            </Button>
          </motion.div>
        </div>
      </AppShell>
    );
  }

  const status = statusConfig[run.status] || statusConfig.pending;

  const summaryStats = run.summary
    ? [
        { label: "Nodes", value: run.summary.node_count, icon: <Network className="w-4 h-4 text-primary" /> },
        { label: "Edges", value: run.summary.edge_count, icon: <GitBranch className="w-4 h-4 text-primary" /> },
        { label: "Clip Candidates", value: run.summary.clip_candidate_count, icon: <Scissors className="w-4 h-4 text-clypt-amber" /> },
        { label: "Rendered Clips", value: run.summary.rendered_clip_count, icon: <Film className="w-4 h-4 text-clypt-green" /> },
      ]
    : [];

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
                  {extractVideoTitle(run.video_url)}
                </h1>
                <div className="flex items-center gap-2.5 mt-2">
                  <span className="text-xs text-muted-foreground font-mono">Run {run.run_id}</span>
                  <span className="w-1 h-1 rounded-full bg-border" />
                  <span className="text-xs text-muted-foreground font-mono">{formatTimestamp(run.created_at)}</span>
                  <a
                    href={run.video_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary flex items-center gap-1 hover:underline"
                  >
                    <ExternalLink className="w-3 h-3" /> source
                  </a>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline" className={`font-mono text-[10px] gap-1 ${status.badgeClass}`}>
                  {status.icon} {status.label}
                </Badge>
                {run.progress_pct > 0 && run.status === "running" && (
                  <Badge variant="outline" className="font-mono text-[10px]">
                    <BarChart3 className="w-3 h-3 mr-1" /> {run.progress_pct}%
                  </Badge>
                )}
                {run.updated_at && (
                  <Badge variant="outline" className="font-mono text-[10px]">
                    <Clock className="w-3 h-3 mr-1" /> Updated {formatTimestamp(run.updated_at)}
                  </Badge>
                )}
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
            {/* Pipeline status & summary */}
            <div className="lg:col-span-2 space-y-3">
              <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em]">Pipeline Status</h2>

              {/* Progress card */}
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="rounded-xl clypt-glass p-5 space-y-4"
              >
                <div className="flex items-center gap-3">
                  <div className="mt-0.5">
                    {run.status === "succeeded" && <CheckCircle2 className="w-5 h-5 text-clypt-green" />}
                    {run.status === "running" && <Loader2 className="w-5 h-5 text-clypt-amber animate-spin" />}
                    {run.status === "failed" && <AlertCircle className="w-5 h-5 text-destructive" />}
                    {run.status === "pending" && <Clock className="w-5 h-5 text-muted-foreground" />}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-display font-semibold text-sm text-foreground">
                      {run.phase ? run.phase.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : status.label}
                    </h3>
                    {run.detail && (
                      <p className="text-xs text-muted-foreground mt-1">{run.detail}</p>
                    )}
                  </div>
                  <span className="text-data font-mono text-lg text-foreground">{run.progress_pct}%</span>
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 rounded-full bg-secondary/60 overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      run.status === "failed" ? "bg-destructive" :
                      run.status === "succeeded" ? "bg-clypt-green" : "bg-primary"
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${run.progress_pct}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
              </motion.div>

              {/* Summary stats */}
              {summaryStats.length > 0 && (
                <>
                  <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em] pt-2">Summary</h2>
                  <div className="grid grid-cols-2 gap-3">
                    {summaryStats.map((stat, i) => (
                      <motion.div
                        key={stat.label}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.08 }}
                        className="rounded-xl clypt-glass p-4 flex items-center gap-3"
                      >
                        {stat.icon}
                        <div>
                          <div className="text-data font-mono text-lg font-bold text-foreground">{stat.value}</div>
                          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{stat.label}</div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Run details sidebar */}
            <div className="space-y-3">
              <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em]">Run Details</h2>
              <div className="rounded-xl clypt-glass p-4 space-y-4">
                {[
                  { label: "Run ID", value: run.run_id },
                  { label: "Status", value: status.label },
                  { label: "Phase", value: run.phase ? run.phase.replace(/_/g, " ") : "—" },
                  { label: "Progress", value: `${run.progress_pct}%` },
                  { label: "Created", value: formatTimestamp(run.created_at) },
                  { label: "Updated", value: formatTimestamp(run.updated_at) },
                  { label: "Creator ID", value: run.creator_id || "—" },
                ].map((item, i) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.04 }}
                    className="flex items-center justify-between"
                  >
                    <span className="text-[11px] text-muted-foreground">{item.label}</span>
                    <span className="text-[11px] font-mono text-foreground truncate max-w-[180px]">{item.value}</span>
                  </motion.div>
                ))}
              </div>

              {/* Latest status detail */}
              {run.detail && (
                <div className="space-y-2">
                  <h2 className="font-display text-xs font-bold text-muted-foreground uppercase tracking-[0.15em]">Latest Status</h2>
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="rounded-xl clypt-glass p-4"
                  >
                    <p className="text-xs text-muted-foreground leading-relaxed">{run.detail}</p>
                  </motion.div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
