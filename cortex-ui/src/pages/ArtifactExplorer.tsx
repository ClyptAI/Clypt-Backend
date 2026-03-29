import { useState } from "react";
import { useParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { FileJson, ChevronDown, ChevronRight, Hash } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { mockArtifacts, type ArtifactFile } from "@/data/mockArtifacts";

function DataView({ data, depth = 0 }: { data: unknown; depth?: number }) {
  if (data === null || data === undefined) {
    return <span className="text-muted-foreground italic">null</span>;
  }

  if (typeof data === "string" || typeof data === "number" || typeof data === "boolean") {
    const color =
      typeof data === "string" ? "text-clypt-teal"
      : typeof data === "number" ? "text-clypt-amber"
      : "text-primary";
    return <span className={`text-data text-xs ${color}`}>{JSON.stringify(data)}</span>;
  }

  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="text-muted-foreground text-xs font-mono">[]</span>;
    if (typeof data[0] === "object" && data[0] !== null) {
      const keys = Object.keys(data[0]);
      return (
        <div className="overflow-x-auto rounded-lg border border-border/40">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-secondary/30">
                {keys.map((k) => (
                  <th key={k} className="px-2.5 py-2 text-left font-mono text-muted-foreground font-normal whitespace-nowrap text-[10px] uppercase tracking-wider">
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={i} className="border-t border-border/30 hover:bg-secondary/10 transition-colors">
                  {keys.map((k) => (
                    <td key={k} className="px-2.5 py-1.5 text-data text-foreground whitespace-nowrap">
                      {typeof (row as Record<string, unknown>)[k] === "object"
                        ? JSON.stringify((row as Record<string, unknown>)[k])
                        : String((row as Record<string, unknown>)[k])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    return (
      <div className="space-y-0.5">
        {data.map((item, i) => (
          <div key={i} className="flex items-start gap-1">
            <span className="text-data text-[10px] text-muted-foreground/50 w-4 shrink-0">{i}</span>
            <DataView data={item} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }

  if (typeof data === "object") {
    return (
      <div className={`space-y-1.5 ${depth > 0 ? "ml-3 pl-3 border-l border-border/30" : ""}`}>
        {Object.entries(data as Record<string, unknown>).map(([key, val]) => (
          <div key={key}>
            <span className="text-[11px] font-mono text-muted-foreground">{key}: </span>
            {typeof val === "object" && val !== null ? (
              <div className="mt-1"><DataView data={val} depth={depth + 1} /></div>
            ) : (
              <DataView data={val} depth={depth + 1} />
            )}
          </div>
        ))}
      </div>
    );
  }

  return <span className="text-xs text-foreground">{String(data)}</span>;
}

function ArtifactCard({ artifact }: { artifact: ArtifactFile }) {
  const [expanded, setExpanded] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  return (
    <div className="rounded-xl clypt-glass overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-5 hover:bg-secondary/20 transition-colors text-left"
      >
        <FileJson className="w-5 h-5 text-primary shrink-0" />
        <div className="flex-1 min-w-0">
          <h3 className="font-display font-semibold text-sm text-foreground">{artifact.name}</h3>
          <p className="text-[11px] text-muted-foreground mt-0.5">{artifact.phase}</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px] font-mono">{artifact.size}</Badge>
          <Badge variant="outline" className="text-[10px] font-mono">
            <Hash className="w-2.5 h-2.5 mr-0.5" />{artifact.itemCount}
          </Badge>
          {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-3">
              <p className="text-xs text-muted-foreground">{artifact.description}</p>
              <div className="rounded-lg bg-secondary/20 p-4 border border-border/30">
                <DataView data={artifact.preview} />
              </div>
              <div className="flex justify-end">
                <Button variant="ghost" size="sm" className="text-[11px] gap-1 font-mono" onClick={() => setShowRaw(!showRaw)}>
                  <FileJson className="w-3 h-3" /> {showRaw ? "Hide" : "View"} raw
                </Button>
              </div>
              <AnimatePresence>
                {showRaw && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <pre className="text-[10px] font-mono text-muted-foreground bg-[hsl(var(--clypt-obsidian))] rounded-lg p-4 overflow-x-auto max-h-64 overflow-y-auto border border-border/30">
                      {JSON.stringify(artifact.preview, null, 2)}
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function ArtifactExplorer() {
  const { id } = useParams();

  return (
    <AppShell runId={id}>
      <div className="h-full overflow-y-auto">
        <div className="max-w-4xl mx-auto p-6">
          <div className="mb-6">
            <h1 className="font-display text-xl font-bold text-foreground">Artifact Explorer</h1>
            <p className="text-xs text-muted-foreground mt-1 font-mono">
              {mockArtifacts.length} pipeline outputs · structured data views
            </p>
          </div>

          <div className="space-y-3">
            {mockArtifacts.map((artifact, i) => (
              <motion.div key={artifact.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <ArtifactCard artifact={artifact} />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
