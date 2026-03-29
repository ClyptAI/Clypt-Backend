import { motion, AnimatePresence } from "framer-motion";
import { X, Clock, User, GitBranch, Star, Play, Heart, MessageCircle } from "lucide-react";
import { type SemanticNodeData, type NodeComment, nodeTypeConfig } from "@/data/mockNodes";
import { type NarrativeEdgeData, relationConfig } from "@/data/mockEdges";
import { Badge } from "@/components/ui/badge";

interface NodeInspectorProps {
  nodeData: SemanticNodeData | null;
  edgeData: (NarrativeEdgeData & { sourceLabel: string; targetLabel: string }) | null;
  onClose: () => void;
}

const accentColorMap: Record<string, { border: string; badge: string; text: string; bg: string }> = {
  hook: { border: "border-clypt-teal/40", badge: "bg-clypt-teal/15 text-clypt-teal border-clypt-teal/30", text: "text-clypt-teal", bg: "bg-clypt-teal" },
  conflict: { border: "border-primary/40", badge: "bg-primary/15 text-primary border-primary/30", text: "text-primary", bg: "bg-primary" },
  punchline: { border: "border-clypt-amber/40", badge: "bg-clypt-amber/15 text-clypt-amber border-clypt-amber/30", text: "text-clypt-amber", bg: "bg-clypt-amber" },
  payoff: { border: "border-clypt-green/40", badge: "bg-clypt-green/15 text-clypt-green border-clypt-green/30", text: "text-clypt-green", bg: "bg-clypt-green" },
  insight: { border: "border-[hsl(270,70%,60%)]/40", badge: "bg-[hsl(270,70%,60%)]/15 text-[hsl(270,70%,65%)] border-[hsl(270,70%,60%)]/30", text: "text-[hsl(270,70%,65%)]", bg: "bg-[hsl(270,70%,60%)]" },
  topic_shift: { border: "border-clypt-slate/40", badge: "bg-clypt-slate/15 text-clypt-slate border-clypt-slate/30", text: "text-clypt-slate", bg: "bg-clypt-slate" },
  speaker_beat: { border: "border-clypt-mist/40", badge: "bg-clypt-mist/15 text-clypt-mist border-clypt-mist/30", text: "text-clypt-mist", bg: "bg-clypt-mist" },
};

export function NodeInspector({ nodeData, edgeData, onClose }: NodeInspectorProps) {
  const hasContent = nodeData || edgeData;
  const accent = nodeData ? accentColorMap[nodeData.type] : null;

  return (
    <AnimatePresence>
      {hasContent && (
        <motion.div
          initial={{ opacity: 0, width: 0 }}
          animate={{ opacity: 1, width: 288 }}
          exit={{ opacity: 0, width: 0 }}
          transition={{ duration: 0 }}
          className={`shrink-0 clypt-surface rounded-lg p-4 overflow-y-auto overflow-x-hidden ${accent ? accent.border : ''}`}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className={`font-display text-xs font-semibold uppercase tracking-wider ${accent ? accent.text : 'text-foreground'}`}>
              {nodeData ? "Node Inspector" : "Edge Detail"}
            </h3>
            <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {nodeData && accent && (
            <div className="space-y-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm">{nodeTypeConfig[nodeData.type].icon}</span>
                  <Badge variant="outline" className={`text-[10px] font-mono ${accent.badge}`}>
                    {nodeTypeConfig[nodeData.type].label}
                  </Badge>
                  {nodeData.clipWorthy && (
                    <Badge className={`text-[10px] font-mono ${accent.badge}`}>
                      Clip-worthy
                    </Badge>
                  )}
                </div>
                <h4 className="font-display font-semibold text-foreground text-sm mt-2">{nodeData.label}</h4>
              </div>

              {/* Video preview */}
              <div className={`rounded-lg overflow-hidden border ${accent.border}`}>
                <div className="relative aspect-video bg-black/60">
                  <img
                    src="https://img.youtube.com/vi/xR4FC5jEMtQ/mqdefault.jpg"
                    alt="Node clip preview"
                    className="w-full h-full object-cover opacity-80"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <button className={`w-10 h-10 rounded-full ${accent.bg}/20 border ${accent.border} flex items-center justify-center backdrop-blur-sm hover:scale-110 transition-transform`}>
                      <Play className={`w-4 h-4 ${accent.text} ml-0.5`} />
                    </button>
                  </div>
                  <div className="absolute bottom-1.5 right-1.5">
                    <span className="text-[10px] font-mono bg-black/70 text-foreground px-1.5 py-0.5 rounded">
                      {Math.floor(nodeData.startTime / 60)}:{String(nodeData.startTime % 60).padStart(2, "0")}
                      {" – "}
                      {Math.floor(nodeData.endTime / 60)}:{String(nodeData.endTime % 60).padStart(2, "0")}
                    </span>
                  </div>
                </div>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">{nodeData.summary}</p>

              <div className={`rounded-md p-3 bg-[hsl(var(--clypt-surface-raised))] border ${accent.border}`}>
                <p className="text-xs text-foreground italic leading-relaxed">{nodeData.transcript}</p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  <span className="text-data">
                    {Math.floor(nodeData.startTime / 60)}:{String(nodeData.startTime % 60).padStart(2, "0")}
                    {" → "}
                    {Math.floor(nodeData.endTime / 60)}:{String(nodeData.endTime % 60).padStart(2, "0")}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <User className="w-3 h-3" />
                  <span>{nodeData.speaker}</span>
                </div>
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <Star className="w-3 h-3" />
                  <span className={`text-data font-semibold ${accent.text}`}>{(nodeData.score * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <GitBranch className="w-3 h-3" />
                  <span className="text-data">{nodeData.relatedNodeIds.length} related</span>
                </div>
              </div>

              {/* Comments section */}
              {nodeData.comments && nodeData.comments.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <MessageCircle className={`w-3 h-3 ${accent.text}`} />
                    <span className={`text-[10px] font-display font-semibold uppercase tracking-wider ${accent.text}`}>
                      Comments ({nodeData.comments.length})
                    </span>
                  </div>
                  <div className="space-y-2">
                    {nodeData.comments.map((comment: NodeComment, i: number) => (
                      <div
                        key={i}
                        className={`rounded-md p-2.5 bg-[hsl(var(--clypt-surface-raised))] border-l-2 ${accent.border.replace('/40', '/60')}`}
                      >
                        <p className="text-xs text-foreground leading-relaxed">{comment.text}</p>
                        <div className="flex items-center gap-2 mt-1.5">
                          <span className="text-[10px] text-muted-foreground font-mono">{comment.username}</span>
                          <span className="w-0.5 h-0.5 rounded-full bg-border" />
                          <div className="flex items-center gap-0.5">
                            <Heart className="w-2.5 h-2.5 text-primary/60" />
                            <span className="text-[10px] text-muted-foreground font-mono">
                              {comment.likes >= 1000 ? `${(comment.likes / 1000).toFixed(1)}K` : comment.likes}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {edgeData && (
            <div className="space-y-4">
              <div>
                <Badge variant="outline" className="text-[10px] font-mono mb-2">
                  {relationConfig[edgeData.relation].label}
                </Badge>
                <h4 className="font-display font-semibold text-foreground text-sm">
                  {edgeData.sourceLabel} → {edgeData.targetLabel}
                </h4>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">{edgeData.description}</p>

              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Strength:</span>
                <div className="flex-1 h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{ width: `${edgeData.strength * 100}%` }}
                  />
                </div>
                <span className="text-data">{(edgeData.strength * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}