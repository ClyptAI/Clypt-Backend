import { SemanticNodeData, nodeTypeConfig, type SemanticNodeType } from "@/data/mockNodes";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Filter, Eye } from "lucide-react";

interface GraphFiltersProps {
  activeTypes: Set<SemanticNodeType>;
  onToggleType: (type: SemanticNodeType) => void;
  scoreThreshold: number;
  onScoreChange: (v: number) => void;
  clipOnly: boolean;
  onClipOnlyChange: (v: boolean) => void;
  viewMode: string;
  onViewModeChange: (v: string) => void;
}

const viewModes = [
  { id: "full", label: "Full Graph" },
  { id: "story", label: "Story Path" },
  { id: "clips", label: "Clip Highlights" },
];

const typeColorMap: Record<string, string> = {
  hook: "bg-clypt-teal/20 text-clypt-teal border-clypt-teal/30",
  conflict: "bg-clypt-coral/20 text-clypt-coral border-clypt-coral/30",
  punchline: "bg-clypt-amber/20 text-clypt-amber border-clypt-amber/30",
  payoff: "bg-clypt-teal/20 text-clypt-teal border-clypt-teal/30",
  insight: "bg-primary/20 text-primary border-primary/30",
  topic_shift: "bg-clypt-slate/20 text-clypt-slate border-clypt-slate/30",
  speaker_beat: "bg-clypt-mist/20 text-clypt-mist border-clypt-mist/30",
};

export function GraphFilters({
  activeTypes, onToggleType, scoreThreshold, onScoreChange,
  clipOnly, onClipOnlyChange, viewMode, onViewModeChange,
}: GraphFiltersProps) {
  return (
    <div className="w-56 shrink-0 clypt-surface rounded-lg p-4 space-y-5 overflow-y-auto">
      <div>
        <h3 className="font-display text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-1.5 mb-3">
          <Filter className="w-3.5 h-3.5" /> Node Types
        </h3>
        <div className="space-y-1.5">
          {(Object.entries(nodeTypeConfig) as [SemanticNodeType, typeof nodeTypeConfig[SemanticNodeType]][]).map(
            ([type, config]) => (
              <button
                key={type}
                onClick={() => onToggleType(type)}
                className={`
                  w-full flex items-center gap-2 px-2.5 py-1.5 rounded text-xs transition-all
                  ${activeTypes.has(type)
                    ? typeColorMap[type] + " border"
                    : "text-muted-foreground opacity-50 hover:opacity-75"
                  }
                `}
              >
                <span>{config.icon}</span>
                <span>{config.label}</span>
              </button>
            )
          )}
        </div>
      </div>

      <div>
        <h3 className="font-display text-xs font-semibold text-foreground uppercase tracking-wider mb-3">
          Score Threshold
        </h3>
        <Slider
          value={[scoreThreshold]}
          onValueChange={([v]) => onScoreChange(v)}
          min={0}
          max={100}
          step={5}
          className="mb-1"
        />
        <span className="text-data text-xs text-muted-foreground">≥ {scoreThreshold}%</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-foreground">Clip candidates only</span>
        <Switch checked={clipOnly} onCheckedChange={onClipOnlyChange} />
      </div>

      <div>
        <h3 className="font-display text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-1.5 mb-3">
          <Eye className="w-3.5 h-3.5" /> View Mode
        </h3>
        <div className="space-y-1">
          {viewModes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => onViewModeChange(mode.id)}
              className={`
                w-full text-left px-2.5 py-1.5 rounded text-xs transition-colors
                ${viewMode === mode.id
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                }
              `}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
