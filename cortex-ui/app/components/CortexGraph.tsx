"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import Link from "next/link";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Node,
  type Edge,
  type Connection,
  BackgroundVariant,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import NodePanel, { EDGE_COLORS } from "./NodePanel";
import EdgePanel from "./EdgePanel";
import SegmentNode from "./SegmentNode";
import ClipsPanel from "./ClipsPanel";

const nodeTypes = { segment: SegmentNode };

// ─── Types ────────────────────────────────────────────────────────────────────

interface RawNode {
  start_time: number;
  end_time: number;
  transcript_segment: string;
  visual_description: string;
  vocal_delivery: string;
  confidence_score: number;
  content_mechanisms: Record<string, { present: boolean; type: string; intensity: number }>;
}

interface RawEdge {
  from_node_start_time: number;
  to_node_start_time: number;
  edge_type: string;
  narrative_classification: string;
  confidence_score: number;
}

interface Payload {
  clip_start_ms: number;
  clip_end_ms: number;
  final_score: number;
  justification: string;
  combined_transcript?: string;
}

interface SelectedEdge {
  id: string;
  edge_type: string;
  narrative_classification: string;
  confidence_score: number;
  from_node_start_time: number;
  to_node_start_time: number;
}

interface GraphResponse {
  nodes?: RawNode[];
  edges?: RawEdge[];
  payloads?: Payload[];
}

interface PipelineStatusResponse {
  status?: "idle" | "running" | "completed" | "failed";
  error?: string;
}

// ─── ZoomController ───────────────────────────────────────────────────────────
// Rendered inside <ReactFlow> so it has access to the flow context via useReactFlow.

interface ZoomHandle {
  zoomToNode: (node: Node) => void;
  zoomToEdge: (edge: Edge, allNodes: Node[]) => void;
  zoomOut: () => void;
}

const NODE_W = 186;
const NODE_H = 108; // approx height of SegmentNode
const ZOOM_IN_DURATION_MS = 820;
const ZOOM_OUT_DURATION_MS = 900;

const ZoomController = forwardRef<ZoomHandle>((_, ref) => {
  const { setCenter, fitView } = useReactFlow();

  useImperativeHandle(ref, () => ({
    zoomToNode(node) {
      setCenter(
        node.position.x + NODE_W / 2,
        node.position.y + NODE_H / 2,
        { zoom: 1.6, duration: ZOOM_IN_DURATION_MS }
      );
    },
    zoomToEdge(edge, allNodes) {
      const src = allNodes.find((n) => n.id === edge.source);
      const tgt = allNodes.find((n) => n.id === edge.target);
      if (!src || !tgt) return;
      const cx = ((src.position.x + NODE_W / 2) + (tgt.position.x + NODE_W / 2)) / 2;
      const cy = ((src.position.y + NODE_H / 2) + (tgt.position.y + NODE_H / 2)) / 2;
      setCenter(cx, cy, { zoom: 1.3, duration: ZOOM_IN_DURATION_MS });
    },
    zoomOut() {
      fitView({ padding: 0.25, duration: ZOOM_OUT_DURATION_MS });
    },
  }));

  return null;
});
ZoomController.displayName = "ZoomController";

// ─── Node / edge builders ─────────────────────────────────────────────────────

const MECHANISM_COLORS: Record<string, string> = {
  humor:    "#f59e0b",
  emotion:  "#ef4444",
  social:   "#3b82f6",
  expertise:"#14b8a6",
};

function buildFlowNodes(rawNodes: RawNode[], payloads: Payload[]): Node[] {
  const count = rawNodes.length;
  const isDense = count >= 50;
  const isVeryDense = count >= 80;

  const NODES_PER_ROW = isVeryDense ? 7 : isDense ? 6 : 5;
  const X_GAP = isDense ? 236 : 224;
  const Y_GAP = isDense ? 166 : 154;
  const ROW_STAGGER = isDense ? 26 : 18;

  const sortedPayloads = [...payloads].sort((a, b) => b.final_score - a.final_score);

  function getClipRank(startMs: number): { rank: number; score: number } | null {
    const startS = startMs / 1000;
    for (let i = 0; i < sortedPayloads.length; i++) {
      const p = sortedPayloads[i];
      if (startS >= p.clip_start_ms / 1000 - 0.5 && startS <= p.clip_end_ms / 1000 + 0.5)
        return { rank: i, score: p.final_score };
    }
    return null;
  }

  const sortedNodes = [...rawNodes].sort((a, b) => a.start_time - b.start_time);

  return sortedNodes.map((n, i) => {
    const col = i % NODES_PER_ROW;
    const row = Math.floor(i / NODES_PER_ROW);
    const x = col * X_GAP + (row % 2 === 0 ? 0 : ROW_STAGGER);
    const y = row * Y_GAP + 60;

    const mechs = Object.entries(n.content_mechanisms).filter(([, m]) => m?.present);
    const topMech = mechs.sort((a, b) => b[1].intensity - a[1].intensity)[0];
    const topColor = topMech ? MECHANISM_COLORS[topMech[0]] || "#0ea5e9" : "#0ea5e9";
    const intensity = topMech ? topMech[1].intensity : 0.2;

    const clip = getClipRank(n.start_time * 1000);
    const isTopClip = clip?.rank === 0;
    const isRecommended = clip !== null;

    const borderColor = isTopClip
      ? "#f59e0b"
      : isRecommended
      ? "#fb7185"
      : `${topColor}${Math.round(40 + intensity * 60).toString(16).padStart(2, "0")}`;

    const bg = isTopClip ? "#f59e0b2b" : isRecommended ? "#fb71850f" : `${topColor}16`;
    const shadow = isTopClip
      ? "0 0 22px #f59e0b66, 0 0 42px #f59e0b33"
      : isRecommended
      ? "0 0 12px #fb718540"
      : intensity > 0.7
      ? `0 0 10px ${topColor}30`
      : "none";

    return {
      id: String(n.start_time),
      type: "segment",
      position: { x, y },
      zIndex: isTopClip ? 40 : isRecommended ? 20 : 10,
      data: { ...n, clipRank: clip?.rank ?? null, clipScore: clip?.score ?? null, topColor, borderColor, bg, shadow },
    };
  });
}

function buildFlowEdges(rawEdges: RawEdge[]): Edge[] {
  return rawEdges.map((e, i) => {
    const color = EDGE_COLORS[e.edge_type] || "#64748b";
    const isLongRange = Math.abs(e.to_node_start_time - e.from_node_start_time) > 60;
    return {
      id: `e-${i}`,
      source: String(e.from_node_start_time),
      target: String(e.to_node_start_time),
      type: "smoothstep",
      style: {
        stroke: color,
        strokeWidth: isLongRange ? 2.5 : 2,
        strokeOpacity: isLongRange ? 1 : 0.8,
        strokeDasharray: e.edge_type === "Transition" ? "4 4" : undefined,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      data: {
        edge_type: e.edge_type,
        narrative_classification: e.narrative_classification,
        confidence_score: e.confidence_score,
        from_node_start_time: e.from_node_start_time,
        to_node_start_time: e.to_node_start_time,
      },
    };
  });
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function CortexGraph() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [rawNodes, setRawNodes] = useState<RawNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<RawNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<SelectedEdge | null>(null);
  const [payloads, setPayloads] = useState<Payload[]>([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [clipsOpen, setClipsOpen] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<"idle" | "running" | "completed" | "failed">("idle");
  const [pipelineError, setPipelineError] = useState("");
  const [loadingGraph, setLoadingGraph] = useState(true);

  const zoomRef = useRef<ZoomHandle>(null);

  const loadGraph = useCallback(async () => {
    const response = await fetch("/api/graph");
    const data = (await response.json()) as GraphResponse;
    const rn = data.nodes ?? [];
    const re = data.edges ?? [];
    const pl = data.payloads ?? [];
    setRawNodes(rn);
    setPayloads(pl);
    setNodes(buildFlowNodes(rn, pl));
    setEdges(buildFlowEdges(re));
  }, [setEdges, setNodes]);

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    async function poll() {
      try {
        const statusResponse = await fetch("/api/pipeline/status");
        const statusData = (await statusResponse.json()) as PipelineStatusResponse;
        if (cancelled) return;

        const status = statusData.status ?? "idle";
        setPipelineStatus(status);
        setPipelineError(statusData.error ?? "");

        await loadGraph();
      } catch (error) {
        if (cancelled) return;
        setPipelineStatus("failed");
        setPipelineError(error instanceof Error ? error.message : "Failed to load graph.");
      } finally {
        if (!cancelled) setLoadingGraph(false);
      }
    }

    poll();
    intervalId = setInterval(poll, 5000);

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [loadGraph]);

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedEdge(null);
      const raw = rawNodes.find((n) => String(n.start_time) === node.id);
      if (raw) setSelectedNode(raw);
      zoomRef.current?.zoomToNode(node);
    },
    [rawNodes]
  );

  const onEdgeClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      setSelectedNode(null);
      const data = edge.data as {
        edge_type?: string;
        narrative_classification?: string;
        confidence_score?: number;
        from_node_start_time?: number;
        to_node_start_time?: number;
      };
      setSelectedEdge({
        id: edge.id,
        edge_type: data?.edge_type || "Transition",
        narrative_classification: data?.narrative_classification || "",
        confidence_score: data?.confidence_score ?? 0.9,
        from_node_start_time: data?.from_node_start_time ?? parseFloat(edge.source),
        to_node_start_time: data?.to_node_start_time ?? parseFloat(edge.target),
      });
      zoomRef.current?.zoomToEdge(edge, nodes);
    },
    [nodes]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedEdge(null);
    zoomRef.current?.zoomOut();
  }, []);

  const onNodeChange = useCallback(
    (updated: RawNode) => {
      setSelectedNode(updated);
      setRawNodes((prev) => prev.map((n) => (n.start_time === updated.start_time ? updated : n)));
      setNodes((prev) =>
        prev.map((n) =>
          n.id === String(updated.start_time) ? { ...n, data: { ...n.data, ...updated } } : n
        )
      );
    },
    [setNodes]
  );

  const onSave = async () => {
    setSaving(true);
    const rawEdges = edges.map((e) => ({
      from_node_start_time: parseFloat(e.source),
      to_node_start_time: parseFloat(e.target),
      edge_type: (e.data as { edge_type?: string })?.edge_type || "Transition",
      narrative_classification: (e.data as { narrative_classification?: string })?.narrative_classification || "",
      confidence_score: (e.data as { confidence_score?: number })?.confidence_score || 0.9,
    }));
    await fetch("/api/graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes: rawNodes, edges: rawEdges }),
    });
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 1800);
  };

  const panelOpen = selectedNode !== null || selectedEdge !== null;
  const recommendedCount = useMemo(
    () => nodes.filter((n) => (n.data as { clipRank?: number | null }).clipRank != null).length,
    [nodes]
  );

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#060b11] text-[#e2e8f0]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,#0f766e28,transparent_34%),radial-gradient(circle_at_bottom_right,#fb71851f,transparent_34%)]" />

      <header className="absolute left-0 right-0 top-0 z-10 border-b border-[#1f2937] bg-[#090f17dd] backdrop-blur">
        <div className="mx-auto flex max-w-[1700px] items-center justify-between px-6 py-3">
          <div>
            <Link href="/" className="inline-block text-lg font-semibold tracking-tight text-white transition hover:text-[#5eead4]">
              Clypt
            </Link>
            <p className="text-xs text-[#94a3b8]">Cortex Map</p>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span
              className={`rounded border px-2 py-1 ${
                pipelineStatus === "running"
                  ? "border-[#f59e0b] bg-[#78350f66] text-[#fcd34d]"
                  : pipelineStatus === "completed"
                  ? "border-[#14532d] bg-[#052e2688] text-[#5eead4]"
                  : pipelineStatus === "failed"
                  ? "border-[#7f1d1d] bg-[#450a0a88] text-[#fda4af]"
                  : "border-[#334155] bg-[#0f172acc] text-[#cbd5e1]"
              }`}
            >
              pipeline: {pipelineStatus}
            </span>
            <span className="rounded border border-[#334155] bg-[#0f172acc] px-2 py-1 text-[#cbd5e1]">
              {nodes.length} nodes
            </span>
            <span className="rounded border border-[#334155] bg-[#0f172acc] px-2 py-1 text-[#cbd5e1]">
              {edges.length} edges
            </span>
            <span className="rounded border border-[#14532d] bg-[#052e2688] px-2 py-1 text-[#5eead4]">
              {recommendedCount} recommended
            </span>
            {payloads.length > 0 && (
              <button
                onClick={() => { setSelectedNode(null); setSelectedEdge(null); setClipsOpen(true); }}
                className="flex items-center gap-1.5 rounded border border-[#a78bfa40] bg-[#a78bfa12] px-3 py-1.5 font-medium text-[#c4b5fd] transition hover:border-[#a78bfa60] hover:bg-[#a78bfa20]"
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#a78bfa]" />
                {payloads.length} Clips
              </button>
            )}
            <button
              onClick={onSave}
              disabled={saving || rawNodes.length === 0}
              className="rounded border border-[#0f766e] bg-[#115e59] px-3 py-1.5 font-medium text-[#ecfeff] transition hover:bg-[#0f766e] disabled:opacity-60"
            >
              {saving ? "Saving..." : saved ? "Saved" : "Save"}
            </button>
          </div>
        </div>
      </header>

      {pipelineError && (
        <div className="absolute left-1/2 top-[60px] z-20 -translate-x-1/2 rounded border border-[#7f1d1d] bg-[#450a0aee] px-3 py-2 text-xs text-[#fecaca] shadow-lg">
          {pipelineError}
        </div>
      )}

      {loadingGraph && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#020617aa] backdrop-blur-[1px]">
          <div className="rounded border border-[#334155] bg-[#0f172a] px-4 py-3 text-sm text-[#cbd5e1]">
            Loading graph data...
          </div>
        </div>
      )}

      <div className="absolute bottom-4 left-4 z-10 flex w-[280px] flex-col gap-3">
        <aside className="rounded border border-[#1f2937] bg-[#0b1220e8] p-3 backdrop-blur">
          <p className="mb-2 text-[11px] uppercase tracking-[0.18em] text-[#94a3b8]">Node Legend</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-[#f59e0b]" />
              <span className="text-[11px] text-[#cbd5e1]">Top clip node</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-[#fb7185]" />
              <span className="text-[11px] text-[#cbd5e1]">Recommended clip node</span>
            </div>
            <div className="mt-2 border-t border-[#1e293b] pt-2">
              <p className="mb-1 text-[10px] uppercase tracking-[0.14em] text-[#94a3b8]">Mechanism Dots</p>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                {Object.entries(MECHANISM_COLORS).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-1.5">
                    <div className="h-2 w-2 rounded-full" style={{ background: color }} />
                    <span className="text-[11px] capitalize text-[#cbd5e1]">{type}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <aside className="rounded border border-[#1f2937] bg-[#0b1220e8] p-3 backdrop-blur">
          <p className="mb-2 text-[11px] uppercase tracking-[0.18em] text-[#94a3b8]">Edge Legend</p>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            {Object.entries(EDGE_COLORS).map(([type, color]) => (
              <div key={type} className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full" style={{ background: color }} />
                <span className="text-[11px] text-[#cbd5e1]">{type}</span>
              </div>
            ))}
          </div>
        </aside>
      </div>

      <div className={`h-full w-full pt-14 ${panelOpen ? "pr-96" : ""}`}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          fitView
          fitViewOptions={{ padding: 0.25, duration: 400 }}
          colorMode="dark"
          panOnScroll
          panOnScrollSpeed={0.6}
          zoomOnScroll={false}
          zoomOnPinch
          zoomActivationKeyCode="Control"
          minZoom={0.08}
          maxZoom={2}
          onlyRenderVisibleElements={false}
        >
          <ZoomController ref={zoomRef} />
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#1e293b" />
          <Controls className="[&>button]:border-[#1e293b] [&>button]:bg-[#0f172a] [&>button]:text-white" />
          <MiniMap
            nodeColor={(n) => {
              const data = n.data as { bg?: string };
              return data.bg || "#0f172a";
            }}
            maskColor="rgba(2, 6, 23, 0.85)"
            style={{ background: "#0f172a", border: "1px solid #1e293b" }}
          />
        </ReactFlow>
      </div>

      <NodePanel
        node={selectedNode}
        onClose={() => {
          setSelectedNode(null);
          zoomRef.current?.zoomOut();
        }}
        onChange={onNodeChange}
      />
      <EdgePanel
        edge={selectedEdge}
        onClose={() => {
          setSelectedEdge(null);
          zoomRef.current?.zoomOut();
        }}
      />
      {clipsOpen && <ClipsPanel payloads={payloads} onClose={() => setClipsOpen(false)} />}
    </div>
  );
}
