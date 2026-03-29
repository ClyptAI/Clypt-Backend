import { useState, useCallback, useMemo, useEffect } from "react";
import { useParams } from "react-router-dom";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  type Node,
  type Edge,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { AppShell } from "@/components/layout/AppShell";
import { SemanticNode } from "@/components/graph/SemanticNode";
import { NarrativeEdge } from "@/components/graph/NarrativeEdge";
import { GraphFilters } from "@/components/graph/GraphFilters";
import { NodeInspector } from "@/components/graph/NodeInspector";
import { TimelineStrip } from "@/components/graph/TimelineStrip";
import { mockNodes, type SemanticNodeData, type SemanticNodeType } from "@/data/mockNodes";
import { mockEdges, type NarrativeEdgeData } from "@/data/mockEdges";

const nodeTypes = { semantic: SemanticNode };
const edgeTypes = { narrative: NarrativeEdge };

const allTypes: SemanticNodeType[] = [
  "hook", "conflict", "punchline", "payoff", "insight", "topic_shift", "speaker_beat",
];

function CortexGraphInner() {
  const { id } = useParams();
  const [nodes, setNodes, onNodesChange] = useNodesState(mockNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(mockEdges);
  const { fitView } = useReactFlow();

  const [activeTypes, setActiveTypes] = useState<Set<SemanticNodeType>>(new Set(allTypes));
  const [scoreThreshold, setScoreThreshold] = useState(0);
  const [clipOnly, setClipOnly] = useState(false);
  const [viewMode, setViewMode] = useState("full");

  const [selectedNodeData, setSelectedNodeData] = useState<SemanticNodeData | null>(null);
  const [selectedEdgeData, setSelectedEdgeData] = useState<
    (NarrativeEdgeData & { sourceLabel: string; targetLabel: string }) | null
  >(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);

  const inspectorOpen = !!(selectedNodeData || selectedEdgeData);

  useEffect(() => {
    fitView({ padding: 0.15, duration: 0 });
  }, [inspectorOpen, fitView]);

  // Compute connected nodes & edges for the hovered node
  const { connectedNodeIds, connectedEdgeIds } = useMemo(() => {
    if (!hoveredNodeId) return { connectedNodeIds: new Set<string>(), connectedEdgeIds: new Set<string>() };
    const nodeIds = new Set<string>();
    const edgeIds = new Set<string>();
    for (const e of mockEdges) {
      if (e.source === hoveredNodeId || e.target === hoveredNodeId) {
        edgeIds.add(e.id);
        nodeIds.add(e.source);
        nodeIds.add(e.target);
      }
    }
    return { connectedNodeIds: nodeIds, connectedEdgeIds: edgeIds };
  }, [hoveredNodeId]);

  const filteredNodes = useMemo(() => {
    return nodes
      .filter((n) => {
        const d = n.data as unknown as SemanticNodeData;
        if (!activeTypes.has(d.type)) return false;
        if (d.score * 100 < scoreThreshold) return false;
        if (clipOnly && !d.clipWorthy) return false;
        return true;
      })
      .map((n) => ({
        ...n,
        data: {
          ...n.data,
          _isHoverTarget: hoveredNodeId === n.id,
          _isHoverConnected: hoveredNodeId ? connectedNodeIds.has(n.id) : false,
          _hasHover: !!hoveredNodeId,
        },
      }));
  }, [nodes, activeTypes, scoreThreshold, clipOnly, hoveredNodeId, connectedNodeIds]);

  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  const filteredEdges = useMemo(() => {
    return edges
      .filter((e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target))
      .map((e) => ({
        ...e,
        data: {
          ...e.data,
          _isHoverHighlighted: hoveredNodeId ? connectedEdgeIds.has(e.id) : false,
          _hasHover: !!hoveredNodeId,
          _isEdgeHovered: hoveredEdgeId === e.id,
        },
      }));
  }, [edges, filteredNodeIds, hoveredNodeId, connectedEdgeIds, hoveredEdgeId]);

  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const d = node.data as unknown as SemanticNodeData;
    setSelectedNodeData(d);
    setSelectedEdgeData(null);
    setSelectedNodeId(node.id);
  }, []);

  const handleEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
    const d = edge.data as unknown as NarrativeEdgeData;
    const sourceNode = mockNodes.find((n) => n.id === edge.source);
    const targetNode = mockNodes.find((n) => n.id === edge.target);
    setSelectedEdgeData({
      ...d,
      sourceLabel: (sourceNode?.data as unknown as SemanticNodeData)?.label ?? edge.source,
      targetLabel: (targetNode?.data as unknown as SemanticNodeData)?.label ?? edge.target,
    });
    setSelectedNodeData(null);
    setSelectedNodeId(null);
    setHoveredEdgeId(null);
    setHoveredNodeId(null);
  }, []);

  const resetInteractionState = useCallback(() => {
    setSelectedNodeData(null);
    setSelectedEdgeData(null);
    setSelectedNodeId(null);
    setHoveredEdgeId(null);
    setHoveredNodeId(null);
    setNodes((prev) => prev.map((node) => (node.selected ? { ...node, selected: false } : node)));
    setEdges((prev) => prev.map((edge) => (edge.selected ? { ...edge, selected: false } : edge)));
  }, [setNodes, setEdges]);

  const handlePaneClick = useCallback(() => {
    resetInteractionState();
  }, [resetInteractionState]);

  const handleNodeMouseEnter = useCallback((_: React.MouseEvent, node: Node) => {
    setHoveredNodeId(node.id);
  }, []);

  const handleNodeMouseLeave = useCallback(() => {
    setHoveredNodeId(null);
  }, []);

  const handleEdgeMouseEnter = useCallback((_: React.MouseEvent, edge: Edge) => {
    setHoveredEdgeId(edge.id);
  }, []);

  const handleEdgeMouseLeave = useCallback(() => {
    setHoveredEdgeId(null);
  }, []);

  const handleTimelineSelect = useCallback((nodeId: string) => {
    const node = mockNodes.find((n) => n.id === nodeId);
    if (node) {
      setSelectedNodeData(node.data as unknown as SemanticNodeData);
      setSelectedEdgeData(null);
      setSelectedNodeId(nodeId);
    }
  }, []);

  const toggleType = useCallback((type: SemanticNodeType) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }, []);

  return (
    <AppShell runId={id}>
      <div className="h-full flex flex-col">
        <div className="flex-1 flex gap-0 p-1 min-h-0">
          <div className="hidden lg:block">
            <GraphFilters
              activeTypes={activeTypes}
              onToggleType={toggleType}
              scoreThreshold={scoreThreshold}
              onScoreChange={setScoreThreshold}
              clipOnly={clipOnly}
              onClipOnlyChange={setClipOnly}
              viewMode={viewMode}
              onViewModeChange={setViewMode}
            />
          </div>

          <div className="flex-1 rounded-xl overflow-hidden border border-border/40 mx-1 bg-[hsl(var(--clypt-obsidian))]">
            <ReactFlow
              nodes={filteredNodes}
              edges={filteredEdges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              onEdgeClick={handleEdgeClick}
              onPaneClick={handlePaneClick}
              onNodeMouseEnter={handleNodeMouseEnter}
              onNodeMouseLeave={handleNodeMouseLeave}
              onEdgeMouseEnter={handleEdgeMouseEnter}
              onEdgeMouseLeave={handleEdgeMouseLeave}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              minZoom={0.3}
              maxZoom={2}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="hsl(0, 0%, 12%)" />
              <Controls position="bottom-left" />
              <MiniMap
                nodeStrokeWidth={3}
                pannable
                zoomable
                position="top-right"
                style={{ width: 120, height: 80 }}
              />
            </ReactFlow>
          </div>

          <div className="hidden md:flex">
            <NodeInspector
              nodeData={selectedNodeData}
              edgeData={selectedEdgeData}
              onClose={resetInteractionState}
            />
          </div>
        </div>

        <TimelineStrip selectedNodeId={selectedNodeId} onSelectNode={handleTimelineSelect} />
      </div>
    </AppShell>
  );
}

export default function CortexGraph() {
  return (
    <ReactFlowProvider>
      <CortexGraphInner />
    </ReactFlowProvider>
  );
}