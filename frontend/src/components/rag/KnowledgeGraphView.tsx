import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Fullscreen,
  Minimize2,
  Network,
  Loader2,
  ArrowLeft,
  AlertTriangle,
} from "lucide-react";
import { api } from "@/lib/api";
import { getEntityTypeColor, getEntityTypeLabel, normalizeEntityType } from "@/lib/entityTypes";
import type { KGFocusGraphData, KGGraphData, KGGraphNode, KGGraphEdge } from "@/types";

type GraphMode = "overview" | "focus";
const OVERVIEW_NODE_BUDGET = 250;
const OVERVIEW_MAX_DEPTH = 3;
const OVERVIEW_LAYOUT_SCALE = 2.8;

function getNodeRadius(degree: number, mode: GraphMode): number {
  if (mode === "overview") {
    return Math.max(3.5, Math.min(10, 3.5 + Math.sqrt(Math.max(0, degree)) * 1.15));
  }
  return Math.max(6, Math.min(18, 6 + degree * 1.5));
}

// ---------------------------------------------------------------------------
// Force simulation types
// ---------------------------------------------------------------------------
interface SimNode extends KGGraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx: number | null; // fixed position (dragging)
  fy: number | null;
}

// ---------------------------------------------------------------------------
// Simple force-directed layout
// ---------------------------------------------------------------------------
function initializeNodes(nodes: KGGraphNode[], width: number, height: number): SimNode[] {
  return nodes.map((n, i) => {
    // Stable sunflower placement avoids putting all overview nodes on one
    // ring before the force simulation has had time to separate them.
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    const radius = Math.sqrt((i + 0.5) / Math.max(nodes.length, 1))
      * Math.min(width, height) * 0.44;
    return {
      ...n,
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle),
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
    };
  });
}

function simulateForces(
  nodes: SimNode[],
  edges: KGGraphEdge[],
  width: number,
  height: number,
  alpha: number,
  mode: GraphMode,
): void {
  const centerX = width / 2;
  const centerY = height / 2;
  const isOverview = mode === "overview";
  const repulsionStrength = isOverview ? 6200 : 900;
  const minimumGap = isOverview ? 58 : 28;

  // Repulsion between all nodes
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = (repulsionStrength * alpha) / (dist * dist);
      const collisionDistance = minimumGap
        + getNodeRadius(nodes[i].degree, mode)
        + getNodeRadius(nodes[j].degree, mode);
      const collisionForce = dist < collisionDistance
        ? (collisionDistance - dist) * 0.035 * alpha
        : 0;
      const fx = (dx / dist) * (force + collisionForce);
      const fy = (dy / dist) * (force + collisionForce);
      nodes[i].vx -= fx;
      nodes[i].vy -= fy;
      nodes[j].vx += fx;
      nodes[j].vy += fy;
    }
  }

  // Spring force for edges
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  for (const edge of edges) {
    const src = nodeMap.get(edge.source);
    const tgt = nodeMap.get(edge.target);
    if (!src || !tgt) continue;
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
    const targetDist = isOverview ? 230 : 120;
    const force = (dist - targetDist) * (isOverview ? 0.006 : 0.01) * alpha;
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;
    src.vx += fx;
    src.vy += fy;
    tgt.vx -= fx;
    tgt.vy -= fy;
  }

  // Center gravity
  for (const node of nodes) {
    node.vx += (centerX - node.x) * (isOverview ? 0.00025 : 0.001) * alpha;
    node.vy += (centerY - node.y) * (isOverview ? 0.00025 : 0.001) * alpha;
  }

  // Apply velocities with damping
  for (const node of nodes) {
    if (node.fx !== null) {
      node.x = node.fx;
      node.vx = 0;
    } else {
      node.vx *= 0.6;
      node.x += node.vx;
      node.x = Math.max(28, Math.min(width - 28, node.x));
    }
    if (node.fy !== null) {
      node.y = node.fy;
      node.vy = 0;
    } else {
      node.vy *= 0.6;
      node.y += node.vy;
      node.y = Math.max(28, Math.min(height - 28, node.y));
    }
  }
}

// ---------------------------------------------------------------------------
// GraphCanvas — SVG rendering
// ---------------------------------------------------------------------------
interface GraphCanvasProps {
  data: KGGraphData;
  width: number;
  height: number;
  highlightEntities?: string[];
  mode: GraphMode;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
}

const GraphCanvas = memo(function GraphCanvas({
  data,
  width,
  height,
  highlightEntities = [],
  mode,
  isFullscreen,
  onToggleFullscreen,
}: GraphCanvasProps) {
  const isOverview = mode === "overview";
  const layoutScale = isOverview ? OVERVIEW_LAYOUT_SCALE : 1;
  const layoutWidth = width * layoutScale;
  const layoutHeight = height * layoutScale;
  const fittedZoom = 1 / layoutScale;
  const initialZoom = isOverview ? 0.62 : 1;
  const initialPan = useMemo(() => ({
    x: (width - layoutWidth * initialZoom) / 2,
    y: (height - layoutHeight * initialZoom) / 2,
  }), [height, initialZoom, layoutHeight, layoutWidth, width]);
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState<string | null>(null);
  const [panning, setPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const frameRef = useRef<number>(0);
  const alphaRef = useRef(1);

  // Initialize nodes
  useEffect(() => {
    setNodes(initializeNodes(data.nodes, layoutWidth, layoutHeight));
    setZoom(initialZoom);
    setPan(initialPan);
    setSelectedNode(null);
    alphaRef.current = 1;
  }, [data.nodes, initialPan, initialZoom, layoutHeight, layoutWidth]);

  // Run simulation
  useEffect(() => {
    if (nodes.length === 0) return;

    const tick = () => {
      if (alphaRef.current > 0.01) {
        setNodes((prev) => {
          const next = prev.map((n) => ({ ...n }));
          simulateForces(
            next,
            data.edges,
            layoutWidth,
            layoutHeight,
            alphaRef.current,
            mode,
          );
          return next;
        });
        alphaRef.current *= 0.99;
        frameRef.current = requestAnimationFrame(tick);
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [nodes.length, data.edges, layoutWidth, layoutHeight, mode]);

  // Node map for edge rendering
  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  // Connected edges for hover highlight
  const connectedEdges = useMemo(() => {
    if (!hoveredNode && !selectedNode) return new Set<number>();
    const target = selectedNode || hoveredNode;
    const set = new Set<number>();
    data.edges.forEach((e, i) => {
      if (e.source === target || e.target === target) set.add(i);
    });
    return set;
  }, [hoveredNode, selectedNode, data.edges]);

  const connectedNodes = useMemo(() => {
    const target = selectedNode || hoveredNode;
    if (!target) return new Set<string>();
    const set = new Set<string>([target]);
    data.edges.forEach((e) => {
      if (e.source === target) set.add(e.target);
      if (e.target === target) set.add(e.source);
    });
    return set;
  }, [hoveredNode, selectedNode, data.edges]);

  // Drag handlers
  const handleNodeMouseDown = useCallback((nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDragging(nodeId);
    alphaRef.current = 0.3; // Reheat
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragging) {
      const svgRect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
      const x = (e.clientX - svgRect.left - pan.x) / zoom;
      const y = (e.clientY - svgRect.top - pan.y) / zoom;
      setNodes((prev) =>
        prev.map((n) => (n.id === dragging ? { ...n, fx: x, fy: y, x, y } : n))
      );
    } else if (panning) {
      setPan({
        x: panStart.current.panX + (e.clientX - panStart.current.x),
        y: panStart.current.panY + (e.clientY - panStart.current.y),
      });
    }
  }, [dragging, panning, pan.x, pan.y, zoom]);

  const handleMouseUp = useCallback(() => {
    if (dragging) {
      setNodes((prev) =>
        prev.map((n) => (n.id === dragging ? { ...n, fx: null, fy: null } : n))
      );
      setDragging(null);
    }
    setPanning(false);
  }, [dragging]);

  const handleSvgMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget || (e.target as Element).tagName === "rect") {
      setPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
      setSelectedNode(null);
    }
  }, [pan]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.3, Math.min(3, z - e.deltaY * 0.001)));
  }, []);

  const legendTypes = useMemo(
    () => [...new Map(
      data.nodes.map((node) => [normalizeEntityType(node.entity_type), node.entity_type])
    ).values()].sort((left, right) =>
      getEntityTypeLabel(left).localeCompare(getEntityTypeLabel(right), "zh-CN")
    ),
    [data.nodes]
  );
  const overviewLabelIds = useMemo(() => new Set(
    [...data.nodes]
      .sort((left, right) => right.degree - left.degree || left.label.localeCompare(right.label))
      .slice(0, Math.max(24, Math.min(42, Math.round(width / 14))))
      .map((node) => node.id)
  ), [data.nodes, width]);

  return (
    <div className="relative w-full h-full">
      {/* Zoom controls */}
      <div className="absolute top-2 right-2 z-10 flex flex-col gap-1">
        <button
          onClick={() => setZoom((z) => Math.min(3, z + 0.2))}
          className="p-1.5 rounded-md border bg-background/80 backdrop-blur-sm hover:bg-muted transition-colors"
          title="放大图谱"
          aria-label="放大图谱"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.3, z - 0.2))}
          className="p-1.5 rounded-md border bg-background/80 backdrop-blur-sm hover:bg-muted transition-colors"
          title="缩小图谱"
          aria-label="缩小图谱"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => { setZoom(fittedZoom); setPan({ x: 0, y: 0 }); }}
          className="p-1.5 rounded-md border bg-background/80 backdrop-blur-sm hover:bg-muted transition-colors"
          title="适应画布"
          aria-label="适应画布"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          onClick={onToggleFullscreen}
          className="p-1.5 rounded-md border bg-background/80 backdrop-blur-sm hover:bg-muted transition-colors"
          title={isFullscreen ? "退出全屏" : "全屏查看图谱"}
          aria-label={isFullscreen ? "退出全屏" : "全屏查看图谱"}
        >
          {isFullscreen
            ? <Minimize2 className="w-3.5 h-3.5" />
            : <Fullscreen className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 z-10 flex max-h-20 max-w-[calc(100%-7rem)] gap-x-2 gap-y-1 overflow-y-auto rounded-md border bg-background/85 px-2 py-1.5 backdrop-blur-sm">
        {legendTypes.map((type) => (
          <div key={type} className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: getEntityTypeColor(type) }} />
            <span className="whitespace-nowrap text-[10px] text-muted-foreground">
              {getEntityTypeLabel(type)}
            </span>
          </div>
        ))}
      </div>

      {/* SVG Canvas */}
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="rounded-lg border bg-card/30 cursor-grab active:cursor-grabbing"
        onMouseDown={handleSvgMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
          {/* Edges */}
          {data.edges.map((edge, i) => {
            const src = nodeMap.get(edge.source);
            const tgt = nodeMap.get(edge.target);
            if (!src || !tgt) return null;
            const highlighted = connectedEdges.has(i);
            const dimmed = (hoveredNode || selectedNode) && !highlighted;
            return (
              <line
                key={`${edge.source}-${edge.target}-${i}`}
                x1={src.x}
                y1={src.y}
                x2={tgt.x}
                y2={tgt.y}
                stroke={highlighted ? getEntityTypeColor(src.entity_type) : "#475569"}
                strokeWidth={highlighted ? 2 : 1}
                strokeOpacity={dimmed ? 0.1 : highlighted ? 0.8 : 0.25}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {/* Nodes */}
          {nodes.map((node) => {
            const r = getNodeRadius(node.degree, mode);
            const color = getEntityTypeColor(node.entity_type);
            const isHovered = hoveredNode === node.id;
            const isSelected = selectedNode === node.id;
            const isHighlighted = highlightEntities.length > 0 &&
              highlightEntities.some((e) => e.toLowerCase() === node.label.toLowerCase());
            const dimmed = highlightEntities.length > 0
              ? !isHighlighted && !isHovered && !isSelected
              : (hoveredNode || selectedNode) && !connectedNodes.has(node.id);
            const showLabel = !isOverview
              || zoom >= 1.05
              || overviewLabelIds.has(node.id)
              || isHovered
              || isSelected
              || isHighlighted;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x},${node.y})`}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onMouseDown={(e) => handleNodeMouseDown(node.id, e)}
                onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                className="cursor-pointer"
              >
                <g transform={`scale(${1 / zoom})`}>
                {/* Glow ring */}
                {(isHovered || isSelected || isHighlighted) && (
                  <circle
                    r={r + (isHighlighted ? 6 : 4)}
                    fill="none"
                    stroke={isHighlighted ? "#fbbf24" : color}
                    strokeWidth={isHighlighted ? 3 : 2}
                    strokeOpacity={isHighlighted ? 0.7 : 0.4}
                  >
                    {isHighlighted && (
                      <animate
                        attributeName="stroke-opacity"
                        values="0.7;0.3;0.7"
                        dur="2s"
                        repeatCount="indefinite"
                      />
                    )}
                  </circle>
                )}
                {/* Node circle */}
                <circle
                  r={r}
                  fill={color}
                  fillOpacity={dimmed ? 0.15 : 0.85}
                  stroke={color}
                  strokeWidth={isSelected ? 2 : 1}
                  strokeOpacity={dimmed ? 0.2 : 1}
                />
                {/* Label (shown when not too zoomed out) */}
                {zoom > 0.35 && showLabel && (
                  <text
                    y={r + 12}
                    textAnchor="middle"
                    fontSize={11}
                    fill="currentColor"
                    fillOpacity={dimmed ? 0.16 : 0.82}
                    stroke="var(--color-background)"
                    strokeWidth={3}
                    strokeLinejoin="round"
                    style={{ paintOrder: "stroke" }}
                    className="pointer-events-none select-none"
                  >
                    {node.label.length > 20 ? node.label.slice(0, 18) + "…" : node.label}
                  </text>
                )}
                </g>
              </g>
            );
          })}
        </g>
      </svg>

      {/* Selected node tooltip */}
      {selectedNode && (() => {
        const node = nodes.find((n) => n.id === selectedNode);
        if (!node) return null;
        return (
          <div className="absolute top-2 left-2 z-10 max-h-[34%] w-[min(280px,calc(100%-4rem))] overflow-y-auto rounded-md border bg-background/95 p-2.5 shadow-lg backdrop-blur-sm">
            <p className="truncate whitespace-nowrap text-sm font-semibold" title={node.label}>{node.label}</p>
            <div className="mt-1.5 flex items-center gap-2 text-[10px]">
              <span
                className="inline-flex rounded-full border px-2 py-0.5 font-medium"
                style={{
                  color: getEntityTypeColor(node.entity_type),
                  borderColor: `color-mix(in srgb, ${getEntityTypeColor(node.entity_type)} 35%, transparent)`,
                  backgroundColor: `color-mix(in srgb, ${getEntityTypeColor(node.entity_type)} 10%, transparent)`,
                }}
              >
                {getEntityTypeLabel(node.entity_type)}
              </span>
              <span className="text-muted-foreground">{node.degree} 个连接</span>
            </div>
            <div className="mt-3 border-t pt-2">
              <p className="mb-1 text-[10px] font-semibold text-muted-foreground">实体说明</p>
              <p className="whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/80">
                {node.description || "暂无实体说明"}
              </p>
            </div>
          </div>
        );
      })()}

      {data.is_truncated && (
        <div className="absolute bottom-2 right-2 z-10 text-[10px] text-amber-400 bg-background/80 backdrop-blur-sm border border-amber-400/30 rounded px-2 py-1">
          图谱已按节点预算截断
        </div>
      )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// KnowledgeGraphView — main export
// ---------------------------------------------------------------------------
interface KnowledgeGraphViewProps {
  projectId: string;
  highlightEntities?: string[];
  highlightDocumentIds?: number[];
  citationId?: number | string | null;
  onClearFocus?: () => void;
}

export const KnowledgeGraphView = memo(function KnowledgeGraphView({
  projectId,
  highlightEntities = [],
  highlightDocumentIds = [],
  citationId,
  onClearFocus,
}: KnowledgeGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const focusEntities = useMemo(
    () => [...new Map(
      highlightEntities
        .map((entity) => entity.trim())
        .filter(Boolean)
        .map((entity) => [entity.toLocaleLowerCase(), entity])
    ).values()].sort((left, right) => left.localeCompare(right)),
    [highlightEntities]
  );
  const focusDocumentIds = useMemo(
    () => [...new Set(highlightDocumentIds)].sort((left, right) => left - right),
    [highlightDocumentIds]
  );
  const isFocusMode = focusEntities.length > 0;

  useEffect(() => {
    if (!isFullscreen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsFullscreen(false);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((current) => !current);
  }, []);

  const { data, isLoading } = useQuery<KGGraphData | KGFocusGraphData>({
    queryKey: isFocusMode
      ? ["kg-graph", "focus", projectId, focusEntities, focusDocumentIds, 1, 80]
      : ["kg-graph", "overview", projectId, OVERVIEW_NODE_BUDGET, OVERVIEW_MAX_DEPTH],
    queryFn: () => isFocusMode
      ? api.post<KGFocusGraphData>(`/rag/graph/${projectId}/focus`, {
          entity_names: focusEntities,
          document_ids: focusDocumentIds,
          max_depth: 1,
          max_nodes: 80,
        })
      : api.get<KGGraphData>(
          `/rag/graph/${projectId}?max_nodes=${OVERVIEW_NODE_BUDGET}&max_depth=${OVERVIEW_MAX_DEPTH}`
        ),
    staleTime: 60_000,
  });

  // The graph container is mounted only after loading completes, so the
  // observer must be attached again when that loading state changes.
  useEffect(() => {
    if (isLoading || !containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 50 && h > 50) {
          setDimensions({ width: w, height: h });
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [isFocusMode, isLoading]);

  const focusData = isFocusMode && data && "selection_mode" in data
    ? data as KGFocusGraphData
    : null;
  const matchedEntities = focusData?.matched_entities ?? [];
  const missingEntities = focusData?.missing_entities ?? [];

  const focusHeader = isFocusMode ? (
    <div className="flex-shrink-0 flex items-center gap-2 border-b bg-amber-500/5 px-3 py-2 text-[11px]">
      <span className="font-medium text-foreground">
        {citationId ? String(citationId) : "引用聚焦"}
      </span>
      {focusData ? (
        <span className="text-muted-foreground">
          已定位 {matchedEntities.length}/{focusData.requested_entities.length} 个实体
        </span>
      ) : (
        <span className="text-muted-foreground">正在定位引用实体…</span>
      )}
      {missingEntities.length > 0 && (
        <span
          className="inline-flex items-center gap-1 text-amber-700"
          title={`未匹配：${missingEntities.join("、")}`}
        >
          <AlertTriangle className="h-3 w-3" />
          未匹配 {missingEntities.length} 个
        </span>
      )}
      <button
        type="button"
        onClick={onClearFocus}
        className="ml-auto inline-flex items-center gap-1 rounded border bg-background px-2 py-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <ArrowLeft className="h-3 w-3" />
        返回全局图谱
      </button>
    </div>
  ) : null;

  if (isLoading) {
    return (
      <div className="h-full flex flex-col">
        {focusHeader}
        <div className="flex flex-1 items-center justify-center py-12">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground mr-2" />
          <span className="text-sm text-muted-foreground">正在加载知识图谱…</span>
        </div>
      </div>
    );
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="h-full flex flex-col">
        {focusHeader}
        <div className="flex flex-1 flex-col items-center justify-center py-10 text-center">
          <Network className="w-10 h-10 text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            {isFocusMode ? "引用实体未在当前图谱中找到" : "暂无知识图谱数据"}
          </p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            {isFocusMode
              ? "未匹配实体已明确列出，不会用其他节点替代"
              : "请先处理文档并生成知识图谱"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={isFullscreen
        ? "fixed inset-0 z-[100] flex min-h-0 flex-col bg-background p-2"
        : "w-full h-full flex flex-col min-h-0 bg-background"}
    >
      {focusHeader}
      <div ref={containerRef} className="flex-1 min-h-0">
        <GraphCanvas
          data={data}
          width={dimensions.width}
          height={dimensions.height}
          highlightEntities={isFocusMode ? matchedEntities : highlightEntities}
          mode={isFocusMode ? "focus" : "overview"}
          isFullscreen={isFullscreen}
          onToggleFullscreen={toggleFullscreen}
        />
      </div>
    </div>
  );
});
