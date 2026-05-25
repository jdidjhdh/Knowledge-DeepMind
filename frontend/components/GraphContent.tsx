"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { usePathname } from "next/navigation";
import { GitGraph, Search, Loader2, ZoomIn, ZoomOut, Maximize2, X, Brain, AlertTriangle, ChevronDown, Clock, Eye, SlidersHorizontal } from "lucide-react";
import { api, GraphData } from "@/lib/api";
import ForceGraphWrapper from "./ForceGraphWrapper";

const COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
];

const RELATION_TYPE_COLORS: Record<string, string> = {
  IS_A: "#8b5cf6",
  PART_OF: "#06b6d4",
  INSTANCE_OF: "#6366f1",
  CAUSES: "#ef4444",
  DEPENDS_ON: "#f59e0b",
  INDICATES: "#3b82f6",
  BELONGS_TO: "#10b981",
  OCCURS_AT: "#ec4899",
  BEFORE: "#84cc16",
  AFTER: "#f97316",
  EVIDENCED_BY: "#10b981",
  CONFIRMED_BY: "#3b82f6",
  CONFLICTS_WITH: "#ef4444",
  ENDORSED_BY: "#06b6d4",
  REVISED_TO: "#8b5cf6",
  RELATED_TO: "#94a3b8",
};

const RELATION_TYPE_CN: Record<string, string> = {
  IS_A: "是…的子类",
  PART_OF: "是…的组成",
  INSTANCE_OF: "是…的实例",
  CAUSES: "导致",
  DEPENDS_ON: "依赖",
  INDICATES: "指示",
  BELONGS_TO: "属于",
  OCCURS_AT: "发生在",
  BEFORE: "早于",
  AFTER: "晚于",
  EVIDENCED_BY: "证据来源于",
  CONFIRMED_BY: "被…确认",
  CONFLICTS_WITH: "与…冲突",
  ENDORSED_BY: "被…认可",
  REVISED_TO: "修正为",
  RELATED_TO: "相关",
};

function hashColor(id: string) {
  if (!id) return COLORS[0];
  let h = 0;
  for (let i = 0; i < id.length; i++) h = ((h << 5) - h + id.charCodeAt(i)) | 0;
  return COLORS[Math.abs(h) % COLORS.length];
}

interface TooltipState {
  x: number; y: number;
  node: { id: string; label: string; degree: number; type?: string } | null;
  edge: { source: string; target: string; relation: string; relation_type?: string; confidence?: number } | null;
}

export default function GraphContent() {
  const pathname = usePathname();
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [entity, setEntity] = useState("");
  const [graphDim, setGraphDim] = useState({ width: 800, height: 580 });
  const [highlightId, setHighlightId] = useState<string | null>(null);
  const [focusEntity, setFocusEntity] = useState<string>("");
  const [tooltip, setTooltip] = useState<TooltipState>({ x: 0, y: 0, node: null, edge: null });
  const [stats, setStats] = useState({ vector_count: 0, node_count: 0 });
  const [showPanel, setShowPanel] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [hops, setHops] = useState(1);
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState("");

  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const prevPathnameRef = useRef<string | null>(null);
  const hasMounted = useRef(false);

  const degreeMap = useMemo(() => {
    const map: Record<string, number> = {};
    const edges = graphData.edges || [];
    for (const e of edges) {
      map[e.source] = (map[e.source] || 0) + 1;
      map[e.target] = (map[e.target] || 0) + 1;
    }
    return map;
  }, [graphData.edges]);

  const { highlightLinks, highlightNodes } = useMemo(() => {
    if (!highlightId) return { highlightLinks: new Set<string>(), highlightNodes: new Set<string>() };
    const nodes = new Set<string>([highlightId]);
    const links = new Set<string>();
    const edges = graphData.edges || [];
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      if (e.source === highlightId || e.target === highlightId) {
        links.add(`${i}`);
        nodes.add(e.source);
        nodes.add(e.target);
      }
    }
    return { highlightLinks: links, highlightNodes: nodes };
  }, [highlightId, graphData.edges]);

  useEffect(() => {
    if (hasMounted.current) return;
    hasMounted.current = true;
    loadGraph();
    loadStats();
    setGraphDim({ width: Math.min(window.innerWidth - 80, 1100), height: 640 });
  }, [pathname]);

  const loadGraph = async (e?: string) => {
    setLoading(true);
    try {
      const data = await api.exploreGraph(e || "", 100, hops);
      setGraphData({ nodes: data?.nodes || [], edges: data?.edges || [] });
    } catch (e) {
      console.error("加载图谱失败:", e);
      setGraphData({ nodes: [], edges: [] });
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const s = await api.getStats();
      setStats(s || { vector_count: 0, node_count: 0 });
    } catch {
      setStats({ vector_count: 0, node_count: 0 });
    }
  };

  const handleNodeClick = useCallback((node: any) => {
    setHighlightId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const handleNodeRightClick = useCallback((node: any) => {
    setFocusEntity(node.id);
    loadGraph(node.id);
  }, [hops]);

  const handleNodeHover = useCallback((node: any | null) => {
    if (node && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setTooltip({
        x: rect.left + (node.x ?? 0),
        y: rect.top + (node.y ?? 0) - 50,
        node: {
          id: node.id,
          label: node.label || node.id,
          degree: degreeMap[node.id] || 0,
          type: node.type,
        },
        edge: null,
      });
    } else {
      setTooltip({ x: 0, y: 0, node: null, edge: null });
    }
  }, [degreeMap]);

  const handleBackgroundClick = useCallback(() => {
    setHighlightId(null);
  }, []);

  const zoomIn = () => graphRef.current?.zoom?.(1.8);
  const zoomOut = () => graphRef.current?.zoom?.(0.6);
  const zoomToFit = () => graphRef.current?.zoomToFit?.(400, 80);

  const graphDataCacheRef = useRef<string>("");
  const graphDataForRenderRef = useRef<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const graphDataForRender = useMemo(() => {
    const validNodes = (graphData.nodes || []).filter((n) => n.id);
    const nodeIds = new Set(validNodes.map((n) => n.id));
    const result = {
      nodes: validNodes.map((n) => {
        const deg = degreeMap[n.id] || 0;
        const communityColor = n.community_id !== undefined
          ? COLORS[n.community_id % COLORS.length]
          : hashColor(n.id);
        return { ...n, name: n.label, degree: deg, color: communityColor, nodeType: n.type || "Entity" };
      }),
      links: (graphData.edges || [])
        .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
        .map((e, i) => ({
          ...e,
          name: e.relation,
          source: e.source,
          target: e.target,
          index: i,
          relColor: RELATION_TYPE_COLORS[e.relation_type || "RELATED_TO"] || "#94a3b8",
        })),
    };
    const key = `${result.nodes.length}|${result.links.length}`;
    if (key === graphDataCacheRef.current) {
      return graphDataForRenderRef.current;
    }
    graphDataCacheRef.current = key;
    graphDataForRenderRef.current = result;
    return result;
  }, [graphData, degreeMap]);

  const getNodeColor = useCallback((node: any) => {
    if (highlightId && !highlightNodes.has(node.id)) return "rgba(150,150,150,0.18)";
    return node.color || "#3b82f6";
  }, [highlightId, highlightNodes]);

  const getLinkColor = useCallback((link: any) => {
    if (highlightId && !highlightLinks.has(String(link.index))) return "rgba(150,150,150,0.06)";
    return link.relColor || "rgba(148,163,184,0.5)";
  }, [highlightId, highlightLinks]);

  const getLinkWidth = useCallback((link: any) => {
    if (highlightId && highlightLinks.has(String(link.index))) return 2.5;
    return 1.2;
  }, [highlightId, highlightLinks]);

  const nodeVal = useCallback((node: any) => Math.max(4, Math.min(14, (node.degree || 0) * 1.5 + 4)), []);

  const relationTypeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const e of graphData.edges) {
      const rt = e.relation_type || "RELATED_TO";
      counts[rt] = (counts[rt] || 0) + 1;
    }
    return counts;
  }, [graphData.edges]);

  return (
    <div className="space-y-4">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <GitGraph className="w-6 h-6 text-primary-600" />
            知识图谱
            {!loading && (graphData.nodes || []).length > 0 && (
              <span className="text-sm font-normal text-gray-400 ml-2">
                {(graphData.nodes || []).length} 节点 · {(graphData.edges || []).length} 关系
              </span>
            )}
          </h1>
          <div className="flex gap-1">
            <button
              onClick={() => setShowLegend(!showLegend)}
              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="图例"
            >
              <SlidersHorizontal className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowPanel(!showPanel)}
              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="图谱分析"
            >
              <Brain className="w-4 h-4" />
            </button>
          </div>
        </div>

        {showLegend && (
          <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg grid grid-cols-3 sm:grid-cols-5 gap-2 text-xs">
            {Object.entries(relationTypeCounts).map(([type, count]) => (
              <div key={type} className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: RELATION_TYPE_COLORS[type] || "#94a3b8" }} />
                <span className="text-gray-600 dark:text-gray-300 truncate">{RELATION_TYPE_CN[type] || type}</span>
                <span className="text-gray-400">({count})</span>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2 flex-wrap">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && loadGraph(entity)}
              placeholder="输入实体名称探索图谱..."
              className="input-field pl-9"
            />
          </div>
          <select
            value={hops}
            onChange={(e) => setHops(Number(e.target.value))}
            className="input-field w-24"
            title="跳数"
          >
            <option value={1}>1跳</option>
            <option value={2}>2跳</option>
            <option value={3}>3跳</option>
          </select>
          <button onClick={() => loadGraph(entity)} className="btn-primary">探索</button>
          <button onClick={() => { setEntity(""); setFocusEntity(""); loadGraph(""); }} className="btn-secondary text-sm">
            全图
          </button>
        </div>
      </div>

      {showPanel && (
        <div className="card">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Brain className="w-4 h-4 text-primary-600" />
            图谱分析工具
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <button
              onClick={async () => {
                try {
                  const r = await api.detectCommunities();
                  alert(`检测到 ${r.community_count} 个社区`);
                  loadGraph();
                } catch (e) { alert("社区检测失败: " + e); }
              }}
              className="btn-secondary text-xs"
            >
              <GitGraph className="w-3 h-3 mr-1 inline" />社区检测
            </button>
            <button
              onClick={async () => {
                try {
                  const r = await api.applyInference();
                  alert(`推理规则应用完成: ${r.total_inferred} 条新三元组`);
                  loadGraph();
                } catch (e) { alert("推理失败: " + e); }
              }}
              className="btn-secondary text-xs"
            >
              <Brain className="w-3 h-3 mr-1 inline" />推理规则
            </button>
            <button
              onClick={async () => {
                try {
                  const r = await api.detectConflicts();
                  alert(`检测到 ${r.count} 个潜在冲突`);
                } catch (e) { alert("冲突检测失败: " + e); }
              }}
              className="btn-secondary text-xs"
            >
              <AlertTriangle className="w-3 h-3 mr-1 inline" />矛盾检测
            </button>
            <button
              onClick={async () => {
                try {
                  const r = await api.syncGraph(true, true);
                  console.log("重新同步完成:", r);
                  await new Promise(r => setTimeout(r, 500));
                  loadGraph();
                  loadStats();
                } catch (e) { console.error("重新同步失败:", e); alert("同步失败: " + e); }
              }}
              className="btn-secondary text-xs"
            >
              <Eye className="w-3 h-3 mr-1 inline" />重新同步
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-24">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      ) : (graphData.nodes || []).length === 0 ? (
        <div className="card text-center py-24 text-gray-500">
          <GitGraph className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>暂无图谱数据</p>
          <p className="text-sm mt-1">上传文件并提取知识后，图谱将自动构建</p>
          <button
            onClick={async () => {
              setSyncing(true);
              setSyncMsg("正在同步图谱数据...");
              try {
                const r = await api.syncGraph(true, true);
                setSyncMsg(`同步完成: ${r.triples || 0} 三元组`);
                await new Promise(r => setTimeout(r, 500));
                await loadGraph();
                await loadStats();
              } catch (e: any) {
                setSyncMsg("同步失败: " + (e?.message || e));
                console.error("同步图谱失败:", e);
              } finally {
                setSyncing(false);
              }
            }}
            disabled={syncing}
            className="btn-primary mt-4"
          >
            {syncing ? "同步中..." : "同步图谱数据"}
          </button>
          {syncMsg && <p className="text-xs mt-2 text-gray-400">{syncMsg}</p>}
        </div>
      ) : (
        <div ref={containerRef} className="card overflow-hidden relative" style={{ height: "700px" }}>
          <div className="absolute top-3 right-3 z-10 flex gap-1">
            <button onClick={zoomIn} className="p-2 rounded-lg bg-white dark:bg-gray-800 shadow border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700" title="放大">
              <ZoomIn className="w-4 h-4" />
            </button>
            <button onClick={zoomOut} className="p-2 rounded-lg bg-white dark:bg-gray-800 shadow border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700" title="缩小">
              <ZoomOut className="w-4 h-4" />
            </button>
            <button onClick={zoomToFit} className="p-2 rounded-lg bg-white dark:bg-gray-800 shadow border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700" title="适应屏幕">
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute bottom-3 left-3 z-10 flex gap-4 text-xs text-gray-500 bg-white/80 dark:bg-gray-800/80 rounded-lg px-3 py-2 backdrop-blur flex-wrap">
            <span title="高置信度 ≥0.7"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: "#10b981" }} />高置信度</span>
            <span title="中置信度 0.4-0.7"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: "#f59e0b" }} />中置信度</span>
            <span title="低置信度 &lt;0.4"><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: "#ef4444" }} />低置信度</span>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <span>节点大小 = 连接数</span>
          </div>

          {focusEntity && (
            <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
              <span className="text-xs bg-primary-50 dark:bg-primary-900/30 rounded-lg px-3 py-1.5 shadow border text-primary-700 dark:text-primary-300">
                聚焦: <strong>{focusEntity.length > 30 ? focusEntity.slice(0, 30) + "..." : focusEntity}</strong>
              </span>
              <button onClick={() => { setFocusEntity(""); loadGraph(""); }} className="text-gray-400 hover:text-gray-600">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {highlightId && (
            <div className="absolute top-12 left-3 z-10 flex items-center gap-2">
              <span className="text-xs bg-white dark:bg-gray-800 rounded-lg px-3 py-1.5 shadow border text-gray-600 dark:text-gray-300">
                选中: <strong className="text-primary-600">{highlightId.length > 30 ? highlightId.slice(0, 30) + "..." : highlightId}</strong>
              </span>
              <button onClick={() => setHighlightId(null)} className="text-gray-400 hover:text-gray-600"><X className="w-3.5 h-3.5" /></button>
            </div>
          )}

          {tooltip.node && (
            <div
              className="absolute z-20 pointer-events-none bg-white dark:bg-gray-800 shadow-lg rounded-lg px-3 py-2 border text-sm transform -translate-x-1/2"
              style={{ left: tooltip.x, top: tooltip.y }}
            >
              <p className="font-semibold text-gray-800 dark:text-gray-100 max-w-[220px] truncate">
                {tooltip.node.label}
              </p>
              <p className="text-xs text-gray-500">
                {tooltip.node.type ? `${tooltip.node.type} · ` : ""}连接数: {tooltip.node.degree}
              </p>
            </div>
          )}

          <ForceGraphWrapper
            ref={graphRef}
            graphData={graphDataForRender}
            nodeLabel="name"
            linkLabel="name"
            nodeColor={getNodeColor}
            linkColor={getLinkColor}
            linkWidth={getLinkWidth}
            nodeVal={nodeVal}
            width={graphDim.width}
            height={660}
            onNodeClick={handleNodeClick}
            onNodeRightClick={handleNodeRightClick}
            onNodeHover={handleNodeHover}
            onBackgroundClick={handleBackgroundClick}
            enableNodeDrag={true}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkDirectionalParticles={0}
            cooldownTicks={50}
            d3VelocityDecay={0.35}
            backgroundColor="transparent"
          />
        </div>
      )}
    </div>
  );
}