"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Clock, Search, Loader2, CalendarDays, Filter,
  Zap, AlertTriangle, GitBranch, ChevronDown, ChevronRight,
  Sparkles
} from "lucide-react";
import { api } from "@/lib/api";
import { getConfidenceDotColor, getConfidenceTier } from "@/lib/confidence";
import { formatDateTime, formatDateShort } from "@/lib/formatTime";

interface TimelineGroup {
  time_key: string;
  label: string;
  record_count: number;
  event_count: number;
  items: Record<string, unknown>[];
  is_gap: boolean;
  is_burst: boolean;
  confidence_avg: number;
}

interface TimelineGap {
  start_date: string;
  end_date: string;
  duration_days: number;
  label: string;
  suggestion: string;
}

interface TimelineBurst {
  center_date: string;
  density_multiplier: number;
  knowledge_count: number;
  top_categories: string[];
  label: string;
}

interface VersionChain {
  entity_name: string;
  versions: Record<string, unknown>[];
  latest_version?: string;
  total_updates: number;
}

interface TimelineData {
  groups: TimelineGroup[];
  total: number;
  mode: string;
  granularity: string;
  gaps: TimelineGap[];
  bursts: TimelineBurst[];
  version_chains: VersionChain[];
}

interface CategoryNode {
  id: string;
  name: string;
  knowledge_count: number;
  children?: CategoryNode[];
}

type Granularity = "year" | "month" | "day";
type TimelineMode = "event_time" | "recorded_at";

const MODE_LABELS: Record<TimelineMode, string> = {
  event_time: "事件时间",
  recorded_at: "记录时间",
};

const GRANULARITY_LABELS: Record<Granularity, string> = {
  year: "年",
  month: "月",
  day: "日",
};

export default function TimelineContent() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState<TimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<TimelineMode>("event_time");
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [activeCategoryId, setActiveCategoryId] = useState<string>("");
  const [categories, setCategories] = useState<CategoryNode[]>([]);
  const [showCategories, setShowCategories] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [showGaps, setShowGaps] = useState(true);
  const [showChains, setShowChains] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.categories.getTimeline({
        mode,
        granularity,
        category_id: activeCategoryId || undefined,
      });
      setData(result);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [mode, granularity, activeCategoryId]);

  const loadCategories = useCallback(async () => {
    try {
      const res = await api.categories.getTree();
      const flatten = (nodes: Record<string, unknown>[]): CategoryNode[] =>
        nodes.map((n) => ({
          id: n.id as string,
          name: n.name as string,
          knowledge_count: n.knowledge_count as number,
          children: n.children ? flatten(n.children as Record<string, unknown>[]) : undefined,
        }));
      setCategories(flatten(res.tree as Record<string, unknown>[]));
    } catch {
      setCategories([]);
    }
  }, []);

  useEffect(() => {
    loadCategories();
  }, [loadCategories]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleExtract = async () => {
    setExtracting(true);
    try {
      const res = await api.categories.extractTimelineTimes(20);
      alert(`时间提取完成: 已更新 ${res.updated} 条，共检查 ${res.total_checked} 条`);
      loadData();
    } catch {
      alert("时间提取失败，请确保 DeepSeek API Key 已配置");
    } finally {
      setExtracting(false);
    }
  };

  const toggleGroup = (key: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleAll = () => {
    if (data && expandedGroups.size > 0) {
      setExpandedGroups(new Set());
    } else if (data) {
      setExpandedGroups(new Set(data.groups.map((g) => g.time_key)));
    }
  };

  return (
    <div className="max-w-6xl mx-auto flex gap-6">
      {showCategories && (
        <div className="w-56 shrink-0">
          <div className="card sticky top-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
              <Filter className="w-4 h-4" /> 分类筛选
            </h3>
            {activeCategoryId && (
              <button
                onClick={() => setActiveCategoryId("")}
                className="text-xs text-primary-600 mb-2 hover:underline"
              >
                清除筛选
              </button>
            )}
            <div className="max-h-[60vh] overflow-y-auto space-y-0.5">
              {categories.length > 0 ? (
                categories.map((node) => (
                  <button
                    key={node.id}
                    onClick={() => setActiveCategoryId(activeCategoryId === node.id ? "" : node.id)}
                    className={`w-full text-left px-2 py-1.5 text-sm rounded flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                      activeCategoryId === node.id
                        ? "bg-primary-50 text-primary-700 dark:bg-primary-900 dark:text-primary-300"
                        : "text-gray-700 dark:text-gray-300"
                    }`}
                  >
                    <span className="truncate flex-1">{node.name}</span>
                    {node.knowledge_count > 0 && (
                      <span className="text-xs text-gray-400 ml-1 shrink-0">({node.knowledge_count})</span>
                    )}
                  </button>
                ))
              ) : (
                <p className="text-xs text-gray-400">暂无分类</p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 space-y-6 min-w-0">
        <div className="card">
          <h1 className="text-xl font-bold flex items-center gap-2 mb-4">
            <Clock className="w-6 h-6 text-primary-600" />
            知识时间线
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
            按事件发生时间串联的知识演化轨迹，展示知识的积累过程与趋势发现
          </p>

          <div className="flex flex-col sm:flex-row gap-3 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索知识点或实体..."
                className="input-field pl-9"
              />
            </div>
            <button onClick={loadData} className="btn-secondary text-sm">刷新</button>
            <button
              onClick={handleExtract}
              disabled={extracting}
              className="btn-secondary text-sm flex items-center gap-1"
            >
              {extracting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              提取事件时间
            </button>
          </div>

          <div className="flex flex-wrap gap-3 items-center">
            <div className="flex items-center gap-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
              {(["event_time", "recorded_at"] as TimelineMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    mode === m
                      ? "bg-white dark:bg-gray-700 text-primary-700 shadow-sm"
                      : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  }`}
                >
                  {MODE_LABELS[m]}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
              {(["year", "month", "day"] as Granularity[]).map((g) => (
                <button
                  key={g}
                  onClick={() => setGranularity(g)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    granularity === g
                      ? "bg-white dark:bg-gray-700 text-primary-700 shadow-sm"
                      : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  }`}
                >
                  {GRANULARITY_LABELS[g]}
                </button>
              ))}
            </div>

            <button
              onClick={() => setShowCategories(!showCategories)}
              className={`btn-secondary text-xs flex items-center gap-1 ${
                showCategories ? "ring-2 ring-primary-400" : ""
              }`}
            >
              <Filter className="w-3.5 h-3.5" />
              分类
            </button>

            <div className="text-xs text-gray-400 ml-auto flex items-center gap-3">
              {data && (
                <>
                  <span>共 {data.total} 条</span>
                  <button onClick={toggleAll} className="hover:text-primary-600 transition-colors">
                    {expandedGroups.size > 0 ? "全部折叠" : "全部展开"}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {data && data.gaps.length > 0 && (
          <div className="card border-l-4 border-l-amber-400 bg-amber-50 dark:bg-amber-900/20">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold flex items-center gap-1.5 text-amber-700 dark:text-amber-400">
                <AlertTriangle className="w-4 h-4" />
                时间空白期 ({data.gaps.length})
              </h3>
              <button
                onClick={() => setShowGaps(!showGaps)}
                className="text-xs text-amber-600 hover:underline"
              >
                {showGaps ? "隐藏" : "显示"}
              </button>
            </div>
            {showGaps && (
              <div className="space-y-2">
                {data.gaps.map((gap, i) => (
                  <div key={i} className="text-xs text-amber-700 dark:text-amber-300">
                    <span className="font-medium">{gap.start_date} ~ {gap.end_date}</span>
                    <span className="mx-2">·</span>
                    {gap.label}
                    <span className="mx-2">·</span>
                    <span className="text-amber-600 dark:text-amber-400">{gap.suggestion}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {data && data.bursts.length > 0 && (
          <div className="card border-l-4 border-l-orange-400 bg-orange-50 dark:bg-orange-900/20">
            <h3 className="text-sm font-semibold flex items-center gap-1.5 text-orange-700 dark:text-orange-400 mb-2">
              <Zap className="w-4 h-4" />
              信息爆发期 ({data.bursts.length})
            </h3>
            <div className="space-y-1.5">
              {data.bursts.map((burst, i) => (
                <div key={i} className="text-xs text-orange-700 dark:text-orange-300 flex items-center gap-2">
                  <span className="font-medium">{burst.label}</span>
                  {burst.top_categories.length > 0 && (
                    <span className="text-orange-500">
                      主要分类: {burst.top_categories.join("、")}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {data && data.version_chains.length > 0 && (
          <div className="card border-l-4 border-l-purple-400 bg-purple-50 dark:bg-purple-900/20">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold flex items-center gap-1.5 text-purple-700 dark:text-purple-400">
                <GitBranch className="w-4 h-4" />
                版本演进链 ({data.version_chains.length})
              </h3>
              <button
                onClick={() => setShowChains(!showChains)}
                className="text-xs text-purple-600 hover:underline"
              >
                {showChains ? "隐藏" : "显示"}
              </button>
            </div>
            {showChains && (
              <div className="space-y-2">
                {data.version_chains.map((chain, i) => (
                  <div key={i} className="text-xs text-purple-700 dark:text-purple-300">
                    <span className="font-medium">{chain.entity_name}</span>
                    <span className="mx-1">·</span>
                    <span>{chain.total_updates}次更新</span>
                    {chain.latest_version && (
                      <>
                        <span className="mx-1">·</span>
                        <span>最新: {chain.latest_version}</span>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {loading && (
          <div className="flex justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          </div>
        )}

        {!loading && data && data.groups.length === 0 && (
          <div className="card text-center py-12 text-gray-500">
            <CalendarDays className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>暂无时间线数据</p>
            <p className="text-sm mt-1">
              {mode === "event_time"
                ? "点击「提取事件时间」按钮，或上传新知识后将自动提取"
                : "上传知识后时间线将自动填充"}
            </p>
          </div>
        )}

        {!loading && data && data.groups.length > 0 && (
          <div className="space-y-6">
            {data.groups
              .filter((g) => {
                if (!query.trim()) return true;
                const q = query.toLowerCase();
                return g.items.some(
                  (item) =>
                    String(item.fact || "").toLowerCase().includes(q) ||
                    String(item.category || "").toLowerCase().includes(q) ||
                    (item.related_entities as string[])?.some((e) => e.toLowerCase().includes(q))
                );
              })
              .map((group) => {
                const isExpanded = expandedGroups.has(group.time_key);
                const filteredItems = query.trim()
                  ? group.items.filter((item) => {
                      const q = query.toLowerCase();
                      return (
                        String(item.fact || "").toLowerCase().includes(q) ||
                        String(item.category || "").toLowerCase().includes(q) ||
                        (item.related_entities as string[])?.some((e) => e.toLowerCase().includes(q))
                      );
                    })
                  : group.items;

                return (
                  <div key={group.time_key}>
                    <div className="flex items-center gap-3 mb-3">
                      <div className={`h-px flex-1 ${
                        group.is_gap ? "bg-amber-300 dark:bg-amber-700" :
                        group.is_burst ? "bg-orange-300 dark:bg-orange-700" :
                        "bg-gray-200 dark:bg-gray-700"
                      }`} />
                      <button
                        onClick={() => toggleGroup(group.time_key)}
                        className={`text-sm font-semibold whitespace-nowrap flex items-center gap-1.5 px-3 py-1 rounded-full transition-colors ${
                          group.is_gap
                            ? "text-amber-700 bg-amber-100 dark:text-amber-400 dark:bg-amber-900"
                            : group.is_burst
                            ? "text-orange-700 bg-orange-100 dark:text-orange-400 dark:bg-orange-900"
                            : "text-primary-600 bg-primary-50 dark:bg-primary-900/30"
                        }`}
                      >
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                        {group.label}
                        {group.is_gap && <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />}
                        {group.is_burst && <Zap className="w-3.5 h-3.5 text-orange-500" />}
                        <span className={`text-xs ${
                          group.is_gap ? "text-amber-600 dark:text-amber-400" :
                          group.is_burst ? "text-orange-600 dark:text-orange-400" :
                          "text-gray-400"
                        }`}>
                          ({filteredItems.length})
                        </span>
                      </button>
                      <div className={`h-px flex-1 ${
                        group.is_gap ? "bg-amber-300 dark:bg-amber-700" :
                        group.is_burst ? "bg-orange-300 dark:bg-orange-700" :
                        "bg-gray-200 dark:bg-gray-700"
                      }`} />
                    </div>

                    {isExpanded && (
                      <div className="relative">
                        <div className={`absolute left-4 top-0 bottom-0 w-0.5 ${
                          group.is_gap ? "bg-amber-200 dark:bg-amber-800" :
                          group.is_burst ? "bg-orange-200 dark:bg-orange-800" :
                          "bg-gray-200 dark:bg-gray-700"
                        }`} />
                        <div className="space-y-3">
                          {filteredItems.map((item, i) => {
                            const conf = (item.confidence as number) || 0.5;
                            const colorClass = getConfidenceDotColor(conf);
                            const hasEventTime = item.event_time as string | undefined;
                            const isRecordOnly = mode === "event_time" && !hasEventTime;

                            return (
                              <div key={(item.id as string) || i} className="relative pl-10">
                                <div
                                  className={`absolute left-2.5 w-3 h-3 rounded-full border-2 border-white dark:border-gray-900 ${
                                    isRecordOnly ? "bg-gray-400" : colorClass
                                  } ${item.status === "archived" ? "opacity-40" : ""}`}
                                />
                                <div className={`card ${group.is_gap ? "border-l-2 border-l-amber-300" : group.is_burst ? "border-l-2 border-l-orange-300" : ""}`}>
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="flex-1 min-w-0">
                                      <p className="font-medium break-words">{item.fact as string}</p>
                                      {isRecordOnly && mode === "event_time" && (
                                        <span className="inline-block text-xs text-gray-400 bg-gray-100 dark:bg-gray-800 rounded px-1.5 py-0.5 mt-1">
                                          无事件日期
                                        </span>
                                      )}
                                    </div>
                                    <span className="text-xs text-gray-400 whitespace-nowrap shrink-0 tabular-nums">
                                      {item.event_time
                                        ? formatDateShort(item.event_time as string)
                                        : formatDateTime(item.created_at as string)}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 flex-wrap">
                                    <span
                                      className={`px-2 py-0.5 rounded-full ${
                                        item.category === "概念" ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" :
                                        item.category === "方法" ? "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300" :
                                        item.category === "事实" ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300" :
                                        item.category === "观点" ? "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300" :
                                        item.category === "待验证" ? "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300" :
                                        "bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"
                                      }`}
                                    >
                                      {item.category as string}
                                    </span>
                                    <span className="flex items-center gap-1">
                                      <span className={`w-2 h-2 rounded-full ${colorClass}`} />
                                      置信度: {conf.toFixed(2)} ({getConfidenceTier(conf).label})
                                    </span>
                                    {(item.time_precision as string) && (
                                      <span className="text-gray-400">
                                        {(item.time_precision as string) === "year" ? "年份级" :
                                         (item.time_precision as string) === "month" ? "月份级" :
                                         (item.time_precision as string) === "day" ? "日期级" : ""}
                                      </span>
                                    )}
                                    <span className="truncate max-w-[160px]">
                                      来源: {item.source as string}
                                    </span>
                                  </div>
                                  {(item.related_entities as string[])?.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-2">
                                      {(item.related_entities as string[]).map((e, j) => (
                                        <span
                                          key={j}
                                          className="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400"
                                        >
                                          {e}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        )}
      </div>
    </div>
  );
}