"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Search, BookOpen, Loader2, Database, Trash2, CheckSquare, Square,
  AlertTriangle, CheckCircle, XCircle, Info,
  PanelLeftOpen, PanelLeftClose,
  Code2, GitFork, Braces, Hash, Tag, Plus, Sparkles, FolderOpen, X,
} from "lucide-react";
import { api, KnowledgePoint } from "@/lib/api";
import { useDebounce } from "@/lib/useDebounce";
import ClassificationPanel from "./ClassificationPanel";
import PaginationBar from "./PaginationBar";
import { SkeletonCard, SkeletonGrid, SkeletonInfoBar } from "./SkeletonCard";

const LOW_CONFIDENCE_THRESHOLD = 0.4;
const DEFAULT_PAGE_SIZE = 20;
const SCROLL_STORAGE_KEY = "wiki_scroll_pos";

const confidenceColor = (c: number) =>
  c >= 0.8 ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
    : c >= 0.6 ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"
    : c >= LOW_CONFIDENCE_THRESHOLD ? "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300"
    : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300";

const categoryColor: Record<string, string> = {
  "概念": "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300",
  "事实": "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
  "方法": "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300",
  "观点": "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300",
  "待验证": "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300",
};

const CODE_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h",
  ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".ipynb", ".sql", ".sh"];

const codeLangMap: Record<string, string> = {
  ".py": "Python", ".js": "JavaScript", ".jsx": "React JSX", ".ts": "TypeScript",
  ".tsx": "React TSX", ".java": "Java", ".go": "Go", ".rs": "Rust",
  ".cpp": "C++", ".c": "C", ".h": "C Header", ".cs": "C#", ".rb": "Ruby",
  ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
  ".ipynb": "Jupyter", ".sql": "SQL", ".sh": "Shell",
};

function isCodeKnowledge(kp: KnowledgePoint): { isCode: boolean; lang: string; ext: string } {
  const source = (kp.source_document_id || kp.source || "").toLowerCase();
  for (const ext of CODE_EXTENSIONS) {
    if (source.endsWith(ext)) {
      return { isCode: true, lang: codeLangMap[ext] || ext.slice(1).toUpperCase(), ext };
    }
  }
  return { isCode: false, lang: "", ext: "" };
}

interface PaginationInfo {
  mode: string;
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  next_cursor?: string;
  prev_cursor?: string;
}

interface EvidenceData {
  knowledge_id: string;
  fact: string;
  confidence: number;
  supporting: Array<{ id: string; fact: string; confidence: number; category: string; source: string }>;
  contradicting: Array<{ id: string; fact: string; confidence: number; category: string; source: string }>;
  supporting_count: number;
  contradicting_count: number;
}

interface CategoryInfo {
  id: string;
  name: string;
  color: string;
  description?: string;
}

function useSelection(items: KnowledgePoint[]) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const safeItems = Array.isArray(items) ? items : [];
  const ids = safeItems.map((kp) => kp.id).filter(Boolean) as string[];

  const toggle = (id: string) =>
    setSelected((prev) => { const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const selectAll = () => setSelected(new Set(ids));
  const clearAll = () => setSelected(new Set());
  const isAllSelected = ids.length > 0 && ids.every((id) => selected.has(id));
  return { selected, toggle, selectAll, clearAll, isAllSelected, ids };
}

export default function WikiContent() {
  const [allPoints, setAllPoints] = useState<KnowledgePoint[]>([]);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 250);
  const [loading, setLoading] = useState(true);
  const [pagination, setPagination] = useState<PaginationInfo>({
    mode: "offset", page: 1, page_size: DEFAULT_PAGE_SIZE, total: 0, total_pages: 0, has_next: false, has_prev: false,
  });
  const [batchDeleting, setBatchDeleting] = useState(false);
  const [categoryRefreshKey, setCategoryRefreshKey] = useState(0);

  const [confirmingIds, setConfirmingIds] = useState<Set<string>>(new Set());
  const [errorMarkingIds, setErrorMarkingIds] = useState<Set<string>>(new Set());
  const [evidenceMap, setEvidenceMap] = useState<Record<string, EvidenceData>>({});
  const [expandedEvidence, setExpandedEvidence] = useState<Set<string>>(new Set());
  const [loadingEvidence, setLoadingEvidence] = useState<Set<string>>(new Set());

  const [correctingId, setCorrectingId] = useState<string | null>(null);
  const [correctText, setCorrectText] = useState("");

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeFilter, setActiveFilter] = useState<Record<string, unknown>>({});
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  const [knowledgeCategoriesMap, setKnowledgeCategoriesMap] = useState<Record<string, CategoryInfo[]>>({});
  const [showCategoryPicker, setShowCategoryPicker] = useState<string | null>(null);
  const [allCategories, setAllCategories] = useState<CategoryInfo[]>([]);
  const [autoCategorizingIds, setAutoCategorizingIds] = useState<Set<string>>(new Set());
  const [draggingKnowledgeId, setDraggingKnowledgeId] = useState<string | null>(null);
  const [batchMoving, setBatchMoving] = useState(false);
  const [showBatchCategoryPicker, setShowBatchCategoryPicker] = useState(false);

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const { selected, toggle, selectAll, clearAll, isAllSelected, ids } = useSelection(allPoints);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (q) setQuery(q);
  }, []);

  useEffect(() => {
    const handleClickOutside = () => {
      setShowCategoryPicker(null);
      setShowBatchCategoryPicker(false);
    };
    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
  }, []);

  const loadPage = useCallback(async (page: number, pageSize: number) => {
    setLoading(true);
    try {
      const filterParams: Record<string, unknown> = { page, page_size: pageSize, order: "desc" };
      if (activeFilter.category_ids) filterParams.category_id = (activeFilter.category_ids as string[])[0];
      if (activeFilter.confidence_min !== undefined) filterParams.confidence_min = activeFilter.confidence_min as number;
      if (activeFilter.confidence_max !== undefined) filterParams.confidence_max = activeFilter.confidence_max as number;
      if (debouncedQuery) filterParams.search = debouncedQuery;

      const res = await api.listKnowledge(filterParams);
      const safeData = Array.isArray(res?.data) ? res.data : [];
      const safePagination = res?.pagination && typeof res.pagination === "object"
        ? {
            mode: res.pagination.mode || "offset",
            page: Number(res.pagination.page) || 1,
            page_size: Number(res.pagination.page_size) || DEFAULT_PAGE_SIZE,
            total: Number(res.pagination.total) || 0,
            total_pages: Number(res.pagination.total_pages) || 0,
            has_next: Boolean(res.pagination.has_next),
            has_prev: Boolean(res.pagination.has_prev),
            next_cursor: res.pagination.next_cursor,
            prev_cursor: res.pagination.prev_cursor,
          }
        : { mode: "offset", page: 1, page_size: DEFAULT_PAGE_SIZE, total: 0, total_pages: 0, has_next: false, has_prev: false };
      setAllPoints(safeData);
      setPagination(safePagination);
    } catch {
      setAllPoints([]);
      setPagination({
        mode: "offset", page: 1, page_size: DEFAULT_PAGE_SIZE, total: 0, total_pages: 0, has_next: false, has_prev: false,
      });
    } finally {
      setLoading(false);
      setIsInitialLoad(false);
    }
  }, [activeFilter, debouncedQuery]);

  useEffect(() => {
    loadPage(1, DEFAULT_PAGE_SIZE);
  }, [loadPage]);

  useEffect(() => {
    if (allPoints.length > 0) {
      loadKnowledgeCategories(allPoints);
      loadAllCategories();
    }
  }, [allPoints]);

  const loadAllCategories = async () => {
    try {
      const res = await api.categories.list();
      const cats = (res.categories || []).map((c: Record<string, unknown>) => ({
        id: c.id as string,
        name: c.name as string,
        color: (c.color as string) || "#6366f1",
        description: c.description as string | undefined,
      }));
      setAllCategories(cats);
    } catch {
      setAllCategories([]);
    }
  };

  const loadKnowledgeCategories = async (points: KnowledgePoint[]) => {
    const ids = points.map((p) => p.id).filter(Boolean) as string[];
    if (ids.length === 0) return;
    try {
      const res = await api.batchGetKnowledgeCategories(ids);
      const map: Record<string, CategoryInfo[]> = {};
      for (const [kid, cats] of Object.entries(res.categories)) {
        map[kid] = (cats as Array<Record<string, unknown>>).map((c) => ({
          id: c.id as string,
          name: c.name as string,
          color: (c.color as string) || "#6366f1",
          description: c.description as string | undefined,
        }));
      }
      setKnowledgeCategoriesMap(map);
    } catch {
      // silently fail
    }
  };

  const handleAssignCategory = async (knowledgeId: string, categoryId: string) => {
    try {
      await api.assignKnowledgeCategories(knowledgeId, [categoryId], undefined, false);
      setKnowledgeCategoriesMap((prev) => {
        const existing = prev[knowledgeId] || [];
        if (existing.find((c) => c.id === categoryId)) return prev;
        const cat = allCategories.find((c) => c.id === categoryId);
        if (!cat) return prev;
        return { ...prev, [knowledgeId]: [...existing, cat] };
      });
      setCategoryRefreshKey((k) => k + 1);
    } catch (err) {
      console.error("分配分类失败:", err);
    }
  };

  const handleRemoveCategory = async (knowledgeId: string, categoryId: string) => {
    try {
      await api.removeKnowledgeCategory(knowledgeId, categoryId);
      setKnowledgeCategoriesMap((prev) => ({
        ...prev,
        [knowledgeId]: (prev[knowledgeId] || []).filter((c) => c.id !== categoryId),
      }));
      setCategoryRefreshKey((k) => k + 1);
    } catch (err) {
      console.error("移除分类失败:", err);
    }
  };

  const handleAutoCategorize = async (knowledgeId: string) => {
    setAutoCategorizingIds((prev) => new Set(prev).add(knowledgeId));
    try {
      await api.autoCategorizeKnowledge(knowledgeId, 0.7, false);
      const res = await api.getKnowledgeCategories(knowledgeId);
      setKnowledgeCategoriesMap((prev) => ({
        ...prev,
        [knowledgeId]: (res.categories || []).map((c: Record<string, unknown>) => ({
          id: c.id as string,
          name: c.name as string,
          color: (c.color as string) || "#6366f1",
          description: c.description as string | undefined,
        })),
      }));
      setCategoryRefreshKey((k) => k + 1);
      await loadAllCategories();
    } catch (err) {
      console.error("自动归类失败:", err);
    } finally {
      setAutoCategorizingIds((prev) => { const next = new Set(prev); next.delete(knowledgeId); return next; });
    }
  };

  const handleBatchMoveToCategory = async (categoryId: string) => {
    const toMove = Array.from(selected).filter((id) => allPoints.some((kp) => kp.id === id));
    if (toMove.length === 0) return;
    setBatchMoving(true);
    try {
      await api.batchAssignCategories(toMove, [categoryId], false);
      clearAll();
      setShowBatchCategoryPicker(false);
      await loadKnowledgeCategories(allPoints);
      setCategoryRefreshKey((k) => k + 1);
    } catch (err) {
      alert("批量归类失败: " + (err instanceof Error ? err.message : "未知错误"));
    } finally {
      setBatchMoving(false);
    }
  };

  const handlePageChange = useCallback((page: number) => {
    loadPage(page, pagination.page_size);
    scrollContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [loadPage, pagination.page_size]);

  const handlePageSizeChange = useCallback((size: number) => {
    loadPage(1, size);
    scrollContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [loadPage]);

  const handleFilterChange = useCallback((filter: Record<string, unknown>) => {
    setActiveFilter(filter);
  }, []);

  function handleDelete(id: string) {
    if (!confirm("确定要删除这条知识吗？此操作不可撤销。")) return;
    api.deleteKnowledge(id).then(() => {
      loadPage(pagination.page, pagination.page_size);
      setCategoryRefreshKey((k) => k + 1);
      selected.delete(id);
    }).catch((err) => {
      alert("删除失败: " + (err instanceof Error ? err.message : "未知错误"));
    });
  }

  async function handleBatchDelete() {
    const toDelete = Array.from(selected).filter((id) => allPoints.some((kp) => kp.id === id));
    if (toDelete.length === 0) return;
    if (!confirm(`确定要删除选中的 ${toDelete.length} 条知识吗？此操作不可撤销。`)) return;
    setBatchDeleting(true);
    try {
      const res = await api.deleteKnowledgeBatch(toDelete);
      clearAll();
      if (res.failed > 0) alert(`已删除 ${res.deleted} 条，${res.failed} 条删除失败`);
      loadPage(pagination.page, pagination.page_size);
      setCategoryRefreshKey((k) => k + 1);
    } catch (err) {
      alert("批量删除失败: " + (err instanceof Error ? err.message : "未知错误"));
    } finally {
      setBatchDeleting(false);
    }
  }

  async function handleResetKnowledge() {
    if (!confirm("确定要清空所有知识库数据吗？\n\n这将删除：\n- 所有知识卡片\n- 代码图谱数据\n- 所有上传文件\n\n此操作不可撤销！")) return;
    setBatchDeleting(true);
    try {
      const res = await api.resetKnowledgeAll();
      setAllPoints([]);
      setPagination({ mode: "offset", page: 1, page_size: DEFAULT_PAGE_SIZE, total: 0, total_pages: 0, has_next: false, has_prev: false });
      clearAll();
      setCategoryRefreshKey((k) => k + 1);
      alert(`已清空: ${res.vector_deleted} 条知识 / ${res.graph_nodes_deleted} 个图谱节点 / ${res.upload_files_deleted} 个文件`);
    } catch (err) {
      alert("清空失败: " + (err instanceof Error ? err.message : "未知错误"));
    } finally {
      setBatchDeleting(false);
    }
  }

  async function handleConfirm(id: string) {
    setConfirmingIds((prev) => new Set(prev).add(id));
    try {
      const res = await api.confirmKnowledge(id);
      setAllPoints((prev) =>
        prev.map((kp) => (kp.id === id ? { ...kp, confidence: res.confidence } : kp))
      );
    } catch (err) {
      alert("确认失败: " + (err instanceof Error ? err.message : "未知错误"));
    } finally {
      setConfirmingIds((prev) => { const next = new Set(prev); next.delete(id); return next; });
    }
  }

  async function handleMarkError(id: string) {
    if (!confirm("确定要将此知识标记为错误示例吗？它将不再参与正常问答。")) return;
    setErrorMarkingIds((prev) => new Set(prev).add(id));
    try {
      await api.markErrorKnowledge(id);
      setAllPoints((prev) =>
        prev.map((kp) => (kp.id === id ? { ...kp, confidence: 0.05, status: "error" } : kp))
      );
    } catch (err) {
      alert("操作失败: " + (err instanceof Error ? err.message : "未知错误"));
    } finally {
      setErrorMarkingIds((prev) => { const next = new Set(prev); next.delete(id); return next; });
    }
  }

  function openCorrectModal(id: string, currentFact: string) {
    setCorrectingId(id);
    setCorrectText(currentFact);
  }

  async function handleCorrect() {
    if (!correctingId || !correctText.trim()) return;
    try {
      await api.correctKnowledge(correctingId, correctText.trim());
      await loadPage(pagination.page, pagination.page_size);
      setCorrectingId(null);
      setCorrectText("");
    } catch (err) {
      alert("修正失败: " + (err instanceof Error ? err.message : "未知错误"));
    }
  }

  async function toggleEvidence(id: string) {
    if (expandedEvidence.has(id)) {
      setExpandedEvidence((prev) => { const next = new Set(prev); next.delete(id); return next; });
      return;
    }
    setExpandedEvidence((prev) => new Set(prev).add(id));
    if (!evidenceMap[id]) {
      setLoadingEvidence((prev) => new Set(prev).add(id));
      try {
        const data = await api.getKnowledgeEvidence(id);
        setEvidenceMap((prev) => ({ ...prev, [id]: data }));
      } catch {
        setExpandedEvidence((prev) => { const next = new Set(prev); next.delete(id); return next; });
      } finally {
        setLoadingEvidence((prev) => { const next = new Set(prev); next.delete(id); return next; });
      }
    }
  }

  const isLowConfidence = (kp: KnowledgePoint) =>
    (kp.confidence < LOW_CONFIDENCE_THRESHOLD) && kp.status !== "error" && kp.status !== "replaced";

  const selectedInView = Array.from(selected).filter((id) => allPoints.some((kp) => kp.id === id)).length;

  const renderKnowledgeCard = (kp: KnowledgePoint, i: number) => {
    const isLow = isLowConfidence(kp);
    const evidence = evidenceMap[kp.id || ""];
    const isLoadingEv = loadingEvidence.has(kp.id || "");
    const codeInfo = isCodeKnowledge(kp);
    const kid = kp.id || "";
    const kpCats = knowledgeCategoriesMap[kid] || [];
    const isDragging = draggingKnowledgeId === kid;
    const isAutoCategorizing = autoCategorizingIds.has(kid);

    return (
      <div
        key={kp.id || i}
        className={`card hover:shadow-md transition-shadow group ${selected.has(kp.id || "") ? "ring-2 ring-primary-400" : ""} ${isDragging ? "opacity-50 scale-95" : ""}`}
        draggable={!!kp.id}
        onDragStart={(e) => {
          if (!kp.id) return;
          setDraggingKnowledgeId(kp.id);
          e.dataTransfer.setData("text/plain", kp.id);
          e.dataTransfer.effectAllowed = "move";
        }}
        onDragEnd={() => setDraggingKnowledgeId(null)}
      >
        {isLow && (
          <div className="-mx-4 -mt-4 mb-3 px-4 py-1.5 bg-orange-50 dark:bg-orange-900/20 border-b border-orange-200 dark:border-orange-800 rounded-t-xl flex items-center justify-between flex-wrap gap-1">
            <span className="text-xs text-orange-700 dark:text-orange-300 font-medium flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />待验证 · 置信度 {kp.confidence.toFixed(2)}
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => kp.id && handleConfirm(kp.id)}
                disabled={confirmingIds.has(kp.id || "")}
                className="text-xs px-2 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50 disabled:opacity-50 flex items-center gap-1"
              >
                {confirmingIds.has(kp.id || "") ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle className="w-3 h-3" />}确认正确
              </button>
              <button
                onClick={() => kp.id && handleMarkError(kp.id)}
                disabled={errorMarkingIds.has(kp.id || "")}
                className="text-xs px-2 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50 disabled:opacity-50 flex items-center gap-1"
              >
                {errorMarkingIds.has(kp.id || "") ? <Loader2 className="w-3 h-3 animate-spin" /> : <XCircle className="w-3 h-3" />}此信息有误
              </button>
              <button
                onClick={() => kp.id && openCorrectModal(kp.id, kp.fact)}
                className="text-xs px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1"
              >
                <Info className="w-3 h-3" />修正
              </button>
            </div>
          </div>
        )}

        {kp.status === "error" && (
          <div className="-mx-4 -mt-4 mb-3 px-4 py-1.5 bg-red-100 dark:bg-red-900/30 border-b border-red-200 dark:border-red-800 rounded-t-xl">
            <span className="text-xs text-red-700 dark:text-red-300 font-medium flex items-center gap-1">
              <XCircle className="w-3.5 h-3.5" />错误示例 · 已排除出正常问答
            </span>
          </div>
        )}

        {kp.status === "replaced" && (
          <div className="-mx-4 -mt-4 mb-3 px-4 py-1.5 bg-gray-100 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 rounded-t-xl">
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" />已被替代 · 保留为历史版本
            </span>
          </div>
        )}

        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <button
              onClick={() => kp.id && toggle(kp.id)}
              className="mt-1 text-gray-300 hover:text-primary-600 shrink-0"
            >
              {kp.id && selected.has(kp.id) ? <CheckSquare className="w-5 h-5 text-primary-600" /> : <Square className="w-5 h-5" />}
            </button>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold mb-2 break-words">
                {codeInfo.isCode && <Code2 className="w-4 h-4 inline mr-1.5 text-primary-500 align-text-bottom" />}
                {kp.fact}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-3 truncate">
                来源: {kp.source}
              </p>
              {codeInfo.isCode && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                  <span className="px-2 py-0.5 text-xs rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 flex items-center gap-1">
                    <Braces className="w-3 h-3" />{codeInfo.lang}
                  </span>
                  <span className="px-2 py-0.5 text-xs rounded-full bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-300 flex items-center gap-1">
                    <GitFork className="w-3 h-3" />代码分析
                  </span>
                </div>
              )}
              {kp.related_entities.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {kp.related_entities.map((e, j) => (
                    <span key={j} className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-700">
                      {e}
                    </span>
                  ))}
                </div>
              )}
              {kpCats.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {kpCats.map((cat) => (
                    <span
                      key={cat.id}
                      className="px-2 py-0.5 text-xs rounded-full flex items-center gap-1 cursor-pointer hover:opacity-80"
                      style={{ backgroundColor: cat.color + "20", color: cat.color, border: `1px solid ${cat.color}40` }}
                      onClick={(e) => { e.stopPropagation(); handleRemoveCategory(kid, cat.id); }}
                      title={`点击移除"${cat.name}"分类`}
                    >
                      {cat.name}
                      <X className="w-3 h-3" />
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="flex flex-col gap-2 shrink-0 items-end">
            <button
              onClick={() => kp.id && handleDelete(kp.id)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 opacity-0 group-hover:opacity-100 transition-all"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowCategoryPicker(showCategoryPicker === kid ? null : kid);
                }}
                className="p-1 rounded-lg text-gray-400 hover:text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 opacity-0 group-hover:opacity-100 transition-all"
                title="添加分类"
              >
                <Plus className="w-4 h-4" />
              </button>
              {showCategoryPicker === kid && (
                <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-gray-800 shadow-xl rounded-lg border border-gray-200 dark:border-gray-700 w-48 max-h-48 overflow-y-auto">
                  {allCategories.filter((c) => !kpCats.find((kc) => kc.id === c.id)).map((cat) => (
                    <button
                      key={cat.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAssignCategory(kid, cat.id);
                        setShowCategoryPicker(null);
                      }}
                      className="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                    >
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cat.color }} />
                      <span className="truncate">{cat.name}</span>
                    </button>
                  ))}
                  {allCategories.filter((c) => !kpCats.find((kc) => kc.id === c.id)).length === 0 && (
                    <p className="text-xs text-gray-400 px-3 py-2">所有分类已分配</p>
                  )}
                </div>
              )}
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (kp.id) handleAutoCategorize(kp.id);
              }}
              disabled={isAutoCategorizing}
              className="p-1 rounded-lg text-gray-400 hover:text-purple-600 hover:bg-purple-50 dark:hover:bg-purple-900/20 opacity-0 group-hover:opacity-100 transition-all disabled:opacity-30"
              title="AI自动归类"
            >
              {isAutoCategorizing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            </button>
            <span className={`px-2 py-1 text-xs rounded-full ${confidenceColor(kp.confidence)}`}>
              置信度: {kp.confidence.toFixed(2)}
            </span>
            <span className={`px-2 py-1 text-xs rounded-full ${categoryColor[kp.category] || categoryColor["待验证"]}`}>
              {kp.category}
            </span>
            {(isLow || kp.status === "error" || kp.status === "replaced") && (
              <button
                onClick={() => kp.id && toggleEvidence(kp.id)}
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
              >
                {isLoadingEv ? <Loader2 className="w-3 h-3 animate-spin" /> : <Info className="w-3 h-3" />}
                关联证据
              </button>
            )}
          </div>
        </div>

        {kp.id && expandedEvidence.has(kp.id) && evidence && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 space-y-2">
            {evidence.supporting.length > 0 && (
              <div>
                <p className="text-xs font-medium text-green-700 dark:text-green-400 mb-1">✅ 支持性证据 ({evidence.supporting_count}条)</p>
                <div className="space-y-1">
                  {evidence.supporting.map((s, si) => (
                    <div key={si} className="text-xs p-2 bg-green-50 dark:bg-green-900/10 rounded">
                      <p className="font-medium">{s.fact}</p>
                      <p className="text-gray-500 mt-0.5">置信度 {s.confidence.toFixed(2)} · {s.category} · {s.source}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {evidence.contradicting.length > 0 && (
              <div>
                <p className="text-xs font-medium text-red-700 dark:text-red-400 mb-1">⚠ 矛盾性证据 ({evidence.contradicting_count}条)</p>
                <div className="space-y-1">
                  {evidence.contradicting.map((c, ci) => (
                    <div key={ci} className="text-xs p-2 bg-red-50 dark:bg-red-900/10 rounded">
                      <p className="font-medium">{c.fact}</p>
                      <p className="text-gray-500 mt-0.5">置信度 {c.confidence.toFixed(2)} · {c.category} · {c.source}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {evidence.supporting.length === 0 && evidence.contradicting.length === 0 && (
              <p className="text-xs text-gray-400">暂无关联证据</p>
            )}
          </div>
        )}

        {kp.id && expandedEvidence.has(kp.id) && !evidence && isLoadingEv && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 flex items-center justify-center py-4">
            <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
          </div>
        )}
      </div>
    );
  };

  const isEmpty = !loading && allPoints.length === 0;
  const isFilteredEmpty = isEmpty && (!!activeFilter.category_ids || !!activeFilter.confidence_min !== undefined || !!debouncedQuery);

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {sidebarOpen && (
        <ClassificationPanel
          onFilterChange={handleFilterChange}
          refreshKey={categoryRefreshKey}
          onKnowledgeDrop={async (knowledgeId, categoryId) => {
            await handleAssignCategory(knowledgeId, categoryId);
          }}
        />
      )}

      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-6 p-6">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-xl font-bold flex items-center gap-2">
                <button
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                  className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                  title={sidebarOpen ? "收起侧栏" : "展开侧栏"}
                >
                  {sidebarOpen ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
                </button>
                <BookOpen className="w-6 h-6 text-primary-600" />知识库
              </h1>
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-500 flex items-center gap-1">
                  <Database className="w-4 h-4" />共 {pagination.total} 条知识
                </span>
                <button
                  onClick={handleResetKnowledge}
                  disabled={batchDeleting || loading}
                  className="text-xs px-2 py-1 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded border border-red-200 dark:border-red-800 flex items-center gap-1"
                  title="清空所有知识、图谱和上传文件"
                >
                  {batchDeleting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                  清空知识库
                </button>
                {selectedInView > 0 && (
                  <button
                    onClick={isAllSelected ? clearAll : selectAll}
                    className="text-xs text-primary-600 hover:underline flex items-center gap-1"
                  >
                    {isAllSelected ? <CheckSquare className="w-3.5 h-3.5" /> : <Square className="w-3.5 h-3.5" />}
                    {isAllSelected ? "取消全选" : "全选"}
                  </button>
                )}
              </div>
            </div>

            {selectedInView > 0 && (
              <div className="mb-4 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg flex items-center justify-between flex-wrap gap-2">
                <span className="text-sm text-primary-700 dark:text-primary-300">
                  已选 <strong>{selectedInView}</strong> 条知识
                </span>
                <div className="flex gap-2">
                  <div className="relative">
                    <button
                      onClick={() => {
                        setShowBatchCategoryPicker(!showBatchCategoryPicker);
                        if (allCategories.length === 0) loadAllCategories();
                      }}
                      className="btn-secondary text-sm flex items-center gap-1"
                    >
                      <FolderOpen className="w-3.5 h-3.5" />
                      归类到
                    </button>
                    {showBatchCategoryPicker && (
                      <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-gray-800 shadow-xl rounded-lg border border-gray-200 dark:border-gray-700 w-48 max-h-60 overflow-y-auto">
                        {allCategories.map((cat) => (
                          <button
                            key={cat.id}
                            onClick={() => handleBatchMoveToCategory(cat.id)}
                            disabled={batchMoving}
                            className="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2 disabled:opacity-50"
                          >
                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cat.color }} />
                            <span className="truncate">{cat.name}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={handleBatchDelete}
                    disabled={batchDeleting || batchMoving}
                    className="btn-danger text-sm flex items-center gap-1"
                  >
                    {batchDeleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                    批量删除
                  </button>
                </div>
              </div>
            )}

            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={pagination.total > 0 ? `搜索 ${pagination.total} 条知识...` : "搜索知识库..."}
                className="input-field pl-10"
              />
            </div>
          </div>

          {loading && isInitialLoad ? (
            <SkeletonGrid count={6} />
          ) : isEmpty ? (
            <div className="card text-center py-12 text-gray-500">
              {isFilteredEmpty ? (
                <>
                  <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>未找到匹配的知识</p>
                  <p className="text-sm mt-1">尝试调整筛选条件或搜索关键词</p>
                </>
              ) : pagination.total === 0 ? (
                <>
                  <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-lg">知识库为空</p>
                  <p className="text-sm mt-1">
                    去<a href="/upload" className="text-primary-600 underline">上传页面</a>上传文件或网页来构建知识库
                  </p>
                </>
              ) : (
                <>
                  <Info className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>当前页无数据</p>
                </>
              )}
            </div>
          ) : (
            <>
              <div className="space-y-4 relative">
                {loading && !isInitialLoad && (
                  <div className="absolute inset-0 bg-white/60 dark:bg-gray-900/40 z-10 rounded-xl flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
                  </div>
                )}
                {allPoints.map((kp, i) => renderKnowledgeCard(kp, i))}
              </div>

              {loading && !isInitialLoad ? (
                <SkeletonInfoBar />
              ) : (
                <PaginationBar
                  page={pagination.page}
                  pageSize={pagination.page_size}
                  total={pagination.total}
                  totalPages={pagination.total_pages}
                  hasPrev={pagination.has_prev}
                  hasNext={pagination.has_next}
                  onPageChange={handlePageChange}
                  onPageSizeChange={handlePageSizeChange}
                  loading={loading}
                />
              )}
            </>
          )}

          {correctingId && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" onClick={() => { setCorrectingId(null); setCorrectText(""); }}>
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-lg shadow-xl" onClick={(e) => e.stopPropagation()}>
                <h3 className="text-lg font-semibold mb-3">修正知识内容</h3>
                <p className="text-sm text-gray-500 mb-4">请修改以下内容，旧版本将保留在历史中</p>
                <textarea
                  value={correctText}
                  onChange={(e) => setCorrectText(e.target.value)}
                  className="input-field w-full min-h-[100px] mb-4"
                  placeholder="请输入修正后的正确内容..."
                />
                <div className="flex justify-end gap-2">
                  <button onClick={() => { setCorrectingId(null); setCorrectText(""); }} className="btn-secondary text-sm">
                    取消
                  </button>
                  <button onClick={handleCorrect} disabled={!correctText.trim()} className="btn-primary text-sm disabled:opacity-50">
                    保存修正
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}