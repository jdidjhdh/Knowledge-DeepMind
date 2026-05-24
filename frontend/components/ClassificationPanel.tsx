"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ChevronRight, ChevronDown, FolderTree, Tag, SlidersHorizontal,
  AlertTriangle, Clock, Database, RefreshCw, Plus, X,
  BarChart3, Search, MoreHorizontal, Edit3, Trash2, GitMerge,
  Sparkles, Share2
} from "lucide-react";
import { api } from "@/lib/api";

interface CategoryNode {
  id: string;
  name: string;
  description?: string;
  category_type: string;
  level: number;
  knowledge_count: number;
  avg_confidence: number;
  is_archived: boolean;
  is_frozen: boolean;
  is_system: boolean;
  color?: string;
  icon?: string;
  children: CategoryNode[];
}

interface UserTag {
  id: string;
  user_id: string;
  name: string;
  color: string;
  dimension: string;
}

interface ClassificationPanelProps {
  onFilterChange: (filter: Record<string, unknown>) => void;
  refreshKey?: number;
  onCategorySelect?: (categoryId: string, categoryName: string) => void;
  onKnowledgeDrop?: (knowledgeId: string, categoryId: string) => void;
}

export default function ClassificationPanel({
  onFilterChange,
  refreshKey = 0,
  onCategorySelect,
  onKnowledgeDrop,
}: ClassificationPanelProps) {
  const [categories, setCategories] = useState<CategoryNode[]>([]);
  const [tags, setTags] = useState<UserTag[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selectedCategoryIds, setSelectedCategoryIds] = useState<Set<string>>(new Set());
  const [selectedTagIds, setSelectedTagIds] = useState<Set<string>>(new Set());
  const [confidenceRange, setConfidenceRange] = useState<[number, number]>([0, 1]);
  const [showTags, setShowTags] = useState(false);
  const [newTagName, setNewTagName] = useState("");
  const [newTagColor, setNewTagColor] = useState("#6366f1");
  const [loadingCategories, setLoadingCategories] = useState(true);
  const [focusMode, setFocusMode] = useState(false);
  const [showHealth, setShowHealth] = useState(false);
  const [healthData, setHealthData] = useState<Array<{
    category_id: string; name: string; knowledge_count: number;
    avg_confidence: number; is_stale: boolean; needs_split: boolean; needs_attention: boolean;
  }>>([]);

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newCatName, setNewCatName] = useState("");
  const [newCatDesc, setNewCatDesc] = useState("");
  const [newCatParentId, setNewCatParentId] = useState("");
  const [newCatColor, setNewCatColor] = useState("#6366f1");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; categoryId: string; categoryName: string } | null>(null);
  const [showMergeSuggestions, setShowMergeSuggestions] = useState(false);
  const [mergeSuggestions, setMergeSuggestions] = useState<Array<Record<string, unknown>>>([]);
  const [categorySearch, setCategorySearch] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [dragOverCategoryId, setDragOverCategoryId] = useState<string | null>(null);

  useEffect(() => {
    loadCategories();
    loadTags();
    const handleClick = () => setContextMenu(null);
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, []);

  useEffect(() => {
    if (refreshKey > 0) {
      loadCategories();
    }
  }, [refreshKey]);

  const loadCategories = async () => {
    setLoadingCategories(true);
    try {
      const res = await api.categories.getPersonalizedTree("default", focusMode);
      setCategories((res.tree || []) as unknown as CategoryNode[]);
    } catch {
      try {
        const res = await api.categories.getTree();
        setCategories((res.tree || []) as unknown as CategoryNode[]);
      } catch {
        setCategories([]);
      }
    } finally {
      setLoadingCategories(false);
    }
  };

  const loadTags = async () => {
    try {
      const res = await api.tags.list();
      setTags(res.tags || []);
    } catch {
      setTags([]);
    }
  };

  const handleCreateCategory = async () => {
    if (!newCatName.trim()) return;
    try {
      await api.categories.create({
        name: newCatName.trim(),
        description: newCatDesc.trim() || undefined,
        parent_id: newCatParentId || undefined,
        color: newCatColor,
        category_type: "structural",
      });
      setNewCatName("");
      setNewCatDesc("");
      setNewCatParentId("");
      setNewCatColor("#6366f1");
      setShowCreateDialog(false);
      await loadCategories();
    } catch (err) {
      console.error("创建分类失败:", err);
    }
  };

  const handleRenameCategory = async (categoryId: string) => {
    const name = prompt("重命名分类为:");
    if (!name?.trim()) return;
    try {
      await api.categories.update(categoryId, { name: name.trim() });
      await loadCategories();
    } catch (err) {
      console.error("重命名失败:", err);
    }
    setContextMenu(null);
  };

  const handleDeleteCategory = async (categoryId: string, categoryName: string) => {
    if (!confirm(`确定删除分类"${categoryName}"吗？该分类下的知识不会被删除。`)) return;
    try {
      await api.categories.delete(categoryId);
      await loadCategories();
    } catch (err) {
      console.error("删除分类失败:", err);
    }
    setContextMenu(null);
  };

  const handleSyncGraph = async () => {
    setSyncing(true);
    try {
      const result = await api.categories.syncGraph();
      alert(`已同步 ${result.synced} 个分类到知识图谱`);
    } catch (err) {
      console.error("图谱同步失败:", err);
    } finally {
      setSyncing(false);
    }
  };

  const handleLoadMergeSuggestions = async () => {
    try {
      const res = await api.categories.getMergeSuggestions();
      setMergeSuggestions(res.suggestions || []);
      setShowMergeSuggestions(true);
    } catch {
      setMergeSuggestions([]);
    }
  };

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleCategory = (id: string) => {
    setSelectedCategoryIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleTag = (id: string) => {
    setSelectedTagIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const emitFilter = useCallback(() => {
    const filter: Record<string, unknown> = {};
    if (selectedCategoryIds.size > 0) {
      filter.category_ids = Array.from(selectedCategoryIds);
    }
    if (selectedTagIds.size > 0) {
      filter.tag_ids = Array.from(selectedTagIds);
    }
    if (confidenceRange[0] > 0 || confidenceRange[1] < 1) {
      filter.confidence_min = confidenceRange[0];
      filter.confidence_max = confidenceRange[1];
    }
    onFilterChange(filter);
  }, [selectedCategoryIds, selectedTagIds, confidenceRange, onFilterChange]);

  useEffect(() => {
    emitFilter();
  }, [emitFilter]);

  const handleCreateTag = async () => {
    if (!newTagName.trim()) return;
    try {
      await api.tags.create({ user_id: "default", name: newTagName.trim(), color: newTagColor, dimension: "custom" });
      setNewTagName("");
      await loadTags();
    } catch (err) {
      console.error("创建标签失败:", err);
    }
  };

  const handleLoadHealth = async () => {
    try {
      const res = await api.categories.getHealth();
      setHealthData(res.health || []);
    } catch {
      setHealthData([]);
    }
    setShowHealth(true);
  };

  const filteredCategories = categorySearch
    ? filterTree(categories, categorySearch.toLowerCase())
    : categories;

  const renderCategoryNode = (node: CategoryNode, depth: number = 0) => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedIds.has(node.id);
    const isSelected = selectedCategoryIds.has(node.id);
    const isDragOver = dragOverCategoryId === node.id;

    return (
      <div key={node.id} className="select-none">
        <div
          className={`flex items-center gap-1 py-1 px-1 rounded cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700/50 text-sm group ${
            isSelected ? "bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300" : ""
          } ${isDragOver ? "bg-primary-100 dark:bg-primary-900/40 ring-2 ring-primary-400" : ""}`}
          style={{ paddingLeft: `${depth * 16 + 4}px` }}
          onClick={() => {
            toggleCategory(node.id);
            onCategorySelect?.(node.id, node.name);
          }}
          onContextMenu={(e) => {
            e.preventDefault();
            if (!node.is_system) {
              setContextMenu({ x: e.clientX, y: e.clientY, categoryId: node.id, categoryName: node.name });
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            setDragOverCategoryId(node.id);
          }}
          onDragLeave={() => setDragOverCategoryId(null)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOverCategoryId(null);
            const knowledgeId = e.dataTransfer.getData("text/plain");
            if (knowledgeId && onKnowledgeDrop) {
              onKnowledgeDrop(knowledgeId, node.id);
            }
          }}
        >
          {hasChildren ? (
            <button
              onClick={(e) => { e.stopPropagation(); toggleExpand(node.id); }}
              className="p-0.5 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
            >
              {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            </button>
          ) : (
            <span className="w-4" />
          )}
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ backgroundColor: node.color || "#6366f1" }}
          />
          <span className="truncate flex-1">{node.name}</span>
          <span className="text-xs text-gray-400 shrink-0">{node.knowledge_count}</span>
          {!node.is_system && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setContextMenu({ x: e.clientX, y: e.clientY, categoryId: node.id, categoryName: node.name });
              }}
              className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
            >
              <MoreHorizontal className="w-3 h-3 text-gray-400" />
            </button>
          )}
        </div>
        {hasChildren && isExpanded && (
          node.children.map((child) => renderCategoryNode(child, depth + 1))
        )}
      </div>
    );
  };

  const clearAllFilters = () => {
    setSelectedCategoryIds(new Set());
    setSelectedTagIds(new Set());
    setConfidenceRange([0, 1]);
  };

  return (
    <div className="w-64 shrink-0 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 h-full overflow-y-auto flex flex-col">
      <div className="p-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <FolderTree className="w-4 h-4 text-primary-600" />分类导航
          </h3>
          <div className="flex gap-1">
            <button
              onClick={() => setShowCreateDialog(true)}
              className="p-1 rounded hover:bg-primary-50 dark:hover:bg-primary-900/20 text-primary-600"
              title="新建分类"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setFocusMode(!focusMode)}
              className={`text-xs px-2 py-0.5 rounded ${
                focusMode
                  ? "bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300"
                  : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
              }`}
            >
              {focusMode ? "专注中" : "专注"}
            </button>
          </div>
        </div>

        <div className="relative mb-2">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400" />
          <input
            type="text"
            value={categorySearch}
            onChange={(e) => setCategorySearch(e.target.value)}
            placeholder="搜索分类..."
            className="w-full pl-6 pr-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-600 bg-transparent"
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span className="flex items-center gap-1"><SlidersHorizontal className="w-3 h-3" />置信度筛选</span>
            <span>
              {confidenceRange[0].toFixed(1)} - {confidenceRange[1].toFixed(1)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidenceRange[0]}
            onChange={(e) => setConfidenceRange([parseFloat(e.target.value), confidenceRange[1]])}
            className="w-full h-1.5 accent-primary-600"
          />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidenceRange[1]}
            onChange={(e) => setConfidenceRange([confidenceRange[0], parseFloat(e.target.value)])}
            className="w-full h-1.5 accent-primary-600"
          />
        </div>
      </div>

      <div className="p-2 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setShowTags(!showTags)}
          className="w-full flex items-center justify-between text-sm font-medium py-1"
        >
          <span className="flex items-center gap-1.5">
            <Tag className="w-4 h-4 text-purple-600" />标签
          </span>
          <ChevronRight className={`w-4 h-4 transition-transform ${showTags ? "rotate-90" : ""}`} />
        </button>
        {showTags && (
          <div className="mt-1 space-y-1">
            {tags.map((tag) => (
              <div
                key={tag.id}
                className={`flex items-center gap-1.5 py-0.5 px-1 rounded cursor-pointer text-xs hover:bg-gray-100 dark:hover:bg-gray-700/50 ${
                  selectedTagIds.has(tag.id) ? "bg-purple-50 dark:bg-purple-900/20" : ""
                }`}
                onClick={() => toggleTag(tag.id)}
              >
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: tag.color }} />
                <span className="truncate flex-1">{tag.name}</span>
              </div>
            ))}
            <div className="flex items-center gap-1 pt-1">
              <input
                type="text"
                value={newTagName}
                onChange={(e) => setNewTagName(e.target.value)}
                placeholder="新建标签..."
                className="flex-1 text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-600 bg-transparent"
                onKeyDown={(e) => { if (e.key === "Enter") handleCreateTag(); }}
              />
              <input
                type="color"
                value={newTagColor}
                onChange={(e) => setNewTagColor(e.target.value)}
                className="w-5 h-5 rounded cursor-pointer border-0 p-0"
              />
              <button
                onClick={handleCreateTag}
                disabled={!newTagName.trim()}
                className="p-1 rounded text-primary-600 hover:bg-primary-50 disabled:opacity-30"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {loadingCategories ? (
          <div className="flex justify-center py-4">
            <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />
          </div>
        ) : filteredCategories.length === 0 ? (
          <div className="text-xs text-gray-400 text-center py-4 space-y-1">
            <p>{categorySearch ? "未找到匹配分类" : "暂无分类数据"}</p>
            {!categorySearch && (
              <button
                onClick={() => setShowCreateDialog(true)}
                className="text-primary-600 hover:underline"
              >
                创建第一个分类
              </button>
            )}
          </div>
        ) : (
          filteredCategories.map((node) => renderCategoryNode(node))
        )}
      </div>

      <div className="p-2 border-t border-gray-200 dark:border-gray-700 space-y-1">
        <button
          onClick={handleSyncGraph}
          disabled={syncing}
          className="w-full text-xs flex items-center gap-1.5 py-1 px-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700/50 text-gray-600 dark:text-gray-400 disabled:opacity-50"
        >
          <Share2 className="w-3.5 h-3.5" />
          {syncing ? "同步中..." : "同步到图谱"}
        </button>
        <button
          onClick={handleLoadMergeSuggestions}
          className="w-full text-xs flex items-center gap-1.5 py-1 px-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700/50 text-gray-600 dark:text-gray-400"
        >
          <GitMerge className="w-3.5 h-3.5" />
          合并建议
        </button>
        <button
          onClick={handleLoadHealth}
          className="w-full text-xs flex items-center justify-between py-1 px-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700/50 text-gray-600 dark:text-gray-400"
        >
          <span className="flex items-center gap-1.5">
            <BarChart3 className="w-3.5 h-3.5" />分类健康
          </span>
          <span>{showHealth ? "▲" : "▼"}</span>
        </button>
        {showHealth && healthData.length > 0 && (
          <div className="mt-1 space-y-1 max-h-40 overflow-y-auto">
            {healthData.map((h) => (
              <div key={h.category_id} className="text-xs flex items-center justify-between py-0.5">
                <span className="truncate flex-1">{h.name}</span>
                <span className="text-gray-400 mx-1">{h.knowledge_count}条</span>
                {h.needs_attention && <AlertTriangle className="w-3 h-3 text-orange-500" />}
                {h.is_stale && <Clock className="w-3 h-3 text-gray-400" />}
              </div>
            ))}
          </div>
        )}
        {(selectedCategoryIds.size > 0 || selectedTagIds.size > 0 || confidenceRange[0] > 0 || confidenceRange[1] < 1) && (
          <button
            onClick={clearAllFilters}
            className="w-full mt-2 text-xs py-1 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center justify-center gap-1"
          >
            <X className="w-3 h-3" />清除所有筛选
          </button>
        )}
      </div>

      {showCreateDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowCreateDialog(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 w-80 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Plus className="w-4 h-4 text-primary-600" />新建分类
            </h3>
            <input
              type="text"
              value={newCatName}
              onChange={(e) => setNewCatName(e.target.value)}
              placeholder="分类名称（如：高等数学库）"
              className="input-field mb-2"
              autoFocus
              onKeyDown={(e) => { if (e.key === "Enter") handleCreateCategory(); }}
            />
            <input
              type="text"
              value={newCatDesc}
              onChange={(e) => setNewCatDesc(e.target.value)}
              placeholder="描述（可选）"
              className="input-field mb-2"
            />
            <div className="flex items-center gap-2 mb-3">
              <label className="text-xs text-gray-500">颜色</label>
              <input
                type="color"
                value={newCatColor}
                onChange={(e) => setNewCatColor(e.target.value)}
                className="w-8 h-6 rounded cursor-pointer border-0 p-0"
              />
            </div>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setShowCreateDialog(false)} className="btn-secondary text-sm">取消</button>
              <button onClick={handleCreateCategory} disabled={!newCatName.trim()} className="btn-primary text-sm">创建</button>
            </div>
          </div>
        </div>
      )}

      {contextMenu && (
        <div
          className="fixed z-50 bg-white dark:bg-gray-800 shadow-xl rounded-lg border border-gray-200 dark:border-gray-700 py-1 w-36"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => handleRenameCategory(contextMenu.categoryId)}
            className="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
          >
            <Edit3 className="w-3 h-3" />重命名
          </button>
          <button
            onClick={() => handleDeleteCategory(contextMenu.categoryId, contextMenu.categoryName)}
            className="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2 text-red-600"
          >
            <Trash2 className="w-3 h-3" />删除
          </button>
        </div>
      )}

      {showMergeSuggestions && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowMergeSuggestions(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 w-96 max-h-[70vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold flex items-center gap-2">
                <GitMerge className="w-4 h-4 text-primary-600" />合并建议
              </h3>
              <button onClick={() => setShowMergeSuggestions(false)}><X className="w-4 h-4" /></button>
            </div>
            {mergeSuggestions.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-4">暂无合并建议，分类状态良好</p>
            ) : (
              <div className="space-y-2">
                {(mergeSuggestions as Array<Record<string, unknown>>).map((s, i) => (
                  <div key={i} className="p-2 rounded-lg bg-gray-50 dark:bg-gray-700/50 text-xs">
                    <p className="font-medium">{(s.category_a as Record<string, unknown>)?.name as string} → {(s.category_b as Record<string, unknown>)?.name as string}</p>
                    <p className="text-gray-500 mt-1">{s.suggestion as string}</p>
                    <div className="flex gap-1 mt-2">
                      <button className="px-2 py-0.5 rounded bg-primary-100 dark:bg-primary-900/30 text-primary-700 text-xs">
                        查看
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function filterTree(nodes: CategoryNode[], query: string): CategoryNode[] {
  return nodes.reduce<CategoryNode[]>((acc, node) => {
    const nameMatch = node.name.toLowerCase().includes(query);
    const descMatch = node.description?.toLowerCase().includes(query);
    const childMatches = filterTree(node.children, query);
    if (nameMatch || descMatch || childMatches.length > 0) {
      acc.push({ ...node, children: childMatches.length > 0 ? childMatches : node.children });
    }
    return acc;
  }, []);
}