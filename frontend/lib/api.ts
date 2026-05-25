const API_BASE = "/api";

import { cachedFetch, invalidateCache } from "./dataCache";

const TTL = {
  highFreq: 30_000,
  normal: 60_000,
  lowFreq: 120_000,
  veryLow: 300_000,
};

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    let msg = text || `HTTP ${res.status}`;
    try {
      const data = JSON.parse(text);
      msg = data.detail || data.message || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export interface KnowledgePoint {
  id?: string;
  fact: string;
  category: string;
  confidence: number;
  related_entities: string[];
  source: string;
  source_document_id?: string;
  created_at?: string;
  calibrated_confidence?: number;
  source_quality?: number;
  consistency_score?: number;
  feedback_alpha?: number;
  feedback_beta?: number;
  status?: string;
  replaced_by?: string;
  history?: Array<{ action: string; timestamp: string; [key: string]: unknown }>;
}

export interface DocumentChunk {
  id?: string;
  content: string;
  source_path: string;
  source_type: string;
  chunk_index: number;
  confidence: number;
}

export interface SearchResult {
  knowledge_points: KnowledgePoint[];
  document_chunks: DocumentChunk[];
  graph_results: Record<string, unknown>[];
}

export interface ConversationResponse {
  answer: string;
  conversation_id: string;
  sources: Record<string, unknown>[];
  related_questions: string[];
  detected_conflicts: string[];
  knowledge_gaps: string[];
  low_confidence_info?: {
    count: number;
    items: Array<{
      id: string;
      fact: string;
      confidence: number;
      source: string;
      source_quality: number;
    }>;
    is_critical: boolean;
  };
}

export interface GraphData {
  nodes: { id: string; label: string; degree?: number; type?: string; community_id?: number }[];
  edges: {
    source: string; target: string; relation: string;
    relation_type?: string; confidence?: number;
    source_knowledge_id?: string; evidence_snippet?: string;
  }[];
  node_count?: number;
  edge_count?: number;
}

export interface MultiHopPathData {
  paths: string[][];
  relations: string[][];
  path_weights: number[];
  source_entity: string;
  target_entity: string;
  max_hops: number;
}

export interface CommunityData {
  community_count: number;
  modularity?: number;
  communities: Record<number, string[]>;
  community_labels: Record<number, string>;
  community_sizes: Record<number, number>;
}

export interface GraphStats {
  node_count: number;
  edge_count: number;
  node_types: Record<string, number>;
}

export interface IngestionTask {
  task_id: string;
  file_path?: string;
  file_type?: string;
  status: string;
  progress?: number;
  result?: { chunks?: Array<{ content: string; chunk_index: number; source_path: string }> } | null;
  error?: string;
}

export interface ModelSettings {
  deepseek_enabled: boolean;
  deepseek_model: string;
  deepseek_api_key: string;
  deepseek_base_url: string;
  vision_enabled: boolean;
  vision_model: string;
  vision_api_key: string;
  vision_base_url: string;
  speech_model: string;
  speech_api_key: string;
  speech_base_url: string;
  speech_enabled: boolean;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  ingestFile: async (file: File, fileType: string): Promise<IngestionTask> => {
    const formData = new FormData();
    formData.append("file", file);
    const isMedia = fileType === "video" || fileType === "audio";
    const timeoutMs = isMedia ? 600000 : 120000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${API_BASE}/ingest/file?file_type=${fileType}`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      if (!res.ok) {
        const errorText = await res.text();
        let msg = errorText || "上传失败";
        try {
          const data = JSON.parse(errorText);
          msg = data.detail || data.message || msg;
        } catch {}
        throw new Error(msg);
      }
      invalidateCache();
      return res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  },

  ingestUrl: async (url: string): Promise<IngestionTask> => {
    const result = await request<IngestionTask>("/ingest/url", {
      method: "POST",
      body: JSON.stringify({ url }),
    });
    invalidateCache();
    return result;
  },

  ingestText: async (content: string, sourceName?: string): Promise<IngestionTask> => {
    const result = await request<IngestionTask>("/ingest/text", {
      method: "POST",
      body: JSON.stringify({ content, source_name: sourceName || "手动输入" }),
    });
    invalidateCache();
    return result;
  },

  chat: (message: string, conversationId?: string) =>
    request<ConversationResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, conversation_id: conversationId, stream: false }),
    }),

  chatStream: async (
    message: string,
    conversationId: string | undefined,
    onChunk: (chunk: string) => void,
    onSources: (sources: Record<string, unknown>[]) => void,
    onLearned: (learnResult: { learned: number; triples: number; message: string }) => void,
    onStatus?: (status: { status: string; message: string }) => void,
    onWebResults?: (results: Record<string, unknown>[]) => void,
    onWarning?: (warning: { message: string; low_confidence: { count: number; items: Array<{ id: string; fact: string; confidence: number; source: string; source_quality: number }>; is_critical: boolean } }) => void,
    onMetadata?: (metadata: Record<string, unknown>) => void,
    enableWebSearch: boolean = true
  ) => {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: conversationId, stream: true, enable_web_search: enableWebSearch }),
    });
    if (!res.ok) {
      const errorText = await res.text();
      let msg = errorText || "对话失败";
      try {
        const data = JSON.parse(errorText);
        msg = data.detail || data.message || msg;
      } catch {}
      throw new Error(msg);
    }
    const reader = res.body?.getReader();
    if (!reader) throw new Error("No stream reader");
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) onChunk(parsed.content);
            if (parsed.sources) onSources(parsed.sources);
            if (parsed.learned) onLearned(parsed.learned);
            if (parsed.type === "status" && onStatus) onStatus(parsed);
            if (parsed.web_results && onWebResults) onWebResults(parsed.web_results);
            if (parsed.type === "warning" && onWarning) onWarning(parsed);
            if (parsed.metadata && onMetadata) onMetadata(parsed.metadata);
          } catch {}
        }
      }
    }
  },

  search: (query: string, topK = 10) =>
    cachedFetch(
      `sr:${query}:${topK}`,
      () => request<SearchResult>("/search", {
        method: "POST",
        body: JSON.stringify({ query, top_k: topK, search_type: "hybrid" }),
      }),
      TTL.normal
    ),

  getKnowledge: (id: string) =>
    request<KnowledgePoint>(`/knowledge/${id}`),

  deleteKnowledge: (id: string) => {
    invalidateCache(["kn", "kl", "ke", "st", "gs", "ge", "lc", "sr"]);
    return request<{ status: string }>(`/knowledge/${id}`, { method: "DELETE" });
  },

  deleteKnowledgeBatch: (ids: string[]) => {
    invalidateCache(["kn", "kl", "ke", "st", "gs", "ge", "lc", "sr"]);
    return request<{ deleted: number; failed: number }>("/knowledge/batch", {
      method: "DELETE",
      body: JSON.stringify({ ids }),
    });
  },

  resetKnowledgeAll: () => {
    invalidateCache();
    return request<{ status: string; vector_deleted: number; graph_nodes_deleted: number; upload_files_deleted: number }>("/knowledge/all", {
      method: "DELETE",
    });
  },

  listKnowledge: (params: {
    page?: number;
    page_size?: number;
    mode?: string;
    cursor?: string;
    direction?: string;
    sort_by?: string;
    order?: string;
    category_id?: string;
    tag?: string;
    confidence_min?: number;
    confidence_max?: number;
    status?: string;
    search?: string;
  } = {}) => {
    const qs = new URLSearchParams();
    if (params.page) qs.set("page", String(params.page));
    if (params.page_size) qs.set("page_size", String(params.page_size));
    if (params.mode) qs.set("mode", params.mode);
    if (params.cursor) qs.set("cursor", params.cursor);
    if (params.direction) qs.set("direction", params.direction);
    if (params.sort_by) qs.set("sort_by", params.sort_by);
    if (params.order) qs.set("order", params.order);
    if (params.category_id) qs.set("category_id", params.category_id);
    if (params.tag) qs.set("tag", params.tag);
    if (params.confidence_min !== undefined) qs.set("confidence_min", String(params.confidence_min));
    if (params.confidence_max !== undefined) qs.set("confidence_max", String(params.confidence_max));
    if (params.status) qs.set("status", params.status);
    if (params.search) qs.set("search", params.search);
    const q = qs.toString();
    const cacheKey = `kl:${q || "all"}`;
    return cachedFetch(
      cacheKey,
      () => request<{
        data: KnowledgePoint[];
        pagination: {
          mode: string;
          page: number;
          page_size: number;
          total: number;
          total_pages: number;
          has_next: boolean;
          has_prev: boolean;
          next_cursor?: string;
          prev_cursor?: string;
        };
      }>(`/knowledge/list${q ? `?${q}` : ""}`),
      TTL.highFreq
    );
  },

  exploreGraph: (entity = "", limit = 50, hops = 1) =>
    cachedFetch(
      `ge:${entity}:${limit}:${hops}`,
      () => request<GraphData>(`/graph/explore?entity=${encodeURIComponent(entity)}&limit=${limit}&hops=${hops}`),
      TTL.lowFreq
    ),

  findPaths: (source: string, target: string, maxHops = 4) =>
    request<MultiHopPathData>(`/graph/paths?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}&max_hops=${maxHops}`),

  detectConflicts: (entity = "", fact = "") =>
    request<{ conflicts: Array<Record<string, unknown>>; contradiction_cycles: string[][]; count: number }>(
      `/graph/conflicts?entity=${encodeURIComponent(entity)}&fact=${encodeURIComponent(fact)}`
    ),

  detectCommunities: () =>
    request<CommunityData>("/graph/communities", { method: "POST" }),

  normalizeEntity: (entityName: string, entityType = "Entity", forceMerge = false) =>
    request<Record<string, unknown>>("/graph/normalize", {
      method: "POST",
      body: JSON.stringify({ entity_name: entityName, entity_type: entityType, force_merge: forceMerge }),
    }),

  executeCypher: (query: string, params?: Record<string, unknown>) =>
    request<{ results: Array<Record<string, unknown>>; count: number }>("/graph/cypher", {
      method: "POST",
      body: JSON.stringify({ query, params }),
    }),

  applyInference: () =>
    request<{ status: string; rules_applied: number; total_inferred: number; rules: Array<Record<string, unknown>> }>(
      "/graph/inference", { method: "POST" }
    ),

  getGraphStats: () =>
    cachedFetch(
      "gs",
      () => request<GraphStats>("/graph/stats"),
      TTL.normal
    ),

  getEntityDetail: (entityName: string) =>
    cachedFetch(
      `ge:${entityName}`,
      () => request<Record<string, unknown>>(`/graph/entity/${encodeURIComponent(entityName)}`),
      TTL.lowFreq
    ),

  scanFusion: (threshold = 0.95) =>
    request<{ similar_pairs: Array<Record<string, unknown>>; count: number }>(
      `/graph/fusion/scan?threshold=${threshold}`, { method: "POST" }
    ),

  syncGraph: (enableNormalization = true, enableEvidenceChain = true) => {
    invalidateCache(["ge", "gs", "gp"]);
    return request<Record<string, unknown>>("/graph/sync", {
      method: "POST",
      body: JSON.stringify({ enable_normalization: enableNormalization, enable_evidence_chain: enableEvidenceChain }),
    });
  },

  getStats: () =>
    cachedFetch(
      "stats",
      () => request<{ vector_count: number; node_count: number }>("/stats"),
      TTL.normal
    ),

  listConversations: () =>
    cachedFetch(
      "convs",
      () => request<{ id: string; title: string; message_count: number; created_at: string; updated_at: string }[]>("/conversations"),
      TTL.normal
    ),

  getConversation: (id: string) =>
    request<{ title: string; messages: Record<string, unknown>[]; created_at: string }>(`/conversations/${id}`),

  saveConversation: (id: string, messages: Record<string, unknown>[], title?: string) => {
    invalidateCache("convs");
    return request<{ status: string }>("/conversations", {
      method: "POST",
      body: JSON.stringify({ id, messages, title }),
    });
  },

  deleteConversation: (id: string) => {
    invalidateCache("convs");
    return request<{ status: string }>(`/conversations/${id}`, { method: "DELETE" });
  },

  confirmKnowledge: (id: string) => {
    invalidateCache(["kn", "kl", "ke", "st", "gs", "ge", "lc", "sr"]);
    return request<{ status: string; id: string; confidence: number; message: string }>(
      `/knowledge/${id}/confirm`,
      { method: "POST" }
    );
  },

  correctKnowledge: (id: string, fact: string, category?: string, source?: string) => {
    invalidateCache(["kn", "kl", "ke", "st", "gs", "ge", "lc", "sr"]);
    return request<{ status: string; old_id: string; new_id: string; old_confidence: number; new_confidence: number; message: string }>(
      `/knowledge/${id}/correct`,
      {
        method: "POST",
        body: JSON.stringify({ fact, category, source }),
      }
    );
  },

  markErrorKnowledge: (id: string) => {
    invalidateCache(["kn", "kl", "ke", "st", "gs", "ge", "lc", "sr"]);
    return request<{ status: string; id: string; confidence: number; message: string }>(
      `/knowledge/${id}/mark-error`,
      { method: "POST" }
    );
  },

  getKnowledgeEvidence: (id: string, limit = 5) =>
    cachedFetch(
      `ke:${id}:${limit}`,
      () => request<{
        knowledge_id: string;
        fact: string;
        confidence: number;
        supporting: Array<{ id: string; fact: string; confidence: number; category: string; source: string }>;
        contradicting: Array<{ id: string; fact: string; confidence: number; category: string; source: string }>;
        supporting_count: number;
        contradicting_count: number;
      }>(`/knowledge/${id}/evidence?limit=${limit}`),
      TTL.lowFreq
    ),

  autoReview: (threshold = 0.4, limit = 10, enableExternalSearch = false) => {
    invalidateCache(["kn", "kl", "ke", "st", "lc"]);
    return request<{
      status: string;
      total_low_confidence: number;
      reviewed: number;
      improved: number;
      improved_items: Array<{ id: string; fact: string; old_confidence: number; new_confidence: number }>;
      unchanged: number;
      top_conflicts: Array<{ id: string; fact: string; confidence: number }>;
      search_evidences: Array<{ knowledge_id: string; fact: string; search_results: Array<{ title: string; snippet: string; url: string }> }>;
      message: string;
      propagated_entities?: number;
    }>("/confidence/auto-review", {
      method: "POST",
      body: JSON.stringify({ threshold, limit, enable_external_search: enableExternalSearch }),
    });
  },

  listLowConfidence: (threshold = 0.5, limit = 20) =>
    cachedFetch(
      `lc:${threshold}:${limit}`,
      () => request<{ count: number; threshold: number; items: Array<{ id: string; fact: string; confidence: number; category: string; source_quality: number }> }>(
        `/confidence/low?threshold=${threshold}&limit=${limit}`
      ),
      TTL.normal
    ),

  categories: {
    list: (categoryType?: string, includeArchived = false) =>
      cachedFetch(
        `cl:${categoryType || ""}:${includeArchived}`,
        () => request<{ categories: Array<{
          id: string; name: string; description?: string; parent_id?: string;
          category_type: string; level: number; knowledge_count: number;
          avg_confidence: number; is_archived: boolean; is_frozen: boolean;
          metadata: Record<string, unknown>;
        }>; total: number }>(
          `/categories?category_type=${categoryType || ""}&include_archived=${includeArchived}`
        ),
        TTL.lowFreq
      ),

    getTree: (userId = "default") =>
      cachedFetch(
        `ct:${userId}`,
        () => request<{ tree: Array<Record<string, unknown>> }>(`/categories/tree?user_id=${userId}`),
        TTL.lowFreq
      ),

    getPersonalizedTree: (userId = "default", focusMode = false) =>
      request<{ tree: Array<Record<string, unknown>> }>(
        `/categories/tree/personalized?user_id=${userId}&focus_mode=${focusMode}`
      ),

    search: (query: string) =>
      request<{ categories: Array<Record<string, unknown>> }>(`/categories/search?q=${encodeURIComponent(query)}`),

    getPath: (id: string) =>
      request<{ path: Array<Record<string, unknown>> }>(`/categories/${id}/path`),

    getMergeSuggestions: () =>
      request<{ suggestions: Array<Record<string, unknown>>; count: number }>("/categories/merge-suggestions"),

    syncGraph: () => {
      invalidateCache(["ge", "gs", "gp"]);
      return request<{ synced: number; status: string }>("/categories/sync-graph", { method: "POST" });
    },

    get: (id: string) =>
      request<Record<string, unknown>>(`/categories/${id}`),

    create: (data: Record<string, unknown>) => {
      invalidateCache(["cl", "ct", "ch", "csg", "tl"]);
      return request<Record<string, unknown>>("/categories", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },

    update: (id: string, updates: Record<string, unknown>) => {
      invalidateCache(["cl", "ct", "ch", "csg", "tl"]);
      return request<Record<string, unknown>>(`/categories/${id}`, {
        method: "PUT",
        body: JSON.stringify(updates),
      });
    },

    delete: (id: string) => {
      invalidateCache(["cl", "ct", "ch", "csg", "tl"]);
      return request<{ status: string }>(`/categories/${id}`, { method: "DELETE" });
    },

    recordVisit: (categoryId: string, userId = "default") =>
      request<{ status: string }>(`/categories/${categoryId}/visit?user_id=${userId}`, { method: "POST" }),

    getHealth: () =>
      cachedFetch(
        "ch",
        () => request<{ health: Array<{
          category_id: string; name: string; knowledge_count: number;
          avg_confidence: number; last_updated?: string; is_stale: boolean;
          needs_split: boolean; needs_attention: boolean;
        }> }>("/categories/health"),
        TTL.normal
      ),

    suggest: (fact: string) =>
      request<{ suggestions: Array<{
        category_id?: string; category_name: string; confidence: number;
        reason: string; is_new: boolean;
      }> }>(`/categories/suggest?fact=${encodeURIComponent(fact)}`),

    runClustering: () =>
      request<Record<string, unknown>>("/categories/cluster", { method: "POST" }),

    checkEvolution: () =>
      request<{ events: Array<Record<string, unknown>>; count: number }>(
        "/categories/evolution/check", { method: "POST" }
      ),

    executeEvolution: (event: Record<string, unknown>) =>
      request<{ status: string; result?: Record<string, unknown> }>(
        "/categories/evolution/execute", { method: "POST", body: JSON.stringify(event) }
      ),

    getTimeline: (params?: { mode?: string; granularity?: string; category_id?: string }) => {
      const qs = new URLSearchParams();
      if (params?.mode) qs.set("mode", params.mode);
      if (params?.granularity) qs.set("granularity", params.granularity);
      if (params?.category_id) qs.set("category_id", params.category_id);
      const q = qs.toString();
      return cachedFetch(
        `tl:${q || "all"}`,
        () => request<{
          groups: Array<{
            time_key: string; label: string; record_count: number; event_count: number;
            items: Array<Record<string, unknown>>; is_gap: boolean; is_burst: boolean;
            confidence_avg: number;
          }>;
          total: number; mode: string; granularity: string;
          gaps: Array<{ start_date: string; end_date: string; duration_days: number; label: string; suggestion: string }>;
          bursts: Array<{ center_date: string; density_multiplier: number; knowledge_count: number; top_categories: string[]; label: string }>;
          version_chains: Array<{ entity_name: string; versions: Array<Record<string, unknown>>; latest_version?: string; total_updates: number }>;
        }>(`/categories/timeline?${qs.toString()}`),
        TTL.normal
      );
    },

    extractTimelineTimes: (batchSize = 20) =>
      request<{ updated: number; total_checked: number }>(`/categories/timeline/extract?batch_size=${batchSize}`),

    getSourceGroups: () =>
      cachedFetch(
        "csg",
        () => request<{ sources: Array<{
          source_name: string; source_type: string; knowledge_count: number;
          avg_confidence: number; knowledge_ids: string[];
          first_added?: string; last_added?: string;
        }> }>("/categories/sources"),
        TTL.lowFreq
      ),

    getSourceComparisons: () =>
      request<{ comparisons: Array<{
        topic: string; source_a: Record<string, unknown>; source_b: Record<string, unknown>;
        conflicting_points: Array<Record<string, unknown>>;
        agreement_points: Array<Record<string, unknown>>;
      }> }>("/categories/sources/compare"),
  },

  tags: {
    list: (userId = "default") =>
      request<{ tags: Array<{
        id: string; user_id: string; name: string; color: string; dimension: string;
      }> }>(`/tags?user_id=${userId}`),

    create: (data: { user_id: string; name: string; color?: string; dimension?: string }) => {
      invalidateCache("tg");
      return request<Record<string, unknown>>("/tags", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },

    delete: (id: string) => {
      invalidateCache("tg");
      return request<{ status: string }>(`/tags/${id}`, { method: "DELETE" });
    },

    assign: (knowledgeId: string, tagId: string, userId: string) =>
      request<Record<string, unknown>>(`/knowledge/${knowledgeId}/tags`, {
        method: "POST",
        body: JSON.stringify({ knowledge_id: knowledgeId, tag_id: tagId, user_id: userId }),
      }),

    remove: (knowledgeId: string, tagId: string) =>
      request<{ status: string }>(`/knowledge/${knowledgeId}/tags/${tagId}`, { method: "DELETE" }),

    getForKnowledge: (knowledgeId: string) =>
      request<{ tags: Array<{ id: string; name: string; color: string }> }>(
        `/knowledge/${knowledgeId}/tags`
      ),
  },

  smartCollections: {
    list: (userId?: string) =>
      request<{ collections: Array<Record<string, unknown>> }>(
        `/smart-collections${userId ? `?user_id=${userId}` : ""}`
      ),

    create: (data: Record<string, unknown>) =>
      request<Record<string, unknown>>("/smart-collections", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    evaluate: (collectionId: string) =>
      request<{ collection_id: string; match_count: number; knowledge_ids: string[] }>(
        `/smart-collections/${collectionId}/evaluate`, { method: "POST" }
      ),
  },

  filter: (flt: Record<string, unknown>) =>
    request<{ items: KnowledgePoint[]; total: number; offset: number; limit: number }>(
      "/knowledge/filter", { method: "POST", body: JSON.stringify(flt) }
    ),

  assignKnowledgeCategories: (knowledgeId: string, categoryIds: string[], primaryCategoryId?: string, isAuto = false) => {
    invalidateCache(["kn", "kl", "ke", "cl", "ct"]);
    return request<{ assignments: Array<Record<string, unknown>>; count: number }>(
      `/knowledge/${knowledgeId}/categories`, {
        method: "POST",
        body: JSON.stringify({ category_ids: categoryIds, primary_category_id: primaryCategoryId, is_auto: isAuto }),
      }
    );
  },

  batchAssignCategories: (knowledgeIds: string[], categoryIds: string[], isAuto = false) => {
    invalidateCache(["kn", "kl", "ke", "cl", "ct"]);
    return request<{ assigned: number; knowledge_count: number }>("/knowledge/batch/categories", {
      method: "POST",
      body: JSON.stringify({ knowledge_ids: knowledgeIds, category_ids: categoryIds, is_auto: isAuto }),
    });
  },

  autoCategorizeKnowledge: (knowledgeId: string, confidenceThreshold = 0.7, autoCreate = false) => {
    invalidateCache(["kn", "kl", "ke", "cl", "ct"]);
    return request<{ knowledge_id: string; assigned_categories: string[]; suggested_new?: string; confidence: number }>(
      `/knowledge/${knowledgeId}/auto-categorize`, {
        method: "POST",
        body: JSON.stringify({ confidence_threshold: confidenceThreshold, auto_create: autoCreate }),
      }
    );
  },

  removeKnowledgeCategory: (knowledgeId: string, categoryId: string) => {
    invalidateCache(["kn", "kl", "ke", "cl", "ct"]);
    return request<{ status: string }>(`/knowledge/${knowledgeId}/categories/${categoryId}`, { method: "DELETE" });
  },

  getKnowledgeCategories: (knowledgeId: string) =>
    request<{ categories: Array<Record<string, unknown>> }>(`/knowledge/${knowledgeId}/categories`),

  batchGetKnowledgeCategories: (knowledgeIds: string[]) =>
    request<{ categories: Record<string, Array<Record<string, unknown>>> }>("/knowledge/categories/batch", {
      method: "POST",
      body: JSON.stringify({ knowledge_ids: knowledgeIds }),
    }),

  getPreferences: (userId = "default") =>
    request<{ preferences: Record<string, Record<string, unknown>> }>(`/categories/preferences?user_id=${userId}`),

  setCategoryPreferences: (categoryId: string, prefs: Record<string, unknown>, userId = "default") =>
    request<{ status: string }>(`/categories/${categoryId}/preferences?user_id=${userId}`, {
      method: "PUT",
      body: JSON.stringify(prefs),
    }),

  getModelSettings: () =>
    cachedFetch(
      "ms",
      () => request<ModelSettings>("/settings/models"),
      TTL.veryLow
    ),

  updateModelSettings: (data: ModelSettings) => {
    invalidateCache("ms");
    return request<{ status: string; message: string; needs_restart: boolean }>("/settings/models", {
      method: "PUT",
      body: JSON.stringify(data),
    });
  },
};