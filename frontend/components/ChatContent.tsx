"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Bot, User, Loader2, AlertTriangle, BookOpen, Sparkles, Plus, Trash2, MessageSquare, ChevronLeft, Search, Globe } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Record<string, unknown>[];
  webResults?: Record<string, unknown>[];
  conflicts?: string[];
  gaps?: string[];
  learned?: string;
  lowConfidence?: {
    count: number;
    items: Array<{ id: string; fact: string; confidence: number; source: string; source_quality: number }>;
    is_critical: boolean;
  };
  answerMetadata?: {
    retrieval_count: number;
    avg_similarity: number;
    max_similarity: number;
    is_inferred: boolean;
    is_below_threshold: boolean;
    multi_source_verified: boolean;
    conflict_count: number;
  };
}

interface ConvMeta {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

const STORAGE_KEY = "kb_current_conv";

function loadConvId(): string | null {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

function saveConvId(id: string) {
  try { localStorage.setItem(STORAGE_KEY, id); } catch {}
}

export default function ChatContent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [convId, setConvId] = useState<string>("");
  const [convs, setConvs] = useState<ConvMeta[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingConv, setLoadingConv] = useState(false);
  const [searchStatus, setSearchStatus] = useState<{ status: string; message: string } | null>(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, searchStatus]);

  useEffect(() => { loadConversationList(); }, []);
  useEffect(() => { if (convId) saveConvId(convId); }, [convId]);

  const loadConversationList = async () => {
    try {
      const list = await api.listConversations();
      setConvs(list);
      const savedId = loadConvId();
      if (savedId && list.some((c) => c.id === savedId)) {
        await switchConversation(savedId);
      } else if (list.length > 0) {
        await switchConversation(list[0].id);
      }
    } catch {
      setConvs([]);
    }
  };

  const switchConversation = async (id: string) => {
    setLoadingConv(true);
    try {
      const conv = await api.getConversation(id);
      const msgs: Message[] = (conv.messages || []).map((m: any) => ({
        id: m.id || Date.now().toString(),
        role: m.role,
        content: m.content || "",
        sources: m.sources,
        webResults: m.webResults,
        learned: m.learned,
        lowConfidence: m.lowConfidence,
      }));
      setMessages(msgs);
      setConvId(id);
    } catch {
      setMessages([]);
      setConvId(id);
    } finally {
      setLoadingConv(false);
    }
  };

  const startNewConversation = () => {
    const newId = Date.now().toString();
    setMessages([]);
    setConvId(newId);
    setConvs((prev) => [{ id: newId, title: "新对话", message_count: 0, created_at: new Date().toISOString(), updated_at: new Date().toISOString() }, ...prev]);
  };

  const saveCurrentConversation = useCallback(async (msgs: Message[]) => {
    if (!convId) return;
    const title = msgs.find((m) => m.role === "user")?.content?.slice(0, 30) || "新对话";
    const data = msgs.map((m) => ({ id: m.id, role: m.role, content: m.content, sources: m.sources, webResults: m.webResults, learned: m.learned, lowConfidence: m.lowConfidence }));
    try {
      await api.saveConversation(convId, data, title);
      setConvs((prev) => {
        const exists = prev.find((c) => c.id === convId);
        const now = new Date().toISOString();
        if (exists) {
          return prev.map((c) => c.id === convId ? { ...c, title, message_count: msgs.length, updated_at: now } : c);
        }
        return [{ id: convId, title, message_count: msgs.length, created_at: now, updated_at: now }, ...prev];
      });
    } catch {}
  }, [convId]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.deleteConversation(id);
      setConvs((prev) => prev.filter((c) => c.id !== id));
      if (id === convId) {
        setMessages([]);
        setConvId("");
      }
    } catch {}
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    if (!convId) startNewConversation();

    const userContent = input.trim();
    const userMsg: Message = { id: Date.now().toString(), role: "user", content: userContent };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    setInput("");
    setLoading(true);
    setSearchStatus(null);

    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: Message = { id: assistantId, role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      await api.chatStream(
        userContent,
        convId,
        (chunk) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m))
          );
        },
        (sources) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          );
        },
        (learnResult) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, learned: learnResult.message } : m))
          );
        },
        (status) => {
          setSearchStatus({ status: status.status, message: status.message });
          if (status.status === "search_done" || status.status === "thinking") {
            setTimeout(() => setSearchStatus(null), 2500);
          }
        },
        (webResults) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, webResults } : m))
          );
        },
        (warning) => {
          if (warning.low_confidence) {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, lowConfidence: warning.low_confidence } : m))
            );
          }
        },
        (metadata) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, answerMetadata: metadata } : m))
          );
        },
        webSearchEnabled
      );
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: "抱歉，请求失败，请检查后端服务是否启动。" } : m
        )
      );
    } finally {
      setLoading(false);
      setSearchStatus(null);
      setMessages((prev) => {
        saveCurrentConversation(prev);
        return prev;
      });
    }
  };

  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso);
      const now = new Date();
      const diff = now.getTime() - d.getTime();
      if (diff < 86400000) return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
      return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
    } catch { return ""; }
  };

  const statusIcon: Record<string, React.ReactNode> = {
    retrieving: <BookOpen className="w-3.5 h-3.5 animate-pulse" />,
    searching: <Search className="w-3.5 h-3.5 animate-spin" />,
    search_done: <Globe className="w-3.5 h-3.5 text-emerald-500" />,
    thinking: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    warning: <AlertTriangle className="w-3.5 h-3.5 text-orange-500" />,
  };

  return (
    <div className="flex gap-4 max-w-5xl mx-auto">
      <div className={`${sidebarOpen ? "w-60" : "w-0"} shrink-0 transition-all duration-200 overflow-hidden`}>
        <div className="card h-full flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-gray-500">对话历史</h2>
            <button onClick={() => setSidebarOpen(false)} className="text-gray-400 hover:text-gray-600"><ChevronLeft className="w-4 h-4" /></button>
          </div>
          <button onClick={startNewConversation} className="btn-primary text-xs w-full mb-3 flex items-center justify-center gap-1">
            <Plus className="w-3 h-3" />新对话
          </button>
          <div className="flex-1 overflow-y-auto space-y-1">
            {convs.map((c) => (
              <motion.div
                key={c.id}
                whileHover={{ x: 3, transition: { duration: 0.15 } }}
                onClick={() => switchConversation(c.id)}
                className={`flex items-center justify-between p-2 rounded-lg cursor-pointer group text-sm ${
                  c.id === convId ? "bg-primary-50 dark:bg-primary-900/20 text-primary-700" : "hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 text-gray-400" />
                  <div className="truncate">
                    <p className="truncate text-xs font-medium">{c.title}</p>
                    <p className="text-xs text-gray-400">{c.message_count} 条 · {formatDate(c.updated_at)}</p>
                  </div>
                </div>
                <button onClick={(e) => handleDelete(c.id, e)} className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 shrink-0">
                  <Trash2 className="w-3 h-3" />
                </button>
              </motion.div>
            ))}
            {convs.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-4">暂无对话历史</p>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="card mb-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold flex items-center gap-2">
                {!sidebarOpen && (
                  <button onClick={() => setSidebarOpen(true)} className="text-gray-400 hover:text-gray-600 mr-1"><MessageSquare className="w-5 h-5" /></button>
                )}
                <Bot className="w-6 h-6 text-primary-600" />
                知识库对话
                <span className="text-xs font-normal text-emerald-500 ml-2 flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />自学习
                </span>
                <span
                  onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                  className={`text-xs font-normal ml-1 flex items-center gap-1 cursor-pointer px-1.5 py-0.5 rounded transition-colors ${
                    webSearchEnabled
                      ? "text-blue-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                      : "text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 line-through"
                  }`}
                  title={webSearchEnabled ? "点击关闭联网搜索" : "点击开启联网搜索"}
                >
                  <Globe className="w-3 h-3" />{webSearchEnabled ? "联网搜索" : "仅知识库"}
                </span>
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">自动检索知识库 + 联网搜索交叉验证</p>
            </div>
            {convId && (
              <button onClick={startNewConversation} className="btn-secondary text-xs flex items-center gap-1">
                <Plus className="w-3 h-3" />新对话
              </button>
            )}
          </div>
        </div>

        <div className="card min-h-[500px] max-h-[580px] overflow-y-auto mb-4 flex flex-col">
          {loadingConv ? (
            <div className="flex-1 flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-primary-600" /></div>
          ) : messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="flex-1 flex items-center justify-center text-gray-400"
            >
              <div className="text-center">
                <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">开始与知识库对话吧</p>
                <p className="text-sm mt-1">AI 会自动检索知识库并联网搜索交叉验证</p>
                <p className="text-xs mt-2 text-emerald-500">提示：说"不对，应该是..."或"补充一下..."来教 AI 新知识</p>
              </div>
            </motion.div>
          ) : (
            <div className="space-y-4">
              <AnimatePresence>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={msg.role === "user" ? { opacity: 0, y: 20, scale: 0.95, x: 30 } : { opacity: 0, y: 20, scale: 0.95, x: -20 }}
                  animate={{ opacity: 1, y: 0, scale: 1, x: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === "user" ? "bg-primary-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}>
                    {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  <motion.div layout className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === "user" ? "bg-gradient-to-br from-primary-500 to-indigo-500 text-white" : "bg-gray-100 dark:bg-gray-800"}`}>
                    {msg.role === "user" ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="markdown-content text-sm">
                        {msg.content ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        ) : (
                          <div className="flex items-center gap-1.5 py-1">
                            <motion.span
                              className="w-2 h-2 rounded-full bg-primary-400"
                              animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                              transition={{ duration: 1.2, repeat: Infinity, delay: 0 }}
                            />
                            <motion.span
                              className="w-2 h-2 rounded-full bg-primary-400"
                              animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                              transition={{ duration: 1.2, repeat: Infinity, delay: 0.2 }}
                            />
                            <motion.span
                              className="w-2 h-2 rounded-full bg-primary-400"
                              animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                              transition={{ duration: 1.2, repeat: Infinity, delay: 0.4 }}
                            />
                          </div>
                        )}
                      </div>
                    )}

                    {msg.learned && (
                      <div className="mt-2 p-2 bg-emerald-50 dark:bg-emerald-900/20 rounded text-xs text-emerald-700 dark:text-emerald-300 flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />{msg.learned}
                      </div>
                    )}

                    {msg.lowConfidence && msg.lowConfidence.count > 0 && (
                      <div className="mt-2 p-2 bg-orange-50 dark:bg-orange-900/20 rounded text-xs">
                        <p className="font-semibold text-orange-700 dark:text-orange-300 flex items-center gap-1 mb-1">
                          <AlertTriangle className="w-3 h-3" />
                          {msg.lowConfidence.is_critical
                            ? `检测到 ${msg.lowConfidence.count} 个低置信度知识，回答已附加不确定性声明`
                            : `检索到 ${msg.lowConfidence.count} 个低置信度知识点，已提示助手谨慎对待`}
                        </p>
                        {msg.lowConfidence.items.slice(0, 3).map((item, i) => (
                          <div key={i} className="mt-1 text-orange-600 dark:text-orange-400 flex items-start gap-1">
                            <span className="shrink-0">⚠</span>
                            <span>
                              <span className="text-orange-500">[{item.confidence.toFixed(2)}]</span>{" "}
                              {item.fact.slice(0, 80)}{item.fact.length > 80 ? "..." : ""}
                              <span className="text-gray-400 ml-1">({item.source.slice(0, 30)}{item.source.length > 30 ? "..." : ""})</span>
                            </span>
                          </div>
                        ))}
                        {msg.lowConfidence.items.length > 3 && (
                          <p className="text-orange-400 mt-1">...还有 {msg.lowConfidence.items.length - 3} 个低置信度知识点</p>
                        )}
                      </div>
                    )}

                    {msg.webResults && msg.webResults.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-2 flex items-center gap-1">
                          <Globe className="w-3 h-3" />网络搜索结果
                        </p>
                        {msg.webResults.map((r: any, i: number) => (
                          <a
                            key={i}
                            href={r.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block mb-1.5 p-2 bg-blue-50 dark:bg-blue-900/10 rounded hover:bg-blue-100 dark:hover:bg-blue-900/20 transition-colors"
                          >
                            <p className="text-xs font-medium text-blue-700 dark:text-blue-300 truncate">{r.title}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2">{r.snippet}</p>
                          </a>
                        ))}
                      </div>
                    )}

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-xs font-semibold text-gray-500 mb-1 flex items-center gap-1">
                          <BookOpen className="w-3 h-3" />参考来源
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {msg.sources.map((s: any, i: number) => (
                            <span key={i} className={`px-2 py-0.5 text-xs rounded ${
                              s.category === "web_search"
                                ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                                : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
                            }`}>
                              {s.category === "web_search" ? "🌐 " : "📚 "}
                              {typeof s.source === "string" && s.source.length > 40 ? s.source.slice(0, 40) + "..." : s.source as string}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {msg.answerMetadata && (
                      <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                        <div className="flex flex-wrap gap-1.5 text-[10px]">
                          <span className="px-1.5 py-0.5 rounded bg-gray-50 dark:bg-gray-800 text-gray-500">
                            📖 {msg.answerMetadata.retrieval_count}个片段
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-gray-50 dark:bg-gray-800 text-gray-500">
                            📊 相似度 {msg.answerMetadata.avg_similarity.toFixed(2)}
                          </span>
                          {msg.answerMetadata.multi_source_verified && (
                            <span className="px-1.5 py-0.5 rounded bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400">
                              ✅ 多源验证
                            </span>
                          )}
                          {msg.answerMetadata.is_inferred && (
                            <span className="px-1.5 py-0.5 rounded bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400">
                              ⚠ 推断结论
                            </span>
                          )}
                          {msg.answerMetadata.is_below_threshold && (
                            <span className="px-1.5 py-0.5 rounded bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400">
                              ⚡ 低质量检索
                            </span>
                          )}
                          {msg.answerMetadata.conflict_count > 0 && (
                            <span className="px-1.5 py-0.5 rounded bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
                              ⚔ {msg.answerMetadata.conflict_count}个冲突
                            </span>
                          )}
                        </div>
                        <button
                          onClick={async () => {
                            try {
                              await fetch("/api/hallucination/challenge", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ query: "标记问题回答", reason: "用户质疑" }),
                              });
                              alert("已记录您的质疑，感谢反馈！");
                            } catch { alert("反馈失败"); }
                          }}
                          className="mt-1 text-[10px] text-gray-400 hover:text-red-500 transition-colors"
                        >
                          🚩 此回答有问题？点击标记
                        </button>
                      </div>
                    )}

                    {msg.conflicts && msg.conflicts.length > 0 && (
                      <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded text-xs text-yellow-700 dark:text-yellow-300">
                        <AlertTriangle className="w-3 h-3 inline mr-1" />
                        检测到矛盾: {msg.conflicts.join("; ")}
                      </div>
                    )}

                    {msg.gaps && msg.gaps.length > 0 && (
                      <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs text-blue-700 dark:text-blue-300">
                        {msg.gaps.join("; ")}
                      </div>
                    )}
                  </motion.div>
                </motion.div>
              ))}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>
          )}

          {searchStatus && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 border-t border-blue-100 dark:border-blue-900/30"
            >
              <p className="text-xs text-blue-700 dark:text-blue-300 flex items-center gap-2">
                {statusIcon[searchStatus.status] || <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                {searchStatus.message}
              </p>
            </motion.div>
          )}
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
            placeholder={webSearchEnabled ? "输入您的问题... AI将自动检索知识库并联网搜索 (Enter 发送)" : "输入您的问题... AI将仅基于知识库内容回答 (Enter 发送)"}
            className="input-field flex-1"
            disabled={loading}
          />
          <motion.button
            whileTap={{ scale: 0.93 }}
            whileHover={{ scale: 1.03 }}
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="btn-primary px-6 flex items-center gap-2 disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            发送
          </motion.button>
        </div>
      </div>
    </div>
  );
}