"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { User, Mail, Calendar, Brain, Trash2, Loader2, LogOut, BookOpen } from "lucide-react";

interface MemoryItem {
  memory_type: string;
  memory_key: string;
  memory_value: string;
  weight: number;
  last_accessed: string;
}

interface UserMemory {
  profile: Record<string, any>;
  memory_items: MemoryItem[];
  item_count: number;
}

export default function ProfilePage() {
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth();
  const router = useRouter();
  const [memory, setMemory] = useState<UserMemory | null>(null);
  const [memLoading, setMemLoading] = useState(false);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [authLoading, isAuthenticated, router]);

  useEffect(() => {
    if (!user) return;
    const fetchMemory = async () => {
      setMemLoading(true);
      try {
        const token = localStorage.getItem("access_token");
        const res = await fetch(`/api/user/memory/${user.user_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          setMemory(await res.json());
        }
      } catch {} finally {
        setMemLoading(false);
      }
    };
    fetchMemory();
  }, [user]);

  const handleDeleteMemory = async (memoryType: string, memoryKey: string) => {
    if (!user) return;
    const token = localStorage.getItem("access_token");
    try {
      const res = await fetch(
        `/api/user/memory/${user.user_id}/item?memory_type=${encodeURIComponent(memoryType)}&memory_key=${encodeURIComponent(memoryKey)}`,
        { method: "DELETE", headers: { Authorization: `Bearer ${token}` } }
      );
      if (res.ok && memory) {
        setMemory({
          ...memory,
          memory_items: memory.memory_items.filter(
            (m) => !(m.memory_type === memoryType && m.memory_key === memoryKey)
          ),
          item_count: memory.item_count - 1,
        });
      }
    } catch {}
  };

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  if (authLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  if (!user) return null;

  const memoryTypeLabels: Record<string, string> = {
    preference: "偏好",
    correction: "纠正",
    term_preference: "术语偏好",
    topic_interest: "兴趣领域",
    style: "风格",
    knowledge_level: "知识水平",
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-14 h-14 rounded-full bg-gradient-to-br from-primary-500 to-indigo-500 flex items-center justify-center">
            <User className="w-7 h-7 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">{user.username}</h2>
            <p className="text-gray-500 dark:text-gray-400 text-sm">{user.email}</p>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Mail className="w-4 h-4" /> {user.email}
          </div>
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Calendar className="w-4 h-4" />
            {user.created_at ? new Date(user.created_at).toLocaleDateString("zh-CN") : "未知"}
          </div>
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <BookOpen className="w-4 h-4" /> ID: {user.user_id.substring(0, 8)}...
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="mt-4 flex items-center gap-2 px-4 py-2 rounded-lg text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-sm font-medium"
        >
          <LogOut className="w-4 h-4" /> 退出登录
        </button>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">AI 记住的偏好</h3>
        </div>

        {memLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
          </div>
        ) : !memory || memory.item_count === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-sm py-4">
            暂无记忆数据。与 AI 对话后，它会自动学习你的偏好和术语习惯。
          </p>
        ) : (
          <div className="space-y-2">
            {memory.memory_items.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300">
                      {memoryTypeLabels[item.memory_type] || item.memory_type}
                    </span>
                    <span className="text-xs text-gray-400">
                      权重 {item.weight.toFixed(1)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1 truncate">
                    {item.memory_value}
                  </p>
                </div>
                <button
                  onClick={() => handleDeleteMemory(item.memory_type, item.memory_key)}
                  className="ml-3 p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors shrink-0"
                  title="删除"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}