"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  Upload, MessageSquare, Search, GitGraph, Clock, ArrowRight,
  Sparkles, FileText, Image, Video, FileCode, Globe,
} from "lucide-react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";

const supportedFormats = [
  { icon: FileText, label: "PDF", color: "text-red-500" },
  { icon: FileText, label: "Word", color: "text-blue-500" },
  { icon: FileText, label: "PPT", color: "text-orange-500" },
  { icon: Image, label: "图片", color: "text-green-500" },
  { icon: Globe, label: "网页", color: "text-purple-500" },
  { icon: FileCode, label: "代码", color: "text-gray-500" },
  { icon: Video, label: "视频", color: "text-pink-500" },
  { icon: Video, label: "音频", color: "text-indigo-500" },
];

const features = [
  { icon: Upload, title: "全格式摄入", desc: "支持 PDF、Word、PPT、图片、网页、代码、视频、音频等所有主流格式文件", href: "/upload" },
  { icon: MessageSquare, title: "智能对话", desc: "基于知识库的深度对话，支持多跳推理、溯源引用和主动提问", href: "/chat" },
  { icon: GitGraph, title: "知识图谱", desc: "自动构建实体关系网络，可视化探索知识间的隐藏关联", href: "/graph" },
  { icon: Search, title: "语义搜索", desc: "支持自然语言的全文语义搜索，精确查找知识库中的内容", href: "/wiki" },
];

export default function HomeContent() {
  const [stats, setStats] = useState({ vector_count: 0, node_count: 0 });
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    api.getStats().then(setStats).catch(() => {});
  }, []);

  return (
    <motion.div
      className="max-w-5xl mx-auto space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-16 pb-12">
      <section className="text-center pt-12 pb-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 text-sm mb-6">
          <Sparkles className="w-4 h-4" />
          基于 DeepSeek AI 的全格式自进化知识库
        </div>
        <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
          全格式自进化知识库智能体
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
          上传任何格式的文件，AI 自动提取知识点，构建知识图谱，通过对话探索您的私人知识库
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center mb-8">
          <Link href="/upload" prefetch={false} className="btn-primary inline-flex items-center gap-2 text-lg px-6 py-3">
            <Upload className="w-5 h-5" />
            上传文件开始
            <ArrowRight className="w-5 h-5" />
          </Link>
          <Link href="/chat" prefetch={false} className="btn-secondary inline-flex items-center gap-2 text-lg px-6 py-3">
            <MessageSquare className="w-5 h-5" />
            开始对话
          </Link>
        </div>

        <div className="max-w-xl mx-auto">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="搜索知识库..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && searchQuery.trim()) {
                  window.location.href = `/wiki?q=${encodeURIComponent(searchQuery)}`;
                }
              }}
              className="input-field pl-10 pr-4 py-3 text-lg"
            />
          </div>
        </div>

        <div className="flex justify-center gap-8 mt-6 text-sm text-gray-500 dark:text-gray-400">
          <span>📚 <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}>{stats.vector_count}</motion.span> 个知识点</span>
          <span>🔗 <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}>{stats.node_count}</motion.span> 个图谱节点</span>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-center mb-8">核心功能</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {features.map((feat, i) => (
            <motion.div
              key={feat.title}
              initial={{ opacity: 0, y: 30, scale: 0.95 }}
              animate={{
                opacity: 1,
                y: 0,
                scale: 1,
                transition: { type: "spring", stiffness: 200, damping: 20, delay: i * 0.1 },
              }}
            >
              <Link
                href={feat.href}
                prefetch={false}
                className="card flex flex-col items-center text-center h-full gap-3 group transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-primary-500/10"
              >
                <div className="w-12 h-12 rounded-xl bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <feat.icon className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                </div>
                <h3 className="font-semibold text-base">{feat.title}</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{feat.desc}</p>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-center mb-8">支持的格式</h2>
        <div className="flex flex-wrap justify-center gap-4">
          {supportedFormats.map((fmt) => (
            <div key={fmt.label} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-800">
              <fmt.icon className={`w-5 h-5 ${fmt.color}`} />
              <span className="font-medium">{fmt.label}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
    </motion.div>
  );
}