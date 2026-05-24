"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, Globe, CheckCircle, XCircle, Loader2, Link } from "lucide-react";
import { api } from "@/lib/api";

const fileTypes = [
  { value: "auto", label: "自动检测（推荐）" },
  { value: "pdf", label: "PDF 文档" }, { value: "word", label: "Word 文档" },
  { value: "ppt", label: "PPT 演示" }, { value: "image", label: "图片" },
  { value: "web", label: "网页文件" }, { value: "table", label: "表格" },
  { value: "code", label: "代码" }, { value: "text", label: "纯文本" },
  { value: "video", label: "视频" }, { value: "audio", label: "音频" },
];

const extTypeMap: Record<string, string> = {
  ".pdf": "pdf",
  ".doc": "word", ".docx": "word",
  ".ppt": "ppt", ".pptx": "ppt",
  ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image", ".tiff": "image",
  ".html": "web", ".htm": "code",
  ".csv": "table", ".xlsx": "table", ".xls": "table",
  ".py": "code", ".js": "code", ".ts": "code", ".tsx": "code", ".jsx": "code", ".java": "code", ".cpp": "code", ".c": "code",
  ".go": "code", ".rs": "code", ".rb": "code", ".php": "code", ".cs": "code", ".swift": "code", ".kt": "code",
  ".sql": "code", ".sh": "code", ".css": "code", ".json": "code", ".xml": "code", ".yaml": "code",
  ".txt": "text", ".md": "text", ".log": "text",
  ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video", ".webm": "video", ".flv": "video", ".wmv": "video",
  ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio", ".ogg": "audio", ".m4a": "audio", ".wma": "audio",
};

function detectFileType(fileName: string): string {
  const ext = fileName.toLowerCase().slice(fileName.lastIndexOf("."));
  return extTypeMap[ext] || "text";
}

interface UploadTask { id: string; fileName: string; status: "uploading" | "processing" | "completed" | "failed"; error?: string; chunks?: number; extracted?: boolean; }

export default function UploadContent() {
  const [fileType, setFileType] = useState("auto");
  const [tasks, setTasks] = useState<UploadTask[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [urlLoading, setUrlLoading] = useState(false);

  const addTask = (fileName: string): string => {
    const taskId = Date.now().toString() + Math.random().toString(36).slice(2);
    setTasks((prev) => [{ id: taskId, fileName, status: "uploading" }, ...prev]);
    return taskId;
  };
  const updateTask = (taskId: string, updates: Partial<UploadTask>) => { setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, ...updates } : t))); };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const taskId = addTask(file.name);
      const detectedType = fileType === "auto" ? detectFileType(file.name) : fileType;
      try {
        const result = await api.ingestFile(file, detectedType);
        const chunks = result.result?.chunks || [];
        const hasContent = chunks.length > 0 && !chunks[0]?.content?.startsWith("[未提取到语音内容]");
        updateTask(taskId, {
          status: result.status === "completed" ? "completed" : "failed",
          error: result.error,
          chunks: chunks.length,
          extracted: hasContent,
        });
      } catch (err: unknown) { updateTask(taskId, { status: "failed", error: err instanceof Error ? err.message : "上传失败" }); }
    }
  }, [fileType]);

  const ingestUrl = async () => {
    const url = urlInput.trim(); if (!url || urlLoading) return;
    setUrlLoading(true); const taskId = addTask(url);
    try {
      const result = await api.ingestUrl(url);
      updateTask(taskId, { status: result.status === "completed" ? "completed" : "failed", error: result.error, chunks: result.result?.chunks?.length });
      if (result.status === "completed") setUrlInput("");
    } catch (err: unknown) { updateTask(taskId, { status: "failed", error: err instanceof Error ? err.message : "抓取失败" }); }
    finally { setUrlLoading(false); }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="card">
        <h1 className="text-xl font-bold flex items-center gap-2 mb-2"><Upload className="w-6 h-6 text-primary-600" />上传文件</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">上传文件后 AI 将自动解析内容、提取知识点、构建知识图谱</p>
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">文件类型</label>
          <select value={fileType} onChange={(e) => setFileType(e.target.value)} className="input-field">
            {fileTypes.map((ft) => (<option key={ft.value} value={ft.value}>{ft.label}</option>))}
          </select>
          <p className="text-xs text-gray-400 mt-1">{fileType === "auto" ? "将根据文件扩展名自动识别类型，无需手动选择" : "已强制指定文件类型，覆盖自动检测"}</p>
        </div>
        <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${isDragActive ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20" : "border-gray-300 dark:border-gray-600 hover:border-primary-400"}`}>
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-lg font-medium mb-2">{isDragActive ? "释放文件以上传" : "拖拽文件到此处，或点击选择"}</p>
          <p className="text-sm text-gray-500">支持 PDF、Word、PPT、图片、网页、代码、视频、音频等全格式</p>
        </div>
      </div>
      <div className="card">
        <h2 className="text-lg font-bold flex items-center gap-2 mb-4"><Globe className="w-5 h-5 text-primary-600" />抓取网页</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">直接输入网址，系统将自动抓取网页内容并提取知识点</p>
        <div className="flex gap-2">
          <div className="relative flex-1"><Link className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" /><input type="url" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ingestUrl()} placeholder="输入网页地址，如 https://example.com/article" className="input-field pl-9" disabled={urlLoading} /></div>
          <button onClick={ingestUrl} disabled={urlLoading || !urlInput.trim()} className="btn-primary px-6 flex items-center gap-2 disabled:opacity-50">{urlLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}抓取</button>
        </div>
      </div>
      {tasks.length > 0 && (
        <div className="card"><h2 className="font-semibold mb-4">上传记录</h2>
          <div className="space-y-3">
            {tasks.map((task) => (
              <div key={task.id} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                <FileText className="w-5 h-5 text-gray-400 shrink-0" />
                <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">{task.fileName}</p>
                    {task.error && <p className="text-xs text-red-500 mt-1">{task.error}</p>}
                    {task.status === "completed" && (
                      <p className={`text-xs mt-1 ${task.extracted === false ? "text-amber-600" : "text-gray-500"}`}>
                        {task.chunks !== undefined && `解析出 ${task.chunks} 个片段`}
                        {task.extracted === false && "  ⚠ 未提取到可读内容"}
                      </p>
                    )}
                  </div>
                  <div className="shrink-0">
                    {task.status === "uploading" ? <Loader2 className="w-5 h-5 animate-spin text-primary-600" />
                    : task.status === "completed" ? task.extracted === false ? <span className="text-amber-500 text-xs font-medium">无内容</span>
                    : <CheckCircle className="w-5 h-5 text-green-500" />
                    : <XCircle className="w-5 h-5 text-red-500" />}
                  </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}