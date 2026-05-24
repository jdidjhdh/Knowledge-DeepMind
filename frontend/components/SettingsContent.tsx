"use client";

import { useState, useEffect } from "react";
import { api, ModelSettings } from "@/lib/api";
import {
  Brain,
  Eye,
  Mic,
  Cloud,
  Settings,
  Save,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ToggleLeft,
  ToggleRight,
} from "lucide-react";

const WHISPER_OPTIONS = [
  { value: "tiny", label: "tiny (最快，精度最低)" },
  { value: "base", label: "base" },
  { value: "small", label: "small" },
  { value: "medium", label: "medium" },
  { value: "large-v2", label: "large-v2" },
  { value: "large-v3", label: "large-v3 (最准，需GPU)" },
];

const defaultSettings: ModelSettings = {
  deepseek_enabled: true,
  deepseek_model: "deepseek-chat",
  deepseek_api_key: "",
  deepseek_base_url: "https://api.deepseek.com",
  vision_enabled: false,
  vision_model: "qwen-vl-max",
  vision_api_key: "",
  vision_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  speech_model: "qwen-audio-turbo-latest",
  speech_api_key: "",
  speech_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  speech_enabled: true,
};

export default function SettingsContent() {
  const [settings, setSettings] = useState<ModelSettings>(defaultSettings);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ type: "success" | "error" | ""; message: string }>({ type: "", message: "" });

  useEffect(() => {
    loadSettings();
  }, []);

  async function loadSettings() {
    try {
      const data = await api.getModelSettings();
      setSettings(data);
    } catch {
      // 使用默认值
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    setSaving(true);
    setStatus({ type: "", message: "" });
    try {
      const result = await api.updateModelSettings(settings);
      setStatus({ type: "success", message: result.message + (result.needs_restart ? "，请重启服务以生效" : "") });
    } catch (e: unknown) {
      setStatus({ type: "error", message: e instanceof Error ? e.message : "保存失败" });
    } finally {
      setSaving(false);
    }
  }

  function handleReset() {
    setSettings(defaultSettings);
    setStatus({ type: "", message: "" });
  }

  function updateField(field: keyof ModelSettings, value: string | boolean) {
    setSettings((prev) => ({ ...prev, [field]: value }));
  }

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Settings className="w-6 h-6 text-primary-600" />
        <h1 className="text-2xl font-bold">模型设置</h1>
      </div>
      <p className="text-sm text-gray-500 dark:text-gray-400">
        选择和管理 AI 模型，保存后需重启服务生效
      </p>

      {/* 状态提示 */}
      {status.type && (
        <div
          className={`flex items-center gap-2 p-3 rounded-lg text-sm ${
            status.type === "success"
              ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800"
              : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800"
          }`}
        >
          {status.type === "success" ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
          {status.message}
        </div>
      )}

      {/* ====== DeepSeek ====== */}
      <SectionCard
        icon={<Brain className="w-5 h-5" />}
        title="基础大模型 (DeepSeek)"
        subtitle="日常对话、文本分析的默认模型"
        enabled={settings.deepseek_enabled}
        onToggle={(v) => updateField("deepseek_enabled", v)}
        defaultExpanded={true}
      >
        <FieldGroup>
          <TextField
            label="模型名称"
            value={settings.deepseek_model}
            onChange={(v) => updateField("deepseek_model", v)}
            placeholder="deepseek-chat"
          />
          <TextField
            label="API Key"
            value={settings.deepseek_api_key}
            onChange={(v) => updateField("deepseek_api_key", v)}
            placeholder="sk-..."
            type="password"
            masked={settings.deepseek_api_key.includes("***")}
          />
          <TextField
            label="API 地址"
            value={settings.deepseek_base_url}
            onChange={(v) => updateField("deepseek_base_url", v)}
            placeholder="https://api.deepseek.com"
          />
        </FieldGroup>
      </SectionCard>

      {/* ====== Qwen VL ====== */}
      <SectionCard
        icon={<Eye className="w-5 h-5" />}
        title="视觉大模型 (通义千问 VL)"
        subtitle="上传图片/视频时启用的多模态模型"
        enabled={settings.vision_enabled}
        onToggle={(v) => updateField("vision_enabled", v)}
      >
        <FieldGroup>
          <TextField
            label="模型名称"
            value={settings.vision_model}
            onChange={(v) => updateField("vision_model", v)}
            placeholder="qwen-vl-max"
          />
          <TextField
            label="API Key"
            value={settings.vision_api_key}
            onChange={(v) => updateField("vision_api_key", v)}
            placeholder="DashScope API Key"
            type="password"
            masked={settings.vision_api_key.includes("***")}
          />
          <TextField
            label="API 地址"
            value={settings.vision_base_url}
            onChange={(v) => updateField("vision_base_url", v)}
            placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
          />
        </FieldGroup>
      </SectionCard>

      {/* ====== Whisper ====== */}
      <SectionCard
        icon={<Mic className="w-5 h-5" />}
        title="语音识别模型 (云端 ASR)"
        subtitle="音频/视频转文字，复用视觉模型 API Key"
        enabled={settings.speech_enabled}
        onToggle={(v) => updateField("speech_enabled", v)}
      >
        <FieldGroup>
          <TextField
            label="模型名称"
            value={settings.speech_model}
            onChange={(v) => updateField("speech_model", v)}
            placeholder="qwen-audio-turbo-latest"
          />
          <TextField
            label="API Key (可选)"
            value={settings.speech_api_key}
            onChange={(v) => updateField("speech_api_key", v)}
            placeholder="留空则复用视觉或DeepSeek Key"
            type="password"
            masked={settings.speech_api_key.includes("***")}
          />
          <TextField
            label="API 地址"
            value={settings.speech_base_url}
            onChange={(v) => updateField("speech_base_url", v)}
            placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
          />
        </FieldGroup>
      </SectionCard>

      {/* 操作按钮 */}
      <div className="flex items-center gap-3 pt-4">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-5 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors"
        >
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          保存配置
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2.5 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg text-sm transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          恢复默认
        </button>
      </div>

      <div className="text-xs text-gray-400 dark:text-gray-500 pt-2 pb-8">
        配置保存到 .env 文件，需重启后端服务才能生效。视觉模型启用后将用于图片分类、语义分析和视频帧描述。
      </div>
    </div>
  );
}

/* ====== 子组件 ====== */

function SectionCard({
  icon,
  title,
  subtitle,
  enabled,
  onToggle,
  children,
  defaultExpanded,
  hideToggle,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  enabled: boolean;
  onToggle: (v: boolean) => void;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  hideToggle?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded ?? enabled);

  useEffect(() => {
    if (enabled) setExpanded(true);
  }, [enabled]);

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <div className={`${enabled ? "text-primary-600 dark:text-primary-400" : "text-gray-400"}`}>
            {icon}
          </div>
          <div>
            <div className="font-medium text-sm">{title}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">{subtitle}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!hideToggle && (
            <span
              onClick={(e) => {
                e.stopPropagation();
                onToggle(!enabled);
              }}
              className="cursor-pointer"
              title={enabled ? "点击关闭" : "点击开启"}
            >
              {enabled ? (
                <ToggleRight className="w-8 h-5 text-primary-600" />
              ) : (
                <ToggleLeft className="w-8 h-5 text-gray-400" />
              )}
            </span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            enabled
              ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
              : "bg-gray-100 dark:bg-gray-800 text-gray-500"
          }`}>
            {enabled ? "已启用" : "未启用"}
          </span>
        </div>
      </button>
      {expanded && <div className="px-4 pb-4 border-t border-gray-100 dark:border-gray-800 pt-4">{children}</div>}
    </div>
  );
}

function FieldGroup({ children }: { children: React.ReactNode }) {
  return <div className="space-y-3">{children}</div>;
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
  type,
  masked,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  masked?: boolean;
}) {
  const [showKey, setShowKey] = useState(false);

  if (masked && !showKey) {
    return (
      <div className="space-y-1.5">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>
        <div className="flex gap-2">
          <input
            type="password"
            value={value}
            disabled
            className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50 px-3 py-2 text-sm text-gray-400"
          />
          <button
            type="button"
            onClick={() => {
              setShowKey(true);
              onChange("");
            }}
            className="px-3 py-2 text-xs text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-colors whitespace-nowrap"
          >
            重新输入
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>
      <input
        type={type || "text"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm placeholder:text-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
      />
    </div>
  );
}