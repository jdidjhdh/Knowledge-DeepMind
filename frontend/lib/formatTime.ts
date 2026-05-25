export function formatRelativeTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 3600000) {
      const mins = Math.floor(diff / 60000);
      return mins <= 0 ? "刚刚" : `${mins}分钟前`;
    }
    if (diff < 86400000) {
      return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    }
    if (diff < 604800000) {
      const days = Math.floor(diff / 86400000);
      return `${days}天前`;
    }
    if (d.getFullYear() === now.getFullYear()) {
      return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
    }
    return d.toLocaleDateString("zh-CN", { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

export function formatDateTime(iso?: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso.slice(0, 19).replace("T", " ");
  }
}

export function formatDateShort(iso?: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  } catch {
    return iso.slice(0, 10);
  }
}

export function formatConversationTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 86400000) {
      return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}