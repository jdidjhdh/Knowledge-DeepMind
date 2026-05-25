"use client";

import { useNetworkStatus } from "@/lib/useNetworkStatus";

export default function NetworkStatusBar() {
  const { isOnline } = useNetworkStatus();

  return (
    <div
      className={`fixed top-0 left-0 right-0 z-[9999] bg-amber-600 text-white text-sm text-center py-1.5 font-medium shadow-md transition-all ${isOnline ? "hidden" : ""}`}
    >
      网络连接已断开，恢复后将自动重试
    </div>
  );
}