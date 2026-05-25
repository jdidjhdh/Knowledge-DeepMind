"use client";

import { useState, useEffect, useCallback } from "react";

export function useNetworkStatus() {
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== "undefined" ? navigator.onLine : true
  );

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const retry = useCallback(async <T>(
    fn: () => Promise<T>,
    maxRetries = 3,
    delayMs = 2000
  ): Promise<T> => {
    let lastError: unknown;
    for (let i = 0; i <= maxRetries; i++) {
      try {
        if (!navigator.onLine) {
          await new Promise((resolve) => setTimeout(resolve, delayMs));
          continue;
        }
        return await fn();
      } catch (err) {
        lastError = err;
        if (i < maxRetries && !navigator.onLine) {
          await new Promise((resolve) => setTimeout(resolve, delayMs));
        } else {
          break;
        }
      }
    }
    throw lastError;
  }, []);

  return { isOnline, retry };
}