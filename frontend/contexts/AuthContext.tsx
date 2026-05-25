"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";

interface User {
  user_id: string;
  email: string;
  username: string;
  avatar_url?: string;
  created_at?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
  logout: () => void;
}

const API_BASE = "/api";

const AuthContext = createContext<AuthContextType>({
  user: null,
  isAuthenticated: false,
  loading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const getAccessToken = useCallback(() => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("access_token");
  }, []);

  const getRefreshToken = useCallback(() => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("refresh_token");
  }, []);

  const saveTokens = useCallback((accessToken: string, refreshToken: string) => {
    localStorage.setItem("access_token", accessToken);
    localStorage.setItem("refresh_token", refreshToken);
  }, []);

  const clearTokens = useCallback(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
  }, []);

  const fetchMe = useCallback(async (token: string): Promise<User | null> => {
    try {
      const res = await fetch(`${API_BASE}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.user;
    } catch {
      return null;
    }
  }, []);

  const tryRefreshToken = useCallback(async (): Promise<string | null> => {
    const rt = getRefreshToken();
    if (!rt) return null;
    try {
      const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: rt }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      saveTokens(data.access_token, data.refresh_token);
      return data.access_token;
    } catch {
      return null;
    }
  }, [getRefreshToken, saveTokens]);

  useEffect(() => {
    const initAuth = async () => {
      let token = getAccessToken();
      if (!token) {
        token = await tryRefreshToken();
      }
      if (token) {
        const u = await fetchMe(token);
        if (u) {
          setUser(u);
        } else {
          const newToken = await tryRefreshToken();
          if (newToken) {
            const u2 = await fetchMe(newToken);
            if (u2) setUser(u2);
            else clearTokens();
          } else {
            clearTokens();
          }
        }
      }
      setLoading(false);
    };
    initAuth();
  }, [getAccessToken, fetchMe, tryRefreshToken, clearTokens]);

  const login = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "登录失败");
    }
    const data = await res.json();
    saveTokens(data.access_token, data.refresh_token);
    const u = await fetchMe(data.access_token);
    if (u) setUser(u);
  }, [saveTokens, fetchMe]);

  const register = useCallback(async (email: string, username: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, username, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "注册失败");
    }
    const data = await res.json();
    saveTokens(data.access_token, data.refresh_token);
    const u = await fetchMe(data.access_token);
    if (u) setUser(u);
  }, [saveTokens, fetchMe]);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
    try {
      localStorage.setItem("auth_logout", Date.now().toString());
    } catch {}
    try {
      const bc = new BroadcastChannel("auth_sync");
      bc.postMessage({ type: "logout" });
      bc.close();
    } catch {}
  }, [clearTokens]);

  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === "auth_logout" && e.newValue) {
        setUser(null);
      }
    };
    window.addEventListener("storage", handleStorage);

    let bc: BroadcastChannel | null = null;
    try {
      bc = new BroadcastChannel("auth_sync");
      bc.onmessage = (event) => {
        if (event.data?.type === "logout") {
          setUser(null);
        }
      };
    } catch {}

    return () => {
      window.removeEventListener("storage", handleStorage);
      if (bc) bc.close();
    };
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        loading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);