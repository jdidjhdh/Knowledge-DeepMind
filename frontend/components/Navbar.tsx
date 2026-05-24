"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { useTheme } from "./ThemeProvider";
import { useAuth } from "@/contexts/AuthContext";
import { useState, useRef, useEffect } from "react";
import {
  Home,
  MessageSquare,
  Upload,
  Search,
  GitGraph,
  Clock,
  Sun,
  Moon,
  BookOpen,
  Settings,
  User,
  LogIn,
  UserPlus,
  LogOut,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const navItems = [
  { href: "/", label: "首页", icon: Home },
  { href: "/chat", label: "对话", icon: MessageSquare },
  { href: "/upload", label: "上传", icon: Upload },
  { href: "/wiki", label: "知识库", icon: BookOpen },
  { href: "/graph", label: "图谱", icon: GitGraph },
  { href: "/timeline", label: "时间线", icon: Clock },
  { href: "/settings", label: "设置", icon: Settings },
];

export default function Navbar() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useTheme();
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <nav className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2 font-bold text-lg text-primary-600" prefetch={false}>
            <BookOpen className="w-6 h-6" />
            <span className="hidden sm:inline">知识库智能体</span>
          </Link>

          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <motion.div key={item.href} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link
                    href={item.href}
                    prefetch={false}
                    className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300"
                        : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="hidden md:inline">{item.label}</span>
                  </Link>
                </motion.div>
              );
            })}

            <motion.button
              onClick={toggleTheme}
              className="ml-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors relative overflow-hidden"
              aria-label="切换主题"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.85 }}
              suppressHydrationWarning
            >
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={theme}
                  initial={{ rotate: -90, opacity: 0, scale: 0.5 }}
                  animate={{ rotate: 0, opacity: 1, scale: 1 }}
                  exit={{ rotate: 90, opacity: 0, scale: 0.5 }}
                  transition={{ duration: 0.35, ease: "easeInOut" }}
                >
                  {theme === "dark" ? (
                    <Sun className="w-4 h-4" suppressHydrationWarning />
                  ) : (
                    <Moon className="w-4 h-4" suppressHydrationWarning />
                  )}
                </motion.div>
              </AnimatePresence>
            </motion.button>

            {authLoading ? (
              <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse ml-1" />
            ) : isAuthenticated && user ? (
              <div className="relative ml-1" ref={menuRef}>
                <button
                  onClick={() => setMenuOpen(!menuOpen)}
                  className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-indigo-500 flex items-center justify-center text-white text-sm font-medium hover:ring-2 hover:ring-primary-300 transition-all"
                >
                  {user.username.charAt(0).toUpperCase()}
                </button>

                {menuOpen && (
                  <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-gray-900 rounded-xl shadow-xl border border-gray-200 dark:border-gray-800 py-1 z-50">
                    <div className="px-4 py-2 border-b border-gray-100 dark:border-gray-800">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {user.username}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                        {user.email}
                      </p>
                    </div>
                    <Link
                      href="/profile"
                      onClick={() => setMenuOpen(false)}
                      className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                    >
                      <User className="w-4 h-4" />
                      个人中心
                    </Link>
                    <button
                      onClick={() => {
                        setMenuOpen(false);
                        logout();
                      }}
                      className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                    >
                      <LogOut className="w-4 h-4" />
                      退出登录
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-1 ml-1">
                <Link
                  href="/login"
                  className="flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                  prefetch={false}
                >
                  <LogIn className="w-4 h-4" />
                  <span className="hidden sm:inline">登录</span>
                </Link>
                <Link
                  href="/register"
                  className="flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium bg-primary-500 text-white hover:bg-primary-600 transition-colors"
                  prefetch={false}
                >
                  <UserPlus className="w-4 h-4" />
                  <span className="hidden sm:inline">注册</span>
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}