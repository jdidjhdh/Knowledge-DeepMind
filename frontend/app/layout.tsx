import type { Metadata } from "next";
import { ThemeProvider } from "@/components/ThemeProvider";
import { AuthProvider } from "@/contexts/AuthContext";
import Navbar from "@/components/Navbar";
import FluidBackground from "@/components/FluidBackground";
import PageTransition from "@/components/PageTransition";
import NetworkStatusBar from "@/components/NetworkStatusBar";
import "./globals.css";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "知识库智能体 - 全格式自进化知识库",
  description: "支持全格式文件摄入、自动知识抽取、对话式知识检索的智能知识库系统",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var t = localStorage.getItem('theme');
                  if (t === 'dark' || t === null) {
                    document.documentElement.classList.add('dark');
                  }
                } catch(e) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
        <NetworkStatusBar />
        <AuthProvider>
          <ThemeProvider>
            <FluidBackground />
            <Navbar />
            <PageTransition>
              <main className="max-w-7xl mx-auto px-4 py-6 relative z-10">
                {children}
              </main>
            </PageTransition>
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  );
}