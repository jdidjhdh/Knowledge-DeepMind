"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { ErrorBoundary } from "@/components/ErrorBoundary";

const WikiContent = dynamic(() => import("@/components/WikiContent"), {
  loading: () => <div className="flex justify-center py-12"><Loader2 className="w-8 h-8 animate-spin text-primary-600" /></div>,
});

export default function WikiPage() {
  return (
    <ErrorBoundary>
      <WikiContent />
    </ErrorBoundary>
  );
}