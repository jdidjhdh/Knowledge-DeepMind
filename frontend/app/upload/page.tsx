"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

const UploadContent = dynamic(() => import("@/components/UploadContent"), {
  loading: () => <div className="flex justify-center py-12"><Loader2 className="w-8 h-8 animate-spin text-primary-600" /></div>,
});

export default function UploadPage() {
  return <UploadContent />;
}