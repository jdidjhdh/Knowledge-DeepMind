"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

const ChatContent = dynamic(() => import("@/components/ChatContent"), {
  loading: () => <div className="flex justify-center py-24"><Loader2 className="w-8 h-8 animate-spin text-primary-600" /></div>,
});

export default function ChatPage() {
  return <ChatContent />;
}