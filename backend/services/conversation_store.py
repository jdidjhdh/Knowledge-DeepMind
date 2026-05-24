import json
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

CONVERSATIONS_FILE = "./data/conversations.json"


class ConversationStore:
    def __init__(self, file_path: str = CONVERSATIONS_FILE):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {}

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存对话失败: {e}")

    def list_conversations(self) -> list[dict]:
        convs = []
        for cid, data in self._data.items():
            convs.append({
                "id": cid,
                "title": data.get("title", "新对话"),
                "message_count": len(data.get("messages", [])),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        convs.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return convs

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        return self._data.get(conv_id)

    def save_messages(self, conv_id: str, messages: list[dict], title: Optional[str] = None):
        now = datetime.now().isoformat()
        if conv_id not in self._data:
            self._data[conv_id] = {"created_at": now, "title": title or "新对话", "messages": []}
        self._data[conv_id]["messages"] = messages
        self._data[conv_id]["updated_at"] = now
        if title:
            self._data[conv_id]["title"] = title
        self._save()

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id in self._data:
            del self._data[conv_id]
            self._save()
            return True
        return False


conversation_store = ConversationStore()