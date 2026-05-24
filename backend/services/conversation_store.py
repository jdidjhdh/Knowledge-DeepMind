import json
import logging
from datetime import datetime
from typing import Optional

from .database import get_db, DB

logger = logging.getLogger(__name__)


class ConversationStore:
    def __init__(self):
        self.db = get_db()

    def list_conversations(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT conv_id, title, length(messages) as msg_count, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        convs = []
        for row in rows:
            convs.append({
                "id": row["conv_id"],
                "title": row["title"],
                "message_count": row["msg_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return convs

    def get_conversation(self, conv_id: str) -> Optional[list]:
        row = self.db.execute(
            "SELECT messages FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
        if row:
            return json.loads(row["messages"])
        return None

    def save_messages(self, conv_id: str, messages: list[dict], title: Optional[str] = None):
        now = datetime.now().isoformat()
        effective_title = title or "新对话"
        messages_json = json.dumps(messages, ensure_ascii=False)
        self.db.execute(
            """INSERT INTO conversations (conv_id, title, messages, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(conv_id) DO UPDATE SET messages=?, title=?, updated_at=?""",
            (conv_id, effective_title, messages_json, now, messages_json, effective_title, now),
        )

    def delete_conversation(self, conv_id: str) -> bool:
        result = self.db.execute(
            "DELETE FROM conversations WHERE conv_id = ?", (conv_id,)
        )
        return result.rowcount > 0


conversation_store = ConversationStore()