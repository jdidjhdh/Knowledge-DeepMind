import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/memory.db"


class MemoryService:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    last_accessed TEXT DEFAULT (datetime('now')),
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(user_id, memory_type, memory_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_items(user_id, memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_conv ON conversation_history(conv_id, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_user ON conversation_history(user_id)
            """)
            conn.commit()
        logger.info(f"[记忆服务] 数据库初始化完成: {self.db_path}")

    def get_profile(self, user_id: str) -> dict:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return json.loads(row["profile_data"])
            return {}

    def save_profile(self, user_id: str, profile: dict):
        now = datetime.now().isoformat()
        data = json.dumps(profile, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO user_profiles (user_id, profile_data, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                   profile_data = excluded.profile_data,
                   updated_at = excluded.updated_at""",
                (user_id, data, now),
            )
            conn.commit()

    def update_profile_field(self, user_id: str, key: str, value: str):
        profile = self.get_profile(user_id)
        profile[key] = value
        self.save_profile(user_id, profile)

    def get_profile_field(self, user_id: str, key: str, default=None):
        profile = self.get_profile(user_id)
        return profile.get(key, default)

    def set_memory_item(self, user_id: str, memory_type: str, memory_key: str, memory_value: str):
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO memory_items (user_id, memory_type, memory_key, memory_value, weight, last_accessed)
                   VALUES (?, ?, ?, ?, 1.0, ?)
                   ON CONFLICT(user_id, memory_type, memory_key) DO UPDATE SET
                   memory_value = excluded.memory_value,
                   last_accessed = excluded.last_accessed,
                   weight = MIN(2.0, weight + 0.1)""",
                (user_id, memory_type, memory_key, memory_value, now),
            )
            conn.commit()
        logger.info(f"[记忆] {user_id[:8]} {memory_type}.{memory_key} = {memory_value[:50]}")

    def get_memory_item(self, user_id: str, memory_type: str, memory_key: str) -> Optional[str]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT memory_value, weight FROM memory_items WHERE user_id = ? AND memory_type = ? AND memory_key = ?",
                (user_id, memory_type, memory_key),
            ).fetchone()
            if row:
                if row["weight"] < 0.1:
                    return None
                conn.execute(
                    "UPDATE memory_items SET last_accessed = ?, weight = MIN(2.0, weight + 0.05) WHERE user_id = ? AND memory_type = ? AND memory_key = ?",
                    (datetime.now().isoformat(), user_id, memory_type, memory_key),
                )
                conn.commit()
                return row["memory_value"]
            return None

    def get_memory_items(self, user_id: str, memory_type: Optional[str] = None) -> list[dict]:
        with self._get_conn() as conn:
            if memory_type:
                rows = conn.execute(
                    """SELECT memory_type, memory_key, memory_value, weight, last_accessed, created_at
                       FROM memory_items WHERE user_id = ? AND memory_type = ? AND weight >= 0.1
                       ORDER BY weight DESC""",
                    (user_id, memory_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT memory_type, memory_key, memory_value, weight, last_accessed, created_at
                       FROM memory_items WHERE user_id = ? AND weight >= 0.1
                       ORDER BY memory_type, weight DESC""",
                    (user_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_all_user_memory(self, user_id: str) -> dict:
        items = self.get_memory_items(user_id)
        profile = self.get_profile(user_id)
        return {
            "profile": profile,
            "memory_items": items,
            "item_count": len(items),
        }

    def delete_memory_item(self, user_id: str, memory_type: str, memory_key: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE user_id = ? AND memory_type = ? AND memory_key = ?",
                (user_id, memory_type, memory_key),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"[记忆] 删除 {user_id[:8]} {memory_type}.{memory_key}")
            return deleted

    def delete_user_memory(self, user_id: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM memory_items WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount

    def decay_memory(self, max_age_days: int = 30, decay_rate: float = 0.95):
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._get_conn() as conn:
            result = conn.execute(
                "UPDATE memory_items SET weight = weight * ? WHERE last_accessed < ? AND weight > 0.01",
                (decay_rate, cutoff),
            )
            decayed = result.rowcount
            conn.execute(
                "DELETE FROM memory_items WHERE weight < 0.1"
            )
            conn.commit()
        if decayed > 0:
            logger.info(f"[记忆衰减] 衰减了 {decayed} 条长期未访问的记忆")

    def save_conversation_message(self, conv_id: str, user_id: str, role: str, content: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversation_history (conv_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (conv_id, user_id, role, content),
            )
            conn.commit()

    def get_conversation_history(self, conv_id: str, limit: int = 20) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation_history WHERE conv_id = ? ORDER BY created_at ASC LIMIT ?",
                (conv_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user_conversations(self, user_id: str, limit: int = 10) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT conv_id, COUNT(*) as msg_count, MIN(created_at) as started_at, MAX(created_at) as last_at
                   FROM conversation_history WHERE user_id = ?
                   GROUP BY conv_id ORDER BY last_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._get_conn() as conn:
            user_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
            item_count = conn.execute("SELECT COUNT(*) FROM memory_items WHERE weight >= 0.1").fetchone()[0]
            conv_count = conn.execute("SELECT COUNT(DISTINCT conv_id) FROM conversation_history").fetchone()[0]
            msg_count = conn.execute("SELECT COUNT(*) FROM conversation_history").fetchone()[0]
            return {
                "user_count": user_count,
                "memory_item_count": item_count,
                "conversation_count": conv_count,
                "message_count": msg_count,
                "db_path": self.db_path,
            }