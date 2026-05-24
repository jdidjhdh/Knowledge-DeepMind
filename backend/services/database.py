import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES = bool(DATABASE_URL)


def use_postgres() -> bool:
    return _USE_POSTGRES


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        try:
            import psycopg2
            import psycopg2.pool
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL,
            )
            logger.info("[DB] PostgreSQL 连接池已创建")
        except Exception as e:
            logger.error(f"[DB] PostgreSQL 连接失败: {e}")
            raise
    return _pg_pool


_pg_pool = None


class DB:
    def __init__(self):
        if _USE_POSTGRES:
            self._pool = _get_pg_pool()
            self._conn = None
        else:
            import sqlite3
            os.makedirs("data", exist_ok=True)
            self._conn = sqlite3.connect("data/knowledge.db")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql: str, params: tuple = ()):
        if _USE_POSTGRES:
            pg_conn = self._pool.getconn()
            try:
                pg_sql, pg_params = _to_pg(sql, list(params))
                with pg_conn.cursor() as cur:
                    cur.execute(pg_sql, pg_params)
                    try:
                        if cur.description:
                            columns = [d[0] for d in cur.description]
                            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
                        else:
                            rows = []
                    except Exception:
                        rows = []
                    pg_conn.commit()
                    return DBCursor(rows, cur.rowcount)
            except Exception:
                pg_conn.rollback()
                raise
            finally:
                self._pool.putconn(pg_conn)
        else:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            rows = [dict(r) for r in cursor.fetchall()]
            return DBCursor(rows, cursor.rowcount)

    def close(self):
        if _USE_POSTGRES and _pg_pool:
            _pg_pool.closeall()
        elif not _USE_POSTGRES and self._conn:
            self._conn.close()


class DBCursor:
    def __init__(self, rows: list[dict], rowcount: int = 0):
        self._rows = rows
        self._rowcount = rowcount

    def fetchone(self) -> Optional[dict]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return self._rows

    @property
    def rowcount(self) -> int:
        return self._rowcount


def _to_pg(sql: str, params: list) -> tuple:
    idx = [0]
    def replacer(m):
        idx[0] += 1
        return f"%s"
    import re
    pg_sql = re.sub(r'\?', replacer, sql)
    pg_sql = pg_sql.replace("datetime('now')", "NOW()")
    pg_sql = pg_sql.replace("AUTOINCREMENT", "SERIAL")
    return pg_sql, tuple(params)


def init_db(db: DB):
    if _USE_POSTGRES:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                avatar_url TEXT,
                is_active INTEGER DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_refresh_hash ON refresh_tokens(token_hash)")

        db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                last_accessed TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, memory_type, memory_key)
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id SERIAL PRIMARY KEY,
                conv_id TEXT NOT NULL,
                user_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_items(user_id, memory_type)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_history_conv ON conversation_history(conv_id, created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON conversation_history(user_id)")

        db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '新对话',
                messages TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS categories_data (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                parent_id TEXT,
                icon TEXT,
                color TEXT,
                sort_order INTEGER DEFAULT 0,
                is_system BOOLEAN DEFAULT false,
                category_type TEXT DEFAULT 'structural',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_categories_user ON categories_data(user_id)")

        db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_categories_data (
                knowledge_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                PRIMARY KEY (knowledge_id, category_id)
            )
        """)
        logger.info("[DB] PostgreSQL 表已初始化")


_db_instance: Optional[DB] = None


def get_db() -> DB:
    global _db_instance
    if _db_instance is None:
        _db_instance = DB()
        if _USE_POSTGRES:
            init_db(_db_instance)
    return _db_instance