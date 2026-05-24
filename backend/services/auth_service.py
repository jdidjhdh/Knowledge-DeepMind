import sqlite3
import os
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from passlib.context import CryptContext
from jose import jwt, JWTError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DEFAULT_DB_PATH = "data/auth.db"
DEFAULT_SECRET_KEY = "dev-secret-key-change-in-production"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        secret_key: str = DEFAULT_SECRET_KEY,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        self.db_path = db_path
        self.secret_key = secret_key
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    avatar_url TEXT,
                    is_active INTEGER DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_hash ON refresh_tokens(token_hash)")
        self._cleanup_expired_tokens()

    def register(self, email: str, username: str, password: str) -> Tuple[str, dict]:
        self._validate_email(email)
        self._validate_username(username)
        self._validate_password(password)

        user_id = str(uuid.uuid4())
        password_hash = pwd_context.hash(password)

        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (id, email, username, password_hash) VALUES (?, ?, ?, ?)",
                    (user_id, email.lower().strip(), username.strip(), password_hash),
                )
            except sqlite3.IntegrityError as e:
                msg = str(e).lower()
                if "email" in msg:
                    raise ValueError("该邮箱已被注册")
                elif "username" in msg:
                    raise ValueError("该用户名已被使用")
                else:
                    raise ValueError("注册失败，请稍后重试")

        tokens = self._create_tokens(user_id)
        return user_id, {
            "user_id": user_id,
            "email": email.lower().strip(),
            "username": username.strip(),
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def login(self, email: str, password: str) -> dict:
        with self._get_conn() as conn:
            user = conn.execute(
                "SELECT id, email, username, password_hash, avatar_url, is_active, failed_attempts, locked_until FROM users WHERE email = ?",
                (email.lower().strip(),),
            ).fetchone()

        if not user:
            raise ValueError("邮箱或密码错误")

        if not user["is_active"]:
            raise ValueError("账号已被禁用，请联系管理员")

        if user["locked_until"]:
            locked_until = datetime.fromisoformat(user["locked_until"])
            if locked_until > datetime.now():
                remaining = int((locked_until - datetime.now()).total_seconds() / 60) + 1
                raise ValueError(f"账号已被锁定，请 {remaining} 分钟后重试")
            else:
                self._reset_failed_attempts(user["id"])

        if not pwd_context.verify(password, user["password_hash"]):
            self._increment_failed_attempts(user["id"])
            remaining = 5 - user["failed_attempts"] - 1
            if remaining > 0:
                raise ValueError(f"邮箱或密码错误，还剩 {remaining} 次尝试机会")
            else:
                self._lock_account(user["id"])
                raise ValueError("登录失败次数过多，账号已被锁定 15 分钟")

        self._reset_failed_attempts(user["id"])
        tokens = self._create_tokens(user["id"])
        return {
            "user_id": user["id"],
            "email": user["email"],
            "username": user["username"],
            "avatar_url": user["avatar_url"],
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def refresh(self, refresh_token: str) -> dict:
        token_hash = _hash_token(refresh_token)

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, user_id, expires_at FROM refresh_tokens WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()

        if not row:
            raise ValueError("无效的刷新令牌")

        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.now():
            self._delete_refresh_token_by_hash(token_hash)
            raise ValueError("刷新令牌已过期，请重新登录")

        self._delete_refresh_token_by_hash(token_hash)
        return self._create_tokens(row["user_id"])

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, email, username, avatar_url, is_active, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "user_id": row["id"],
            "email": row["email"],
            "username": row["username"],
            "avatar_url": row["avatar_url"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }

    def verify_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("user_id")
            if not user_id:
                return None
            user = self.get_user(user_id)
            if not user or not user["is_active"]:
                return None
            return user_id
        except JWTError:
            return None

    def _create_tokens(self, user_id: str) -> dict:
        access_token = self._create_access_token(user_id)
        refresh_token = self._create_refresh_token(user_id)
        return {"access_token": access_token, "refresh_token": refresh_token}

    def _create_access_token(self, user_id: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def _create_refresh_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(64)
        token_hash = _hash_token(token)
        expires_at = (datetime.now() + timedelta(days=self.refresh_token_expire_days)).isoformat()

        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), user_id, token_hash, expires_at),
            )

        return token

    def _delete_refresh_token_by_hash(self, token_hash: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,))

    def _cleanup_expired_tokens(self):
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM refresh_tokens WHERE expires_at < datetime('now')"
            )

    def _increment_failed_attempts(self, user_id: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE id = ?",
                (user_id,),
            )

    def _reset_failed_attempts(self, user_id: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?",
                (user_id,),
            )

    def _lock_account(self, user_id: str):
        locked_until = (datetime.now() + timedelta(minutes=15)).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE users SET locked_until = ? WHERE id = ?",
                (locked_until, user_id),
            )

    @staticmethod
    def _validate_email(email: str):
        import re
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            raise ValueError("邮箱格式不正确")

    @staticmethod
    def _validate_username(username: str):
        if len(username.strip()) < 2:
            raise ValueError("用户名至少 2 个字符")
        if len(username.strip()) > 30:
            raise ValueError("用户名最多 30 个字符")

    @staticmethod
    def _validate_password(password: str):
        if len(password) < 8:
            raise ValueError("密码至少 8 位")
        if not any(c.isalpha() for c in password):
            raise ValueError("密码必须包含字母")
        if not any(c.isdigit() for c in password):
            raise ValueError("密码必须包含数字")