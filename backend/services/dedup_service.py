import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np

from config import Settings
from models import DedupCheckResult, FileHashRecord, DedupStats, DocumentType

logger = logging.getLogger(__name__)


class DedupService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.hash_store_path = "dedup_store"
        self.hash_registry: dict[str, dict] = {}
        self.duplicate_log: list[dict] = []

        self._embedding_model = None

    async def initialize(self):
        os.makedirs(self.hash_store_path, exist_ok=True)
        self._load_hash_registry()
        self._load_duplicate_log()
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            logger.info(f"[去重] 向量模型加载成功: {self.settings.embedding_model}")
        except Exception as e:
            logger.warning(f"[去重] 向量模型加载失败: {e}, 仅使用哈希去重")
            self._embedding_model = None

    def _compute_file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _text_to_vector(self, text: str) -> Optional[list[float]]:
        if self._embedding_model and text.strip():
            embedding = self._embedding_model.encode(text[:2000], normalize_embeddings=True)
            return embedding.tolist()
        return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        a_np = np.array(a)
        b_np = np.array(b)
        dot = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def check_file_hash(self, content: bytes) -> tuple[bool, Optional[dict]]:
        if not self.settings.dedup_file_hash_enabled:
            return False, None
        file_hash = self._compute_file_hash(content)
        if file_hash in self.hash_registry:
            return True, self.hash_registry[file_hash]
        return False, None

    async def check_content_similarity(
        self, content_text: str, content_vector: Optional[list[float]] = None
    ) -> tuple[float, list[dict]]:
        if not self.settings.dedup_content_similarity_enabled:
            return 0.0, []

        if content_vector is None:
            content_vector = self._text_to_vector(content_text)

        if content_vector is None:
            return 0.0, []

        similarities = []
        for file_hash, record in self.hash_registry.items():
            existing_vector = record.get("content_vector")
            if existing_vector:
                sim = self._cosine_similarity(content_vector, existing_vector)
                if sim >= self.settings.dedup_warn_threshold:
                    similarities.append({
                        "file_name": record.get("file_name", ""),
                        "file_hash": file_hash[:16],
                        "similarity": round(sim, 4),
                        "file_type": record.get("file_type", ""),
                        "created_at": record.get("created_at", ""),
                    })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        max_sim = similarities[0]["similarity"] if similarities else 0.0
        return max_sim, similarities

    def determine_action(self, hash_match: bool, max_similarity: float) -> tuple[str, str, bool]:
        mode = self.settings.dedup_mode
        strict_threshold = self.settings.dedup_strict_threshold
        warn_threshold = self.settings.dedup_warn_threshold

        if hash_match:
            if mode == "strict":
                return "skip", "文件哈希相同，严格模式下自动跳过", True
            else:
                return "warn", "检测到完全相同的文件(哈希匹配)，建议跳过或覆盖更新", True

        if max_similarity >= strict_threshold:
            if mode == "strict":
                return "skip", f"内容高度相似({max_similarity:.1%})，严格模式下自动跳过", True
            elif mode == "loose":
                return "warn", f"内容高度相似({max_similarity:.1%})，已标记为可能重复", True
            else:
                return "warn", f"内容高度相似({max_similarity:.1%})，请确认是否继续", True

        if max_similarity >= warn_threshold:
            return "warn", f"内容部分相似({max_similarity:.1%})，可能是更新版本或相关内容", False

        return "proceed", "未检测到重复内容", False

    async def register_file(
        self, content: bytes, file_name: str, file_size: int,
        file_type: DocumentType, saved_path: str, task_id: str,
        content_text: str = "",
    ):
        file_hash = self._compute_file_hash(content)
        content_vector = self._text_to_vector(content_text)

        if file_hash in self.hash_registry:
            existing = self.hash_registry[file_hash]
            existing["updated_at"] = datetime.now().isoformat()
            existing["version"] = existing.get("version", 1) + 1
            logger.info(f"[去重] 更新已有文件记录: {file_name} (v{existing['version']})")
        else:
            self.hash_registry[file_hash] = {
                "file_hash": file_hash,
                "file_name": file_name,
                "file_size": file_size,
                "file_type": file_type.value if hasattr(file_type, 'value') else str(file_type),
                "saved_path": saved_path,
                "task_id": task_id,
                "content_text": content_text[:5000] if content_text else "",
                "content_vector": content_vector,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": 1,
            }
            logger.info(f"[去重] 注册新文件: {file_name} ({file_hash[:16]})")

        self._save_hash_registry()

    async def log_duplicate(self, file_name: str, reason: str, matched_file: str = ""):
        self.duplicate_log.append({
            "id": str(uuid.uuid4()),
            "file_name": file_name,
            "reason": reason,
            "matched_file": matched_file,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.duplicate_log) > 1000:
            self.duplicate_log = self.duplicate_log[-500:]
        self._save_duplicate_log()

    async def get_stats(self) -> DedupStats:
        total = len(self.hash_registry)
        duplicates = len(self.duplicate_log)
        strict_skipped = sum(1 for d in self.duplicate_log if "strict" in d.get("reason", "").lower() or "严格" in d.get("reason", ""))
        return DedupStats(
            total_files_tracked=total,
            duplicates_found=duplicates,
            strict_skipped=strict_skipped,
            hash_store_size=total,
        )

    async def clear_registry(self) -> int:
        count = len(self.hash_registry)
        self.hash_registry = {}
        self.duplicate_log = []
        self._save_hash_registry()
        self._save_duplicate_log()
        return count

    def _save_hash_registry(self):
        try:
            data = {}
            for k, v in self.hash_registry.items():
                data[k] = {key: val for key, val in v.items() if key != "content_vector"}
            with open(os.path.join(self.hash_store_path, "hashes.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[去重] 保存哈希注册表失败: {e}")

    def _load_hash_registry(self):
        path = os.path.join(self.hash_store_path, "hashes.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.hash_registry = json.load(f)
                logger.info(f"[去重] 加载哈希注册表: {len(self.hash_registry)} 个文件")
            except Exception as e:
                logger.error(f"[去重] 加载哈希注册表失败: {e}")

    def _save_duplicate_log(self):
        try:
            with open(os.path.join(self.hash_store_path, "duplicates.json"), "w", encoding="utf-8") as f:
                json.dump(self.duplicate_log, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[去重] 保存重复日志失败: {e}")

    def _load_duplicate_log(self):
        path = os.path.join(self.hash_store_path, "duplicates.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.duplicate_log = json.load(f)
                logger.info(f"[去重] 加载重复日志: {len(self.duplicate_log)} 条")
            except Exception as e:
                logger.error(f"[去重] 加载重复日志失败: {e}")