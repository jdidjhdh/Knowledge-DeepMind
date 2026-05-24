import json
import logging
import uuid
import os
import numpy as np
from typing import Optional

from config import Settings
from models import KnowledgePoint, DocumentChunk

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_model = None
        self.vector_store_path = "vector_store"
        self.knowledge_index = {}
        self.document_index = {}

    async def initialize(self):
        os.makedirs(self.vector_store_path, exist_ok=True)
        self._load_index()
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.settings.embedding_model)
            logger.info(f"向量模型 {self.settings.embedding_model} 加载成功")
        except Exception as e:
            logger.warning(f"向量模型加载失败: {e}, 使用简单哈希向量")
            self.embedding_model = None

    def _text_to_vector(self, text: str) -> list[float]:
        if self.embedding_model:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        else:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
            if len(arr) < self.settings.vector_dim:
                arr = np.pad(arr, (0, self.settings.vector_dim - len(arr)))
            else:
                arr = arr[:self.settings.vector_dim]
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        dot = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    async def index_knowledge_point(self, kp: KnowledgePoint):
        kp_id = kp.id or str(uuid.uuid4())
        vector = self._text_to_vector(kp.fact)
        self.knowledge_index[kp_id] = {
            "id": kp_id,
            "vector": vector,
            "data": kp.model_dump(),
        }
        self._save_index()

    async def index_document_chunk(self, chunk: DocumentChunk):
        chunk_id = chunk.id or str(uuid.uuid4())
        vector = self._text_to_vector(chunk.content[:2000])
        self.document_index[chunk_id] = {
            "id": chunk_id,
            "vector": vector,
            "data": chunk.model_dump(),
        }

    async def search_knowledge(self, query: str, top_k: int = 10) -> list[KnowledgePoint]:
        query_vector = self._text_to_vector(query)
        scores = []
        for kp_id, item in self.knowledge_index.items():
            sim = self._cosine_similarity(query_vector, item["vector"])
            scores.append((sim, item["data"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, data in scores[:top_k]:
            data["confidence"] = max(data.get("confidence", 0.5), sim)
            results.append(KnowledgePoint(**data))
        return results

    async def search_documents(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        query_vector = self._text_to_vector(query)
        scores = []
        for chunk_id, item in self.document_index.items():
            sim = self._cosine_similarity(query_vector, item["vector"])
            scores.append((sim, item["data"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [DocumentChunk(**data) for sim, data in scores[:top_k]]

    async def get_knowledge_point(self, kp_id: str) -> Optional[dict]:
        item = self.knowledge_index.get(kp_id)
        if item:
            return item["data"]
        return None

    async def delete_knowledge_point(self, kp_id: str):
        if kp_id in self.knowledge_index:
            del self.knowledge_index[kp_id]
            self._save_index()

    async def delete_all(self):
        count = len(self.knowledge_index)
        self.knowledge_index = {}
        self._save_index()
        return count

    async def update_knowledge_point(self, kp_id: str, updated: KnowledgePoint):
        self.knowledge_index[kp_id] = {
            "id": kp_id,
            "vector": self._text_to_vector(updated.fact),
            "data": updated.model_dump(),
        }
        self._save_index()

    async def count(self) -> int:
        return len(self.knowledge_index)

    async def list_all(self, offset: int = 0, limit: int = 50) -> list[dict]:
        items = list(self.knowledge_index.values())
        items.sort(key=lambda x: str(x["data"].get("created_at", "")), reverse=True)
        return [item["data"] for item in items[offset:offset + limit]]

    def _save_index(self):
        try:
            data = {
                "knowledge": self.knowledge_index,
                "documents": self.document_index,
            }
            with open(os.path.join(self.vector_store_path, "index.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")

    def _load_index(self):
        index_path = os.path.join(self.vector_store_path, "index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.knowledge_index = data.get("knowledge", {})
                self.document_index = data.get("documents", {})
                logger.info(f"加载向量索引: {len(self.knowledge_index)} 个知识点, {len(self.document_index)} 个文档片段")
            except Exception as e:
                logger.error(f"加载向量索引失败: {e}")