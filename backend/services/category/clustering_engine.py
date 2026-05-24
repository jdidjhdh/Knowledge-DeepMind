import logging
import math
import re
from typing import Optional
from collections import Counter

import numpy as np

from models import ClusteringResult, CategoryType

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "的", "是", "了", "在", "和", "与", "或", "不", "这", "那", "也", "就", "都",
    "对", "及", "把", "被", "从", "以", "而", "且", "但", "所", "为", "其",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "just",
}


class ClusteringEngine:
    def __init__(self, vector_service=None, graph_service=None):
        self.vector_service = vector_service
        self.graph_service = graph_service
        self.hdbscan = None
        self._init_hdbscan()

    def _init_hdbscan(self):
        try:
            import hdbscan
            self.hdbscan = hdbscan
            logger.info("HDBSCAN 聚类引擎初始化成功")
        except ImportError:
            logger.warning("HDBSCAN 未安装，使用 KMeans 回退方案")
            self.hdbscan = None

    async def structural_cluster(
        self,
        knowledge_points: list[dict],
        existing_categories: list[dict],
    ) -> dict[str, list[str]]:
        """第一阶段：基于图谱实体类型的结构分类"""
        entity_type_groups: dict[str, list[str]] = {
            "技术知识": [],
            "人物关系": [],
            "组织信息": [],
            "事件记录": [],
            "概念理论": [],
            "方法流程": [],
        }
        for kp in knowledge_points:
            entities = kp.get("related_entities", [])
            fact = kp.get("fact", "")
            kp_id = kp.get("id", "")
            if not kp_id:
                continue
            category = kp.get("category", "待验证")
            cat_map = {
                "方法": "方法流程",
                "技术": "技术知识",
                "概念": "概念理论",
                "事实": "事件记录",
                "观点": "概念理论",
            }
            mapped = cat_map.get(category, "概念理论")
            if mapped in entity_type_groups:
                entity_type_groups[mapped].append(kp_id)
        for group_name in list(entity_type_groups.keys()):
            if len(entity_type_groups[group_name]) < 3:
                entity_type_groups.pop(group_name)
        return entity_type_groups

    async def semantic_cluster(
        self,
        vectors: list[list[float]],
        min_cluster_size: int = 5,
        min_samples: int = 3,
    ) -> ClusteringResult:
        """第二阶段：HDBSCAN 语义微调聚类"""
        if not vectors or len(vectors) < min_cluster_size:
            return ClusteringResult(
                num_clusters=0,
                labels=[-1] * len(vectors),
                noise_count=len(vectors),
                cluster_keywords={},
                cluster_sizes={},
                suggested_names={},
            )
        try:
            vectors_np = np.array(vectors, dtype=np.float64)
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors_np = vectors_np / norms
        except Exception:
            vectors_np = np.array(vectors, dtype=np.float64)

        if self.hdbscan:
            try:
                clusterer = self.hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_epsilon=0.5,
                )
                labels = clusterer.fit_predict(vectors_np)
                labels = labels.tolist()
            except Exception as e:
                logger.warning(f"HDBSCAN 聚类失败: {e}, 使用 KMeans 回退")
                labels = self._kmeans_fallback(vectors_np, min_cluster_size)
        else:
            labels = self._kmeans_fallback(vectors_np, min_cluster_size)

        unique_labels = [l for l in set(labels) if l >= 0]
        num_clusters = len(unique_labels)
        noise_count = sum(1 for l in labels if l < 0)

        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = sum(1 for l in labels if l == label)

        cluster_keywords = {}
        suggested_names = {}
        for label in unique_labels:
            cluster_keywords[int(label)] = [f"聚类-{label}"]
            suggested_names[int(label)] = f"新兴主题 {label + 1}"

        return ClusteringResult(
            num_clusters=num_clusters,
            labels=labels,
            noise_count=noise_count,
            cluster_keywords=cluster_keywords,
            cluster_sizes=cluster_sizes,
            suggested_names=suggested_names,
        )

    def _kmeans_fallback(self, vectors_np, min_cluster_size: int) -> list[int]:
        try:
            from sklearn.cluster import KMeans
            n = len(vectors_np)
            n_clusters = max(2, min(n // min_cluster_size, 15))
            if n_clusters < 2:
                return [0] * n
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors_np)
            return labels.tolist()
        except Exception as e:
            logger.warning(f"KMeans 回退失败: {e}")
            return [0] * len(vectors_np)

    async def boundary_optimize(
        self,
        knowledge_vectors: dict[str, list[float]],
        category_centroids: dict[str, list[float]],
        threshold: float = 0.1,
    ) -> dict[str, list[str]]:
        """第三阶段：边界优化，支持多归属"""
        multi_assignments: dict[str, list[str]] = {}
        for kp_id, vec in knowledge_vectors.items():
            similarities = []
            for cat_id, centroid in category_centroids.items():
                sim = self._cosine(vec, centroid)
                similarities.append((cat_id, sim))
            if not similarities:
                continue
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_id, best_sim = similarities[0]
            assignments = [best_id]
            for cat_id, sim in similarities[1:]:
                if best_sim - sim < threshold:
                    assignments.append(cat_id)
                else:
                    break
            if len(assignments) > 1:
                multi_assignments[kp_id] = assignments
        return multi_assignments

    def _cosine(self, a: list[float], b: list[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        dot = np.dot(a_np, b_np)
        na = np.linalg.norm(a_np)
        nb = np.linalg.norm(b_np)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    async def compute_category_centroid(
        self,
        knowledge_points: list[dict],
    ) -> Optional[list[float]]:
        """计算一组知识点的平均向量作为分类中心"""
        vectors = []
        for kp in knowledge_points:
            vec = kp.get("vector")
            if vec:
                vectors.append(vec)
        if not vectors:
            return None
        avg = np.mean(vectors, axis=0).tolist()
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = (np.array(avg) / norm).tolist()
        return avg

    async def extract_keywords(self, texts: list[str], top_n: int = 5) -> list[str]:
        """从文本列表中提取关键词"""
        word_counter = Counter()
        for text in texts:
            words = re.findall(r"[\u4e00-\u9fff\w]+", text.lower())
            for word in words:
                if len(word) >= 2 and word not in STOP_WORDS:
                    word_counter[word] += 1
        return [word for word, _ in word_counter.most_common(top_n)]

    async def compute_internal_diversity(
        self,
        vectors: list[list[float]],
    ) -> float:
        """计算簇内语义差异度，用于判断是否需要分裂"""
        if len(vectors) < 2:
            return 0.0
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = self._cosine(vectors[i], vectors[j])
                similarities.append(sim)
        if not similarities:
            return 0.0
        return 1.0 - float(np.mean(similarities))