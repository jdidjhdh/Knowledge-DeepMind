import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

from models import (
    Category, CategoryType, CategoryRelation, CategoryRelationType,
    EvolutionAction, CategoryEvolutionEvent, ClusteringResult,
)

logger = logging.getLogger(__name__)

SPLIT_THRESHOLD = 200
MERGE_SIMILARITY_THRESHOLD = 0.85
MERGE_IDLE_DAYS = 10
NEW_CLUSTER_MIN_SIZE = 15
ARCHIVE_IDLE_DAYS = 90


class EvolutionManager:
    def __init__(self, vector_service=None, graph_service=None):
        self.vector_service = vector_service
        self.graph_service = graph_service

    async def check_evolution_triggers(
        self,
        categories: list[Category],
        category_members: dict[str, list[dict]],
        clustering_engine,
    ) -> list[CategoryEvolutionEvent]:
        """检查所有进化触发条件，返回建议的进化事件列表"""
        events = []

        for cat in categories:
            if cat.is_frozen or cat.is_archived:
                continue

            members = category_members.get(cat.id, [])
            member_count = len(members)

            split_event = await self._check_split(cat, members, clustering_engine)
            if split_event:
                events.append(split_event)

            if cat.category_type == CategoryType.STRUCTURAL:
                archive_event = await self._check_archive(cat)
                if archive_event:
                    events.append(archive_event)

        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat_a = categories[i]
                cat_b = categories[j]
                if cat_a.is_frozen or cat_b.is_frozen:
                    continue
                if cat_a.is_archived or cat_b.is_archived:
                    continue
                merge_event = await self._check_merge(
                    cat_a, cat_b, categories, clustering_engine
                )
                if merge_event:
                    events.append(merge_event)

        new_category_event = await self._check_new_category(
            categories, category_members, clustering_engine
        )
        if new_category_event:
            events.append(new_category_event)

        return events

    async def _check_split(
        self,
        category: Category,
        members: list[dict],
        clustering_engine,
    ) -> Optional[CategoryEvolutionEvent]:
        """检查是否需要分裂"""
        if len(members) < SPLIT_THRESHOLD:
            return None

        vectors = [m.get("vector") for m in members if m.get("vector")]
        if len(vectors) < SPLIT_THRESHOLD:
            return None

        diversity = await clustering_engine.compute_internal_diversity(vectors)
        if diversity < 0.35:
            return None

        result = await clustering_engine.semantic_cluster(
            vectors, min_cluster_size=8, min_samples=3
        )
        valid_clusters = [
            l for l in set(result.labels)
            if l >= 0 and result.cluster_sizes.get(int(l), 0) >= 8
        ]
        if len(valid_clusters) < 2:
            return None

        return CategoryEvolutionEvent(
            category_id=category.id,
            action=EvolutionAction.SPLIT,
            details={
                "category_name": category.name,
                "member_count": len(members),
                "internal_diversity": diversity,
                "suggested_subcategories": result.suggested_names,
                "sub_cluster_sizes": {
                    str(k): v for k, v in result.cluster_sizes.items()
                },
            },
        )

    async def _check_merge(
        self,
        cat_a: Category,
        cat_b: Category,
        all_categories: list[Category],
        clustering_engine,
    ) -> Optional[CategoryEvolutionEvent]:
        """检查两个分类是否需要合并"""
        if cat_a.semantic_vector and cat_b.semantic_vector:
            sim = clustering_engine._cosine(
                cat_a.semantic_vector, cat_b.semantic_vector
            )
        else:
            sim = 0.0
            if cat_a.metadata.get("keywords") and cat_b.metadata.get("keywords"):
                keywords_a = set(cat_a.metadata["keywords"])
                keywords_b = set(cat_b.metadata["keywords"])
                if keywords_a and keywords_b:
                    sim = len(keywords_a & keywords_b) / max(
                        len(keywords_a | keywords_b), 1
                    )
        if sim < MERGE_SIMILARITY_THRESHOLD:
            return None

        updated_a = cat_a.updated_at
        updated_b = cat_b.updated_at
        now = datetime.now()
        idle_a = (now - updated_a).days if updated_a else 999
        idle_b = (now - updated_b).days if updated_b else 999
        if idle_a < MERGE_IDLE_DAYS and idle_b < MERGE_IDLE_DAYS:
            return None

        return CategoryEvolutionEvent(
            action=EvolutionAction.MERGE,
            details={
                "category_a": {"id": cat_a.id, "name": cat_a.name},
                "category_b": {"id": cat_b.id, "name": cat_b.name},
                "similarity": sim,
                "idle_days_a": idle_a,
                "idle_days_b": idle_b,
            },
        )

    async def _check_archive(self, category: Category) -> Optional[CategoryEvolutionEvent]:
        """检查分类是否应该归档"""
        now = datetime.now()
        last_accessed = category.last_accessed_at
        if not last_accessed:
            return None
        days_idle = (now - last_accessed).days
        if days_idle < ARCHIVE_IDLE_DAYS:
            return None
        if not category.updated_at:
            return None
        days_since_update = (now - category.updated_at).days
        if days_since_update < ARCHIVE_IDLE_DAYS:
            return None
        if category.avg_confidence < 0.6:
            return None

        return CategoryEvolutionEvent(
            category_id=category.id,
            action=EvolutionAction.ARCHIVE,
            details={
                "category_name": category.name,
                "days_idle": days_idle,
                "days_since_update": days_since_update,
                "avg_confidence": category.avg_confidence,
            },
        )

    async def _check_new_category(
        self,
        categories: list[Category],
        category_members: dict[str, list[dict]],
        clustering_engine,
    ) -> Optional[CategoryEvolutionEvent]:
        """检查是否有未分类知识需要创建新分类"""
        unclassified = category_members.get("__unclassified__", [])
        if len(unclassified) < NEW_CLUSTER_MIN_SIZE:
            return None

        vectors = [m.get("vector") for m in unclassified if m.get("vector")]
        if len(vectors) < NEW_CLUSTER_MIN_SIZE:
            return None

        result = await clustering_engine.semantic_cluster(
            vectors, min_cluster_size=NEW_CLUSTER_MIN_SIZE, min_samples=5
        )
        if result.num_clusters == 0:
            return None

        texts = [m.get("data", {}).get("fact", "") for m in unclassified]
        keywords_by_cluster = {}
        for label in set(result.labels):
            if label < 0:
                continue
            cluster_texts = [
                texts[i] for i, l in enumerate(result.labels) if l == label
            ]
            keywords = await clustering_engine.extract_keywords(cluster_texts)
            keywords_by_cluster[int(label)] = keywords

        suggested_names = {}
        for label, kw in keywords_by_cluster.items():
            if kw:
                suggested_names[label] = " · ".join(kw[:3])

        return CategoryEvolutionEvent(
            action=EvolutionAction.CREATE,
            details={
                "unclassified_count": len(unclassified),
                "num_new_clusters": result.num_clusters,
                "suggested_names": suggested_names,
                "cluster_sizes": result.cluster_sizes,
            },
        )

    async def execute_split(
        self,
        category_id: str,
        sub_cluster_assignments: dict[int, list[str]],
        sub_cluster_names: dict[int, str],
    ) -> list[Category]:
        """执行分裂：创建子分类并重新分配成员"""
        new_categories = []
        for label, member_ids in sub_cluster_assignments.items():
            name = sub_cluster_names.get(label, f"子分类 {label + 1}")
            new_cat = Category(
                id=str(uuid.uuid4()),
                name=name,
                parent_id=category_id,
                category_type=CategoryType.STRUCTURAL,
                level=1,
                metadata={"split_from": category_id, "split_label": label},
            )
            new_categories.append(new_cat)
        return new_categories

    async def execute_merge(
        self,
        cat_id_a: str,
        cat_id_b: str,
        new_name: Optional[str] = None,
    ) -> Category:
        """执行合并：合并两个分类为一个"""
        merged = Category(
            id=str(uuid.uuid4()),
            name=new_name or "合并分类",
            category_type=CategoryType.STRUCTURAL,
            metadata={
                "merged_from": [cat_id_a, cat_id_b],
                "merged_at": datetime.now().isoformat(),
            },
        )
        return merged

    async def execute_archive(self, category_id: str) -> None:
        """执行归档"""
        pass

    async def freeze_category(self, category_id: str) -> None:
        """冻结分类，阻止自动进化"""
        pass

    async def unfreeze_category(self, category_id: str) -> None:
        """解冻分类"""
        pass