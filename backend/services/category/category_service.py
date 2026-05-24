import json
import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

from openai import AsyncOpenAI

from config import Settings
from models import (
    Category, CategoryType, CategoryRelation, CategoryRelationType,
    KnowledgeCategoryAssignment, UserTag, KnowledgeTagAssignment,
    SmartCollection, UserCategoryPrefs, CategoryTree, CategoryHealth,
    MultiDimensionFilter, CategoryEvolutionEvent, ClusteringResult,
    KnowledgeTimeline, SourceGroup, SourceComparison,
    ConfidenceTier, EvolutionAction,
)
from .clustering_engine import ClusteringEngine
from .evolution_manager import EvolutionManager
from .time_extractor import extract_event_times, standardize_event_time, build_timeline_groups
from .source_classifier import SourceClassifier

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURAL_CATEGORIES = [
    {"name": "技术知识", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "技术方法、工具、编程语言等"},
    {"name": "概念理论", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "概念定义、理论框架、学术知识"},
    {"name": "事件记录", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "历史事件、项目里程碑、时间记录"},
    {"name": "人物关系", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "人物信息、组织内关系"},
    {"name": "组织信息", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "公司、机构、团队信息"},
    {"name": "方法流程", "type": CategoryType.STRUCTURAL, "level": 0,
     "description": "方法论、流程、步骤指南"},
]

DEFAULT_META_CATEGORIES = [
    {"name": "⚠️ 待验证知识库", "type": CategoryType.META, "level": 0,
     "description": "置信度低于阈值的知识点，需要定期审查",
     "metadata": {"confidence_max": 0.4}},
    {"name": "✅ 已验证知识", "type": CategoryType.META, "level": 0,
     "description": "置信度较高的可靠知识",
     "metadata": {"confidence_min": 0.7}},
    {"name": "📅 时间归档", "type": CategoryType.TEMPORAL, "level": 0,
     "description": "按事件时间组织的知识"},
    {"name": "📂 按来源浏览", "type": CategoryType.SOURCE, "level": 0,
     "description": "按来源文件组织知识"},
]

CATEGORY_SUGGESTION_PROMPT = """你是一个知识分类专家。请为以下知识点推荐最合适的分类。

现有分类列表：
{categories}

新知识点：
{fact}

请推荐1-3个最合适的分类（从现有分类中选择），并给出推荐理由和置信度。
只输出JSON数组，格式：
[{{"category_name": "分类名", "confidence": 0.85, "reason": "理由"}}]

如果没有合适的现有分类，可以建议新建分类：
[{{"category_name": "新建: 建议的分类名", "confidence": 0.6, "reason": "理由"}}]"""


class CategoryService:
    def __init__(self, settings: Settings, vector_service=None, graph_service=None, confidence_calculator=None):
        self.settings = settings
        self.vector_service = vector_service
        self.graph_service = graph_service
        self.confidence_calculator = confidence_calculator
        self.clustering_engine = ClusteringEngine(vector_service, graph_service)
        self.evolution_manager = EvolutionManager(vector_service, graph_service)
        self.source_classifier = SourceClassifier(vector_service)

        self._categories: dict[str, Category] = {}
        self._knowledge_category: dict[str, list[str]] = defaultdict(list)
        self._category_relations: list[CategoryRelation] = []
        self._tags: dict[str, UserTag] = {}
        self._knowledge_tags: dict[str, list[str]] = defaultdict(list)
        self._smart_collections: dict[str, SmartCollection] = {}
        self._user_prefs: dict[str, dict[str, UserCategoryPrefs]] = {}
        self._evolution_log: list[CategoryEvolutionEvent] = []

        self._data_file = "vector_store/categories.json"

        self.llm_client = None
        if settings.deepseek_api_key:
            self.llm_client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

    async def initialize(self):
        self._load()
        if not self._categories:
            await self._init_default_categories()
        logger.info(f"分类服务初始化完成: {len(self._categories)} 个分类")

    async def _init_default_categories(self):
        for cat_data in DEFAULT_STRUCTURAL_CATEGORIES + DEFAULT_META_CATEGORIES:
            cat_id = str(uuid.uuid4())
            cat = Category(
                id=cat_id,
                name=cat_data["name"],
                description=cat_data.get("description"),
                category_type=cat_data["type"],
                level=cat_data.get("level", 0),
                metadata=cat_data.get("metadata", {}),
            )
            self._categories[cat_id] = cat

        struct_ids = [
            cid for cid, c in self._categories.items()
            if c.category_type == CategoryType.STRUCTURAL
        ]
        for i in range(len(struct_ids)):
            for j in range(i + 1, len(struct_ids)):
                self._category_relations.append(CategoryRelation(
                    source_category_id=struct_ids[i],
                    target_category_id=struct_ids[j],
                    relation_type=CategoryRelationType.RELATED_TO,
                    weight=0.3,
                ))
        self._save()
        logger.info(f"初始化 {len(self._categories)} 个默认分类")

    def _load(self):
        try:
            import os
            if os.path.exists(self._data_file):
                with open(self._data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._categories = {
                    k: Category(**v) for k, v in data.get("categories", {}).items()
                }
                self._knowledge_category = defaultdict(
                    list, data.get("knowledge_category", {})
                )
                self._category_relations = [
                    CategoryRelation(**r)
                    for r in data.get("category_relations", [])
                ]
                self._tags = {
                    k: UserTag(**v) for k, v in data.get("tags", {}).items()
                }
                self._knowledge_tags = defaultdict(
                    list, data.get("knowledge_tags", {})
                )
                self._smart_collections = {
                    k: SmartCollection(**v)
                    for k, v in data.get("smart_collections", {}).items()
                }
                self._user_prefs = {}
                for uid, prefs in data.get("user_prefs", {}).items():
                    self._user_prefs[uid] = {
                        cid: UserCategoryPrefs(**p)
                        for cid, p in prefs.items()
                    }
        except Exception as e:
            logger.warning(f"加载分类数据失败: {e}")
            self._categories = {}

    def _save(self):
        try:
            import os
            os.makedirs("vector_store", exist_ok=True)
            data = {
                "categories": {
                    k: v.model_dump() for k, v in self._categories.items()
                },
                "knowledge_category": dict(self._knowledge_category),
                "category_relations": [
                    r.model_dump() for r in self._category_relations
                ],
                "tags": {k: v.model_dump() for k, v in self._tags.items()},
                "knowledge_tags": dict(self._knowledge_tags),
                "smart_collections": {
                    k: v.model_dump()
                    for k, v in self._smart_collections.items()
                },
                "user_prefs": {
                    uid: {cid: p.model_dump() for cid, p in prefs.items()}
                    for uid, prefs in self._user_prefs.items()
                },
            }
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
        except Exception as e:
            logger.error(f"保存分类数据失败: {e}")

    # ================================================================
    #  分类 CRUD
    # ================================================================

    def get_all_categories(
        self,
        category_type: Optional[CategoryType] = None,
        include_archived: bool = False,
    ) -> list[Category]:
        cats = list(self._categories.values())
        if category_type:
            cats = [c for c in cats if c.category_type == category_type]
        if not include_archived:
            cats = [c for c in cats if not c.is_archived]
        return cats

    def get_category(self, category_id: str) -> Optional[Category]:
        return self._categories.get(category_id)

    def create_category(self, category: Category) -> Category:
        if not category.id:
            category.id = str(uuid.uuid4())
        category.created_at = category.created_at or datetime.now()
        category.updated_at = datetime.now()
        self._categories[category.id] = category
        if category.parent_id and category.parent_id in self._categories:
            self._category_relations.append(CategoryRelation(
                source_category_id=category.id,
                target_category_id=category.parent_id,
                relation_type=CategoryRelationType.CHILD_OF,
            ))
        self._save()
        self._sync_category_to_graph(category)
        return category

    def update_category(self, category_id: str, updates: dict) -> Optional[Category]:
        cat = self._categories.get(category_id)
        if not cat:
            return None
        for key, value in updates.items():
            if hasattr(cat, key):
                setattr(cat, key, value)
        cat.updated_at = datetime.now()
        self._categories[category_id] = cat
        self._save()
        self._sync_category_to_graph(cat)
        return cat

    def delete_category(self, category_id: str) -> bool:
        if category_id not in self._categories:
            return False
        del self._categories[category_id]
        self._knowledge_category.pop(category_id, None)
        self._category_relations = [
            r for r in self._category_relations
            if r.source_category_id != category_id
            and r.target_category_id != category_id
        ]
        self._save()
        self._sync_category_removal_to_graph(category_id)
        return True

    # ================================================================
    #  多对多知识-分类关联
    # ================================================================

    def assign_knowledge_to_categories(
        self,
        knowledge_id: str,
        category_ids: list[str],
        primary_category_id: Optional[str] = None,
        is_auto: bool = True,
    ) -> list[KnowledgeCategoryAssignment]:
        assignments = []
        for cat_id in category_ids:
            if cat_id not in self._categories:
                continue
            if cat_id not in self._knowledge_category.get(knowledge_id, []):
                self._knowledge_category[knowledge_id].append(cat_id)
            assignments.append(KnowledgeCategoryAssignment(
                knowledge_id=knowledge_id,
                category_id=cat_id,
                is_primary=(cat_id == primary_category_id),
                is_auto_assigned=is_auto,
                assigned_at=datetime.now(),
            ))
        self._update_category_counts()
        self._save()
        for cat_id in category_ids:
            if cat_id in self._categories:
                self._sync_knowledge_category_to_graph(knowledge_id, cat_id, is_auto)
        return assignments

    def remove_knowledge_from_category(
        self,
        knowledge_id: str,
        category_id: str,
    ) -> bool:
        if knowledge_id in self._knowledge_category:
            if category_id in self._knowledge_category[knowledge_id]:
                self._knowledge_category[knowledge_id].remove(category_id)
                self._update_category_counts()
                self._save()
                return True
        return False

    def get_knowledge_categories(self, knowledge_id: str) -> list[Category]:
        cat_ids = self._knowledge_category.get(knowledge_id, [])
        return [self._categories[cid] for cid in cat_ids if cid in self._categories]

    def get_category_members(self, category_id: str) -> list[str]:
        members = []
        for kid, cat_ids in self._knowledge_category.items():
            if category_id in cat_ids:
                members.append(kid)
        return members

    def _update_category_counts(self):
        counts: dict[str, int] = defaultdict(int)
        conf_sums: dict[str, float] = defaultdict(float)
        for kid, cat_ids in self._knowledge_category.items():
            for cid in cat_ids:
                counts[cid] += 1
        for cid, cat in self._categories.items():
            cat.knowledge_count = counts.get(cid, 0)
            if cat.knowledge_count > 0:
                cat.avg_confidence = conf_sums.get(cid, 2.5) / cat.knowledge_count
            cat.updated_at = datetime.now()

    # ================================================================
    #  图谱同步 (Neo4j)
    # ================================================================

    def _sync_category_to_graph(self, category: Category):
        if not self.graph_service or not self.graph_service.driver:
            return
        try:
            props = {
                "name": category.name,
                "description": category.description or "",
                "user_id": category.user_id or "default",
                "category_id": category.id,
                "category_type": category.category_type.value if hasattr(category.category_type, 'value') else str(category.category_type),
                "knowledge_count": category.knowledge_count,
                "color": category.color,
                "icon": category.icon or "",
                "is_system": category.is_system,
            }
            self.graph_service._run_cypher("""
                MERGE (c:Category {category_id: $category_id})
                SET c.name = $name, c.description = $description,
                    c.user_id = $user_id, c.category_type = $category_type,
                    c.knowledge_count = $knowledge_count, c.color = $color,
                    c.icon = $icon, c.is_system = $is_system,
                    c.updated_at = datetime()
            """, props)
            if category.parent_id:
                self.graph_service._run_cypher("""
                    MATCH (c:Category {category_id: $child_id})
                    MATCH (p:Category {category_id: $parent_id})
                    MERGE (c)-[r:HAS_PARENT]->(p)
                """, {"child_id": category.id, "parent_id": category.parent_id})
        except Exception as e:
            logger.warning(f"同步Category节点到图谱失败: {e}")

    def _sync_category_removal_to_graph(self, category_id: str):
        if not self.graph_service or not self.graph_service.driver:
            return
        try:
            self.graph_service._run_cypher("""
                MATCH (c:Category {category_id: $category_id})
                DETACH DELETE c
            """, {"category_id": category_id})
        except Exception as e:
            logger.warning(f"从图谱删除Category节点失败: {e}")

    def _sync_knowledge_category_to_graph(self, knowledge_id: str, category_id: str, is_auto: bool):
        if not self.graph_service or not self.graph_service.driver:
            return
        try:
            source = "ai" if is_auto else "manual"
            self.graph_service._run_cypher("""
                MATCH (k:KnowledgeAtom {id: $knowledge_id})
                MATCH (c:Category {category_id: $category_id})
                MERGE (k)-[r:BELONGS_TO]->(c)
                SET r.source = $source, r.assigned_at = datetime()
            """, {"knowledge_id": knowledge_id, "category_id": category_id, "source": source})
        except Exception as e:
            logger.warning(f"同步知识-分类关系到图谱失败: {e}")

    def sync_all_categories_to_graph(self) -> int:
        if not self.graph_service or not self.graph_service.driver:
            return 0
        count = 0
        for cat in self._categories.values():
            try:
                self._sync_category_to_graph(cat)
                count += 1
            except Exception:
                pass
        for r in self._category_relations:
            if r.relation_type == CategoryRelationType.CHILD_OF and \
               r.source_category_id in self._categories and \
               r.target_category_id in self._categories:
                try:
                    self.graph_service._run_cypher("""
                        MATCH (c:Category {category_id: $child_id})
                        MATCH (p:Category {category_id: $parent_id})
                        MERGE (c)-[r:HAS_PARENT]->(p)
                    """, {"child_id": r.source_category_id, "parent_id": r.target_category_id})
                except Exception:
                    pass
        return count

    _CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "技术知识": ["技术"],
        "概念理论": ["概念"],
        "事件记录": ["事实", "观点"],
        "人物关系": ["人物"],
        "组织信息": ["组织", "公司"],
        "方法流程": ["方法", "流程"],
    }

    def _match_knowledge_to_category_ids(self, knowledge: dict) -> list[str]:
        cat_field = knowledge.get("category", "")
        confidence = knowledge.get("confidence", 0.5)
        matched: list[str] = []
        for cid, cat_obj in self._categories.items():
            if cat_obj.is_archived:
                continue
            if cat_obj.category_type in (CategoryType.STRUCTURAL, None):
                keywords = self._CATEGORY_KEYWORDS.get(cat_obj.name, [])
                if cat_field in keywords:
                    matched.append(cid)
            elif cat_obj.category_type == CategoryType.META:
                meta = cat_obj.metadata or {}
                if "confidence_max" in meta:
                    if confidence <= float(meta["confidence_max"]) or cat_field == "待验证":
                        matched.append(cid)
                elif "confidence_min" in meta and confidence >= float(meta["confidence_min"]):
                    matched.append(cid)
        return matched

    def refresh_counts_from_data(self, knowledge_items: list[dict]):
        counts: dict[str, int] = defaultdict(int)
        for kp in knowledge_items:
            for cid in self._match_knowledge_to_category_ids(kp):
                counts[cid] += 1
        for cid, cat in self._categories.items():
            cat.knowledge_count = counts.get(cid, 0)
            cat.updated_at = datetime.now()

    # ================================================================
    #  分类关系管理 (DAG)
    # ================================================================

    def add_relation(self, relation: CategoryRelation) -> CategoryRelation:
        existing = [
            r for r in self._category_relations
            if r.source_category_id == relation.source_category_id
            and r.target_category_id == relation.target_category_id
            and r.relation_type == relation.relation_type
        ]
        if existing:
            return existing[0]
        self._category_relations.append(relation)
        self._save()
        return relation

    def get_relations(
        self,
        category_id: str,
        relation_type: Optional[CategoryRelationType] = None,
    ) -> list[CategoryRelation]:
        relations = [
            r for r in self._category_relations
            if r.source_category_id == category_id
            or r.target_category_id == category_id
        ]
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        return relations

    # ================================================================
    #  标签管理
    # ================================================================

    def create_tag(self, tag: UserTag) -> UserTag:
        if not tag.id:
            tag.id = str(uuid.uuid4())
        self._tags[tag.id] = tag
        self._save()
        return tag

    def get_user_tags(self, user_id: str) -> list[UserTag]:
        return [t for t in self._tags.values() if t.user_id == user_id]

    def delete_tag(self, tag_id: str) -> bool:
        if tag_id not in self._tags:
            return False
        del self._tags[tag_id]
        self._knowledge_tags = defaultdict(
            list,
            {k: [tid for tid in v if tid != tag_id]
             for k, v in self._knowledge_tags.items()},
        )
        self._save()
        return True

    def assign_tag(self, assignment: KnowledgeTagAssignment) -> KnowledgeTagAssignment:
        if assignment.tag_id not in self._tags:
            raise ValueError(f"标签不存在: {assignment.tag_id}")
        current = self._knowledge_tags.get(assignment.knowledge_id, [])
        if assignment.tag_id not in current:
            self._knowledge_tags[assignment.knowledge_id].append(assignment.tag_id)
        self._save()
        return assignment

    def remove_tag(self, knowledge_id: str, tag_id: str) -> bool:
        if knowledge_id in self._knowledge_tags and tag_id in self._knowledge_tags[knowledge_id]:
            self._knowledge_tags[knowledge_id].remove(tag_id)
            self._save()
            return True
        return False

    def get_knowledge_tags(self, knowledge_id: str) -> list[UserTag]:
        tag_ids = self._knowledge_tags.get(knowledge_id, [])
        return [self._tags[tid] for tid in tag_ids if tid in self._tags]

    # ================================================================
    #  智能集合
    # ================================================================

    def create_smart_collection(self, collection: SmartCollection) -> SmartCollection:
        if not collection.id:
            collection.id = str(uuid.uuid4())
        self._smart_collections[collection.id] = collection
        self._save()
        return collection

    def get_smart_collections(self, user_id: Optional[str] = None) -> list[SmartCollection]:
        collections = list(self._smart_collections.values())
        if user_id:
            collections = [
                c for c in collections
                if c.user_id == user_id or c.is_system
            ]
        return collections

    def evaluate_smart_collection(
        self,
        collection_id: str,
        all_knowledge: dict[str, dict],
    ) -> list[str]:
        collection = self._smart_collections.get(collection_id)
        if not collection:
            return []
        rules = collection.filter_rules
        result_ids = []
        for kid, kp in all_knowledge.items():
            if self._matches_rules(kp, rules):
                result_ids.append(kid)
        return result_ids

    def _matches_rules(self, kp: dict, rules: dict) -> bool:
        if "confidence_min" in rules:
            if kp.get("confidence", 0) < rules["confidence_min"]:
                return False
        if "confidence_max" in rules:
            if kp.get("confidence", 1) > rules["confidence_max"]:
                return False
        if "days_within" in rules:
            created = kp.get("created_at")
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if created and (datetime.now() - created).days > rules["days_within"]:
                return False
        if "category_ids" in rules:
            kid = kp.get("id", "")
            if kid:
                kid_cats = set(self._knowledge_category.get(kid, []))
                if not kid_cats.intersection(set(rules["category_ids"])):
                    return False
        if "tag_ids" in rules:
            kid = kp.get("id", "")
            if kid:
                kid_tags = set(self._knowledge_tags.get(kid, []))
                if not kid_tags.intersection(set(rules["tag_ids"])):
                    return False
        if "source_contains" in rules:
            source = kp.get("source", "")
            if rules["source_contains"] not in source:
                return False
        return True

    # ================================================================
    #  分类树构建 (DAG -> Tree)
    # ================================================================

    def build_category_tree(
        self,
        user_id: Optional[str] = None,
        root_id: Optional[str] = None,
    ) -> list[CategoryTree]:
        cats = self.get_all_categories(include_archived=False)
        children_map: dict[str, list[Category]] = defaultdict(list)
        root_cats = []
        for cat in cats:
            if cat.parent_id and cat.parent_id in self._categories:
                children_map[cat.parent_id].append(cat)
            else:
                root_cats.append(cat)

        if root_id and root_id in self._categories:
            root_cats = [self._categories[root_id]]

        user_prefs_map = self._user_prefs.get(user_id or "default", {})

        def build_node(cat: Category) -> CategoryTree:
            children = [build_node(c) for c in children_map.get(cat.id, [])]
            relations = self.get_relations(cat.id)
            return CategoryTree(
                id=cat.id or "",
                name=cat.name,
                description=cat.description,
                category_type=cat.category_type,
                level=cat.level,
                knowledge_count=cat.knowledge_count,
                avg_confidence=cat.avg_confidence,
                is_archived=cat.is_archived,
                is_frozen=cat.is_frozen,
                children=children,
                relations=relations,
            )

        tree = [build_node(cat) for cat in root_cats]
        return tree

    # ================================================================
    #  多维度筛选
    # ================================================================

    def filter_knowledge(
        self,
        flt: MultiDimensionFilter,
        all_knowledge: dict[str, dict],
    ) -> list[dict]:
        results = []
        for kid, kp in all_knowledge.items():
            if not self._matches_filter(flt, kp, kid):
                continue
            results.append(kp)

        reverse = flt.sort_order == "desc"
        sort_key = flt.sort_by
        if sort_key == "confidence":
            results.sort(key=lambda x: x.get("confidence", 0.5), reverse=reverse)
        elif sort_key == "created_at":
            results.sort(
                key=lambda x: str(x.get("created_at", "")),
                reverse=reverse,
            )
        elif sort_key == "updated_at":
            results.sort(
                key=lambda x: str(x.get("updated_at", "")),
                reverse=reverse,
            )

        total = len(results)
        return results[flt.offset:flt.offset + flt.limit]

    def _matches_filter(
        self,
        flt: MultiDimensionFilter,
        kp: dict,
        kid: str,
    ) -> bool:
        if flt.category_ids:
            kid_cats = set(self._knowledge_category.get(kid, []))
            if not kid_cats.intersection(set(flt.category_ids)):
                return False
        if flt.tag_ids:
            kid_tags = set(self._knowledge_tags.get(kid, []))
            if not kid_tags.intersection(set(flt.tag_ids)):
                return False
        conf = kp.get("confidence", 0.5)
        if flt.confidence_min is not None and conf < flt.confidence_min:
            return False
        if flt.confidence_max is not None and conf > flt.confidence_max:
            return False
        if flt.confidence_tier:
            if flt.confidence_tier == ConfidenceTier.VERIFIED and conf < 0.7:
                return False
            if flt.confidence_tier == ConfidenceTier.PENDING and (conf < 0.4 or conf >= 0.7):
                return False
            if flt.confidence_tier == ConfidenceTier.DOUBTFUL and conf >= 0.4:
                return False
        if flt.source_type and kp.get("source", "") != flt.source_type:
            return False
        if flt.search_text:
            fact = kp.get("fact", "")
            if flt.search_text.lower() not in fact.lower():
                return False
        if flt.status and kp.get("status", "active") != flt.status:
            return False
        if flt.date_from or flt.date_to:
            created = kp.get("created_at")
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if flt.date_from and created and created < flt.date_from:
                return False
            if flt.date_to and created and created > flt.date_to:
                return False
        return True

    # ================================================================
    #  混合聚类引擎
    # ================================================================

    async def run_hybrid_clustering(
        self,
        knowledge_points: list[dict],
    ) -> dict:
        """三阶段混合聚类"""
        structural_groups = await self.clustering_engine.structural_cluster(
            knowledge_points,
            [c.model_dump() for c in self._categories.values()],
        )

        vectors = []
        remaining_kps = []
        assigned_ids = set()
        for group_ids in structural_groups.values():
            assigned_ids.update(group_ids)
        for kp in knowledge_points:
            kid = kp.get("id", "")
            if kid not in assigned_ids and kp.get("vector"):
                vectors.append(kp["vector"])
                remaining_kps.append(kp)
        semantic_result = None
        if len(remaining_kps) >= 5:
            semantic_result = await self.clustering_engine.semantic_cluster(
                vectors, min_cluster_size=5, min_samples=3
            )

        category_centroids = {}
        for cid, cat in self._categories.items():
            if cat.semantic_vector:
                category_centroids[cid] = cat.semantic_vector
        kp_vectors = {
            kp.get("id", ""): kp.get("vector")
            for kp in knowledge_points
            if kp.get("vector")
        }
        boundary_assignments = {}
        if category_centroids and kp_vectors:
            boundary_assignments = await self.clustering_engine.boundary_optimize(
                kp_vectors, category_centroids
            )

        return {
            "structural_groups": structural_groups,
            "semantic_clusters": (
                semantic_result.model_dump() if semantic_result else None
            ),
            "boundary_multi_assignments": boundary_assignments,
        }

    # ================================================================
    #  大模型辅助分类建议
    # ================================================================

    async def suggest_categories(
        self,
        fact: str,
        max_suggestions: int = 3,
    ) -> list[dict]:
        """使用 DeepSeek 推荐分类"""
        cats = self.get_all_categories(CategoryType.STRUCTURAL)
        if not cats:
            return []
        cat_names = "\n".join(
            [f"- ID:{c.id} | {c.name}: {c.description or ''}" for c in cats]
        )
        if not self.llm_client:
            return self._fallback_suggest(fact, cats)
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[
                    {"role": "system", "content": "你是知识分类专家，只输出JSON数组。"},
                    {"role": "user", "content": CATEGORY_SUGGESTION_PROMPT.format(
                        categories=cat_names, fact=fact[:500],
                    )},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            result_text = response.choices[0].message.content or ""
            json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
                result = []
                for s in suggestions[:max_suggestions]:
                    name = s.get("category_name", "")
                    if name.startswith("新建: "):
                        result.append({
                            "category_name": name[4:],
                            "confidence": s.get("confidence", 0.6),
                            "reason": s.get("reason", ""),
                            "is_new": True,
                        })
                    else:
                        matched = next(
                            (c for c in cats if c.name == name), None
                        )
                        result.append({
                            "category_id": matched.id if matched else None,
                            "category_name": name,
                            "confidence": s.get("confidence", 0.5),
                            "reason": s.get("reason", ""),
                            "is_new": False,
                        })
                return result
        except Exception as e:
            logger.warning(f"LLM 分类建议失败: {e}")
        return self._fallback_suggest(fact, cats)

    def _fallback_suggest(self, fact: str, categories: list[Category]) -> list[dict]:
        keywords = {
            "技术": ["技术", "算法", "编程", "代码", "架构", "系统", "API", "框架"],
            "概念理论": ["定义", "理论", "概念", "原理", "模型", "定理", "定律"],
            "事件记录": ["发生", "举办", "发布", "会议", "事件", "历史"],
            "人物关系": ["创始人", "CEO", "作者", "领导", "经理", "教授"],
            "组织信息": ["公司", "企业", "机构", "部门", "团队", "组织"],
            "方法流程": ["步骤", "方法", "流程", "指南", "实践", "策略"],
        }
        scores = []
        for cat in categories:
            kw_list = keywords.get(cat.name, [])
            score = sum(1 for kw in kw_list if kw in fact)
            if score > 0:
                scores.append({
                    "category_id": cat.id,
                    "category_name": cat.name,
                    "confidence": min(0.8, 0.4 + score * 0.15),
                    "reason": "关键词匹配",
                    "is_new": False,
                })
        scores.sort(key=lambda x: x["confidence"], reverse=True)
        return scores[:3]

    # ================================================================
    #  自动分类集成
    # ================================================================

    async def auto_categorize_knowledge(
        self,
        knowledge_id: str,
        fact: str,
        confidence_threshold: float = 0.7,
        auto_create: bool = False,
    ) -> dict:
        suggestions = await self.suggest_categories(fact, max_suggestions=3)
        assigned = []
        suggested_new = None
        for s in suggestions:
            if s.get("is_new"):
                if auto_create:
                    new_cat = Category(
                        name=s["category_name"],
                        description=f"AI自动创建: {s.get('reason', '')}",
                        category_type=CategoryType.STRUCTURAL,
                        metadata={"source": "ai", "auto_created": True},
                    )
                    new_cat = self.create_category(new_cat)
                    self.assign_knowledge_to_categories(knowledge_id, [new_cat.id], is_auto=True)
                    assigned.append(new_cat.id)
                    suggested_new = s["category_name"]
                else:
                    suggested_new = s["category_name"]
            elif s.get("category_id"):
                confidence = s.get("confidence", 0)
                if confidence >= confidence_threshold:
                    self.assign_knowledge_to_categories(knowledge_id, [s["category_id"]], is_auto=True)
                    assigned.append(s["category_id"])
                elif confidence >= 0.35:
                    self.assign_knowledge_to_categories(knowledge_id, [s["category_id"]], is_auto=True)
                    assigned.append(s["category_id"])
        return {
            "knowledge_id": knowledge_id,
            "assigned_categories": assigned,
            "suggested_new": suggested_new,
            "confidence": suggestions[0].get("confidence", 0) if suggestions else 0,
        }

    async def suggest_category_merges(
        self,
        similarity_threshold: float = 0.85,
    ) -> list[dict]:
        cats = self.get_all_categories(CategoryType.STRUCTURAL)
        suggestions = []
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = cats[i], cats[j]
                if a.is_archived or b.is_archived:
                    continue
                if a.name == b.name:
                    suggestions.append({
                        "category_a": a.model_dump(),
                        "category_b": b.model_dump(),
                        "similarity": 1.0,
                        "overlapping_knowledge": [],
                        "suggestion": "名称完全相同，建议合并",
                    })
                    continue
                if a.name in b.name or b.name in a.name:
                    suggestions.append({
                        "category_a": a.model_dump(),
                        "category_b": b.model_dump(),
                        "similarity": 0.9,
                        "overlapping_knowledge": [],
                        "suggestion": "名称高度相似，建议确认是否合并",
                    })
        return suggestions

    def batch_assign_categories(
        self,
        knowledge_ids: list[str],
        category_ids: list[str],
        is_auto: bool = False,
    ) -> int:
        count = 0
        for kid in knowledge_ids:
            try:
                self.assign_knowledge_to_categories(kid, category_ids, is_auto=is_auto)
                count += 1
            except Exception:
                pass
        return count

    def search_categories(self, query: str, limit: int = 20) -> list[Category]:
        q = query.lower()
        results = []
        for cat in self._categories.values():
            if q in cat.name.lower() or (cat.description and q in cat.description.lower()):
                results.append(cat)
            if len(results) >= limit:
                break
        return results

    def get_category_path_to_root(self, category_id: str) -> list[Category]:
        path = []
        current = self._categories.get(category_id)
        visited = set()
        while current and current.id not in visited:
            path.append(current)
            visited.add(current.id)
            if current.parent_id:
                current = self._categories.get(current.parent_id)
            else:
                break
        path.reverse()
        return path

    # ================================================================
    #  分类健康检查
    # ================================================================

    def get_category_health(self) -> list[CategoryHealth]:
        health_list = []
        now = datetime.now()
        for cat in self._categories.values():
            if cat.is_archived:
                continue
            is_stale = False
            if cat.updated_at:
                days_since = (now - cat.updated_at).days
                is_stale = days_since > 30
            needs_split = cat.knowledge_count > 200
            needs_attention = (
                cat.category_type == CategoryType.STRUCTURAL
                and cat.knowledge_count == 0
            )
            health_list.append(CategoryHealth(
                category_id=cat.id or "",
                name=cat.name,
                knowledge_count=cat.knowledge_count,
                avg_confidence=cat.avg_confidence,
                last_updated=cat.updated_at,
                is_stale=is_stale,
                needs_split=needs_split,
                needs_attention=needs_attention,
            ))
        return health_list

    # ================================================================
    #  时间线分类
    # ================================================================

    def build_timeline(
        self,
        knowledge_points: list[dict],
    ) -> list[KnowledgeTimeline]:
        timeline_groups = build_timeline_groups(knowledge_points)
        timelines = []
        for time_key, kid_list in timeline_groups.items():
            parts = time_key.split("-")
            year = int(parts[0]) if parts[0].isdigit() else 0
            month = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            label = f"{year}年"
            if month:
                label = f"{year}年{month}月"
            timelines.append(KnowledgeTimeline(
                year=year,
                month=month,
                knowledge_count=len(kid_list),
                event_label=label,
                knowledge_ids=kid_list,
            ))
        timelines.sort(key=lambda t: (t.year, t.month or 0), reverse=True)
        return timelines

    # ================================================================
    #  来源分类
    # ================================================================

    def build_source_groups(
        self,
        knowledge_points: list[dict],
    ) -> list[SourceGroup]:
        return self.source_classifier.group_by_source(knowledge_points)

    def find_source_comparisons(
        self,
        knowledge_points: list[dict],
    ) -> list[SourceComparison]:
        groups = self.source_classifier.group_by_source(knowledge_points)
        kp_map = {kp.get("id", ""): kp for kp in knowledge_points if kp.get("id")}
        return self.source_classifier.find_source_conflicts(groups, kp_map)

    # ================================================================
    #  用户个性化
    # ================================================================

    def get_user_category_prefs(self, user_id: str) -> dict[str, UserCategoryPrefs]:
        return self._user_prefs.get(user_id, {})

    def set_user_category_prefs(
        self,
        user_id: str,
        category_id: str,
        prefs: UserCategoryPrefs,
    ):
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = {}
        self._user_prefs[user_id][category_id] = prefs
        self._save()

    def record_category_visit(self, user_id: str, category_id: str):
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = {}
        if category_id not in self._user_prefs[user_id]:
            self._user_prefs[user_id][category_id] = UserCategoryPrefs(
                user_id=user_id,
                category_id=category_id,
            )
        prefs = self._user_prefs[user_id][category_id]
        prefs.visit_count += 1
        prefs.last_visited_at = datetime.now()
        if category_id in self._categories:
            self._categories[category_id].last_accessed_at = datetime.now()
        self._save()

    def get_personalized_tree(
        self,
        user_id: str,
        focused_topics: Optional[list[str]] = None,
        focus_mode: bool = False,
    ) -> list[CategoryTree]:
        """根据用户画像构建个性化分类树"""
        raw_tree = self.build_category_tree(user_id)
        prefs = self.get_user_category_prefs(user_id)
        if not focused_topics:
            focused_topics = []
        if focus_mode and focused_topics:
            return self._apply_focus_mode(raw_tree, focused_topics)
        return self._apply_personalization(raw_tree, prefs)

    def _apply_personalization(
        self,
        tree: list[CategoryTree],
        prefs: dict[str, UserCategoryPrefs],
    ) -> list[CategoryTree]:
        def sort_node(node: CategoryTree) -> float:
            p = prefs.get(node.id)
            if p:
                return p.visit_count
            return 0

        result = []
        for node in tree:
            node.children = self._apply_personalization(node.children, prefs)
            result.append(node)

        result.sort(key=sort_node, reverse=True)

        for p in prefs.values():
            if not p.is_visible:
                result = [
                    n for n in result
                    if n.id != p.category_id
                ]

        return result

    def _apply_focus_mode(
        self,
        tree: list[CategoryTree],
        focused_topics: list[str],
    ) -> list[CategoryTree]:
        relevant = []
        for node in tree:
            is_match = any(t in node.name for t in focused_topics)
            if is_match:
                relevant.append(node)
            matching_children = self._apply_focus_mode(
                node.children, focused_topics
            )
            if matching_children:
                node.children = matching_children
                if node not in relevant:
                    relevant.append(node)
        return relevant

    # ================================================================
    #  进化触发与执行
    # ================================================================

    async def trigger_evolution_check(self) -> list[CategoryEvolutionEvent]:
        cats = list(self._categories.values())
        members_map = {}
        for cid in self._categories:
            members_map[cid] = [
                {"id": kid} for kid in self.get_category_members(cid)
            ]
        events = await self.evolution_manager.check_evolution_triggers(
            cats, members_map, self.clustering_engine,
        )
        for event in events:
            self._evolution_log.append(event)
        self._save()
        return events

    async def execute_evolution(
        self,
        event: CategoryEvolutionEvent,
    ) -> Optional[Category]:
        if event.action == EvolutionAction.SPLIT:
            sub_assignments = event.details.get("sub_cluster_assignments", {})
            sub_names = event.details.get("suggested_names", {})
            new_cats = await self.evolution_manager.execute_split(
                event.category_id or "",
                {int(k): v for k, v in sub_assignments.items()},
                {int(k): v for k, v in sub_names.items()},
            )
            for cat in new_cats:
                self.create_category(cat)
            return None
        elif event.action == EvolutionAction.MERGE:
            cat_a = event.details.get("category_a", {}).get("id", "")
            cat_b = event.details.get("category_b", {}).get("id", "")
            merged = await self.evolution_manager.execute_merge(cat_a, cat_b)
            self.create_category(merged)
            return merged
        elif event.action == EvolutionAction.ARCHIVE:
            if event.category_id:
                self.update_category(event.category_id, {"is_archived": True})
            return None
        return None

    # ================================================================
    #  置信度联动
    # ================================================================

    def get_confidence_distribution(
        self,
        category_id: str,
    ) -> dict[str, int]:
        member_ids = self.get_category_members(category_id)
        distribution = {"verified": 0, "pending": 0, "doubtful": 0}
        for kid in member_ids:
            kp = None
            if self.vector_service:
                kp = None
            conf = 0.5
            if conf >= 0.7:
                distribution["verified"] += 1
            elif conf >= 0.4:
                distribution["pending"] += 1
            else:
                distribution["doubtful"] += 1
        return distribution