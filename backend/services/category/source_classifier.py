import logging
from collections import defaultdict
from typing import Optional

from models import SourceGroup, SourceComparison

logger = logging.getLogger(__name__)

SOURCE_TYPE_LABELS = {
    "pdf": "PDF文档",
    "word": "Word文档",
    "ppt": "PPT演示",
    "image": "图片",
    "web": "网页",
    "table": "表格",
    "code": "代码",
    "text": "文本",
    "手动输入": "手动录入",
    "对话学习": "对话学习",
}


class SourceClassifier:
    def __init__(self, vector_service=None):
        self.vector_service = vector_service

    def group_by_source(
        self,
        knowledge_points: list[dict],
    ) -> list[SourceGroup]:
        """按来源分组知识"""
        groups: dict[str, dict] = {}
        for kp in knowledge_points:
            source = kp.get("source", "未知来源")
            source_doc = kp.get("source_document_id", source)
            group_key = source_doc if source_doc else source
            if group_key not in groups:
                groups[group_key] = {
                    "source_name": source,
                    "source_type": self._infer_source_type(source),
                    "knowledge_ids": [],
                    "confidences": [],
                    "first_added": None,
                    "last_added": None,
                }
            g = groups[group_key]
            g["knowledge_ids"].append(kp.get("id", ""))
            g["confidences"].append(kp.get("confidence", 0.5))
            created = kp.get("created_at")
            if created:
                if g["first_added"] is None or created < g["first_added"]:
                    g["first_added"] = created
                if g["last_added"] is None or created > g["last_added"]:
                    g["last_added"] = created

        source_groups = []
        for group_key, data in groups.items():
            avg_conf = (
                sum(data["confidences"]) / len(data["confidences"])
                if data["confidences"]
                else 0.5
            )
            source_groups.append(SourceGroup(
                source_name=data["source_name"],
                source_type=data["source_type"],
                knowledge_count=len(data["knowledge_ids"]),
                avg_confidence=round(avg_conf, 3),
                knowledge_ids=data["knowledge_ids"],
                first_added=data["first_added"],
                last_added=data["last_added"],
            ))

        source_groups.sort(key=lambda g: g.knowledge_count, reverse=True)
        return source_groups

    def find_source_conflicts(
        self,
        source_groups: list[SourceGroup],
        all_knowledge: dict[str, dict],
    ) -> list[SourceComparison]:
        """发现不同来源对同一主题的冲突观点"""
        comparisons = []
        for i in range(len(source_groups)):
            for j in range(i + 1, len(source_groups)):
                sa = source_groups[i]
                sb = source_groups[j]
                conflicting, agreeing = self._compare_sources(
                    sa, sb, all_knowledge
                )
                if conflicting or agreeing:
                    comparisons.append(SourceComparison(
                        topic=self._extract_common_topic(
                            sa, sb, all_knowledge
                        ),
                        source_a=sa,
                        source_b=sb,
                        conflicting_points=conflicting,
                        agreement_points=agreeing,
                    ))
        return comparisons

    def _compare_sources(
        self,
        sa: SourceGroup,
        sb: SourceGroup,
        all_knowledge: dict[str, dict],
    ) -> tuple[list[dict], list[dict]]:
        """比较两个来源的观点"""
        conflicting = []
        agreeing = []
        facts_a = {}
        for kid in sa.knowledge_ids:
            kp = all_knowledge.get(kid, {})
            fact = kp.get("fact", "")[:60]
            if fact:
                facts_a[kid] = {"fact": fact, "confidence": kp.get("confidence", 0.5)}

        facts_b = {}
        for kid in sb.knowledge_ids:
            kp = all_knowledge.get(kid, {})
            fact = kp.get("fact", "")[:60]
            if fact:
                facts_b[kid] = {"fact": fact, "confidence": kp.get("confidence", 0.5)}

        for kid_a, fa in facts_a.items():
            for kid_b, fb in facts_b.items():
                sim = self._jaccard_similarity(fa["fact"], fb["fact"])
                if sim > 0.5:
                    if fa["confidence"] > 0.7 and fb["confidence"] > 0.7:
                        agreeing.append({
                            "source_a_fact": fa["fact"],
                            "source_b_fact": fb["fact"],
                            "similarity": sim,
                        })
                elif 0.15 < sim <= 0.4:
                    if abs(fa["confidence"] - fb["confidence"]) > 0.2:
                        conflicting.append({
                            "source_a_fact": fa["fact"],
                            "source_a_confidence": fa["confidence"],
                            "source_b_fact": fb["fact"],
                            "source_b_confidence": fb["confidence"],
                            "similarity": sim,
                        })

        return conflicting, agreeing

    def _extract_common_topic(
        self,
        sa: SourceGroup,
        sb: SourceGroup,
        all_knowledge: dict[str, dict],
    ) -> str:
        """提取两个来源的共同主题"""
        all_facts = []
        for kid in sa.knowledge_ids + sb.knowledge_ids:
            kp = all_knowledge.get(kid, {})
            fact = kp.get("fact", "")
            if fact:
                all_facts.append(fact)
        if all_facts:
            return all_facts[0][:80]
        return f"「{sa.source_name}」vs「{sb.source_name}」"

    def _jaccard_similarity(self, a: str, b: str) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _infer_source_type(self, source: str) -> str:
        for key, label in SOURCE_TYPE_LABELS.items():
            if key in source.lower():
                return label
        if source.endswith(".pdf"):
            return "PDF文档"
        if source.endswith(".docx") or source.endswith(".doc"):
            return "Word文档"
        if source.endswith(".pptx") or source.endswith(".ppt"):
            return "PPT演示"
        if source.startswith("http"):
            return "网页"
        return "文本"

    def calculate_source_trust_bonus(
        self,
        source_name: str,
        source_type: str,
        knowledge_count: int,
        avg_confidence: float,
        manual_corrections: int = 0,
    ) -> float:
        """计算来源可信度加成因子"""
        base_trust = 0.5
        counts_bonus = min(0.15, knowledge_count * 0.01)
        quality_categories = {
            "PDF文档": 0.1,
            "Word文档": 0.05,
            "PPT演示": 0.03,
            "网页": -0.05,
            "图片": -0.1,
        }
        type_bonus = quality_categories.get(source_type, 0.0)
        trust_bonus = -0.2 * min(manual_corrections, 5) * 0.04
        total = base_trust + counts_bonus + type_bonus + trust_bonus
        return max(0.2, min(1.0, total))