"""
置信度优化计算器 — 三层架构
Layer 1: 信号精细化（多采样一致性 + 段落级来源分 + 图谱传播一致性）
Layer 2: 贝叶斯反馈更新 + 时间衰减 + 多源证据融合
Layer 3: ML 模型预测（预留接口）
"""
import math
import logging
import hashlib
from datetime import datetime
from typing import Optional

import numpy as np

from config import Settings
from models import KnowledgePoint, KnowledgeTriple, FactCategory

logger = logging.getLogger(__name__)

SOURCE_QUALITY_BASE = {
    "pdf": 0.85,
    "学术论文": 0.90,
    "word": 0.75,
    "ppt": 0.65,
    "image": 0.50,
    "web": 0.45,
    "社交网页": 0.35,
    "table": 0.70,
    "code": 0.60,
    "text": 0.50,
    "手动输入": 0.50,
    "用户上传": 0.60,
    "个人笔记": 0.55,
    "OCR识别": 0.45,
}

SUBJECTIVE_WORDS = ["我认为", "我觉得", "可能", "也许", "大概", "应该", "据说", "听说", "传闻"]
AUTHORITY_WORDS = ["研究表", "数据显", "证明", "实验证", "统计", "据XX", "文献", "官方", "权威"]


class ConfidenceCalculator:
    def __init__(self, settings: Settings, graph_service=None, vector_service=None):
        self.settings = settings
        self.graph_service = graph_service
        self.vector_service = vector_service
        self.calibration_model = None

    # ================================================================
    #  第一层：信号精细化
    # ================================================================

    def calculate_model_confidence(self, raw_model_score: float,
                                   extractions: list[dict] = None) -> float:
        """M_cal: 校准后的模型置信度"""
        calibrated = raw_model_score
        if extractions and len(extractions) > 1:
            consistency = self._multi_sample_consistency(extractions)
            calibrated = 0.7 * raw_model_score + 0.3 * consistency
        if self.calibration_model:
            try:
                calibrated = float(self.calibration_model.predict([[raw_model_score]])[0])
            except Exception:
                pass
        return max(0.1, min(1.0, calibrated))

    def _multi_sample_consistency(self, extractions: list[dict]) -> float:
        """多次采样的语义一致性"""
        if len(extractions) < 2:
            return 0.5
        facts = [e.get("fact", "") for e in extractions if e.get("fact")]
        if len(facts) < 2:
            return 0.5
        overlaps = 0
        total = 0
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                total += 1
                if self._jaccard_similarity(facts[i], facts[j]) > 0.3:
                    overlaps += 1
        if total == 0:
            return 0.5
        return overlaps / total

    def _jaccard_similarity(self, a: str, b: str) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def calculate_source_quality(self, source_type: str, text_content: str = "",
                                  user_interactions: int = 0) -> float:
        """S_fine: 段落级来源质量分"""
        base = SOURCE_QUALITY_BASE.get(source_type, 0.5)
        text_lower = text_content.lower()
        subjective_penalty = sum(1 for w in SUBJECTIVE_WORDS if w in text_lower) * 0.05
        authority_bonus = sum(1 for w in AUTHORITY_WORDS if w in text_lower) * 0.08
        has_data = any(c.isdigit() for c in text_content)
        data_bonus = 0.05 if has_data else 0.0
        score = base - subjective_penalty + authority_bonus + data_bonus
        user_boost = min(0.15, user_interactions * 0.02)
        return max(0.2, min(1.0, score + user_boost))

    async def calculate_graph_consistency(self, knowledge_point: KnowledgePoint) -> float:
        """C_prop: 基于图谱传播的一致性分数"""
        entities = knowledge_point.related_entities
        if not entities or not self.graph_service or not self.graph_service.driver:
            return 0.5
        try:
            consistency_scores = []
            async with self.graph_service.driver.session() as session:
                for entity in entities:
                    result = await session.run(
                        """
                        MATCH (e:Entity {name: $name})-[r]-(neighbor:Entity)
                        RETURN avg(r.confidence) as avg_confidence,
                               count(neighbor) as degree
                        """,
                        name=entity,
                    )
                    record = await result.single()
                    if record and record["degree"] and record["degree"] > 0:
                        avg_conf = record["avg_confidence"] or 0.5
                        degree_penalty = 1.0 / (1.0 + math.exp(-0.5 * (record["degree"] - 3)))
                        consistency_scores.append(avg_conf * degree_penalty)
            if consistency_scores:
                return float(np.mean(consistency_scores))
            return 0.5
        except Exception as e:
            logger.warning(f"图谱一致性计算失败: {e}")
            return 0.5

    async def propagate_confidence(self, iterations: int = 3) -> dict[str, float]:
        """图谱置信度传播: PageRank 风格迭代"""
        if not self.vector_service or not self.graph_service or not self.graph_service.driver:
            return {}
        try:
            all_kps = await self.vector_service.list_all(0, 10000)
            conf_map: dict[str, float] = {}
            for kp_data in all_kps:
                kp_id = kp_data.get("id", "")
                conf_map[kp_id] = kp_data.get("confidence", 0.5)

            async with self.graph_service.driver.session() as session:
                for _ in range(iterations):
                    new_conf = dict(conf_map)
                    for kp_id, old_conf in conf_map.items():
                        kp_data = next((k for k in all_kps if k.get("id") == kp_id), None)
                        if not kp_data:
                            continue
                        entities = kp_data.get("related_entities", [])
                        neighbor_confs = []
                        for entity in entities:
                            result = await session.run(
                                "MATCH (e:Entity {name: $name})-[r]-(n:Entity) "
                                "RETURN avg(r.confidence) as nc",
                                name=entity,
                            )
                            record = await result.single()
                            if record and record["nc"]:
                                neighbor_confs.append(record["nc"])
                        if neighbor_confs:
                            neighbor_avg = float(np.mean(neighbor_confs))
                            new_conf[kp_id] = 0.6 * old_conf + 0.4 * neighbor_avg
                    conf_map = new_conf
            return conf_map
        except Exception as e:
            logger.warning(f"置信度传播失败: {e}")
            return {}

    # ================================================================
    #  第二层：贝叶斯更新 + 时间衰减
    # ================================================================

    def bayesian_update(self, kp: KnowledgePoint, feedback: str) -> tuple[float, float]:
        """基于用户反馈的贝叶斯更新 Beta(alpha, beta)"""
        alpha, beta = kp.feedback_alpha, kp.feedback_beta
        if feedback == "positive":
            alpha += 1
        elif feedback == "negative":
            beta += 1
        elif feedback == "correct":
            alpha += 1
            beta = max(1.0, beta - 0.5)
        kp.feedback_alpha = alpha
        kp.feedback_beta = beta
        kp.interaction_count += 1
        feedback_conf = alpha / (alpha + beta)
        return alpha, feedback_conf

    def apply_time_decay(self, kp: KnowledgePoint,
                          domain_decay_rate: float = 0.005) -> float:
        """时间衰减因子 e^(-lambda * delta_t)"""
        if not kp.created_at:
            return 1.0
        now = datetime.now()
        dt = kp.created_at if hasattr(kp.created_at, 'timestamp') else kp.created_at
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                return 1.0
        days = max(0, (now - dt).days)
        decay = math.exp(-domain_decay_rate * days)
        return decay

    def merge_multi_source(self, confidences: list[float]) -> float:
        """多源证据融合: 贝叶斯联合概率"""
        if not confidences:
            return 0.5
        if len(confidences) == 1:
            return confidences[0]
        product_term = 1.0
        complement_product = 1.0
        for c in confidences:
            c = max(0.01, min(0.99, c))
            product_term *= c
            complement_product *= (1.0 - c)
        if complement_product == 0:
            return 1.0
        return 1.0 / (1.0 + complement_product / product_term)

    # ================================================================
    #  综合计算
    # ================================================================

    async def compute(self, kp: KnowledgePoint,
                       source_type: str = "text",
                       apply_feedback: bool = True,
                       apply_decay: bool = True) -> float:
        """综合置信度 = alpha*M_cal + beta*S_fine + gamma*C_prop, 再经贝叶斯+衰减修正"""

        m_cal = self.calculate_model_confidence(kp.model_confidence_raw)
        s_fine = self.calculate_source_quality(source_type, kp.fact,
                                                kp.interaction_count)
        c_prop = kp.consistency_score

        alpha, beta_w, gamma = 0.4, 0.3, 0.3
        base_conf = alpha * m_cal + beta_w * s_fine + gamma * c_prop

        if apply_feedback:
            if kp.feedback_alpha > 2.0 or kp.feedback_beta > 2.0:
                feedback_conf = kp.feedback_alpha / (kp.feedback_alpha + kp.feedback_beta)
                base_conf = (base_conf + feedback_conf) / 2

        if apply_decay:
            decay = self.apply_time_decay(kp)
            base_conf *= decay

        kp.model_confidence_raw = m_cal
        kp.source_quality = s_fine
        kp.consistency_score = c_prop
        kp.confidence = max(0.1, min(1.0, round(base_conf, 4)))
        kp.calibrated_confidence = kp.confidence
        kp.last_updated = datetime.now()

        return kp.confidence

    async def compute_batch(self, knowledge_points: list[KnowledgePoint],
                             source_type: str = "text") -> list[KnowledgePoint]:
        """批量计算置信度"""
        for kp in knowledge_points:
            if kp.consistency_score == 0.5:
                kp.consistency_score = await self.calculate_graph_consistency(kp)
            await self.compute(kp, source_type)
        return knowledge_points

    def contradiction_penalty(self, kp: KnowledgePoint,
                               conflicting_count: int = 1) -> float:
        """矛盾惩罚因子: 每发现一个矛盾，折扣因子 0.8"""
        penalty = 0.8 ** conflicting_count
        kp.confidence *= penalty
        kp.confidence = max(0.1, min(1.0, kp.confidence))
        return kp.confidence

    # ================================================================
    #  第三层：ML 模型接口（预留）
    # ================================================================

    def extract_features(self, kp: KnowledgePoint) -> dict:
        return {
            "model_confidence_raw": kp.model_confidence_raw,
            "source_quality": kp.source_quality,
            "consistency_score": kp.consistency_score,
            "fact_length": len(kp.fact),
            "entity_count": len(kp.related_entities),
            "feedback_alpha": kp.feedback_alpha,
            "feedback_beta": kp.feedback_beta,
            "interaction_count": kp.interaction_count,
            "category_code": list(FactCategory).index(kp.category) if kp.category else 0,
        }

    def load_calibration_model(self, model_path: str):
        try:
            import joblib
            self.calibration_model = joblib.load(model_path)
            logger.info(f"校准模型加载成功: {model_path}")
        except Exception as e:
            logger.warning(f"校准模型加载失败: {e}")