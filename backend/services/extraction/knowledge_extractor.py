import json
import re
import logging
import uuid
from datetime import datetime
from typing import Optional
from openai import AsyncOpenAI

from config import Settings
from models import DocumentChunk, KnowledgePoint, KnowledgeTriple, FactCategory, RelationType

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """你是知识提取专家。从以下文本中提取所有有价值的知识点，严格按JSON数组输出。

## 思维链要求（必须内化，不在输出中体现）
1. 先识别内容类型（对话转录/演讲/文档/表格/图像描述）
2. 找出核心主张和证据
3. 提取可独立使用的知识条目

## 输出字段
- fact: 确切的陈述句（主语+谓语+宾语，包含时间、地点、数值）
- category: 概念/事实/方法/观点/待验证
- confidence: 0-1之间，基于内容的具体性、一致性和可验证性
- related_entities: 涉及的实体名称列表

## 重要规则
- fact必须完整独立，脱离上下文也能被理解
- 数值必须精确（如35%，不要约30%）
- 多媒体内容中标注有"说话人"或"时间戳"的，尽量在fact中体现
- 画面描述中的观察结果，转化为可验证的事实陈述
- 只输出JSON数组，不要解释

文本：
{text}"""

TRIPLE_EXTRACTION_PROMPT = """从以下知识点列表中提取三元组关系（实体A → 关系 → 实体B）。

对每个知识点，提取所有可能的实体关系三元组。关系类型必须是以下之一：
IS_A（是一种/层级）、PART_OF（组成部分）、INSTANCE_OF（实例）、CAUSES（导致/引起）、
DEPENDS_ON（依赖）、INDICATES（表明/指示）、BELONGS_TO（属于）、OCCURS_AT（发生在）、
BEFORE（在...之前）、AFTER（在...之后）、CONFLICTS_WITH（与...矛盾）、
CONFIRMED_BY（被...证实）、ENDORSED_BY（被...支持）

若无法归类到以上类型，输出 RELATED_TO（通用关联），并标记 confidence 降为 0.6。

输出一个JSON数组，格式如下：
[{{"subject": "实体A", "relation": "关系描述", "object": "实体B", "relation_type": "IS_A|CAUSES|...", "confidence": 0.8}}]

知识点：
{knowledge_points}"""


class KnowledgeExtractor:
    def __init__(self, settings: Settings, confidence_calculator=None, category_service=None):
        self.settings = settings
        self.confidence_calculator = confidence_calculator
        self.category_service = category_service
        self.client: Optional[AsyncOpenAI] = None
        if settings.deepseek_api_key:
            self.client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

    async def extract_from_chunks(self, chunks: list[DocumentChunk]) -> list[KnowledgePoint]:
        all_points = []
        for chunk in chunks:
            if chunk.confidence is not None and chunk.confidence <= 0.0:
                continue
            points = await self._extract_from_text(chunk.content, chunk.source_path, chunk.source_type.value)
            for p in points:
                p.source_document_id = chunk.source_path
            all_points.extend(points)
        merged = self._merge_similar(all_points)
        if self.confidence_calculator:
            merged = await self.confidence_calculator.compute_batch(
                merged, chunks[0].source_type.value if chunks else "text"
            )
        if self.category_service:
            for kp in merged:
                try:
                    await self.category_service.auto_categorize_knowledge(
                        kp.id, kp.fact, confidence_threshold=0.5, auto_create=False
                    )
                except Exception:
                    pass
        return merged

    async def _extract_from_text(self, text: str, source: str, source_type: str = "text") -> list[KnowledgePoint]:
        if len(text) > 4000:
            text = text[:4000]

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识提取专家，擅长从文本中提取结构化的知识点。"},
                    {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            result_text = response.choices[0].message.content or ""
            json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                points = []
                for item in data:
                    category = item.get("category", "待验证")
                    cat_map = {
                        "概念": FactCategory.CONCEPT,
                        "事实": FactCategory.FACT,
                        "方法": FactCategory.METHOD,
                        "观点": FactCategory.OPINION,
                        "待验证": FactCategory.PENDING,
                    }
                    model_conf = float(item.get("confidence", 0.5))
                    point = KnowledgePoint(
                        id=str(uuid.uuid4()),
                        fact=item.get("fact", ""),
                        category=cat_map.get(category, FactCategory.PENDING),
                        confidence=model_conf,
                        model_confidence_raw=model_conf,
                        related_entities=item.get("related_entities", []),
                        source=source,
                        source_quality=self._estimate_source_quality(source_type, item.get("fact", "")),
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                    )
                    points.append(point)
                return points

        except Exception as e:
            logger.warning(f"DeepSeek API 调用失败: {e}, 使用简单的文本分割代替")

        sentences = re.split(r"[。！？\n]+", text)
        points = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 5:
                points.append(KnowledgePoint(
                    id=str(uuid.uuid4()),
                    fact=sent[:200],
                    category=FactCategory.PENDING,
                    confidence=0.5,
                    model_confidence_raw=0.5,
                    related_entities=[],
                    source=source,
                    source_quality=0.4,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                ))
        return points

    def _estimate_source_quality(self, source_type: str, text: str) -> float:
        base_map = {
            "pdf": 0.85, "word": 0.75, "ppt": 0.65, "image": 0.55,
            "web": 0.45, "table": 0.70, "code": 0.75, "text": 0.50,
            "video": 0.65, "audio": 0.60,
        }
        base = base_map.get(source_type, 0.5)
        if any(w in text for w in ["研究表", "数据显", "证明", "官方", "权威"]):
            base += 0.08
        if any(w in text for w in ["我认为", "我觉得", "可能", "也许", "据说"]):
            base -= 0.05
        return max(0.2, min(1.0, base))

    async def extract_triples(self, knowledge_points: list[KnowledgePoint]) -> list[KnowledgeTriple]:
        points_text = "\n".join([f"- {p.fact}" for p in knowledge_points[:15]])
        if not points_text:
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[
                    {"role": "system", "content": "你是一个知识图谱构建专家，擅长从文本中提取实体关系三元组。"},
                    {"role": "user", "content": TRIPLE_EXTRACTION_PROMPT.format(knowledge_points=points_text)},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            result_text = response.choices[0].message.content or ""
            json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                triples = []
                kp_ids = ",".join([kp.id for kp in knowledge_points if hasattr(kp, "id") and kp.id])
                for item in data:
                    rel_type_str = item.get("relation_type", "RELATED_TO")
                    conf = float(item.get("confidence", 0.8))
                    triples.append(KnowledgeTriple(
                        subject=item.get("subject", ""),
                        relation=item.get("relation", ""),
                        object=item.get("object", ""),
                        relation_type=RelationType(rel_type_str) if rel_type_str in {rt.value for rt in RelationType} else RelationType.RELATED_TO,
                        confidence=conf,
                        source_knowledge_id=kp_ids,
                    ))
                return triples

        except Exception as e:
            logger.warning(f"三元组提取失败: {e}")

        triples = []
        for kp in knowledge_points:
            entities = kp.related_entities
            if len(entities) >= 2:
                for i in range(len(entities) - 1):
                    triples.append(KnowledgeTriple(
                        subject=entities[i],
                        relation="关联",
                        object=entities[i + 1],
                        confidence=kp.confidence,
                        source_knowledge_id=kp.id or "",
                    ))
        return triples

    def _merge_similar(self, points: list[KnowledgePoint]) -> list[KnowledgePoint]:
        seen = set()
        merged = []
        for p in points:
            key = p.fact[:50]
            if key not in seen:
                seen.add(key)
                merged.append(p)
        return merged
CODE_EXTRACTION_PROMPT = """你是代码知识提取专家。从以下代码分析报告中提取有价值的知识点。

## 输入包含
- 静态分析结果（函数签名、类结构、调用关系、复杂度）
- LLM语义理解（功能摘要、算法、业务逻辑）

## 输出要求
输出JSON数组，每个元素包含：
- fact: 完整独立的知识陈述
- category: 概念/事实/方法/观点/待验证
- confidence: 0-1
- related_entities: 函数名、类名、模块名、算法名

## 重要规则
- 优先提取高复杂度函数的知识点
- 调用关系、继承关系作为独立知识
- 算法实现必须标注具体算法名
- 只输出JSON数组

文本：
{text}"""
