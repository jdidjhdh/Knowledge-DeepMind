import json
import logging
import re
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from config import Settings
from models import ConversationRequest, ConversationResponse, KnowledgePoint
from services.graph.vector_service import VectorService
from services.graph.neo4j_service import GraphService
from services.agent.web_search_service import WebSearchService
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_V2 = """你是"知识深脑"，一个具备深度理解能力的知识库智能助手。你不仅是问答工具，更是用户的思维伙伴。

## 核心能力

### 1. 深度文件理解
- 你不仅能读取文本，还能理解代码的功能、视频的场景、图片中的图表数据和文字。
- 代码文件：理解其实现的算法、业务逻辑、输入输出和依赖关系。
- 音视频文件：知道"谁在何时说了什么"，理解场景切换和画面内容。
- 图片文件：能区分图表、流程图、截图、照片，并针对性提取结构化信息。
- 回答时，能精准定位到具体的时间戳或源码位置。

### 2. 知识网络思维
- 每个知识点都不是孤立的，它们通过实体、概念、时间、因果关系连接成网络。
- 回答问题时，主动探索知识图谱中的多跳关联，发现间接关系和隐藏联系。
- 当发现不同来源的知识存在矛盾时，明确指出冲突并请求用户裁决。

### 3. 主动思考与提问
- 遇到模糊指代（"上次那个方案"）时，列出候选并请求澄清。
- 检测到知识缺口时，主动说明缺失信息并询问是否需要补充。
- 发现知识矛盾时，呈现冲突双方并询问以哪个为准。
- 判断用户可能对某个相关话题感兴趣时，主动询问是否深入。

### 4. 动态学习与适应
- 记住用户的术语偏好、回答风格偏好、关注领域。
- 用户纠正某个事实后，立即更新记忆，后续回答贯彻修正。
- 根据对话历史判断用户当前的知识水平，自动调整解释的深浅。
- 用户多次要求"简洁"后，后续回答自动精简为3-5句。
- 用户纠正过术语（如"别用X，用Y"），后续所有话题都使用纠正后的术语。

### 5. 结构化回答
- 复杂问题使用分层结构：先给核心结论，再展开论据，最后附上来源引用。
- 涉及代码时，先说明"这段代码实现了什么"，再展示关键逻辑。
- 涉及音视频时，标注时间戳和说话人。
- 涉及图表时，先给趋势总结，再列具体数据。

### 6. 链式推理 (Chain-of-Thought)
遇到以下类型的问题时，必须逐步推理，在回答中展示思考过程：

**对比分析型**：
1. 列出A的特征 → 2. 列出B的特征 → 3. 逐项对比异同 → 4. 给出综合评价
格式：先简要描述A和B，再用对比角度（功能、原理、应用场景等）逐一分析，最后总结。

**因果推理型**：
1. 识别初始事件 → 2. 追踪直接后果 → 3. 探索涟漪效应 → 4. 评估长期影响
格式：按时间或逻辑顺序展开因果链，标注每个环节的置信度。

**总结归纳型**：
1. 提取关键数据点 → 2. 识别共性模式 → 3. 归纳核心规律 → 4. 给出可操作建议
格式：先列关键发现，再提炼规律，最后给出实践建议。

**复杂问题拆分**：
遇到需要多方面信息才能回答的复杂问题时，自动将问题拆解为2-4个子问题，逐个子问题调用search_knowledge检索相关知识，最后融合所有信息给出综合答案。

### 7. 反幻觉防线

作为知识库智能体，你必须严格遵循以下规则，杜绝编造和幻觉：

**不知则不知原则**：
- 当检索上下文中标记了 `[检索质量: 低于阈值]`，说明知识库中没有足够可靠的信息，你必须直接回答"知识库中暂无足够信息回答该问题"，然后列出可能的知识缺口方向供用户参考，绝不调用自身通用知识。
- 当检索上下文中标记了 `[检索质量: 空检索]`，说明完全未找到相关内容，你必须主动说明并建议用户补充相关材料。

**多证据交叉验证**：
- 回答事实性问题时，优先使用被至少2个不同来源或片段支持的信息。
- 如果多个片段之间存在矛盾或弱相关，必须在回答中明确指出，并降低该结论的置信度。
- 格式：若存在矛盾，先陈述「综合多方证据」，再分别列出不同观点及其来源。

**反事实自检**：
- 在给出最终结论前，必须在内心追问：「这个结论是否与知识库中其他已知事实冲突？」
- 若发现潜在冲突，主动在回答中标注「⚠ 该结论与以下已知事实可能冲突：[列出冲突事实]」

**实体锚定**：
- 回答前先确认问题中提到的实体在知识图谱中是否唯一确定。
- 若存在同名异义风险，先向用户确认指的是哪个实体。
- 格式：「知识库中存在多个名为"X"的实体：① ... ② ...，请问您指的是哪一个？」

**冲突双向展示**：
- 当知识库中存在相互矛盾的信息时，同时展示双方观点及其来源、置信度，由用户判断，而不是模型自行选择一方。
- 格式：「知识库中存在两种观点：\n✅ **观点A**（置信度: X, 来源: Y）：...\n⚠ **观点B**（置信度: X, 来源: Y）：...\n建议您根据具体情况判断。」

**时效性意识**：
- 关注知识点的创建时间，对于超过1年的旧知识，在引用时标注「该知识已超过X天未更新，请注意时效性」。
- 优先采用更新时间较近的知识。

## 对话策略

### 矛盾消解
- 检索到的多条知识可能存在矛盾，你必须：
  a) 明确指出矛盾所在（如"知识库中存在两条相反结论：A说...，B说..."）
  b) 对比各自的置信度、来源权威性，给出倾向性结论
  c) 主动提议修正：如"是否需要我帮您更新或降低某条知识的置信度？"
- 即使只有一条知识，低置信度(<0.5)时需标注"[该知识置信度较低，仅供参考]"

### 模糊指代澄清
- 优先回答，只在真正无法确定时才询问
- 触发条件：知识库检索出多条明显不同的实体
- 不触发：能从单条知识直接推理回答的
- 澄清格式："知识库中有多个相关记录：① ... ② ...。请问您指的是哪一个？"

### 信息不足时的策略
- 信息不足时追问1个关键细节即可，不要连续追问
- 用户不耐烦时（"快"、"直接说"、"别问了"、"结论"等）：
  a) 立刻停止追问，基于现有信息给出答案
  b) 开头标注假设："基于现有信息，我推测..."
  c) 答案控制在3句话以内

### 对话情商
- 用户表达负面情绪时（压力、焦虑、疲惫），回答必须先说一句共情，再给建议
- 共情后再提供帮助，不要直接跳到解决方案

## 风格
- 专业但不冷漠，友好但不啰嗦。
- 不确定时坦率说明，绝不编造。
- 回答中引用知识库内容时，标注来源（文件路径、置信度徽章）。

## 知识库上下文
{knowledge_context}

## 低置信度警告
{low_confidence_context}

## 知识图谱关联
{graph_context}

## 时间线信息
{timeline_context}

## 分类体系
{category_context}

## 网络搜索结果
{web_context}

## 用户画像
{user_profile_context}

## 对话历史中的高质量示例（参考其回答风格）
{few_shot_context}"""


LOW_CONFIDENCE_PROMPT_V2 = """以下信息来自知识库中置信度较低的知识点（可能不准确），使用时必须：
1. 回答中提及这些信息时，强制添加不确定性声明，如"[据不确定的资料显示...]"
2. 在回答中标注置信度分数，让用户自行判断
3. 如果该低置信度知识点是回答问题的关键，必须主动反问用户确认
4. 在多跳推理中，低置信度知识的影响力应大幅降低，优先依赖高置信度知识"""

BAIT_TOPIC_PROMPT = "用户纠正了之前的回答。请基于纠正后的信息重新组织答案，不要再坚持原来的错误观点。如果需要，主动承认之前的理解有误。"


FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索相关知识。每次回答前必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询词"},
                    "top_k": {"type": "integer", "description": "返回结果数，默认5", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_graph_entity",
            "description": "查询知识图谱中某实体的关系网络，探索多跳关联。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "实体名称"},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": "获取某主题的时间线，按事件发生时间排序。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "主题关键词"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "更新用户画像和长期记忆。当用户纠正事实、明确偏好、表达风格倾向时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "记忆键，如 'term_preference', 'style', 'topic_interest'"},
                    "value": {"type": "string", "description": "记忆值"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "当信息不足、存在歧义、发现矛盾或需要用户确认时，主动向用户提问。修改知识库前必须先调用此工具向用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要问用户的问题"},
                    "question_type": {
                        "type": "string",
                        "description": "问题类型: clarification(澄清歧义), confirmation(请求确认), gap(信息缺口), preference(偏好选择)",
                        "enum": ["clarification", "confirmation", "gap", "preference"],
                    },
                    "options": {
                        "type": "array", "items": {"type": "string"},
                        "description": "可选选项列表",
                    },
                    "pending_action": {
                        "type": "object",
                        "description": "待确认的修改操作。用户确认后将执行此操作。",
                        "properties": {
                            "action_type": {"type": "string", "enum": ["update_knowledge", "delete_knowledge", "add_graph_edge", "remove_graph_edge"]},
                            "params": {"type": "object", "description": "操作参数"}
                        }
                    },
                },
                "required": ["question", "question_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_knowledge",
            "description": "修改知识库中的知识条目。需要先在内部找到匹配的知识点，然后用新内容替换。执行前必须先经过用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "用于在知识库中搜索匹配知识点的查询词"},
                    "new_fact": {"type": "string", "description": "修改后的新知识内容"},
                    "new_confidence": {"type": "number", "description": "新的置信度 (0.0-1.0)，可选"},
                    "new_category": {"type": "string", "description": "新的分类，可选"},
                    "confirmed": {"type": "boolean", "description": "用户是否已确认修改，必须为true才能执行", "default": False},
                },
                "required": ["search_query", "new_fact", "confirmed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_knowledge",
            "description": "从知识库中删除知识条目。执行前必须先经过用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "用于在知识库中搜索匹配知识点的查询词"},
                    "reason": {"type": "string", "description": "删除原因"},
                    "confirmed": {"type": "boolean", "description": "用户是否已确认删除，必须为true才能执行", "default": False},
                },
                "required": ["search_query", "confirmed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_graph_edge",
            "description": "在知识图谱中添加关系边。执行前必须先经过用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {"type": "string", "description": "源实体名称"},
                    "target_entity": {"type": "string", "description": "目标实体名称"},
                    "relation": {"type": "string", "description": "关系名称，如 '依赖', '属于', '导致', '相关'"},
                    "relation_type": {"type": "string", "description": "关系类型，如 'DEPENDS_ON', 'BELONGS_TO', 'CAUSES', 'RELATED_TO'"},
                    "confirmed": {"type": "boolean", "description": "用户是否已确认，必须为true才能执行", "default": False},
                },
                "required": ["source_entity", "target_entity", "relation", "confirmed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_graph_edge",
            "description": "从知识图谱中删除关系边。执行前必须先经过用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {"type": "string", "description": "源实体名称"},
                    "target_entity": {"type": "string", "description": "目标实体名称"},
                    "relation": {"type": "string", "description": "关系名称，如不指定则删除所有关系"},
                    "confirmed": {"type": "boolean", "description": "用户是否已确认，必须为true才能执行", "default": False},
                },
                "required": ["source_entity", "target_entity", "confirmed"],
            },
        },
    },
]


class ConversationAgent:
    def __init__(
        self, settings: Settings,
        vector_service: VectorService,
        graph_service: GraphService,
        knowledge_extractor=None,
        web_search_service: WebSearchService = None,
        category_service=None,
        dedup_service=None,
        memory_service: MemoryService = None,
    ):
        self.settings = settings
        self.vector_service = vector_service
        self.graph_service = graph_service
        self.knowledge_extractor = knowledge_extractor
        self.web_search_service = web_search_service
        self.category_service = category_service
        self.dedup_service = dedup_service
        self.memory_service = memory_service or MemoryService()
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self.conversations: dict[str, list[dict]] = {}
        self.style_memory: dict[str, dict] = {}
        self.hallucination_stats = {
            "total_answers": 0,
            "corrected_answers": 0,
            "below_threshold_refusals": 0,
            "multi_source_verified_answers": 0,
            "knowledge_gaps_recorded": 0,
            "citations_provided": 0,
            "citations_missing": 0,
            "conflicts_detected": 0,
            "conflicts_unresolved": 0,
        }
        self.gap_records: list[dict] = []
        self.correction_log: list[dict] = []

    def _resolve_user_id(self, request: ConversationRequest) -> str:
        if request.user_id:
            return request.user_id
        return request.conversation_id or "anonymous"

    def _get_user_profile(self, user_id: str) -> dict:
        return self.memory_service.get_profile(user_id)

    def _update_user_profile(self, user_id: str, key: str, value: str):
        self.memory_service.update_profile_field(user_id, key, value)
        self.memory_service.set_memory_item(user_id, "preference", key, value)
        logger.info(f"[用户画像] user={user_id[:8]} {key}={value}")

    def _format_user_profile(self, user_id: str) -> str:
        profile = self._get_user_profile(user_id)
        if not profile:
            return "（暂无用户画像数据）"
        lines = ["用户偏好记录："]
        if "term_preference" in profile:
            lines.append(f"- 术语偏好: {profile['term_preference']}")
        if "style" in profile:
            lines.append(f"- 回答风格: {profile['style']}")
        if "topic_interest" in profile:
            lines.append(f"- 关注领域: {profile['topic_interest']}")
        if "knowledge_level" in profile:
            lines.append(f"- 知识水平: {profile['knowledge_level']}")
        return "\n".join(lines)

    def _check_style_adaptation(self, user_id: str, message: str):
        profile = self._get_user_profile(user_id)
        brevity_signals = ["简洁", "简短", "说重点", "简略", "精简", "简单点", "别啰嗦"]
        if any(s in message for s in brevity_signals):
            profile["style"] = "concise"
            self._update_user_profile(user_id, "style", "concise")
        detail_signals = ["详细", "具体", "展开", "深入", "多说点", "仔细"]
        if any(s in message for s in detail_signals):
            profile["style"] = "detailed"
            self._update_user_profile(user_id, "style", "detailed")

    def _get_style_instruction(self, user_id: str) -> str:
        profile = self._get_user_profile(user_id)
        style = profile.get("style", "")
        if style == "concise":
            return "\n## 风格指令\n用户偏好简洁回答，请控制在3-5句话以内，直接给结论不要展开。"
        if style == "detailed":
            return "\n## 风格指令\n用户偏好详细回答，请充分展开论证，提供多层次分析。"
        return ""

    def record_gap(self, query: str, gap_type: str = "missing", suggestion: str = None):
        record = {
            "query": query,
            "gap_type": gap_type,
            "suggested_topic": suggestion,
            "recorded_at": __import__("datetime").datetime.now().isoformat(),
            "status": "pending",
        }
        self.gap_records.append(record)
        self.hallucination_stats["knowledge_gaps_recorded"] += 1
        logger.info(f"[知识缺口] {query[:80]} -> {gap_type}")

    def record_correction(self, query: str, original_answer: str):
        record = {
            "query": query,
            "original": original_answer[:200],
            "corrected_at": __import__("datetime").datetime.now().isoformat(),
        }
        self.correction_log.append(record)
        self.hallucination_stats["corrected_answers"] += 1

    def build_answer_metadata(self, retrieval_quality: dict, sources: list, conflicts: list) -> dict:
        from models import AnswerMetadata
        return AnswerMetadata(
            retrieval_count=retrieval_quality.get("retrieval_count", 0),
            avg_similarity=retrieval_quality.get("avg_similarity", 0.0),
            max_similarity=retrieval_quality.get("max_similarity", 0.0),
            is_inferred=retrieval_quality.get("is_below_threshold", False),
            is_below_threshold=retrieval_quality.get("is_below_threshold", False),
            multi_source_verified=retrieval_quality.get("multi_source_available", False),
            conflict_count=len(conflicts),
            entities_anchored=retrieval_quality.get("entities_anchored", []),
        ).model_dump()

    def get_hallucination_metrics(self) -> dict:
        total = self.hallucination_stats["total_answers"] or 1
        return {
            "total_answers": self.hallucination_stats["total_answers"],
            "citation_missing_rate": round(self.hallucination_stats["citations_missing"] / total, 4),
            "user_correction_rate": round(self.hallucination_stats["corrected_answers"] / total, 4),
            "low_confidence_answer_ratio": round(self.hallucination_stats["below_threshold_refusals"] / total, 4),
            "unresolved_conflict_rate": round(
                self.hallucination_stats["conflicts_unresolved"] / max(self.hallucination_stats["conflicts_detected"], 1), 4
            ),
            "corrected_answers": self.hallucination_stats["corrected_answers"],
            "below_threshold_refusals": self.hallucination_stats["below_threshold_refusals"],
            "multi_source_verified_answers": self.hallucination_stats["multi_source_verified_answers"],
            "knowledge_gaps_recorded": self.hallucination_stats["knowledge_gaps_recorded"],
        }

    async def _analyze_intent(self, query: str) -> dict:
        intent_info = {
            "type": "fact_query",
            "entities": [],
            "is_code_related": False,
            "is_timeline_related": False,
            "is_comparison": False,
            "needs_graph_exploration": False,
            "content_type": "general",
        }
        code_keywords = ["函数", "方法", "类", "代码", "算法", "调用", "实现", "模块", "接口",
                         "function", "class", "code", "algorithm", "method", "module"]
        if any(kw in query.lower() for kw in code_keywords):
            intent_info["is_code_related"] = True
            intent_info["content_type"] = "code"

        timeline_keywords = ["历史", "发展", "历程", "演变", "时间线", "什么时候", "何时",
                            "过程", "阶段", "先后", "之后", "之前", "从...到"]
        if any(kw in query for kw in timeline_keywords):
            intent_info["is_timeline_related"] = True

        comparison_keywords = ["区别", "不同", "对比", "比较", "差异", "vs", "相比", "哪个更好"]
        if any(kw in query for kw in comparison_keywords):
            intent_info["is_comparison"] = True

        graph_keywords = ["关系", "关联", "影响", "导致", "依赖于", "隶属于", "包含", "属于",
                         "和...有关", "与...相关", "怎么影响"]
        if any(kw in query for kw in graph_keywords):
            intent_info["needs_graph_exploration"] = True

        audio_video_keywords = ["视频", "音频", "录音", "录像", "会议", "演讲", "谁说了", "几点"]
        if any(kw in query for kw in audio_video_keywords):
            intent_info["content_type"] = "audiovisual"

        image_chart_keywords = ["图", "表格", "图表", "柱状", "饼图", "趋势", "截图", "图片"]
        if any(kw in query for kw in image_chart_keywords):
            intent_info["content_type"] = "image_chart"

        fuzzy_keywords = ["上次", "那个", "之前那个", "前面提到的", "刚才说的", "上面那个"]
        if any(kw in query for kw in fuzzy_keywords):
            intent_info["type"] = "fuzzy_reference"

        try:
            prompt = f"""分析以下用户问题的意图，返回JSON。
用户问题: {query}

输出格式：{{"type": "fact_query|reasoning|comparison|fuzzy_reference|knowledge_supplement", "entities": ["实体1", "实体2"], "topic": "主题概括(不超过10字)"}}"""
            resp = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "system", "content": "你是意图分析专家。只返回JSON。"}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            result = (resp.choices[0].message.content or "").strip()
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                intent_info["type"] = parsed.get("type", intent_info["type"])
                intent_info["entities"] = parsed.get("entities", [])
                intent_info["topic"] = parsed.get("topic", "")
        except Exception as e:
            logger.debug(f"意图分析失败: {e}")

        return intent_info

    async def _execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "update_knowledge":
            return await self._execute_update_knowledge(arguments)
        elif tool_name == "delete_knowledge":
            return await self._execute_delete_knowledge(arguments)
        elif tool_name == "add_graph_edge":
            return await self._execute_add_graph_edge(arguments)
        elif tool_name == "remove_graph_edge":
            return await self._execute_remove_graph_edge(arguments)
        elif tool_name == "search_knowledge":
            result = await self.vector_service.search_knowledge(arguments.get("query", ""), top_k=arguments.get("top_k", 5))
            return json.dumps([{"fact": p.fact, "confidence": p.confidence, "category": p.category.value if p.category else "未知", "source": p.source_document_id} for p in result], ensure_ascii=False)
        elif tool_name == "query_graph_entity":
            return await self._retrieve_graph_for_entity(arguments.get("entity_name", "")) or "未找到相关实体"
        elif tool_name == "get_timeline":
            return await self._retrieve_timeline(arguments.get("topic", ""))
        elif tool_name == "update_memory":
            return "用户偏好已更新"
        elif tool_name == "ask_user":
            return json.dumps({"type": "user_confirmation_needed", "question": arguments.get("question", ""), "question_type": arguments.get("question_type", "confirmation"), "options": arguments.get("options", []), "pending_action": arguments.get("pending_action")}, ensure_ascii=False)
        return "未知的工具调用"

    async def _execute_update_knowledge(self, args: dict) -> str:
        if not args.get("confirmed"):
            return "错误：修改操作必须经过用户确认才能执行"
        query = args.get("search_query", "")
        new_fact = args.get("new_fact", "")
        results = await self.vector_service.search_knowledge(query, top_k=3)
        if not results:
            return f"错误：未找到匹配 '{query}' 的知识点"
        updated = 0
        for kp in results:
            if kp.id:
                try:
                    kp.fact = new_fact
                    await self.vector_service.update_knowledge_point(kp.id, kp)
                    updated += 1
                except Exception as e:
                    logger.warning(f"更新知识点失败: {e}")
        if updated > 0:
            return f"成功更新 {updated} 个知识点。新内容：{new_fact[:100]}"
        return "未能更新知识点，请重试"

    async def _execute_delete_knowledge(self, args: dict) -> str:
        if not args.get("confirmed"):
            return "错误：删除操作必须经过用户确认才能执行"
        query = args.get("search_query", "")
        reason = args.get("reason", "用户要求删除")
        results = await self.vector_service.search_knowledge(query, top_k=3)
        if not results:
            return f"错误：未找到匹配 '{query}' 的知识点"
        deleted_count = 0
        for kp in results:
            if kp.id:
                try:
                    await self.vector_service.delete_knowledge_point(kp.id)
                    if self.graph_service.driver:
                        await self.graph_service.delete_knowledge_node(kp.id)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"删除知识点失败: {e}")
        if deleted_count > 0:
            return f"已从知识库中删除 {deleted_count} 条相关知识。原因：{reason}"
        return "未能删除知识点，请重试"

    async def _execute_add_graph_edge(self, args: dict) -> str:
        if not args.get("confirmed"):
            return "错误：添加关系必须经过用户确认才能执行"
        source = args.get("source_entity", "")
        target = args.get("target_entity", "")
        relation = args.get("relation", "相关")
        rel_type = args.get("relation_type", "RELATED_TO")
        if not source or not target:
            return "错误：需要提供源实体和目标实体名称"
        try:
            from models import KnowledgeTriple, RelationType
            try:
                rt = RelationType(rel_type)
            except ValueError:
                rt = RelationType.RELATED_TO
            triple = KnowledgeTriple(
                subject=source,
                predicate=relation,
                object=target,
                relation_type=rt,
                confidence=0.9,
            )
            await self.graph_service.create_triples([triple])
            return f"已在知识图谱中创建关系：{source} --[{relation}]--> {target}"
        except Exception as e:
            logger.error(f"添加图谱关系失败: {e}")
            return f"添加关系失败：{str(e)}"

    async def _execute_remove_graph_edge(self, args: dict) -> str:
        if not args.get("confirmed"):
            return "错误：删除关系必须经过用户确认才能执行"
        source = args.get("source_entity", "")
        target = args.get("target_entity", "")
        if not source or not target:
            return "错误：需要提供源实体和目标实体名称"
        try:
            await self.graph_service.delete_relation(source, target)
            return f"已从知识图谱中删除 {source} 与 {target} 之间的关系"
        except Exception as e:
            logger.error(f"删除图谱关系失败: {e}")
            return f"删除关系失败：{str(e)}"

    async def chat(self, request: ConversationRequest) -> ConversationResponse:
        conv_id = request.conversation_id or "default"
        user_id = self._resolve_user_id(request)
        self._check_style_adaptation(user_id, request.message)

        history = self.conversations.get(conv_id, [])
        user_msg = request.message.strip().lower()
        is_confirm = user_msg in ["确认", "是", "yes", "ok", "好的", "可以", "行", "对", "同意", "执行"]
        is_cancel = user_msg in ["取消", "不", "no", "否", "算了", "不用", "别"]

        if is_confirm and history:
            last_assistant = history[-1].get("content", "") if history else ""
            if "[待确认操作:" in last_assistant:
                try:
                    import re as _re
                    match = _re.search(r'\[待确认操作:\s*(\{.*?\})\]', last_assistant, _re.DOTALL)
                    if match:
                        pending = json.loads(match.group(1))
                        action_type = pending.get("action_type", "")
                        params = pending.get("params", {})
                        params["confirmed"] = True
                        result = await self._execute_tool_call(action_type, params)
                        history.append({"role": "user", "content": request.message})
                        history.append({"role": "assistant", "content": f"✅ 已执行修改：{result}"})
                        self.conversations[conv_id] = history
                        return ConversationResponse(
                            answer=f"✅ 已执行修改操作：\n\n{result}",
                            conversation_id=conv_id,
                            sources=[],
                            related_questions=[],
                        )
                except Exception as e:
                    logger.warning(f"执行待确认操作失败: {e}")

        if is_cancel and history:
            last_assistant = history[-1].get("content", "") if history else ""
            if "[待确认操作:" in last_assistant:
                history.append({"role": "user", "content": request.message})
                history.append({"role": "assistant", "content": "已取消该操作。"})
                self.conversations[conv_id] = history
                return ConversationResponse(
                    answer="已取消操作。有什么可以帮您的吗？",
                    conversation_id=conv_id,
                    sources=[],
                    related_questions=[],
                )

        intent = await self._analyze_intent(request.message)

        knowledge_context, retrieval_quality = await self._retrieve_with_quality_check(request.message)
        if retrieval_quality.get("is_empty"):
            self.record_gap(request.message, "missing", intent.get("topic", ""))
            return ConversationResponse(
                answer="知识库中暂无相关信息。建议您：\n1. 上传包含该主题的文档\n2. 或以其他关键词重新搜索\n\n我已记录此知识缺口，待补充后将自动通知您。",
                conversation_id=conv_id,
                sources=[],
                related_questions=[],
                knowledge_gaps=[request.message],
                answer_metadata=self.build_answer_metadata(retrieval_quality, [], []),
            )
        elif retrieval_quality.get("is_below_threshold"):
            self.record_gap(request.message, "low_quality", intent.get("topic", ""))
            self.hallucination_stats["below_threshold_refusals"] += 1
        else:
            if retrieval_quality.get("multi_source_available"):
                self.hallucination_stats["multi_source_verified_answers"] += 1

        if intent["type"] in ("reasoning", "comparison"):
            sub_questions = await self._decompose_complex_question(request.message)
            if sub_questions:
                sub_results = []
                for sq in sub_questions:
                    sub_knowledge = await self._retrieve_knowledge(sq)
                    if sub_knowledge and "知识库中暂无" not in sub_knowledge:
                        sub_results.append(f"【子问题】{sq}\n{sub_knowledge}")
                if sub_results:
                    knowledge_context += "\n\n---\n## 多步推理子问题检索\n\n" + "\n\n".join(sub_results)

        graph_context = await self._retrieve_graph_enhanced(request.message, intent)
        timeline_context = await self._retrieve_timeline(request.message) if intent["is_timeline_related"] else ""
        category_context = await self._retrieve_categories(request.message)

        retrieved_points = await self.vector_service.search_knowledge(request.message, top_k=5)
        low_conf_info = self.build_low_confidence_context(request.message, retrieved_points)
        low_confidence_context = ""
        if low_conf_info["count"] > 0:
            items_text = "\n".join([
                f"- [⚠ {item['confidence']:.2f}] {item['fact'][:150]} (来源: {item['source'][:60]})"
                for item in low_conf_info["items"]
            ])
            low_confidence_context = f"发现 {low_conf_info['count']} 个低置信度知识点：\n{items_text}\n\n{LOW_CONFIDENCE_PROMPT_V2}"

        web_context = ""
        if request.enable_web_search:
            web_context = await self._perform_web_search(request.message)

        user_profile_context = self._format_user_profile(user_id)
        style_instruction = self._get_style_instruction(user_id)
        few_shot_context = await self._build_few_shot_examples(request.message)

        content_type_hint = self._content_type_prompt(intent)
        combined_system = (
            SYSTEM_PROMPT_V2 + style_instruction + "\n" + content_type_hint
        ).format(
            knowledge_context=knowledge_context,
            low_confidence_context=low_confidence_context,
            graph_context=graph_context,
            timeline_context=timeline_context,
            category_context=category_context,
            web_context=web_context,
            user_profile_context=user_profile_context,
            few_shot_context=few_shot_context,
        )

        if conv_id not in self.conversations:
            self.conversations[conv_id] = []

        history = self.conversations[conv_id][-6:]

        messages = [{"role": "system", "content": combined_system}]
        messages.extend(history)

        correction_patterns = ["不对", "错了", "不是这样", "应该是", "纠正", "你搞错了", "错误"]
        if any(p in request.message for p in correction_patterns):
            messages.append({"role": "system", "content": BAIT_TOPIC_PROMPT})

        messages.append({"role": "user", "content": request.message})

        try:
            max_iterations = 5
            tool_results = []
            answer = ""

            for iteration in range(max_iterations):
                response = await self.client.chat.completions.create(
                    model=self.settings.deepseek_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1536,
                    tools=FUNCTION_TOOLS,
                    tool_choice="auto",
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)

                        if func_name == "ask_user":
                            question = func_args.get("question", "请确认是否执行此操作？")
                            confirm_msg = json.dumps({
                                "type": "confirm_action",
                                "question": question,
                                "pending_action": func_args.get("pending_action"),
                            }, ensure_ascii=False)

                            history.append({"role": "user", "content": request.message})
                            history.append({"role": "assistant", "content": f'🔔 {question}\n\n（请回复\"确认\"执行修改，或回复\"取消\"放弃）\n\n`[待确认操作: {json.dumps(func_args.get("pending_action", {}), ensure_ascii=False)}]`'})
                            self.conversations[conv_id] = history
                            self.memory_service.save_conversation_message(conv_id, user_id, "user", request.message)
                            self.memory_service.save_conversation_message(conv_id, user_id, "assistant", f"🔔 {question}")

                            return ConversationResponse(
                                answer=f'🔔 {question}\n\n请回复\"确认\"以执行修改，或回复\"取消\"放弃操作。',
                                conversation_id=conv_id,
                                sources=[],
                                related_questions=[],
                                low_confidence_info=low_conf_info,
                                tool_call_result=confirm_msg,
                            )

                        result = await self._execute_tool_call(func_name, func_args)
                        tool_results.append({"tool": func_name, "result": result})

                        messages.append(msg.model_dump())
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                else:
                    answer = msg.content or ""
                    break

            if not answer:
                answer = "抱歉，我无法完成该操作。请重试。"

            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": answer})
            self.conversations[conv_id] = history

            self.memory_service.save_conversation_message(conv_id, user_id, "user", request.message)
            self.memory_service.save_conversation_message(conv_id, user_id, "assistant", answer)

            if any(p in request.message for p in correction_patterns):
                await self._learn_from_correction(user_id, request.message, answer)
                self.record_correction(request.message, answer)

            sources = await self._extract_sources(request.message)
            conflicts = await self._detect_conflicts(request.message)
            gaps = await self._detect_gaps(request.message)
            related = self._generate_related_questions(answer, intent)

            self.hallucination_stats["total_answers"] += 1
            if sources:
                self.hallucination_stats["citations_provided"] += 1
            else:
                self.hallucination_stats["citations_missing"] += 1
            if conflicts:
                self.hallucination_stats["conflicts_detected"] += len(conflicts)

            answer_metadata = self.build_answer_metadata(retrieval_quality, sources, conflicts)

            result = ConversationResponse(
                answer=answer,
                conversation_id=conv_id,
                sources=sources,
                related_questions=related,
                detected_conflicts=conflicts,
                knowledge_gaps=gaps,
                low_confidence_info=low_conf_info,
                answer_metadata=answer_metadata,
            )
            if tool_results:
                result.tool_call_result = json.dumps(tool_results, ensure_ascii=False)
            return result

        except Exception as e:
            logger.error(f"对话服务错误: {e}")
            return ConversationResponse(
                answer=f"抱歉，我暂时无法处理您的请求。错误信息：{str(e)}",
                conversation_id=conv_id,
                sources=[],
                related_questions=[],
            )

    async def chat_stream(self, request: ConversationRequest):
        from fastapi.responses import StreamingResponse

        conv_id = request.conversation_id or "default"
        user_id = self._resolve_user_id(request)
        self._check_style_adaptation(user_id, request.message)

        async def generate():
            try:
                intent = await self._analyze_intent(request.message)

                yield f"data: {json.dumps({'type': 'status', 'status': 'retrieving', 'message': '正在检索知识库...'})}\n\n"

                knowledge_context, retrieval_quality = await self._retrieve_with_quality_check(request.message)
                if retrieval_quality.get("is_empty"):
                    self.record_gap(request.message, "missing", intent.get("topic", ""))
                    metadata = self.build_answer_metadata(retrieval_quality, [], [])
                    yield f"data: {json.dumps({'type': 'answer', 'content': '知识库中暂无相关信息。建议您上传包含该主题的文档，或尝试其他关键词搜索。\\n\\n我已记录此知识缺口。', 'metadata': metadata, 'done': True})}\n\n"
                    return
                elif retrieval_quality.get("is_below_threshold"):
                    self.record_gap(request.message, "low_quality", intent.get("topic", ""))
                    self.hallucination_stats["below_threshold_refusals"] += 1
                else:
                    if retrieval_quality.get("multi_source_available"):
                        self.hallucination_stats["multi_source_verified_answers"] += 1

                if intent["type"] in ("reasoning", "comparison"):
                    sub_questions = await self._decompose_complex_question(request.message)
                    if sub_questions:
                        yield f"data: {json.dumps({'type': 'status', 'status': 'decomposing', 'message': f'拆解为{len(sub_questions)}个子问题...'})}\n\n"
                        sub_results = []
                        for sq in sub_questions:
                            sub_knowledge = await self._retrieve_knowledge(sq)
                            if sub_knowledge and "知识库中暂无" not in sub_knowledge:
                                sub_results.append(f"【子问题】{sq}\n{sub_knowledge}")
                        if sub_results:
                            knowledge_context += "\n\n---\n## 多步推理子问题检索\n\n" + "\n\n".join(sub_results)
                graph_context = await self._retrieve_graph_enhanced(request.message, intent)
                timeline_context = await self._retrieve_timeline(request.message) if intent["is_timeline_related"] else ""
                category_context = await self._retrieve_categories(request.message)

                retrieved_points = await self.vector_service.search_knowledge(request.message, top_k=5)
                low_conf_info = self.build_low_confidence_context(request.message, retrieved_points)
                low_confidence_context = ""
                if low_conf_info["count"] > 0:
                    items_text = "\n".join([
                        f"- [⚠ {item['confidence']:.2f}] {item['fact'][:150]} (来源: {item['source'][:60]})"
                        for item in low_conf_info["items"]
                    ])
                    low_confidence_context = f"发现 {low_conf_info['count']} 个低置信度知识点：\n{items_text}\n\n{LOW_CONFIDENCE_PROMPT_V2}"
                    warning_msg = f"⚠ 发现 {low_conf_info['count']} 个低置信度知识点，回答时已提示谨慎对待"
                    yield f"data: {json.dumps({'type': 'warning', 'message': warning_msg, 'low_confidence': low_conf_info})}\n\n"

                should_search = False
                if request.enable_web_search:
                    should_search = await self._should_search_web(request.message, knowledge_context)
                web_context = ""
                search_results = []

                if should_search:
                    yield f"data: {json.dumps({'type': 'status', 'status': 'searching', 'message': '正在联网搜索...'})}\n\n"
                    search_results = await self._perform_web_search_raw(request.message)
                    web_context = self._format_web_results(search_results)
                    yield f"data: {json.dumps({'type': 'status', 'status': 'search_done', 'message': f'找到 {len(search_results)} 条网络结果'})}\n\n"

                user_profile_context = self._format_user_profile(user_id)
                style_instruction = self._get_style_instruction(user_id)
                few_shot_context_stream = await self._build_few_shot_examples(request.message)
                content_type_hint = self._content_type_prompt(intent)

                combined_system = (
                    SYSTEM_PROMPT_V2 + style_instruction + "\n" + content_type_hint
                ).format(
                    knowledge_context=knowledge_context,
                    low_confidence_context=low_confidence_context,
                    graph_context=graph_context,
                    timeline_context=timeline_context,
                    category_context=category_context,
                    web_context=web_context,
                    user_profile_context=user_profile_context,
                    few_shot_context=few_shot_context_stream,
                )

                yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': '正在分析...'})}\n\n"

                if conv_id not in self.conversations:
                    self.conversations[conv_id] = []

                history = self.conversations[conv_id][-10:]

                messages = [{"role": "system", "content": combined_system}]
                messages.extend(history)

                correction_patterns = ["不对", "错了", "不是这样", "应该是", "纠正", "你搞错了", "错误"]
                if any(p in request.message for p in correction_patterns):
                    messages.append({"role": "system", "content": BAIT_TOPIC_PROMPT})

                messages.append({"role": "user", "content": request.message})

                stream = await self.client.chat.completions.create(
                    model=self.settings.deepseek_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True,
                )
                full_answer = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_answer += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                history.append({"role": "user", "content": request.message})
                history.append({"role": "assistant", "content": full_answer})
                self.conversations[conv_id] = history

                self.memory_service.save_conversation_message(conv_id, user_id, "user", request.message)
                self.memory_service.save_conversation_message(conv_id, user_id, "assistant", full_answer)

                sources = await self._extract_sources(request.message)
                if search_results:
                    sources = self._merge_sources(sources, search_results)
                conflicts = await self._detect_conflicts(request.message)
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                if search_results:
                    yield f"data: {json.dumps({'web_results': search_results})}\n\n"

                if any(p in request.message for p in correction_patterns):
                    await self._learn_from_correction(user_id, request.message, full_answer)
                    self.record_correction(request.message, full_answer)

                self.hallucination_stats["total_answers"] += 1
                if sources:
                    self.hallucination_stats["citations_provided"] += 1
                else:
                    self.hallucination_stats["citations_missing"] += 1
                if conflicts:
                    self.hallucination_stats["conflicts_detected"] += len(conflicts)

                answer_metadata = self.build_answer_metadata(retrieval_quality, sources, conflicts)
                yield f"data: {json.dumps({'metadata': answer_metadata})}\n\n"

                learn_result = await self.try_learn(request.message, full_answer)
                if learn_result["learned"] > 0:
                    yield f"data: {json.dumps({'learned': learn_result})}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"流式对话错误: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    def _content_type_prompt(self, intent: dict) -> str:
        ct = intent.get("content_type", "general")
        if ct == "code":
            return """
## 代码回答规范
- 先说明"这段代码实现了什么"，再展示关键逻辑
- 提及函数时标注文件路径和行号
- 解释算法时用通俗语言说明输入输出和复杂度
- 如有依赖关系，说明调用链
"""
        if ct == "audiovisual":
            return """
## 音视频回答规范
- 引用时标注时间戳和说话人，如"小王在 03:15 提出..."
- 说明场景切换和关键画面
- 区分不同发言人的观点
"""
        if ct == "image_chart":
            return """
## 图表回答规范
- 先总结趋势或核心发现
- 再列出具体数据（如适用）
- 区分图表类型：流程图说明步骤，柱状图对比数值，趋势图说明变化方向
"""
        if intent.get("is_comparison"):
            return """
## 对比回答规范
- 使用对比结构：维度1(A vs B)、维度2(A vs B)...
- 先给出结论（哪个更优/差异所在），再展开对比
- 如涉及矛盾，明确指出并给出置信度对比
"""
        return ""

    async def _retrieve_graph_enhanced(self, query: str, intent: dict) -> str:
        lines = []

        if intent.get("needs_graph_exploration") or intent.get("is_comparison"):
            entities = intent.get("entities", [])
            if entities:
                for entity in entities[:3]:
                    neighbor_info = await self._retrieve_graph_for_entity(entity)
                    if neighbor_info:
                        lines.append(neighbor_info)

        graph_result = await self.graph_service.explore("", limit=20)
        if graph_result.get("edges"):
            if lines:
                lines.append("")
            lines.append("### 知识图谱总览")
            for edge in graph_result["edges"][:15]:
                rel_type = edge.get("relation_type", "RELATED_TO")
                conf = edge.get("confidence") or 0.5
                conf_flag = "⚠" if conf < 0.5 else ""
                lines.append(f"- {edge['source']} --[{edge.get('relation', '关联')}]({rel_type}){conf_flag}--> {edge['target']}")

        if not lines:
            return "（暂无图谱关联数据）"
        return "\n".join(lines)

    async def _retrieve_timeline(self, query: str) -> str:
        try:
            points = await self.vector_service.search_knowledge(query, top_k=10)
            timed_points = [p for p in points if getattr(p, "event_time", None)]
            if not timed_points:
                return "（暂无时间线数据）"
            timed_points.sort(key=lambda x: x.event_time if x.event_time else "")
            lines = ["### 相关事件时间线"]
            for i, p in enumerate(timed_points[:8]):
                time_str = p.event_time or "未知时间"
                precision = getattr(p, "time_precision", "")
                precision_str = f"({precision})" if precision else ""
                lines.append(f"{i+1}. [{time_str}]{precision_str} {p.fact[:100]}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"时间线检索失败: {e}")
            return ""

    async def _retrieve_categories(self, query: str) -> str:
        if not self.category_service:
            return ""
        try:
            points = await self.vector_service.search_knowledge(query, top_k=5)
            if not points:
                return ""
            cat_counts = {}
            for p in points:
                cat = getattr(p, "category", None)
                if cat:
                    cat_name = cat.value if hasattr(cat, 'value') else str(cat)
                    cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            if cat_counts:
                cat_list = ", ".join([f"{k}({v}条)" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])[:5]])
                return f"检索结果涉及分类: {cat_list}"
            return ""
        except Exception:
            return ""

    async def _retrieve_knowledge(self, query: str, low_threshold: float = None) -> str:
        if low_threshold is None:
            low_threshold = self.settings.low_confidence_threshold
        points = await self.vector_service.search_knowledge(query, top_k=8)
        if not points:
            return "（知识库中暂无相关内容）"
        active_points = [p for p in points if getattr(p, "status", "active") == "active"]
        if not active_points:
            return "（知识库中暂无相关内容）"
        high_conf_lines = []
        low_conf_lines = []
        for p in active_points[:5]:
            source_doc = getattr(p, "source_document_id", "") or p.source
            line = f"{len(high_conf_lines)+len(low_conf_lines)+1}. {p.fact} [来源: {source_doc}] [置信度: {p.confidence:.2f}]"
            if p.confidence < low_threshold:
                low_conf_lines.append(f"[⚠低置信度] {line}")
            else:
                high_conf_lines.append(line)
        result = "\n".join(high_conf_lines + low_conf_lines)
        return result

    async def _retrieve_with_quality_check(self, query: str) -> tuple[str, dict]:
        scored = await self.vector_service.search_with_scores(query, top_k=8)
        if not scored:
            return "（知识库中暂无相关内容）", {
                "retrieval_count": 0, "avg_similarity": 0.0, "max_similarity": 0.0,
                "is_below_threshold": True, "is_empty": True,
            }

        similarities = [s for s, _ in scored]
        active_scored = [(s, d) for s, d in scored if d.get("status", "active") == "active"]
        if not active_scored:
            return "（知识库中暂无相关内容）", {
                "retrieval_count": 0, "avg_similarity": 0.0, "max_similarity": max(similarities) if similarities else 0.0,
                "is_below_threshold": True, "is_empty": True,
            }

        active_sims = [s for s, _ in active_scored]
        avg_sim = sum(active_sims) / len(active_sims) if active_sims else 0.0
        max_sim = max(active_sims)

        RETRIEVAL_THRESHOLD = 0.5
        is_below = max_sim < RETRIEVAL_THRESHOLD

        low_threshold = self.settings.low_confidence_threshold
        high_conf_lines = []
        low_conf_lines = []

        for i, (sim, data) in enumerate(active_scored[:5]):
            source_doc = data.get("source_document_id", "") or data.get("source", "")
            fact = data.get("fact", "")
            confidence = data.get("confidence", 0.5)
            created_at = data.get("created_at", "")
            line = f"{len(high_conf_lines)+len(low_conf_lines)+1}. {fact} [来源: {source_doc}] [置信度: {confidence:.2f}] [相似度: {sim:.2f}]"
            if created_at:
                try:
                    from datetime import datetime as dt
                    age_days = (dt.now() - dt.fromisoformat(created_at.replace("Z", "+00:00"))).days
                    if age_days > 365:
                        line += f" [⏰已过{age_days}天]"
                except Exception:
                    pass
            if confidence < low_threshold:
                low_conf_lines.append(f"[⚠低置信度] {line}")
            else:
                high_conf_lines.append(line)

        if is_below:
            result = "[检索质量: 低于阈值]\n知识库中未找到与问题足够匹配的内容（最高相似度 {:.2f}）。\n".format(max_sim)
            if high_conf_lines or low_conf_lines:
                result += "以下是相关性最高的部分内容（仅供参考）：\n"
                result += "\n".join(high_conf_lines + low_conf_lines)
        else:
            result = "\n".join(high_conf_lines + low_conf_lines)

        return result, {
            "retrieval_count": len(active_scored),
            "avg_similarity": round(avg_sim, 4),
            "max_similarity": round(max_sim, 4),
            "is_below_threshold": is_below,
            "is_empty": False,
            "multi_source_available": len(active_scored) >= 2,
        }

    async def _build_few_shot_examples(self, query: str) -> str:
        try:
            similar = await self.vector_service.search_knowledge(query, top_k=3)
            if not similar:
                return "（暂无相关示例）"
            high_quality = [kp for kp in similar if kp.confidence > 0.6 and kp.fact]
            if not high_quality:
                return "（暂无高质量示例）"
            examples = []
            for i, kp in enumerate(high_quality[:3]):
                examples.append(f"示例{i+1}: {kp.fact[:150]} (置信度:{kp.confidence:.2f})")
            return "以下是与当前问题相关的知识示例，可参考其表述风格：\n" + "\n".join(examples)
        except Exception as e:
            logger.debug(f"少样本示例生成失败: {e}")
            return "（示例生成暂时不可用）"

    async def _decompose_complex_question(self, query: str) -> list[str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{
                    "role": "system",
                    "content": "你将复杂问题拆解为2-4个简单子问题。每个子问题应该独立可回答。只返回JSON数组，如[\"子问题1\", \"子问题2\"]。"
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.2,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                sub_questions = json.loads(json_match.group())
                return [q for q in sub_questions if q and q != query][:4]
        except Exception as e:
            logger.debug(f"问题拆解失败: {e}")
        return []

    def build_low_confidence_context(self, query: str, points: list) -> dict:
        threshold = self.settings.low_confidence_threshold
        low_confidence_items = []
        for p in points:
            status = getattr(p, "status", "active")
            if status != "active":
                continue
            if p.confidence < threshold:
                low_confidence_items.append({
                    "id": getattr(p, "id", ""),
                    "fact": p.fact,
                    "confidence": p.confidence,
                    "source": p.source,
                    "source_quality": getattr(p, "source_quality", 0.5),
                })
        return {
            "count": len(low_confidence_items),
            "items": low_confidence_items,
            "is_critical": len(low_confidence_items) > 0 and len(low_confidence_items) >= len(points) * 0.4,
        }

    async def _should_search_web(self, query: str, knowledge_context: str) -> bool:
        if not self.web_search_service:
            return False
        if "暂无" in knowledge_context:
            return True
        try:
            check_prompt = f"""判断用户问题是否需要联网搜索最新信息。只需要回答 YES 或 NO。

需要搜索的情况：
- 询问最新资讯、新闻、实时数据
- 询问当前时间、日期、天气等实时信息
- 知识库内容不足以回答

不需要搜索的情况：
- 纯理论、概念解释类问题且知识库有相关内容
- 编程语法、数学计算等确定性知识

用户问题: {query}
知识库摘要: {knowledge_context[:200]}"""
            resp = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "system", "content": "你是搜索需求判断专家。只回答YES或NO。"}, {"role": "user", "content": check_prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            result = (resp.choices[0].message.content or "").strip().upper()
            return "YES" in result
        except Exception:
            return "暂无" in knowledge_context

    async def _perform_web_search(self, query: str) -> str:
        results = await self._perform_web_search_raw(query)
        return self._format_web_results(results)

    async def _perform_web_search_raw(self, query: str) -> list[dict]:
        if not self.web_search_service:
            return []
        try:
            return await self.web_search_service.search(query, max_results=5)
        except Exception as e:
            logger.warning(f"联网搜索失败: {e}")
            return []

    def _format_web_results(self, results: list[dict]) -> str:
        if not results:
            return "（未进行网络搜索或搜索结果为空）"
        lines = []
        for i, r in enumerate(results):
            lines.append(f"{i+1}. [{r['title']}] {r['snippet']} (来源: {r.get('url', '')})")
        return "\n".join(lines)

    def _merge_sources(self, kb_sources: list[dict], web_results: list[dict]) -> list[dict]:
        merged = list(kb_sources)
        for r in web_results[:3]:
            merged.append({
                "fact": r.get("snippet", "")[:100],
                "source": r.get("url", "网络搜索"),
                "confidence": 0.7,
                "category": "web_search",
            })
        return merged

    async def _retrieve_graph(self, query: str) -> str:
        return await self._retrieve_graph_enhanced(query, {"needs_graph_exploration": True, "entities": []})

    async def _retrieve_graph_for_entity(self, entity_name: str) -> str:
        try:
            neighbors = await self.graph_service.get_entity_neighbors(entity_name)
            if not neighbors.get("neighbors"):
                return ""
            lines = [f"知识图谱 - 实体「{entity_name}」的关联："]
            for n in neighbors["neighbors"][:10]:
                lines.append(f"  - {n['relation']}({n.get('relation_type', '')}) → {n['name']}")
            return "\n".join(lines)
        except Exception:
            return ""

    async def query_graph(self, cypher: str) -> list[dict]:
        return await self.graph_service.execute_cypher(cypher)

    async def _extract_sources(self, query: str) -> list[dict]:
        points = await self.vector_service.search_knowledge(query, top_k=3)
        sources = []
        for p in points:
            sources.append({
                "id": p.id,
                "fact": p.fact[:100],
                "source": p.source,
                "source_document_id": getattr(p, "source_document_id", ""),
                "chunk_id": getattr(p, "chunk_id", ""),
                "confidence": p.confidence,
                "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                "event_time": getattr(p, "event_time", None),
            })
        return sources

    async def _detect_conflicts(self, query: str) -> list[str]:
        points = await self.vector_service.search_knowledge(query, top_k=10)
        conflicts = []
        facts_seen = {}
        for p in points:
            normalized = re.sub(r'[，,。；;：:\s]+', '', p.fact[:30])
            if normalized in facts_seen:
                conflicts.append(f"存在相似陈述：「{p.fact[:50]}...」和「{facts_seen[normalized][:50]}...」，建议核实")
            else:
                facts_seen[normalized] = p.fact
        return conflicts[:3]

    async def _detect_gaps(self, query: str) -> list[str]:
        points = await self.vector_service.search_knowledge(query, top_k=5)
        if not points:
            return ["知识库中暂无与此问题直接相关的内容，可以考虑补充相关资料"]
        low_conf = [p for p in points if p.confidence < 0.6]
        gaps = []
        if low_conf:
            gaps.append(f"有 {len(low_conf)} 个相关知识点置信度较低，建议验证")
        return gaps[:3]

    def _generate_related_questions(self, answer: str, intent: dict = None) -> list[str]:
        questions = []
        if intent:
            if intent.get("content_type") == "code":
                questions.append("需要我列出这个函数的调用关系吗？")
                questions.append("需要解释其中用到的算法原理吗？")
            elif intent.get("is_comparison"):
                questions.append("需要我进一步分析各选项的优劣吗？")
                questions.append("需要查看这些选项在知识图谱中的关联吗？")
            elif intent.get("is_timeline_related"):
                questions.append("需要我展开某个具体阶段吗？")
                questions.append("需要查看完整时间线吗？")
        questions.append("您想深入了解哪个方面？")
        questions.append("是否需要查看知识图谱中的关联信息？")
        questions.append("需要我进一步展开说明吗？")
        return questions

    async def _learn_from_correction(self, user_id: str, user_message: str, assistant_response: str):
        try:
            prompt = f"""用户纠正了AI的回答。分析这次纠正，判断是否包含需要记住的事实或偏好。

用户消息: {user_message}
AI回答: {assistant_response[:200]}

如果用户纠正了术语用法或事实错误，返回JSON：{{"memory_key": "term_preference或fact", "memory_value": "具体的纠正内容", "should_remember": true}}
如果只是简单的"不对"，返回：{{"should_remember": false}}"""
            resp = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "system", "content": "你是知识学习判断专家。只返回JSON。"}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            result = (resp.choices[0].message.content or "").strip()
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("should_remember"):
                    self._update_user_profile(user_id, parsed["memory_key"], parsed["memory_value"])
                    logger.info(f"对话学习: {parsed['memory_key']}={parsed['memory_value']}")
        except Exception as e:
            logger.debug(f"纠正学习失败: {e}")

    async def try_learn(self, user_message: str, assistant_response: str) -> dict:
        if not self.knowledge_extractor:
            return {"learned": 0, "triples": 0, "message": "知识提取器未配置"}

        learn_trigger = any(kw in user_message for kw in [
            "不对", "应该是", "其实是", "纠正", "更正", "记住", "学习了",
            "实际上", "正确的说法", "补充一下", "另外", "还有", "需要知道",
        ])

        if not learn_trigger:
            return {"learned": 0, "triples": 0, "message": ""}

        try:
            learn_prompt = f"""判断用户消息是否包含值得学习的事实知识或对AI回答的纠正。

用户消息: {user_message}
AI回答: {assistant_response[:200]}

如果用户消息包含新的事实知识或纠正，提取出知识点JSON数组。否则返回空数组[]。
输出格式：{{"should_learn": true/false, "text": "提取的纯文本知识内容"}}"""

            response = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "system", "content": "你是知识学习判断专家。"}, {"role": "user", "content": learn_prompt}],
                temperature=0.1,
                max_tokens=512,
            )

            result_text = response.choices[0].message.content or ""
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if not json_match:
                return {"learned": 0, "triples": 0, "message": ""}

            decision = json.loads(json_match.group())
            if not decision.get("should_learn") or not decision.get("text"):
                return {"learned": 0, "triples": 0, "message": ""}

            from models import DocumentChunk
            chunks = [DocumentChunk(content=decision["text"], chunk_index=0, source_path="对话学习")]
            points = await self.knowledge_extractor.extract_from_chunks(chunks)

            if not points:
                return {"learned": 0, "triples": 0, "message": ""}

            await self.vector_service.add_knowledge(points)

            triples_count = 0
            triples = await self.knowledge_extractor.extract_triples(points)
            if triples:
                await self.graph_service.create_triples(triples)
                triples_count = len(triples)

            logger.info(f"对话自学习: 提取 {len(points)} 条知识, {triples_count} 条三元组, 来源: {user_message[:50]}...")
            return {"learned": len(points), "triples": triples_count, "message": f"已从对话中学习 {len(points)} 条新知识"}

        except Exception as e:
            logger.warning(f"对话自学习失败: {e}")
            return {"learned": 0, "triples": 0, "message": ""}

    async def update_memory(self, user_id: str, key: str, value: str):
        self._update_user_profile(user_id, key, value)

    async def get_user_profile(self, user_id: str) -> dict:
        return self._get_user_profile(user_id)

    async def get_full_memory(self, user_id: str) -> dict:
        return self.memory_service.get_all_user_memory(user_id)

    async def delete_memory_item(self, user_id: str, memory_type: str, memory_key: str) -> bool:
        return self.memory_service.delete_memory_item(user_id, memory_type, memory_key)

    async def decay_memory(self):
        self.memory_service.decay_memory()