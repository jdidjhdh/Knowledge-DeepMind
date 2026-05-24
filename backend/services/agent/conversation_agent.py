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
{user_profile_context}"""


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
            "description": "当信息不足、存在歧义、发现矛盾或需要用户确认时，主动向用户提问。",
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
                },
                "required": ["question", "question_type"],
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

    async def chat(self, request: ConversationRequest) -> ConversationResponse:
        conv_id = request.conversation_id or "default"
        user_id = self._resolve_user_id(request)
        self._check_style_adaptation(user_id, request.message)

        intent = await self._analyze_intent(request.message)

        knowledge_context = await self._retrieve_knowledge(request.message)
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
            response = await self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1536,
            )

            answer = response.choices[0].message.content or ""

            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": answer})
            self.conversations[conv_id] = history

            self.memory_service.save_conversation_message(conv_id, user_id, "user", request.message)
            self.memory_service.save_conversation_message(conv_id, user_id, "assistant", answer)

            if any(p in request.message for p in correction_patterns):
                await self._learn_from_correction(user_id, request.message, answer)

            sources = await self._extract_sources(request.message)
            conflicts = await self._detect_conflicts(request.message)
            gaps = await self._detect_gaps(request.message)
            related = self._generate_related_questions(answer, intent)

            return ConversationResponse(
                answer=answer,
                conversation_id=conv_id,
                sources=sources,
                related_questions=related,
                detected_conflicts=conflicts,
                knowledge_gaps=gaps,
                low_confidence_info=low_conf_info,
            )

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

                knowledge_context = await self._retrieve_knowledge(request.message)
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
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                if search_results:
                    yield f"data: {json.dumps({'web_results': search_results})}\n\n"

                if any(p in request.message for p in correction_patterns):
                    await self._learn_from_correction(user_id, request.message, full_answer)

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
                conf = edge.get("confidence", 0.5)
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