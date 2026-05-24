import logging
import os
import base64
import uuid
import json
import math
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, HttpUrl

from config import get_settings
from models import (
    ConversationRequest, ConversationResponse, SearchRequest, SearchResult,
    IngestionTask, DocumentType, DocumentChunk, KnowledgePoint, KnowledgeFeedback,
    KnowledgeCorrectRequest, LowConfidenceReviewRequest, FactCategory,
    Category, CategoryType, CategoryRelation, CategoryRelationType,
    UserTag, KnowledgeTagAssignment, SmartCollection, UserCategoryPrefs,
    CategoryTree, CategoryHealth, CategoryEvolutionEvent, ClusteringResult,
    KnowledgeTimeline, MultiDimensionFilter, KnowledgeListResponse,
    SourceGroup, SourceComparison, TimelineGroup, TimelineGap,
    TimelineBurst, VersionChain, TimelineResponse, ConfidenceTier,
    PaginationResponse,
    EntityType, RelationType, KnowledgeTriple, GraphNodeDetail, GraphEdgeDetail,
    MultiHopPath, CommunityResult, EntityNormalizeRequest, RuleInference,
    GraphSyncRequest,
    DedupCheckResult, DedupStats, DedupMode,
    CategoryMergeSuggestion, BatchCategoryAssignRequest, AutoCategorizeResult,
)
from services.ingestion.file_handler import FileHandler
from services.extraction.knowledge_extractor import KnowledgeExtractor
from services.graph.neo4j_service import GraphService
from services.graph.vector_service import VectorService
from services.agent.conversation_agent import ConversationAgent
from services.agent.search_service import SearchService
from services.agent.web_search_service import WebSearchService
from services.confidence.confidence_calculator import ConfidenceCalculator
from services.category import CategoryService
from services.extraction.time_extraction_service import TimeExtractionService
from services.dedup_service import DedupService
from services.memory_service import MemoryService
from services.auth_service import AuthService
from middleware.auth import get_current_user, get_optional_user

logger = logging.getLogger(__name__)

settings = get_settings()

file_handler: FileHandler = None
knowledge_extractor: KnowledgeExtractor = None
graph_service: GraphService = None
vector_service: VectorService = None
conversation_agent: ConversationAgent = None
search_service: SearchService = None
web_search_service: WebSearchService = None
confidence_calculator: ConfidenceCalculator = None
category_service: CategoryService = None
time_extraction_service: TimeExtractionService = None
dedup_service: DedupService = None
memory_service: MemoryService = None
auth_service: AuthService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global file_handler, knowledge_extractor, graph_service, vector_service
    global conversation_agent, search_service, web_search_service, confidence_calculator
    global category_service, dedup_service, memory_service, auth_service

    logger.info("正在初始化知识库智能体服务...")
    file_handler = FileHandler(settings)
    graph_service = GraphService(settings)
    vector_service = VectorService(settings)
    dedup_service = DedupService(settings)
    memory_service = MemoryService()
    auth_service = AuthService(
        db_path="data/auth.db",
        secret_key=settings.secret_key,
        access_token_expire_minutes=settings.access_token_expire_minutes,
    )
    web_search_service = WebSearchService()
    confidence_calculator = ConfidenceCalculator(settings, graph_service, vector_service)
    category_service = CategoryService(settings, vector_service, graph_service, confidence_calculator)
    knowledge_extractor = KnowledgeExtractor(settings, confidence_calculator, category_service)
    conversation_agent = ConversationAgent(settings, vector_service, graph_service, knowledge_extractor, web_search_service, category_service, dedup_service, memory_service)
    search_service = SearchService(vector_service, graph_service)
    time_extraction_service = TimeExtractionService()

    await graph_service.initialize()
    await vector_service.initialize()
    await dedup_service.initialize()
    await category_service.initialize()

    logger.info("知识库智能体服务初始化完成")
    yield

    await graph_service.close()
    logger.info("知识库智能体服务已关闭")


app = FastAPI(
    title="全格式自进化知识库智能体",
    description="支持全格式文件摄入、自动知识抽取、对话式知识检索的智能知识库系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "knowledge-brain"}


class ModelSettings(BaseModel):
    deepseek_enabled: bool = True
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    vision_enabled: bool = False
    vision_model: str = "qwen-vl-max"
    vision_api_key: str = ""
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    speech_model: str = "qwen-audio-turbo-latest"
    speech_api_key: str = ""
    speech_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    speech_enabled: bool = True


def _get_env_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_file() -> dict[str, str]:
    env_path = _get_env_path()
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
    return env_vars


def _write_env_file(env_vars: dict[str, str]) -> None:
    env_path = _get_env_path()
    sections = [
        ("# ====== 基础 LLM 配置（DeepSeek）======", [
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        ]),
        ("# ====== 语音识别配置（云端 ASR）======", [
            "SPEECH_ENABLED", "SPEECH_MODEL", "SPEECH_API_KEY", "SPEECH_BASE_URL",
        ]),
        ("# ====== 视觉模型配置（通义千问 VL）======", [
            "VISION_ENABLED", "VISION_MODEL", "VISION_API_KEY", "VISION_BASE_URL",
        ]),
    ]
    non_model_keys = {"POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
                      "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
                      "REDIS_URL", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET",
                      "UPLOAD_DIR", "MAX_UPLOAD_SIZE_MB",
                      "EMBEDDING_MODEL", "VECTOR_DIM",
                      "SECRET_KEY", "ACCESS_TOKEN_EXPIRE_MINUTES",
                      "LOW_CONFIDENCE_THRESHOLD", "AUTO_REVIEW_INTERVAL_DAYS", "ENABLE_EXTERNAL_SEARCH"}

    lines = []
    for comment, keys in sections:
        lines.append(comment)
        for k in keys:
            v = env_vars.get(k, _get_default_value(k))
            lines.append(f"{k}={v}")
        lines.append("")

    remaining = {k: v for k, v in env_vars.items()
                 if k not in {kk for _, kks in sections for kk in kks}
                 and k not in non_model_keys}
    if remaining:
        lines.append("# ====== 其他配置 ======")
        for k, v in remaining.items():
            lines.append(f"{k}={v}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _get_default_value(key: str) -> str:
    defaults = {
        "DEEPSEEK_API_KEY": "",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "WHISPER_MODEL": "large-v3",
        "VISION_ENABLED": "false",
        "VISION_MODEL": "qwen-vl-max",
        "VISION_API_KEY": "",
        "VISION_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "SPEECH_MODEL": "qwen-audio-turbo-latest",
        "SPEECH_API_KEY": "",
        "SPEECH_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "SPEECH_ENABLED": "true",
    }
    return defaults.get(key, "")


@app.get("/api/settings/models")
async def get_model_settings():
    return ModelSettings(
        deepseek_enabled=bool(settings.deepseek_api_key),
        deepseek_model=settings.deepseek_model,
        deepseek_api_key=settings.deepseek_api_key[:8] + "***" if settings.deepseek_api_key else "",
        deepseek_base_url=settings.deepseek_base_url,
        vision_enabled=settings.vision_enabled,
        vision_model=settings.vision_model,
        vision_api_key=settings.vision_api_key[:8] + "***" if settings.vision_api_key else "",
        vision_base_url=settings.vision_base_url,
        speech_model=settings.speech_model,
        speech_api_key=settings.speech_api_key[:8] + "***" if settings.speech_api_key else "",
        speech_base_url=settings.speech_base_url,
        speech_enabled=settings.speech_enabled,
    )


@app.put("/api/settings/models")
async def update_model_settings(body: ModelSettings):
    env_vars = _read_env_file()

    env_vars["DEEPSEEK_MODEL"] = body.deepseek_model
    env_vars["DEEPSEEK_BASE_URL"] = body.deepseek_base_url
    if body.deepseek_api_key and "***" not in body.deepseek_api_key:
        env_vars["DEEPSEEK_API_KEY"] = body.deepseek_api_key

    env_vars["SPEECH_MODEL"] = body.speech_model
    env_vars["SPEECH_BASE_URL"] = body.speech_base_url
    env_vars["SPEECH_ENABLED"] = "true" if body.speech_enabled else "false"
    if body.speech_api_key and "***" not in body.speech_api_key:
        env_vars["SPEECH_API_KEY"] = body.speech_api_key

    env_vars["VISION_ENABLED"] = "true" if body.vision_enabled else "false"
    env_vars["VISION_MODEL"] = body.vision_model
    env_vars["VISION_BASE_URL"] = body.vision_base_url
    if body.vision_api_key and "***" not in body.vision_api_key:
        env_vars["VISION_API_KEY"] = body.vision_api_key

    _write_env_file(env_vars)

    global settings
    from config import Settings
    settings = Settings()

    logger.info("[设置] 模型配置已更新，需重启服务以完全生效")

    return {
        "status": "saved",
        "message": "配置已保存到 .env 文件",
        "needs_restart": True,
    }


@app.post("/api/ingest/file", response_model=IngestionTask)
async def ingest_file(
    file: UploadFile = File(...),
    file_type: DocumentType = Query(default=DocumentType.TEXT),
    force_ingest: bool = Query(default=False),
):
    if not file_handler:
        raise HTTPException(status_code=503, detail="服务未初始化")
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    content = await file.read()
    file_size = len(content)
    await file.seek(0)

    dedup_info = None
    if dedup_service and not force_ingest:
        hash_match, matched = dedup_service.check_file_hash(content)
        if hash_match and matched:
            matched_name = matched.get("file_name", "未知")
            action, message, is_dup = dedup_service.determine_action(True, 0.0)
            if action == "skip":
                await dedup_service.log_duplicate(file.filename, f"哈希匹配严格跳过: {matched_name}", matched_name)
                return IngestionTask(
                    task_id=str(uuid.uuid4()),
                    file_path=file.filename or "",
                    file_type=file_type,
                    status="skipped",
                    error=f"[去重] {message} (匹配文件: {matched_name})",
                    result={"dedup": {"action": "skip", "reason": "hash_match", "matched_file": matched_name}},
                )
            dedup_info = {"hash_match": True, "matched_file": matched_name, "message": message}

    try:
        result = await file_handler.process_file(file, file_type)
        if result.status == "completed":
            try:
                chunks_data = result.result.get("chunks", [])
                if chunks_data:
                    chunks = [DocumentChunk(**c) for c in chunks_data]
                    knowledge_points = await knowledge_extractor.extract_from_chunks(chunks)

                    if dedup_service and not force_ingest:
                        combined_text = " ".join([kp.fact for kp in knowledge_points])[:3000]
                        max_sim, similar = await dedup_service.check_content_similarity(combined_text)
                        if max_sim > 0:
                            action, message, is_dup = dedup_service.determine_action(
                                dedup_info.get("hash_match", False) if dedup_info else False, max_sim
                            )
                            dedup_info = dedup_info or {}
                            dedup_info["content_similarity"] = max_sim
                            dedup_info["similar_files"] = similar
                            dedup_info["action"] = action
                            dedup_info["message"] = message
                            if action == "skip":
                                await dedup_service.log_duplicate(file.filename, f"内容相似严格跳过(sim={max_sim:.2f})")
                                result.result["dedup"] = dedup_info
                                result.status = "skipped"
                                result.error = f"[去重] {message}"
                                return result

                    for kp in knowledge_points:
                        await vector_service.index_knowledge_point(kp)
                    triples = await knowledge_extractor.extract_triples(knowledge_points)
                    try:
                        await graph_service.create_triples(triples)
                    except Exception as e:
                        logger.warning(f"[图谱] 三元组创建失败: {e}")

                code_analysis = result.result.get("code_analysis")
                if code_analysis and graph_service and graph_service.driver:
                    try:
                        await graph_service.create_code_structure(
                            file_path=code_analysis["file_path"],
                            filename=code_analysis["filename"],
                            language=code_analysis["language"],
                            static_result=code_analysis["static_result"],
                            semantic=code_analysis["semantic"],
                        )
                        logger.info(f"[图谱] 代码结构已存入: {code_analysis['filename']}")
                    except Exception as e:
                        logger.warning(f"[图谱] 代码结构存入失败: {e}")

                if dedup_service:
                    combined_text = " ".join([
                        c.get("content", "") for c in chunks_data
                    ])[:5000]
                    await dedup_service.register_file(
                        content=content,
                        file_name=file.filename or "",
                        file_size=file_size,
                        file_type=file_type,
                        saved_path=result.result.get("saved_path", ""),
                        task_id=result.task_id,
                        content_text=combined_text,
                    )

                if dedup_info:
                    result.result["dedup"] = dedup_info

            except Exception as e:
                logger.error(f"[摄取] 知识提取/索引失败 (文件已保存): {e}")
                result.result["indexing_error"] = str(e)
        return result
    except Exception as e:
        logger.error(f"[摄取] 未预期错误: {e}", exc_info=True)
        return IngestionTask(
            task_id=str(uuid.uuid4()),
            file_path=file.filename or "",
            file_type=file_type,
            status="failed",
            error=f"服务器内部错误: {str(e)}",
        )


class IngestUrlRequest(BaseModel):
    url: str


class IngestTextRequest(BaseModel):
    content: str
    source_name: str = "手动输入"
    format: str = "natural"


@app.post("/api/ingest/text", response_model=IngestionTask)
async def ingest_text(request: IngestTextRequest):
    if not file_handler or not knowledge_extractor:
        raise HTTPException(status_code=503, detail="服务未初始化")
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")

    import uuid
    task_id = str(uuid.uuid4())

    chunk = DocumentChunk(
        content=request.content.strip(),
        source_path=request.source_name,
        source_type=DocumentType.TEXT,
        chunk_index=0,
    )

    try:
        knowledge_points = await knowledge_extractor.extract_from_chunks([chunk])
        for kp in knowledge_points:
            kp.source = request.source_name
            await vector_service.index_knowledge_point(kp)
        if time_extraction_service:
            import asyncio
            async def _extract_time(kp):
                try:
                    event_times = await time_extraction_service.extract_event_times(kp.fact)
                    if event_times:
                        ev_time, precision = time_extraction_service.pick_best_event_time(event_times)
                        kp_data = vector_service.knowledge_index.get(kp.id)
                        if kp_data:
                            kp_data["data"]["event_time"] = ev_time
                            kp_data["data"]["time_precision"] = precision
                            kp_data["data"]["event_times"] = event_times
                            vector_service.knowledge_index[kp.id] = kp_data
                except Exception:
                    pass
            for kp in knowledge_points:
                asyncio.create_task(_extract_time(kp))
        triples = await knowledge_extractor.extract_triples(knowledge_points)
        if graph_service.driver:
            await graph_service.create_triples(triples)
    except Exception as e:
        logger.warning(f"文本注入部分失败: {e}")

    return IngestionTask(
        task_id=task_id,
        file_path=request.source_name,
        file_type=DocumentType.TEXT,
        status="completed",
        progress=1.0,
        result={"chunks": [{"content": request.content, "chunk_index": 0, "source_path": request.source_name}]},
    )


@app.post("/api/ingest/url", response_model=IngestionTask)
async def ingest_url(request: IngestUrlRequest):
    if not file_handler:
        raise HTTPException(status_code=503, detail="服务未初始化")
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL不能为空")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    result = await file_handler.process_url(url)
    if result.status == "completed":
        chunks_data = result.result.get("chunks", [])
        if chunks_data:
            chunks = [DocumentChunk(**c) for c in chunks_data]
            knowledge_points = await knowledge_extractor.extract_from_chunks(chunks)
            for kp in knowledge_points:
                await vector_service.index_knowledge_point(kp)
            triples = await knowledge_extractor.extract_triples(knowledge_points)
            await graph_service.create_triples(triples)
    return result


@app.post("/api/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    if not conversation_agent:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    if request.stream:
        return await conversation_agent.chat_stream(request)
    return await conversation_agent.chat(request)


@app.get("/api/conversations")
async def list_conversations():
    from services.conversation_store import conversation_store
    return conversation_store.list_conversations()


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    from services.conversation_store import conversation_store
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


@app.post("/api/conversations")
async def save_conversation(data: dict):
    from services.conversation_store import conversation_store
    conv_id = data.get("id", "")
    messages = data.get("messages", [])
    title = data.get("title")
    if not conv_id:
        raise HTTPException(status_code=400, detail="缺少对话ID")
    conversation_store.save_messages(conv_id, messages, title)
    return {"status": "saved", "id": conv_id}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    from services.conversation_store import conversation_store
    if not conversation_store.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "deleted"}


@app.post("/api/search", response_model=SearchResult)
async def search(request: SearchRequest):
    if not search_service:
        raise HTTPException(status_code=503, detail="搜索服务未初始化")
    return await search_service.search(request)


@app.get("/api/knowledge/list", response_model=KnowledgeListResponse)
async def list_knowledge(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    mode: str = Query(default="offset"),
    cursor: str = Query(default=None),
    direction: str = Query(default="next"),
    sort_by: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    category_id: str = Query(default=None),
    tag: str = Query(default=None),
    confidence_min: float = Query(default=None),
    confidence_max: float = Query(default=None),
    status: str = Query(default=None),
    search: str = Query(default=None),
):
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    all_items = list(vector_service.knowledge_index.values())
    data_items = [item["data"] for item in all_items]

    if category_id and category_service:
        cat = category_service._categories.get(category_id)
        if cat:
            cat_name = cat.name
            cat_meta = cat.metadata or {}
            if cat.category_type == "meta":
                if "confidence_max" in cat_meta:
                    cmax = float(cat_meta["confidence_max"])
                    data_items = [d for d in data_items if d.get("confidence", 0.5) <= cmax or str(d.get("category", "")) == "待验证"]
                elif "confidence_min" in cat_meta:
                    cmin = float(cat_meta["confidence_min"])
                    data_items = [d for d in data_items if d.get("confidence", 0.5) >= cmin]
            elif cat.category_type in ("temporal", "source"):
                member_ids = set(category_service.get_category_members(category_id))
                if member_ids:
                    data_items = [d for d in data_items if d.get("id") in member_ids]
            else:
                member_ids = set(category_service.get_category_members(category_id))
                if member_ids:
                    data_items = [d for d in data_items if d.get("id") in member_ids]
                else:
                    keyword_map = {
                        "技术知识": ["技术"],
                        "概念理论": ["概念"],
                        "事件记录": ["事实", "观点"],
                        "人物关系": ["人物"],
                        "组织信息": ["组织", "公司"],
                        "方法流程": ["方法", "流程"],
                    }
                    keywords = keyword_map.get(cat_name, [])
                    if keywords:
                        data_items = [
                            d for d in data_items
                            if d.get("category", "") in keywords
                        ]
    if tag:
        data_items = [d for d in data_items if tag.lower() in str(d.get("category", "")).lower() or tag.lower() in d.get("fact", "").lower()]
    if confidence_min is not None:
        data_items = [d for d in data_items if d.get("confidence", 0.5) >= confidence_min]
    if confidence_max is not None:
        data_items = [d for d in data_items if d.get("confidence", 0.5) <= confidence_max]
    if status:
        data_items = [d for d in data_items if d.get("status", "") == status]
    if search:
        search_lower = search.lower()
        data_items = [d for d in data_items if search_lower in d.get("fact", "").lower()]

    reverse = order == "desc"
    def sort_key(d):
        val = d.get(sort_by)
        if sort_by == "created_at":
            return (str(val or ""), str(d.get("id", "")))
        if sort_by == "confidence":
            return (float(val or 0.5), str(d.get("id", "")))
        return (str(val or ""), str(d.get("id", "")))
    data_items.sort(key=sort_key, reverse=reverse)

    total = len(data_items)

    if mode == "cursor":
        start_idx = 0
        if cursor:
            decoded = base64.b64decode(cursor).decode("utf-8")
            parts = decoded.split("|", 2)
            cursor_created = parts[0] if len(parts) > 0 else ""
            cursor_id = parts[1] if len(parts) > 1 else ""
            cursor_pos = total
            for i, d in enumerate(data_items):
                d_created = str(d.get("created_at", ""))
                d_id = str(d.get("id", ""))
                if d_created == cursor_created and d_id == cursor_id:
                    cursor_pos = i
                    break
            if direction == "next":
                start_idx = min(cursor_pos + 1, total)
            else:
                start_idx = max(0, cursor_pos - page_size + 1)

        all_items = data_items
        data_items = all_items[start_idx:start_idx + page_size]

        has_next = (start_idx + page_size) < total
        has_prev = start_idx > 0
        next_cursor = None
        prev_cursor = None
        if data_items:
            if has_next:
                next_cursor = base64.b64encode(
                    f"{data_items[-1].get('created_at', '')}|{data_items[-1].get('id', '')}|0".encode()
                ).decode()
            if has_prev:
                prev_item = all_items[start_idx - 1]
                prev_cursor = base64.b64encode(
                    f"{prev_item.get('created_at', '')}|{prev_item.get('id', '')}|0".encode()
                ).decode()

        pagination = PaginationResponse(
            mode="cursor",
            page=1,
            page_size=page_size,
            total=total,
            total_pages=math.ceil(total / page_size) if page_size else 0,
            has_next=has_next,
            has_prev=has_prev,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor,
        )
        return KnowledgeListResponse(data=data_items, pagination=pagination)

    start = (page - 1) * page_size
    paged = data_items[start:start + page_size]
    total_pages = math.ceil(total / page_size) if page_size else 0

    pagination = PaginationResponse(
        mode="offset",
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
    return KnowledgeListResponse(data=paged, pagination=pagination)


@app.get("/api/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str):
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    result = await vector_service.get_knowledge_point(knowledge_id)
    if not result:
        raise HTTPException(status_code=404, detail="知识点不存在")
    return result


@app.delete("/api/knowledge/all")
async def reset_knowledge():
    vc, gc, uc = 0, 0, 0

    if vector_service:
        try:
            vc = await vector_service.delete_all()
        except Exception as e:
            logger.warning(f"向量库清空失败: {e}")

    if graph_service:
        try:
            gc = await graph_service.delete_all()
        except Exception as e:
            logger.warning(f"图谱清空失败: {e}")

    upload_dir = settings.upload_dir
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            try:
                fp = os.path.join(upload_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                    uc += 1
            except Exception:
                pass

    return {
        "status": "reset",
        "vector_deleted": vc,
        "graph_nodes_deleted": gc,
        "upload_files_deleted": uc,
    }


@app.delete("/api/knowledge/batch")
async def delete_knowledge_batch(data: dict):
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    ids = data.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="缺少ids列表")
    deleted = []
    failed = []
    for kid in ids:
        try:
            await vector_service.delete_knowledge_point(kid)
            if graph_service:
                try:
                    await graph_service.delete_knowledge_node(kid)
                except Exception:
                    pass
            deleted.append(kid)
        except Exception as e:
            failed.append({"id": kid, "error": str(e)})
    return {"deleted": len(deleted), "failed": len(failed), "details": {"deleted": deleted, "failed": failed}}


@app.delete("/api/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str):
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    result = await vector_service.get_knowledge_point(knowledge_id)
    if not result:
        raise HTTPException(status_code=404, detail="知识点不存在")
    await vector_service.delete_knowledge_point(knowledge_id)
    if graph_service:
        try:
            await graph_service.delete_knowledge_node(knowledge_id)
        except Exception:
            pass
    return {"status": "deleted", "id": knowledge_id}


@app.get("/api/graph/explore")
async def explore_graph(entity: str = Query(default=""), limit: int = Query(default=50), hops: int = Query(default=1)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    await graph_service.ensure_connected()
    return await graph_service.explore(entity, limit, hops)


@app.post("/api/graph/sync")
async def sync_graph(request: GraphSyncRequest = Body(default_factory=GraphSyncRequest)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    connected = await graph_service.ensure_connected()
    if not connected:
        return {"status": "unavailable", "synced": 0, "triples": 0, "message": "Neo4j 未连接，知识图谱同步不可用。数据已保存在向量库中，知识图谱功能需要连接 Neo4j。"}
    if not vector_service or not knowledge_extractor:
        raise HTTPException(status_code=503, detail="服务未初始化")

    all_data = await vector_service.list_all(0, 10000)
    if not all_data:
        return {"status": "empty", "synced": 0, "triples": 0, "message": "无知识数据可同步"}

    knowledge_points = [KnowledgePoint(**data) for data in all_data]
    triples = await knowledge_extractor.extract_triples(knowledge_points)
    if request.enable_evidence_chain:
        chunk_id = request.chunk_id
        for kp in knowledge_points:
            kp_id = kp.id or ""
            cid = chunk_id or f"chunk_{kp.source_document_id or 'unknown'}"
            await graph_service.create_evidence_chain(
                kp_id, cid, kp.source[:200], kp.source_document_id or kp.source
            )
    if request.enable_normalization:
        all_entities = set()
        for t in triples:
            all_entities.add(t.subject)
            all_entities.add(t.object)
        import asyncio
        norm_tasks = [graph_service.normalize_entity(e) for e in list(all_entities)[:50]]
        await asyncio.gather(*norm_tasks, return_exceptions=True)
    await graph_service.create_triples(triples)
    stats = await graph_service.get_graph_stats()

    return {
        "status": "synced",
        "knowledge_count": len(knowledge_points),
        "triples": len(triples),
        **stats,
    }


@app.get("/api/graph/paths")
async def find_graph_paths(source: str = Query(...), target: str = Query(...), max_hops: int = Query(default=4)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    result = await graph_service.find_paths(source, target, max_hops)
    return result.model_dump()


@app.get("/api/graph/conflicts")
async def detect_graph_conflicts(entity: str = Query(default=""), fact: str = Query(default="")):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    conflicts = await graph_service.detect_conflicts(fact=fact, entity=entity)
    cycles = await graph_service.detect_contradiction_cycles()
    return {"conflicts": conflicts, "contradiction_cycles": cycles, "count": len(conflicts)}


@app.post("/api/graph/communities")
async def detect_graph_communities():
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    result = await graph_service.detect_communities()
    return result.model_dump()


@app.post("/api/graph/normalize")
async def normalize_graph_entity(request: EntityNormalizeRequest):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.normalize_entity(request.entity_name, request.entity_type.value, request.force_merge)


@app.post("/api/graph/cypher")
async def execute_cypher_query(query: str = Body(..., embed=True), params: dict = Body(default=None, embed=True)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    try:
        results = await graph_service.execute_cypher(query, params)
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询执行失败: {str(e)}")


@app.post("/api/graph/inference")
async def apply_graph_inference():
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    rule_results = await graph_service.apply_inference_rules()
    return {
        "status": "completed",
        "rules_applied": len(rule_results),
        "total_inferred": sum(r.inferred_count for r in rule_results),
        "rules": [r.model_dump() for r in rule_results],
    }


@app.get("/api/graph/stats")
async def get_graph_stats():
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    await graph_service.ensure_connected()
    return await graph_service.get_graph_stats()


@app.get("/api/graph/entity/{entity_name:path}")
async def get_entity_detail(entity_name: str):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    await graph_service.ensure_connected()
    neighbors = await graph_service.get_entity_neighbors(entity_name)
    aliases = await graph_service.get_entity_aliases(entity_name)
    conflicts = await graph_service.detect_conflicts(entity=entity_name)
    return {**neighbors, "aliases": aliases, "conflicts": conflicts}


@app.post("/api/graph/fusion/scan")
async def scan_similar_atoms(threshold: float = Query(default=0.95)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    pairs = await graph_service.find_similar_atoms(threshold)
    return {"similar_pairs": pairs, "count": len(pairs)}


# ================================================================
#  代码知识图谱 API
# ================================================================

@app.get("/api/code/structure")
async def get_code_structure(file: str = Query(..., description="文件路径")):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.get_file_code_structure(file)


@app.get("/api/code/functions/search")
async def search_code_functions(name: str = Query(...), limit: int = Query(default=20)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.find_functions_by_name(name, limit)


@app.get("/api/code/functions/by-algorithm")
async def find_functions_by_algorithm(algorithm: str = Query(...)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.find_functions_by_algorithm(algorithm)


@app.get("/api/code/functions/by-concept")
async def find_functions_by_concept(concept: str = Query(...)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.find_functions_by_concept(concept)


@app.get("/api/code/callgraph")
async def get_call_graph(file: str = Query(default=""), limit: int = Query(default=100)):
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.get_code_call_graph(file, limit)


@app.get("/api/code/algorithms")
async def list_code_algorithms():
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.list_algorithms()


@app.get("/api/code/modules")
async def list_code_modules():
    if not graph_service:
        raise HTTPException(status_code=503, detail="图谱服务未初始化")
    return await graph_service.list_code_modules()


@app.get("/api/stats")
async def get_stats():
    stats = {}
    if vector_service:
        stats["vector_count"] = await vector_service.count()
    if graph_service:
        stats["node_count"] = await graph_service.count_nodes()
    return stats


@app.post("/api/knowledge/{knowledge_id}/feedback")
async def feedback_knowledge(knowledge_id: str, feedback: KnowledgeFeedback):
    """用户反馈端点：positive=点赞, negative=点踩, correct=纠错"""
    if not vector_service or not confidence_calculator:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")
    kp = KnowledgePoint(**kp_data)
    alpha, _ = confidence_calculator.bayesian_update(kp, feedback.feedback_type)
    new_conf = await confidence_calculator.compute(kp, apply_feedback=True, apply_decay=True)
    if feedback.feedback_type == "negative":
        conflicting_count = 1
        confidence_calculator.contradiction_penalty(kp, conflicting_count)
        new_conf = kp.confidence
    await vector_service.update_knowledge_point(knowledge_id, kp)
    return {
        "status": "updated",
        "id": knowledge_id,
        "new_confidence": new_conf,
        "feedback_alpha": kp.feedback_alpha,
        "feedback_beta": kp.feedback_beta,
        "interaction_count": kp.interaction_count,
    }


@app.post("/api/confidence/recalculate")
async def recalculate_confidence():
    """重新计算所有知识点的置信度（含图谱传播）"""
    if not confidence_calculator or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    prop_map = {}
    if graph_service and graph_service.driver:
        prop_map = await graph_service.propagate_confidence(iterations=3)

    all_data = await vector_service.list_all(0, 10000)
    updated_count = 0
    conf_distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}

    for kp_data in all_data:
        kp = KnowledgePoint(**kp_data)
        kp.consistency_score = await confidence_calculator.calculate_graph_consistency(kp)
        await confidence_calculator.compute(kp, apply_feedback=True, apply_decay=True)
        kp_id = kp.id or kp_data.get("id", "")
        await vector_service.update_knowledge_point(kp_id, kp)
        updated_count += 1
        conf = kp.confidence
        if conf <= 0.2:
            conf_distribution["0.0-0.2"] += 1
        elif conf <= 0.4:
            conf_distribution["0.2-0.4"] += 1
        elif conf <= 0.6:
            conf_distribution["0.4-0.6"] += 1
        elif conf <= 0.8:
            conf_distribution["0.6-0.8"] += 1
        else:
            conf_distribution["0.8-1.0"] += 1

    return {
        "status": "recalculated",
        "updated": updated_count,
        "propagated_entities": len(prop_map),
        "confidence_distribution": conf_distribution,
    }


@app.post("/api/confidence/recalculate/{knowledge_id}")
async def recalculate_single_confidence(knowledge_id: str):
    """重新计算单个知识点的置信度"""
    if not confidence_calculator or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")
    kp = KnowledgePoint(**kp_data)
    kp.consistency_score = await confidence_calculator.calculate_graph_consistency(kp)
    new_conf = await confidence_calculator.compute(kp, apply_feedback=True, apply_decay=True)
    await vector_service.update_knowledge_point(knowledge_id, kp)
    return {
        "id": knowledge_id,
        "confidence": new_conf,
        "model_confidence_raw": kp.model_confidence_raw,
        "source_quality": kp.source_quality,
        "consistency_score": kp.consistency_score,
        "feedback_alpha": kp.feedback_alpha,
        "feedback_beta": kp.feedback_beta,
        "time_decayed": confidence_calculator.apply_time_decay(kp) < 1.0,
    }


@app.get("/api/confidence/low")
async def get_low_confidence_knowledge(threshold: float = Query(default=0.5), limit: int = Query(default=20)):
    """获取低置信度知识点列表，供人工审核"""
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_data = await vector_service.list_all(0, 10000)
    low_conf = []
    for kp_data in all_data:
        conf = kp_data.get("confidence", kp_data.get("calibrated_confidence", 0.5))
        if conf <= threshold:
            low_conf.append({
                "id": kp_data.get("id"),
                "fact": kp_data.get("fact", "")[:200],
                "confidence": conf,
                "category": kp_data.get("category", ""),
                "source_quality": kp_data.get("source_quality", 0.5),
            })
    low_conf.sort(key=lambda x: x["confidence"])
    return {"count": len(low_conf), "threshold": threshold, "items": low_conf[:limit]}


@app.post("/api/knowledge/{knowledge_id}/confirm")
async def confirm_knowledge(knowledge_id: str):
    """一键确认：将低置信度知识直接提升至0.9，记录为人工强证据"""
    if not vector_service or not confidence_calculator:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")
    kp = KnowledgePoint(**kp_data)
    kp.status = "active"
    kp.feedback_alpha = 10.0
    kp.feedback_beta = 1.0
    kp.interaction_count += 1
    history_entry = {
        "action": "confirmed",
        "previous_confidence": kp_data.get("confidence", 0.5),
        "timestamp": str(datetime.now()),
    }
    kp.history = kp_data.get("history", []) + [history_entry]
    new_conf = await confidence_calculator.compute(kp, apply_feedback=True, apply_decay=True)
    kp.confidence = max(0.9, kp.confidence)
    await vector_service.update_knowledge_point(knowledge_id, kp)
    return {
        "status": "confirmed",
        "id": knowledge_id,
        "confidence": kp.confidence,
        "message": "该知识已被人工确认，置信度提升至" + f"{kp.confidence:.2f}",
    }


@app.post("/api/knowledge/{knowledge_id}/correct")
async def correct_knowledge(knowledge_id: str, request: KnowledgeCorrectRequest):
    """修正知识内容：旧版本降为0.1并标记'已被替代'，新版本继承高置信度0.8"""
    if not vector_service or not confidence_calculator:
        raise HTTPException(status_code=503, detail="服务未初始化")
    if not request.fact.strip():
        raise HTTPException(status_code=400, detail="修正内容不能为空")

    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")

    old_kp = KnowledgePoint(**kp_data)
    new_id = str(uuid.uuid4())
    old_kp.status = "replaced"
    old_kp.replaced_by = new_id
    old_kp.confidence = 0.1
    old_kp.feedback_alpha = 1.0
    old_kp.feedback_beta = 10.0
    history_entry = {
        "action": "replaced",
        "new_id": new_id,
        "previous_confidence": kp_data.get("confidence", 0.5),
        "timestamp": str(datetime.now()),
    }
    old_kp.history = kp_data.get("history", []) + [history_entry]
    await vector_service.update_knowledge_point(knowledge_id, old_kp)

    category = FactCategory(request.category) if request.category else old_kp.category
    source_name = request.source or old_kp.source
    now = datetime.now()
    new_kp = KnowledgePoint(
        id=new_id,
        fact=request.fact.strip(),
        category=category,
        confidence=0.8,
        related_entities=old_kp.related_entities,
        source=f"{source_name} (用户修正)",
        source_document_id=old_kp.source_document_id,
        created_at=now,
        source_quality=0.85,
        feedback_alpha=8.0,
        feedback_beta=2.0,
        interaction_count=1,
        history=[{
            "action": "created_from_correction",
            "original_id": knowledge_id,
            "timestamp": str(now),
        }],
    )
    await vector_service.index_knowledge_point(new_kp)
    await confidence_calculator.compute(new_kp, apply_feedback=True, apply_decay=True)
    await vector_service.update_knowledge_point(new_id, new_kp)

    triples = await knowledge_extractor.extract_triples([new_kp]) if knowledge_extractor else []
    if triples and graph_service:
        await graph_service.create_triples(triples)
    if graph_service and graph_service.driver:
        version = len(old_kp.history or []) + 1
        await graph_service.soft_delete_knowledge(knowledge_id)
        await graph_service.create_new_version(knowledge_id, new_id, new_kp.fact, version)

    return {
        "status": "corrected",
        "old_id": knowledge_id,
        "new_id": new_id,
        "old_confidence": 0.1,
        "new_confidence": new_kp.confidence,
        "message": "知识已修正，旧版本已归档",
    }


@app.post("/api/knowledge/{knowledge_id}/mark-error")
async def mark_error_knowledge(knowledge_id: str):
    """标记为错误示例：保留供模型学习，但不参与正常问答"""
    if not vector_service or not confidence_calculator:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")
    kp = KnowledgePoint(**kp_data)
    kp.status = "error"
    kp.confidence = 0.05
    kp.feedback_alpha = 0.5
    kp.feedback_beta = 10.0
    kp.interaction_count += 1
    history_entry = {
        "action": "marked_error",
        "previous_confidence": kp_data.get("confidence", 0.5),
        "timestamp": str(datetime.now()),
    }
    kp.history = kp_data.get("history", []) + [history_entry]
    await vector_service.update_knowledge_point(knowledge_id, kp)
    return {
        "status": "marked_error",
        "id": knowledge_id,
        "confidence": 0.05,
        "message": "该知识已被标记为错误示例，不再参与正常问答",
    }


@app.get("/api/knowledge/{knowledge_id}/evidence")
async def get_knowledge_evidence(knowledge_id: str, limit: int = Query(default=5)):
    """获取支持或反驳该知识的关联证据（图谱+向量检索）"""
    if not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp_data = await vector_service.get_knowledge_point(knowledge_id)
    if not kp_data:
        raise HTTPException(status_code=404, detail="知识点不存在")

    kp = KnowledgePoint(**kp_data)
    query_text = kp.fact
    related = await vector_service.search_knowledge(query_text, top_k=limit + 1)

    supporting = []
    contradicting = []
    for r in related:
        if r.id == knowledge_id:
            continue
        item = {
            "id": r.id,
            "fact": r.fact[:200],
            "confidence": r.confidence,
            "category": r.category.value if hasattr(r.category, 'value') else str(r.category),
            "source": r.source,
        }
        if r.confidence >= settings.low_confidence_threshold:
            supporting.append(item)
        else:
            contradicting.append(item)

    return {
        "knowledge_id": knowledge_id,
        "fact": kp.fact[:200],
        "confidence": kp.confidence,
        "supporting": supporting[:limit],
        "contradicting": contradicting[:limit],
        "supporting_count": len(supporting),
        "contradicting_count": len(contradicting),
    }


@app.post("/api/confidence/auto-review")
async def auto_review_low_confidence(request: LowConfidenceReviewRequest = None):
    """自动复核流水线：重算低置信度知识+图谱传播+外部搜索佐证"""
    if not confidence_calculator or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if request is None:
        request = LowConfidenceReviewRequest()

    threshold = request.threshold or settings.low_confidence_threshold
    all_data = await vector_service.list_all(0, 10000)

    low_conf_ids = []
    for kp_data in all_data:
        conf = kp_data.get("confidence", 0.5)
        status = kp_data.get("status", "active")
        if conf < threshold and status == "active":
            low_conf_ids.append((kp_data.get("id"), conf))

    if not low_conf_ids:
        return {"status": "none", "message": "当前没有需要复核的低置信度知识", "reviewed": 0}

    reviewed_ids = [kid for kid, _ in low_conf_ids[:request.limit]]
    improved = []
    unchanged = []
    search_evidences = []

    prop_map = {}
    if graph_service and graph_service.driver:
        prop_map = await graph_service.propagate_confidence(iterations=3)

    for kid, old_conf in low_conf_ids[:request.limit]:
        kp_data = await vector_service.get_knowledge_point(kid)
        if not kp_data:
            continue
        kp = KnowledgePoint(**kp_data)
        kp.consistency_score = await confidence_calculator.calculate_graph_consistency(kp)
        new_conf = await confidence_calculator.compute(kp, apply_feedback=True, apply_decay=True)
        await vector_service.update_knowledge_point(kid, kp)

        if new_conf > old_conf + 0.02:
            improved.append({"id": kid, "fact": kp.fact[:100], "old_confidence": old_conf, "new_confidence": new_conf})
        else:
            unchanged.append({"id": kid, "fact": kp.fact[:100], "confidence": new_conf})

        if request.enable_external_search and web_search_service and settings.enable_external_search:
            try:
                search_results = await web_search_service.search(kp.fact[:100], max_results=3)
                if search_results:
                    search_evidences.append({
                        "knowledge_id": kid,
                        "fact": kp.fact[:100],
                        "search_results": [
                            {"title": r.get("title", ""), "snippet": r.get("snippet", "")[:200],
                             "url": r.get("url", "")}
                            for r in search_results
                        ],
                    })
            except Exception as e:
                logger.warning(f"外部搜索证据获取失败 (kid={kid}): {e}")

    top_conflicts = sorted(unchanged, key=lambda x: x["confidence"])[:10]

    return {
        "status": "reviewed",
        "total_low_confidence": len(low_conf_ids),
        "reviewed": len(reviewed_ids),
        "improved": len(improved),
        "improved_items": improved,
        "unchanged": len(unchanged),
        "top_conflicts": top_conflicts,
        "propagated_entities": len(prop_map),
        "search_evidences": search_evidences,
        "message": f"已复核 {len(reviewed_ids)} 条，其中 {len(improved)} 条置信度提升",
    }


# ================================================================
#  分类系统 API
# ================================================================

@app.get("/api/categories")
async def list_categories(
    category_type: Optional[str] = Query(default=None),
    include_archived: bool = Query(default=False),
):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    ct = CategoryType(category_type) if category_type else None
    category_service.refresh_counts_from_data([v["data"] for v in vector_service.knowledge_index.values()])
    cats = category_service.get_all_categories(ct, include_archived)
    return {"categories": [c.model_dump() for c in cats], "total": len(cats)}


@app.get("/api/categories/tree")
async def get_category_tree(user_id: str = Query(default="default")):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    category_service.refresh_counts_from_data([v["data"] for v in vector_service.knowledge_index.values()])
    tree = category_service.build_category_tree(user_id)
    return {"tree": [t.model_dump() for t in tree]}


@app.get("/api/categories/tree/personalized")
async def get_personalized_tree(
    user_id: str = Query(default="default"),
    focus_mode: bool = Query(default=False),
):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    category_service.refresh_counts_from_data([v["data"] for v in vector_service.knowledge_index.values()])
    tree = category_service.get_personalized_tree(user_id, focus_mode=focus_mode)
    return {"tree": [t.model_dump() for t in tree]}


@app.post("/api/categories")
async def create_category(category: Category):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    return category_service.create_category(category).model_dump()


@app.put("/api/categories/{category_id}")
async def update_category(category_id: str, updates: dict):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    result = category_service.update_category(category_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="分类不存在")
    return result.model_dump()


@app.delete("/api/categories/{category_id}")
async def delete_category(category_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    if not category_service.delete_category(category_id):
        raise HTTPException(status_code=404, detail="分类不存在")
    return {"status": "deleted", "id": category_id}


@app.post("/api/categories/{category_id}/visit")
async def record_category_visit(category_id: str, user_id: str = Query(default="default")):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    category_service.record_category_visit(user_id, category_id)
    return {"status": "recorded"}


@app.post("/api/knowledge/{knowledge_id}/categories")
async def assign_knowledge_categories(knowledge_id: str, data: dict):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    category_ids = data.get("category_ids", [])
    primary_id = data.get("primary_category_id")
    is_auto = data.get("is_auto", False)
    assignments = category_service.assign_knowledge_to_categories(
        knowledge_id, category_ids, primary_id, is_auto,
    )
    return {"assignments": [a.model_dump() for a in assignments], "count": len(assignments)}


@app.delete("/api/knowledge/{knowledge_id}/categories/{category_id}")
async def remove_knowledge_category(knowledge_id: str, category_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    if not category_service.remove_knowledge_from_category(knowledge_id, category_id):
        raise HTTPException(status_code=404, detail="关联不存在")
    return {"status": "removed"}


@app.get("/api/knowledge/{knowledge_id}/categories")
async def get_knowledge_categories(knowledge_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    cats = category_service.get_knowledge_categories(knowledge_id)
    return {"categories": [c.model_dump() for c in cats]}


@app.post("/api/knowledge/categories/batch")
async def batch_get_knowledge_categories(data: dict):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    knowledge_ids = data.get("knowledge_ids", [])
    result = {}
    for kid in knowledge_ids:
        cats = category_service.get_knowledge_categories(kid)
        result[kid] = [c.model_dump() for c in cats]
    return {"categories": result}


@app.post("/api/categories/relations")
async def add_category_relation(relation: CategoryRelation):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    return category_service.add_relation(relation).model_dump()


@app.get("/api/categories/{category_id}/relations")
async def get_category_relations(
    category_id: str,
    relation_type: Optional[str] = Query(default=None),
):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    rt = CategoryRelationType(relation_type) if relation_type else None
    relations = category_service.get_relations(category_id, rt)
    return {"relations": [r.model_dump() for r in relations]}


@app.post("/api/tags")
async def create_tag(tag: UserTag):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    return category_service.create_tag(tag).model_dump()


@app.get("/api/tags")
async def list_tags(user_id: str = Query(default="default")):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    tags = category_service.get_user_tags(user_id)
    return {"tags": [t.model_dump() for t in tags]}


@app.delete("/api/tags/{tag_id}")
async def delete_tag(tag_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    if not category_service.delete_tag(tag_id):
        raise HTTPException(status_code=404, detail="标签不存在")
    return {"status": "deleted", "id": tag_id}


@app.post("/api/knowledge/{knowledge_id}/tags")
async def assign_knowledge_tag(knowledge_id: str, assignment: KnowledgeTagAssignment):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    assignment.knowledge_id = knowledge_id
    return category_service.assign_tag(assignment).model_dump()


@app.delete("/api/knowledge/{knowledge_id}/tags/{tag_id}")
async def remove_knowledge_tag(knowledge_id: str, tag_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    if not category_service.remove_tag(knowledge_id, tag_id):
        raise HTTPException(status_code=404, detail="标签关联不存在")
    return {"status": "removed"}


@app.get("/api/knowledge/{knowledge_id}/tags")
async def get_knowledge_tags(knowledge_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    tags = category_service.get_knowledge_tags(knowledge_id)
    return {"tags": [t.model_dump() for t in tags]}


@app.post("/api/smart-collections")
async def create_smart_collection(collection: SmartCollection):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    return category_service.create_smart_collection(collection).model_dump()


@app.get("/api/smart-collections")
async def list_smart_collections(user_id: Optional[str] = Query(default=None)):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    collections = category_service.get_smart_collections(user_id)
    return {"collections": [c.model_dump() for c in collections]}


@app.post("/api/smart-collections/{collection_id}/evaluate")
async def evaluate_smart_collection(collection_id: str):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_kps = {}
    all_data = await vector_service.list_all(0, 10000)
    for kp_data in all_data:
        all_kps[kp_data.get("id", "")] = kp_data
    result_ids = category_service.evaluate_smart_collection(collection_id, all_kps)
    return {"collection_id": collection_id, "match_count": len(result_ids), "knowledge_ids": result_ids}


@app.post("/api/knowledge/filter")
async def filter_knowledge(flt: MultiDimensionFilter):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_kps = {}
    all_data = await vector_service.list_all(0, 10000)
    for kp_data in all_data:
        all_kps[kp_data.get("id", "")] = kp_data
    results = category_service.filter_knowledge(flt, all_kps)
    return {"items": results, "total": len(results), "offset": flt.offset, "limit": flt.limit}


@app.get("/api/categories/suggest")
async def suggest_categories(fact: str = Query(..., description="知识内容")):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    suggestions = await category_service.suggest_categories(fact)
    return {"suggestions": suggestions}


@app.post("/api/categories/cluster")
async def run_clustering():
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_data = await vector_service.list_all(0, 10000)
    kps_with_vectors = []
    for kp_data in all_data:
        vec = kp_data.get("vector")
        if vec:
            kps_with_vectors.append({"id": kp_data.get("id", ""), "vector": vec, **kp_data})
    result = await category_service.run_hybrid_clustering(kps_with_vectors)
    return result


@app.post("/api/categories/evolution/check")
async def check_evolution():
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    events = await category_service.trigger_evolution_check()
    return {"events": [e.model_dump() for e in events], "count": len(events)}


@app.post("/api/categories/evolution/execute")
async def execute_evolution(event: dict):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    action = EvolutionAction(event.get("action", "create"))
    evt = CategoryEvolutionEvent(
        category_id=event.get("category_id"),
        action=action,
        details=event.get("details", {}),
        triggered_by=event.get("triggered_by", "api"),
    )
    result = await category_service.execute_evolution(evt)
    return {"status": "executed", "result": result.model_dump() if result else None}


@app.get("/api/categories/health")
async def get_category_health():
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    health = category_service.get_category_health()
    return {"health": [h.model_dump() for h in health]}


@app.get("/api/categories/timeline")
async def get_timeline(
    mode: str = Query("event_time", description="event_time 或 recorded_at"),
    granularity: str = Query("month", description="year / month / day"),
    category_id: Optional[str] = Query(None),
    limit: int = Query(10000, ge=1, le=50000),
):
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_data = await vector_service.list_all(0, limit)
    items = [v["data"] for v in vector_service.knowledge_index.values()]
    groups = _build_timeline_groups(items, mode, granularity)
    gaps = _detect_timeline_gaps(groups, mode)
    bursts = _detect_timeline_bursts(groups)
    chains = _detect_version_chains(items)
    if category_id:
        groups = _filter_groups_by_category(groups, category_id, category_service)
    return TimelineResponse(
        groups=groups,
        total=len(items),
        mode=mode,
        granularity=granularity,
        gaps=gaps,
        bursts=bursts,
        version_chains=chains,
    ).model_dump()


@app.get("/api/categories/timeline/extract")
async def extract_timeline_times(
    batch_size: int = Query(20, ge=1, le=100),
):
    if not vector_service or not time_extraction_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    items = [v["data"] for v in vector_service.knowledge_index.values()]
    updated = 0
    import asyncio
    from models import KnowledgePoint
    for i in range(0, min(len(items), 200), batch_size):
        batch = items[i:i + batch_size]
        tasks = []
        for item in batch:
            kp_id = item.get("id", "")
            kp_data = vector_service.knowledge_index.get(kp_id)
            if not kp_data:
                continue
            text = item.get("fact", "")
            tasks.append(time_extraction_service.extract_event_times(text))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, result in enumerate(results):
            if isinstance(result, Exception) or not result:
                continue
            kp_id = batch[j].get("id", "")
            kp_data = vector_service.knowledge_index.get(kp_id)
            if not kp_data or kp_data["data"].get("event_time"):
                continue
            event_time, time_precision = time_extraction_service.pick_best_event_time(result)
            if event_time:
                kp_data["data"]["event_time"] = event_time
                kp_data["data"]["time_precision"] = time_precision
                kp_data["data"]["event_times"] = result
                vector_service.knowledge_index[kp_id] = kp_data
                updated += 1
    if updated:
        vector_service._save_index()
    return {"updated": updated, "total_checked": min(len(items), 200)}


@app.post("/api/categories/auto-categorize")
async def auto_categorize_knowledge(data: dict):
    """大模型自动对知识点进行分类"""
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    knowledge_id = data.get("knowledge_id", "")
    if not knowledge_id:
        raise HTTPException(status_code=400, detail="knowledge_id 必填")
    confidence_threshold = float(data.get("confidence_threshold", 0.7))
    auto_create = data.get("auto_create", False)
    result = await category_service.auto_categorize_knowledge(
        knowledge_id,
        data.get("fact", ""),
        confidence_threshold=confidence_threshold,
        auto_create=auto_create,
    )
    return result


@app.get("/api/categories/search")
async def search_categories(q: str = Query(..., description="搜索关键词")):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    results = category_service.search_categories(q)
    return {"categories": [c.model_dump() for c in results]}


@app.get("/api/categories/{category_id}/path")
async def get_category_path(category_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    path = category_service.get_category_path_to_root(category_id)
    return {"path": [c.model_dump() for c in path]}


@app.get("/api/categories/merge-suggestions")
async def get_merge_suggestions():
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    suggestions = await category_service.suggest_category_merges()
    return {"suggestions": suggestions, "count": len(suggestions)}


@app.post("/api/knowledge/batch/categories")
async def batch_assign_categories(data: dict):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    knowledge_ids = data.get("knowledge_ids", [])
    category_ids = data.get("category_ids", [])
    is_auto = data.get("is_auto", False)
    if not knowledge_ids or not category_ids:
        raise HTTPException(status_code=400, detail="knowledge_ids 和 category_ids 必填")
    count = category_service.batch_assign_categories(knowledge_ids, category_ids, is_auto)
    return {"assigned": count, "knowledge_count": len(knowledge_ids)}


@app.post("/api/categories/sync-graph")
async def sync_categories_to_graph():
    """将所有分类和关联同步到Neo4j知识图谱"""
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    count = category_service.sync_all_categories_to_graph()
    return {"synced": count, "status": "completed"}


@app.post("/api/knowledge/{knowledge_id}/auto-categorize")
async def auto_categorize_single(knowledge_id: str, data: dict):
    """对单条知识进行AI自动分类"""
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    kp = vector_service.knowledge_index.get(knowledge_id)
    if not kp:
        raise HTTPException(status_code=404, detail="知识点不存在")
    result = await category_service.auto_categorize_knowledge(
        knowledge_id,
        kp["data"].get("fact", ""),
        confidence_threshold=float(data.get("confidence_threshold", 0.7)),
        auto_create=data.get("auto_create", False),
    )
    return result

def _build_timeline_groups(items: list[dict], mode: str, granularity: str) -> list[TimelineGroup]:
    from collections import defaultdict
    from datetime import datetime
    groups_dict: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        time_key = _get_time_key(item, mode, granularity)
        if time_key:
            groups_dict[time_key].append(item)
    avg_density = len(items) / max(len(groups_dict), 1)
    result = []
    for time_key in sorted(groups_dict.keys(), reverse=True):
        group_items = groups_dict[time_key]
        conf_sum = sum(it.get("confidence", 0.5) for it in group_items)
        result.append(TimelineGroup(
            time_key=time_key,
            label=_format_time_label(time_key, granularity),
            record_count=sum(1 for it in group_items if it.get("created_at")),
            event_count=len(group_items),
            items=group_items,
            confidence_avg=conf_sum / len(group_items) if group_items else 0,
        ))
    _mark_gaps_and_bursts(result, avg_density)
    return result


def _get_time_key(item: dict, mode: str, granularity: str) -> str:
    dt_str = ""
    if mode == "recorded_at":
        dt_str = str(item.get("created_at", "") or "")
    else:
        dt_str = item.get("event_time", "") or str(item.get("created_at", "") or "")
    if not dt_str:
        return ""
    dt_str = dt_str[:10].replace("T", " ")[:10]
    if granularity == "year":
        return dt_str[:4]
    elif granularity == "day":
        return dt_str[:10]
    else:
        return dt_str[:7]


def _format_time_label(time_key: str, granularity: str) -> str:
    parts = time_key.replace("/", "-").split("-")
    if granularity == "year":
        return f"{parts[0]}年"
    elif granularity == "day":
        if len(parts) >= 3:
            return f"{parts[0]}年{int(parts[1])}月{int(parts[2])}日"
        return time_key
    else:
        if len(parts) >= 2:
            return f"{parts[0]}年{int(parts[1])}月"
        return time_key


def _mark_gaps_and_bursts(groups: list[TimelineGroup], avg_density: float):
    for g in groups:
        if g.event_count > avg_density * 2.5:
            g.is_burst = True
    for i in range(1, len(groups)):
        try:
            t1 = datetime.strptime(groups[i - 1].time_key + ("-01" if len(groups[i - 1].time_key) <= 7 else ""), "%Y-%m-%d" if "-" in groups[i - 1].time_key and len(groups[i - 1].time_key) > 7 else "%Y-%m")
            t2 = datetime.strptime(groups[i].time_key + ("-01" if len(groups[i].time_key) <= 7 else ""), "%Y-%m-%d" if "-" in groups[i].time_key and len(groups[i].time_key) > 7 else "%Y-%m")
            delta = abs((t1 - t2).days)
            if delta > 90:
                groups[i].is_gap = True
        except (ValueError, IndexError):
            pass


def _detect_timeline_gaps(groups: list[TimelineGroup], mode: str) -> list[TimelineGap]:
    gaps = []
    for i in range(1, len(groups)):
        try:
            t1 = datetime.strptime(groups[i - 1].time_key + (("-01" if len(groups[i - 1].time_key) <= 7 else "")), "%Y-%m-%d" if "-" in groups[i - 1].time_key and len(groups[i - 1].time_key) > 7 else "%Y-%m")
            t2 = datetime.strptime(groups[i].time_key + (("-01" if len(groups[i].time_key) <= 7 else "")), "%Y-%m-%d" if "-" in groups[i].time_key and len(groups[i].time_key) > 7 else "%Y-%m")
            delta = abs((t1 - t2).days)
            if delta > 60:
                gaps.append(TimelineGap(
                    start_date=min(t1, t2).strftime("%Y-%m-%d"),
                    end_date=max(t1, t2).strftime("%Y-%m-%d"),
                    duration_days=delta,
                    label=f"{delta}天空白期",
                    suggestion="建议补充该时段的知识数据",
                ))
        except (ValueError, IndexError):
            pass
    return gaps


def _detect_timeline_bursts(groups: list[TimelineGroup]) -> list[TimelineBurst]:
    if not groups:
        return []
    counts = [g.event_count for g in groups]
    avg = sum(counts) / len(counts)
    std = (sum((c - avg) ** 2 for c in counts) / len(counts)) ** 0.5
    bursts = []
    for g in groups:
        if g.event_count > avg + 2 * std and g.event_count >= 5:
            categories: dict[str, int] = {}
            for item in g.items:
                cat = str(item.get("category", ""))
                categories[cat] = categories.get(cat, 0) + 1
            top_cats = sorted(categories, key=categories.get, reverse=True)[:3]
            multiplier = g.event_count / max(avg, 1)
            bursts.append(TimelineBurst(
                center_date=g.time_key,
                density_multiplier=round(multiplier, 1),
                knowledge_count=g.event_count,
                top_categories=top_cats,
                label=f"信息爆发期 ({g.event_count}条, {multiplier:.1f}x)",
            ))
    return bursts


def _detect_version_chains(items: list[dict]) -> list[VersionChain]:
    from collections import defaultdict
    entity_versions: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        entities = item.get("related_entities", [])
        for entity in entities:
            entity_versions[entity].append(item)
    chains = []
    for entity, vers in entity_versions.items():
        if len(vers) >= 2:
            sorted_vers = sorted(vers, key=lambda v: str(v.get("created_at", "") or ""), reverse=True)
            latest = str(sorted_vers[0].get("created_at", "") or "")[:10] if sorted_vers else None
            chains.append(VersionChain(
                entity_name=entity,
                versions=sorted_vers,
                latest_version=latest,
                total_updates=len(sorted_vers),
            ))
    chains.sort(key=lambda c: c.total_updates, reverse=True)
    return chains[:20]


def _filter_groups_by_category(groups: list[TimelineGroup], category_id: str, cat_service) -> list[TimelineGroup]:
    from services.category import CategoryService
    filtered = []
    for g in groups:
        filtered_items = [
            item for item in g.items
            if category_id in cat_service._match_knowledge_to_category_ids(item)
        ]
        if filtered_items:
            conf_sum = sum(it.get("confidence", 0.5) for it in filtered_items)
            filtered.append(TimelineGroup(
                time_key=g.time_key,
                label=g.label,
                record_count=sum(1 for it in filtered_items if it.get("created_at")),
                event_count=len(filtered_items),
                items=filtered_items,
                is_gap=g.is_gap,
                is_burst=g.is_burst,
                confidence_avg=conf_sum / len(filtered_items) if filtered_items else 0,
            ))
    return filtered


@app.get("/api/categories/sources")
async def get_source_groups():
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_data = await vector_service.list_all(0, 10000)
    groups = category_service.build_source_groups(all_data)
    return {"sources": [g.model_dump() for g in groups]}


@app.get("/api/categories/sources/compare")
async def get_source_comparisons():
    if not category_service or not vector_service:
        raise HTTPException(status_code=503, detail="服务未初始化")
    all_data = await vector_service.list_all(0, 10000)
    comparisons = category_service.find_source_comparisons(all_data)
    return {"comparisons": [c.model_dump() for c in comparisons]}


@app.get("/api/categories/preferences")
async def get_user_preferences(user_id: str = Query(default="default")):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    prefs = category_service.get_user_category_prefs(user_id)
    return {"preferences": {k: v.model_dump() for k, v in prefs.items()}}


@app.put("/api/categories/{category_id}/preferences")
async def set_category_preferences(
    category_id: str,
    prefs: UserCategoryPrefs,
    user_id: str = Query(default="default"),
):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    prefs.user_id = user_id
    prefs.category_id = category_id
    category_service.set_user_category_prefs(user_id, category_id, prefs)
    return {"status": "saved"}


@app.get("/api/categories/{category_id}")
async def get_category(category_id: str):
    if not category_service:
        raise HTTPException(status_code=503, detail="分类服务未初始化")
    cat = category_service.get_category(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="分类不存在")
    return cat.model_dump()


class DedupSettingsUpdate(BaseModel):
    dedup_file_hash_enabled: Optional[bool] = None
    dedup_content_similarity_enabled: Optional[bool] = None
    dedup_strict_threshold: Optional[float] = None
    dedup_warn_threshold: Optional[float] = None
    dedup_mode: Optional[str] = None


@app.get("/api/dedup/stats", response_model=DedupStats)
async def get_dedup_stats():
    if not dedup_service:
        raise HTTPException(status_code=503, detail="去重服务未初始化")
    return await dedup_service.get_stats()


@app.post("/api/dedup/check")
async def check_dedup(file_name: str = Query(...), file_size: int = Query(...)):
    if not dedup_service:
        raise HTTPException(status_code=503, detail="去重服务未初始化")
    return {"status": "ok", "message": "请在文件上传时自动执行去重检查"}


@app.post("/api/dedup/settings")
async def update_dedup_settings(body: DedupSettingsUpdate = Body(...)):
    global settings
    updates = body.model_dump(exclude_none=True)
    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    upper_updates = {k.upper(): v for k, v in updates.items()}
    try:
        for env_key, env_value in upper_updates.items():
            key_line = f"{env_key}="
            found = False
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                with open(env_path, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.startswith(key_line):
                            f.write(f"{env_key}={env_value}\n")
                            found = True
                        else:
                            f.write(line)
                    if not found:
                        f.write(f"\n{env_key}={env_value}\n")
            else:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(f"\n{env_key}={env_value}\n")
    except Exception as e:
        logger.warning(f"[去重] 保存配置到 .env 失败: {e}")
    return {"status": "saved", "settings": {k: str(getattr(settings, k)) for k in updates}}


@app.get("/api/dedup/settings")
async def get_dedup_settings():
    return {
        "dedup_file_hash_enabled": settings.dedup_file_hash_enabled,
        "dedup_content_similarity_enabled": settings.dedup_content_similarity_enabled,
        "dedup_strict_threshold": settings.dedup_strict_threshold,
        "dedup_warn_threshold": settings.dedup_warn_threshold,
        "dedup_mode": settings.dedup_mode,
    }


@app.delete("/api/dedup/registry")
async def clear_dedup_registry():
    if not dedup_service:
        raise HTTPException(status_code=503, detail="去重服务未初始化")
    count = await dedup_service.clear_registry()
    return {"status": "cleared", "removed_count": count}


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/api/auth/register")
async def register_user(req: RegisterRequest):
    if not auth_service:
        raise HTTPException(status_code=503, detail="认证服务未初始化")
    try:
        user_id, result = auth_service.register(req.email, req.username, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login_user(req: LoginRequest):
    if not auth_service:
        raise HTTPException(status_code=503, detail="认证服务未初始化")
    try:
        result = auth_service.login(req.email, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/refresh")
async def refresh_token(req: RefreshRequest):
    if not auth_service:
        raise HTTPException(status_code=503, detail="认证服务未初始化")
    try:
        tokens = auth_service.refresh(req.refresh_token)
        return tokens
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/auth/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    return current_user


@app.post("/api/user/memory/decay")
async def trigger_memory_decay():
    if not conversation_agent:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    await conversation_agent.decay_memory()
    return {"status": "decayed"}


@app.get("/api/user/conversations/{user_id}")
async def get_user_conversations(user_id: str):
    if not memory_service:
        raise HTTPException(status_code=503, detail="记忆服务未初始化")
    conversations = memory_service.get_user_conversations(user_id)
    return {"user_id": user_id, "conversations": conversations, "count": len(conversations)}


@app.get("/api/user/memory/{user_id}")
async def get_user_memory(user_id: str):
    if not conversation_agent:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    data = await conversation_agent.get_full_memory(user_id)
    return {
        "user_id": user_id,
        "profile": data["profile"],
        "memory_items": data["memory_items"],
        "item_count": data["item_count"],
    }


@app.post("/api/user/memory/{user_id}")
async def update_user_memory(user_id: str, key: str = Query(...), value: str = Query(...)):
    if not conversation_agent:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    await conversation_agent.update_memory(user_id, key, value)
    return {"status": "updated", "user_id": user_id, "key": key, "value": value}


@app.delete("/api/user/memory/{user_id}/item")
async def delete_user_memory_item(
    user_id: str, memory_type: str = Query(...), memory_key: str = Query(...)
):
    if not conversation_agent:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    deleted = await conversation_agent.delete_memory_item(user_id, memory_type, memory_key)
    return {"status": "deleted" if deleted else "not_found"}


@app.get("/api/memory/stats")
async def get_memory_stats():
    if not memory_service:
        raise HTTPException(status_code=503, detail="记忆服务未初始化")
    return memory_service.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)