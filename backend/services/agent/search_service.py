import logging
import asyncio
from models import SearchRequest, SearchResult, KnowledgePoint
from services.graph.vector_service import VectorService
from services.graph.neo4j_service import GraphService
from services.graph.bm25_service import BM25Service

logger = logging.getLogger(__name__)

RRF_K = 60


class SearchService:
    def __init__(self, vector_service: VectorService, graph_service: GraphService, bm25_service: BM25Service = None):
        self.vector_service = vector_service
        self.graph_service = graph_service
        self.bm25_service = bm25_service
        self.query_cache: dict[str, list[dict]] = {}
        self.cache_size = 100

    def _rrf_fusion(self, ranked_lists: list[list[dict]], id_key: str = "id", k: int = RRF_K) -> list[dict]:
        score_map: dict[str, tuple[float, dict]] = {}
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                item_id = str(item.get(id_key, item.get("metadata", {}).get("id", str(rank))))
                rrf_score = 1.0 / (k + rank + 1)
                if item_id in score_map:
                    score_map[item_id] = (score_map[item_id][0] + rrf_score, score_map[item_id][1])
                else:
                    score_map[item_id] = (rrf_score, item)

        fused = sorted(score_map.items(), key=lambda x: x[1][0], reverse=True)
        return [{"id": item_id, "score": score, "item": data} for item_id, (score, data) in fused]

    async def search(self, request: SearchRequest) -> SearchResult:
        query = request.query
        top_k = request.top_k

        cache_key = f"{query}:{top_k}"
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            return SearchResult(
                knowledge_points=cached.get("kps", []),
                document_chunks=cached.get("chunks", []),
                graph_results=cached.get("graph", []),
            )

        tasks = []
        tasks.append(self._vector_search(query, top_k * 2))
        tasks.append(self._bm25_search(query, top_k * 2))
        tasks.append(self._graph_search(query, min(top_k, 20)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_results = results[0] if not isinstance(results[0], Exception) else []
        bm25_results = results[1] if not isinstance(results[1], Exception) else []
        graph_results = results[2] if not isinstance(results[2], Exception) else []

        vector_ranked = [
            {"id": r.id or "", "score": getattr(r, "confidence", 0.5), "item": r}
            for r in vector_results
        ]
        bm25_ranked = [{"id": r["id"], "score": r["score"], "item": r} for r in bm25_results]

        fused = self._rrf_fusion(
            [vector_ranked, bm25_ranked, graph_results],
            id_key="id",
        )

        kp_ids = set()
        knowledge_points = []
        document_chunks = []
        for item in fused[:top_k]:
            raw = item["item"]
            if isinstance(raw, KnowledgePoint):
                if raw.id not in kp_ids:
                    kp_ids.add(raw.id)
                    knowledge_points.append(raw)
            elif isinstance(raw, dict):
                if raw.get("id") not in kp_ids:
                    kp_ids.add(raw["id"])
                    metadata = raw.get("metadata", {})
                    kp = KnowledgePoint(
                        id=metadata.get("id", raw.get("id", "")),
                        fact=metadata.get("fact", raw.get("text", "")),
                        confidence=metadata.get("confidence", 0.5),
                        source=metadata.get("source", ""),
                        related_entities=metadata.get("related_entities", []),
                    )
                    knowledge_points.append(kp)

        chunk_data = await self.vector_service.search_documents(query, top_k=min(top_k, 5))
        document_chunks = chunk_data

        if self.query_cache:
            cache_entry = {"kps": knowledge_points, "chunks": document_chunks, "graph": graph_results[:10]}
            self.query_cache[cache_key] = cache_entry
            if len(self.query_cache) > self.cache_size:
                first_key = next(iter(self.query_cache))
                del self.query_cache[first_key]

        return SearchResult(
            knowledge_points=knowledge_points,
            document_chunks=document_chunks,
            graph_results=graph_results[:10],
        )

    async def _vector_search(self, query: str, top_k: int) -> list[KnowledgePoint]:
        try:
            return await self.vector_service.search_knowledge(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"向量检索异常: {e}")
            return []

    async def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self.bm25_service:
            return []
        try:
            return self.bm25_service.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"BM25检索异常: {e}")
            return []

    async def _graph_search(self, query: str, limit: int) -> list[dict]:
        try:
            graph_data = await self.graph_service.explore("", limit=limit)
            edges = graph_data.get("edges", [])
            return [{"id": f"g_{i}", "score": 0.3, "item": e} for i, e in enumerate(edges)]
        except Exception as e:
            logger.warning(f"图谱检索异常: {e}")
            return []

    async def rewrite_queries(self, query: str, llm_client=None, user_profile: dict = None) -> list[str]:
        variants = [query]

        if llm_client:
            try:
                profile_hint = ""
                if user_profile:
                    interests = user_profile.get("topic_interest", "")
                    if interests:
                        profile_hint = f"\n用户关注领域：{interests}，优先生成与用户关注相关的变体。"
                response = await llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{
                        "role": "system",
                        "content": f"你是一个查询改写助手。将用户问题改写为2-3种不同表述，包含：1)更正式的术语表达 2)口语化表达 3)简短关键词形式。只返回JSON数组，不要其他内容。{profile_hint}"
                    }, {
                        "role": "user",
                        "content": query
                    }],
                    temperature=0.3,
                    max_tokens=300,
                )
                import json
                content = response.choices[0].message.content.strip()
                if content.startswith("["):
                    rewritten = json.loads(content)
                    variants.extend([r for r in rewritten if r not in variants])
            except Exception as e:
                logger.warning(f"查询重写失败: {e}")

        return variants[:4]

    def sync_bm25_from_vector(self):
        if not self.bm25_service:
            return
        self.bm25_service.clear()
        for kp_id, item in self.vector_service.knowledge_index.items():
            data = item.get("data", {})
            fact = data.get("fact", "")
            if fact:
                self.bm25_service.index_document(kp_id, fact, data)
        logger.info(f"BM25索引同步完成: {self.bm25_service.count()} 条")

    def clear_cache(self):
        self.query_cache.clear()