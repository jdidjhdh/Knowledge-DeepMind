import logging
from models import SearchRequest, SearchResult
from services.graph.vector_service import VectorService
from services.graph.neo4j_service import GraphService

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, vector_service: VectorService, graph_service: GraphService):
        self.vector_service = vector_service
        self.graph_service = graph_service

    async def search(self, request: SearchRequest) -> SearchResult:
        knowledge_points = await self.vector_service.search_knowledge(request.query, top_k=request.top_k)
        document_chunks = await self.vector_service.search_documents(request.query, top_k=request.top_k)

        graph_results = []
        if request.search_type in ("hybrid", "graph"):
            graph_data = await self.graph_service.explore("", limit=min(request.top_k, 20))
            graph_results = graph_data.get("edges", [])

        return SearchResult(
            knowledge_points=knowledge_points,
            document_chunks=document_chunks,
            graph_results=graph_results,
        )