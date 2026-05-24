import json
import logging
from services.graph.vector_service import VectorService
from services.graph.neo4j_service import GraphService

logger = logging.getLogger(__name__)


class WebGenService:
    def __init__(self, vector_service: VectorService, graph_service: GraphService):
        self.vector_service = vector_service
        self.graph_service = graph_service

    async def generate_knowledge_card(self, kp_id: str) -> dict:
        kp = await self.vector_service.get_knowledge_point(kp_id)
        if not kp:
            return {}
        return {
            "id": kp_id,
            "fact": kp.get("fact", ""),
            "category": kp.get("category", ""),
            "confidence": kp.get("confidence", 0.5),
            "source": kp.get("source", ""),
            "related_entities": kp.get("related_entities", []),
            "created_at": str(kp.get("created_at", "")),
        }

    async def generate_mindmap_data(self, entity: str = "") -> dict:
        graph_data = await self.graph_service.explore(entity, limit=50)
        return graph_data

    async def generate_timeline_data(self) -> list:
        all_points = []
        for kp_id, item in self.vector_service.knowledge_index.items():
            data = item["data"]
            all_points.append({
                "id": kp_id,
                "fact": data.get("fact", ""),
                "category": data.get("category", ""),
                "created_at": str(data.get("created_at", "")),
            })
        all_points.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return all_points[:100]

    async def get_stats(self) -> dict:
        return {
            "total_knowledge_points": len(self.vector_service.knowledge_index),
            "total_documents": len(self.vector_service.document_index),
            "graph_nodes": await self.graph_service.count_nodes(),
        }