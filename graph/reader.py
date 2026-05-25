"""GraphReader — convenience read-only interface over GraphPort."""

from __future__ import annotations

from core.config_loader import Config
from core.models import GraphNode, GraphStats
from graph.base import GraphPort


class GraphReader:
    """High-level read API used by the Retriever and other modules."""

    def __init__(self, graph: GraphPort, config: Config) -> None:
        self.graph = graph
        self.config = config

    async def get_node(self, node_id: str) -> GraphNode | None:
        return await self.graph.get_node(node_id)

    async def find_by_name(self, name: str) -> GraphNode | None:
        return await self.graph.find_node_by_name(name)

    async def find_by_type(
        self, node_type: str, limit: int = 20
    ) -> list[GraphNode]:
        return await self.graph.find_nodes_by_type(node_type, limit)

    async def find_by_namespace(
        self, namespace: str, limit: int = 20
    ) -> list[GraphNode]:
        return await self.graph.find_nodes_by_namespace(namespace, limit)

    async def get_neighbors(
        self, node_id: str, hops: int | None = None
    ) -> list[GraphNode]:
        h = hops if hops is not None else self.config.retrieval.hop_depth
        return await self.graph.get_neighbors(node_id, h)

    async def semantic_search(
        self, vector: list[float], top_k: int | None = None
    ) -> list[GraphNode]:
        k = top_k if top_k is not None else self.config.retrieval.top_k_nodes
        return await self.graph.semantic_search(vector, k)

    async def get_stats(self) -> GraphStats:
        return await self.graph.get_stats()

    async def get_all_nodes(
        self, limit: int | None = None
    ) -> list[GraphNode]:
        return await self.graph.get_all_nodes(limit)
