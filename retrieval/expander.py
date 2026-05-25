"""NeighborhoodExpander — expands seed nodes to 1-2 hop neighbors."""

from __future__ import annotations

from core.config_loader import Config
from core.models import GraphNode
from graph.base import GraphPort


class NeighborhoodExpander:
    """Expand seed nodes by fetching their graph neighbors."""

    def __init__(self, graph: GraphPort, config: Config) -> None:
        self.graph = graph
        self.config = config

    async def expand(
        self,
        seed_nodes: list[GraphNode],
        hops: int | None = None,
    ) -> list[GraphNode]:
        """Return *seed_nodes* enriched with their neighbors up to *hops*."""
        h = hops if hops is not None else self.config.retrieval.hop_depth
        all_nodes: dict[str, GraphNode] = {n.node_id: n for n in seed_nodes}

        for node in seed_nodes:
            neighbors = await self.graph.get_neighbors(node.node_id, hops=h)
            for neighbor in neighbors:
                if neighbor.node_id not in all_nodes:
                    all_nodes[neighbor.node_id] = neighbor

        return list(all_nodes.values())
