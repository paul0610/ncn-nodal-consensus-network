"""Retriever — orchestrates the full retrieval pipeline.

Pipeline:
    query text
      → embed
      → semantic search with scores (top-K)
      → **Relevance Gate**: filter out nodes below threshold
      → expand neighborhood (1-2 hops)
      → rank by temperature
      → apply token budget
      → serialize to bullet points
      → GraphContext (with max_relevance)
"""

from __future__ import annotations

from loguru import logger

from core.config_loader import Config
from core.models import GraphContext
from graph.base import GraphPort
from retrieval.embedder import Embedder
from retrieval.expander import NeighborhoodExpander
from retrieval.serializer import NodeSerializer


class Retriever:
    """End-to-end retrieval from query text to serialised graph context."""

    def __init__(
        self,
        graph: GraphPort,
        config: Config,
        embedder: Embedder | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.embedder = embedder or Embedder(config)
        self.expander = NeighborhoodExpander(graph, config)
        self.serializer = NodeSerializer(graph, config)

    async def retrieve(
        self, query: str, namespace: str | None = None
    ) -> GraphContext:
        """Run the full retrieval pipeline for *query*.

        If *namespace* is provided, restrict retrieval to that namespace —
        prevents cross-domain contamination between isolated knowledge bases.
        """

        # 1. Embed query
        query_vector = await self.embedder.embed(query)

        # 2. Top-K nodes with similarity scores (namespace-scoped if provided)
        scored = await self.graph.semantic_search_with_scores(
            vector=query_vector,
            top_k=self.config.retrieval.top_k_nodes,
            namespace=namespace,
        )

        # Record max similarity BEFORE filtering (for logging/debugging)
        max_relevance = max((s for _, s in scored), default=0.0)

        # Debug: log top-3 scores with node names to diagnose cross-contamination
        top3 = scored[:3]
        top3_str = ", ".join(
            f"'{n.name[:30]}'={s:.3f}" for n, s in top3
        )
        logger.info(f"Retriever query='{query[:50]}' → top3: [{top3_str}]")

        # 3. Relevance Gate: filter out nodes below threshold
        threshold = self.config.retrieval.relevance_threshold
        relevant = [(n, s) for n, s in scored if s >= threshold]

        if not relevant:
            logger.info(
                f"Relevance Gate: no nodes above threshold {threshold:.2f} "
                f"(max similarity was {max_relevance:.2f}) — empty context"
            )
            return GraphContext(
                nodes=[],
                serialized_context="",
                avg_confidence=0.0,
                nodes_used=[],
                max_relevance=max_relevance,
            )

        seed_nodes = [n for n, _ in relevant]
        logger.info(
            f"Relevance Gate: {len(relevant)}/{len(scored)} nodes passed "
            f"threshold={threshold:.2f} (max similarity: {max_relevance:.3f})"
        )

        # 4. Expand neighborhood
        expanded_nodes = await self.expander.expand(seed_nodes=seed_nodes)

        # 5. Rank by temperature (Pareto of the graph)
        ranked_nodes = sorted(
            expanded_nodes, key=lambda n: n.temperature, reverse=True
        )

        # 6. Apply token budget
        selected_nodes = await self.serializer.apply_budget(
            nodes=ranked_nodes,
            budget=self.config.retrieval.token_budget,
        )

        # 7. Serialize as bullet points
        serialized = await self.serializer.serialize(selected_nodes)

        avg_confidence = (
            sum(n.confidence for n in selected_nodes) / len(selected_nodes)
            if selected_nodes
            else 0.0
        )

        return GraphContext(
            nodes=selected_nodes,
            serialized_context=serialized,
            avg_confidence=avg_confidence,
            nodes_used=[n.node_id for n in selected_nodes],
            max_relevance=max_relevance,
        )
