"""GraphPort — abstract interface for graph database adapters.

The rest of the system ONLY talks to GraphPort.
NEVER import a concrete adapter directly from outside the graph/ package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import GraphNode, GraphRelation, GraphStats


class GraphPort(ABC):
    """Abstract port for graph database operations."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema / tables if they don't exist yet."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_node(self, node: GraphNode) -> str:
        """Persist a node. Returns its ``node_id``."""

    @abstractmethod
    async def create_relation(self, relation: GraphRelation) -> None:
        """Persist a relation between two existing nodes."""

    @abstractmethod
    async def update_node(self, node_id: str, updates: dict) -> None:
        """Update one or more properties of an existing node."""

    # ------------------------------------------------------------------
    # Read — single
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        """Return a node by id, or ``None``."""

    @abstractmethod
    async def find_node_by_name(self, name: str) -> GraphNode | None:
        """Return the first node whose ``name`` matches exactly.

        Global (namespace-agnostic) lookup. Use
        :meth:`find_node_by_name_in_namespace` when you need strict
        namespace isolation.
        """

    @abstractmethod
    async def find_node_by_name_in_namespace(
        self, name: str, namespace: str
    ) -> GraphNode | None:
        """Return the first node matching ``name`` AND ``namespace`` exactly.

        Used by namespace-scoped upserts so the same entity name can
        coexist as distinct nodes in different namespaces (e.g. ``water``
        in ``survival`` is a separate node from ``water`` in ``cooking``).
        """

    @abstractmethod
    async def count_relations_in_namespace(self, namespace: str) -> int:
        """Count edges where BOTH endpoints are in the given namespace.

        Must be implemented as a single aggregate DB query — iterating
        over nodes and calling ``get_relations_for_node`` per node is
        too slow for graphs with thousands of nodes (timeout risk).
        """

    # ------------------------------------------------------------------
    # Read — multiple
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_nodes_by_type(
        self, node_type: str, limit: int = 20
    ) -> list[GraphNode]:
        """Return nodes with the given ``node_type``."""

    @abstractmethod
    async def find_nodes_by_namespace(
        self, namespace: str, limit: int = 20
    ) -> list[GraphNode]:
        """Return nodes in the given namespace."""

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, hops: int = 1
    ) -> list[GraphNode]:
        """Return neighbor nodes up to *hops* away (both directions)."""

    @abstractmethod
    async def get_all_nodes(
        self, limit: int | None = None
    ) -> list[GraphNode]:
        """Return all nodes (optional limit)."""

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @abstractmethod
    async def semantic_search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[GraphNode]:
        """Return the *top_k* nodes closest to *vector* by cosine similarity.

        If *namespace* is provided, restrict the search to nodes whose
        ``namespace`` equals the given string.
        """

    @abstractmethod
    async def semantic_search_with_scores(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[tuple[GraphNode, float]]:
        """Return *top_k* nodes with their cosine similarity scores (0.0-1.0).

        Used by the Relevance Gate to filter out semantically unrelated nodes
        and prevent cross-contamination hallucinations.

        If *namespace* is provided, restrict the search to that namespace.
        """

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_relations_for_node(
        self, node_id: str
    ) -> list[dict]:
        """Return outgoing relations with target info.

        Each dict contains: ``predicate``, ``confidence``, ``date``,
        ``contested``, ``target_name``, ``target_node_id``.
        """

    # ------------------------------------------------------------------
    # Ontology
    # ------------------------------------------------------------------

    @abstractmethod
    async def register_node_type(self, node_type: str) -> None:
        """Persist a new ontology node type as graph metadata."""

    @abstractmethod
    async def get_registered_node_types(self) -> list[str]:
        """Return all registered node types."""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_stats(self) -> GraphStats:
        """Aggregate statistics about the graph."""

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @abstractmethod
    def begin_transaction(self):
        """Return an **async context-manager** for transaction scope."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    async def rollback(self) -> None:
        """Roll back the current transaction."""
