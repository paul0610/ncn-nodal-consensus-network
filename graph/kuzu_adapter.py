"""Kùzu DB adapter — concrete implementation of GraphPort."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import kuzu
import numpy as np
from loguru import logger

from core.config_loader import Config
from core.exceptions import (
    GraphConnectionError,
    GraphSchemaError,
    GraphWriteError,
)
from core.models import GraphNode, GraphRelation, GraphStats
from graph.base import GraphPort

# Column order returned by  ``RETURN n.*``  (we use explicit columns instead
# to stay portable across Kùzu versions).
_NODE_COLUMNS = (
    "node_id",
    "name",
    "node_type",
    "aliases",
    "namespace",
    "confidence",
    "temperature",
    "embedding",
    "source",
    "verified_by",
    "uncertain",
    "created_at",
    "last_accessed",
    "metadata",
)


class KuzuAdapter(GraphPort):
    """Embedded Kùzu graph database adapter."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._in_transaction = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        db_path = self._config.graph.kuzu.path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._db = kuzu.Database(db_path)
            self._conn = kuzu.Connection(self._db)
        except Exception as exc:
            raise GraphConnectionError(
                f"Cannot open Kùzu database at {db_path}: {exc}"
            ) from exc
        await self._ensure_schema()

    async def close(self) -> None:
        self._conn = None
        self._db = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        # --- Create tables if they don't exist ---
        await self._try_execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                node_id          STRING,
                name             STRING,
                node_type        STRING,
                aliases          STRING[],
                namespace        STRING,
                confidence       DOUBLE,
                temperature      DOUBLE,
                embedding        DOUBLE[],
                source           STRING,
                source_authority STRING,
                verified_by      STRING[],
                uncertain        BOOLEAN,
                created_at       STRING,
                last_accessed    STRING,
                metadata         STRING,
                PRIMARY KEY (node_id)
            )
            """,
            schema=True,
        )
        await self._try_execute(
            """
            CREATE REL TABLE IF NOT EXISTS Relation(
                FROM Entity TO Entity,
                relation_id      STRING,
                predicate        STRING,
                weight           DOUBLE,
                confidence       DOUBLE,
                temporal         BOOLEAN,
                date             STRING,
                contested        BOOLEAN,
                source           STRING,
                source_authority STRING,
                created_at       STRING
            )
            """,
            schema=True,
        )
        await self._try_execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Meta(
                key   STRING,
                value STRING,
                PRIMARY KEY (key)
            )
            """,
            schema=True,
        )

        # --- Migrations: add columns to existing tables ---
        await self._migrate_schema()

    async def _migrate_schema(self) -> None:
        """Add columns that may not exist in older databases.

        Each migration is idempotent — if the column already exists the
        ALTER TABLE fails silently.  New columns added in future versions
        should be appended here.
        """
        migrations = [
            # v0.1.1 — Source Authority
            "ALTER TABLE Entity ADD source_authority STRING DEFAULT 'slm_single'",
            "ALTER TABLE Relation ADD source_authority STRING DEFAULT 'slm_single'",
        ]
        for stmt in migrations:
            await self._try_execute(stmt, schema=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def create_node(self, node: GraphNode) -> str:
        params = {
            "node_id": node.node_id,
            "name": node.name,
            "node_type": node.node_type,
            "aliases": node.aliases,
            "namespace": node.namespace,
            "confidence": node.confidence,
            "temperature": node.temperature,
            "embedding": node.embedding or [],
            "source": node.source or "",
            "source_authority": node.source_authority,
            "verified_by": node.verified_by,
            "uncertain": node.uncertain,
            "created_at": node.created_at.isoformat(),
            "last_accessed": node.last_accessed.isoformat(),
            "metadata": json.dumps(node.metadata),
        }
        await self._execute(
            """
            CREATE (n:Entity {
                node_id:          $node_id,
                name:             $name,
                node_type:        $node_type,
                aliases:          $aliases,
                namespace:        $namespace,
                confidence:       $confidence,
                temperature:      $temperature,
                embedding:        $embedding,
                source:           $source,
                source_authority: $source_authority,
                verified_by:      $verified_by,
                uncertain:        $uncertain,
                created_at:       $created_at,
                last_accessed:    $last_accessed,
                metadata:         $metadata
            })
            """,
            params,
        )
        return node.node_id

    async def create_relation(self, relation: GraphRelation) -> None:
        params = {
            "src": relation.source_node_id,
            "tgt": relation.target_node_id,
            "relation_id": relation.relation_id,
            "predicate": relation.predicate,
            "weight": relation.weight,
            "confidence": relation.confidence,
            "temporal": relation.temporal,
            "date": relation.date or "",
            "contested": relation.contested,
            "source": relation.source or "",
            "source_authority": relation.source_authority,
            "created_at": relation.created_at.isoformat(),
        }
        await self._execute(
            """
            MATCH (a:Entity), (b:Entity)
            WHERE a.node_id = $src AND b.node_id = $tgt
            CREATE (a)-[:Relation {
                relation_id:      $relation_id,
                predicate:        $predicate,
                weight:           $weight,
                confidence:       $confidence,
                temporal:         $temporal,
                date:             $date,
                contested:        $contested,
                source:           $source,
                source_authority: $source_authority,
                created_at:       $created_at
            }]->(b)
            """,
            params,
        )

    async def update_node(self, node_id: str, updates: dict) -> None:
        if not updates:
            return
        set_clauses = []
        params: dict = {"node_id": node_id}
        for key, value in updates.items():
            param_name = f"v_{key}"
            if key == "metadata" and isinstance(value, dict):
                value = json.dumps(value)
            set_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
        set_str = ", ".join(set_clauses)
        await self._execute(
            f"MATCH (n:Entity) WHERE n.node_id = $node_id SET {set_str}",
            params,
        )

    # ------------------------------------------------------------------
    # Read — single
    # ------------------------------------------------------------------

    async def get_node(self, node_id: str) -> GraphNode | None:
        rows = await self._query_nodes(
            "MATCH (n:Entity) WHERE n.node_id = $node_id RETURN n",
            {"node_id": node_id},
        )
        return rows[0] if rows else None

    async def find_node_by_name(self, name: str) -> GraphNode | None:
        rows = await self._query_nodes(
            "MATCH (n:Entity) WHERE n.name = $name RETURN n LIMIT 1",
            {"name": name},
        )
        return rows[0] if rows else None

    async def find_node_by_name_in_namespace(
        self, name: str, namespace: str
    ) -> GraphNode | None:
        rows = await self._query_nodes(
            "MATCH (n:Entity) WHERE n.name = $name AND n.namespace = $ns "
            "RETURN n LIMIT 1",
            {"name": name, "ns": namespace},
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Read — multiple
    # ------------------------------------------------------------------

    async def find_nodes_by_type(
        self, node_type: str, limit: int = 20
    ) -> list[GraphNode]:
        return await self._query_nodes(
            "MATCH (n:Entity) WHERE n.node_type = $node_type "
            "RETURN n LIMIT $lim",
            {"node_type": node_type, "lim": limit},
        )

    async def find_nodes_by_namespace(
        self, namespace: str, limit: int = 20
    ) -> list[GraphNode]:
        return await self._query_nodes(
            "MATCH (n:Entity) WHERE n.namespace = $ns RETURN n LIMIT $lim",
            {"ns": namespace, "lim": limit},
        )

    async def get_neighbors(
        self, node_id: str, hops: int = 1
    ) -> list[GraphNode]:
        seen_ids: set[str] = set()
        current_ids: set[str] = {node_id}

        for _ in range(hops):
            next_ids: set[str] = set()
            for cid in current_ids:
                for query in (
                    "MATCH (a:Entity)-[:Relation]->(b:Entity) "
                    "WHERE a.node_id = $id RETURN b",
                    "MATCH (a:Entity)<-[:Relation]-(b:Entity) "
                    "WHERE a.node_id = $id RETURN b",
                ):
                    nodes = await self._query_nodes(query, {"id": cid})
                    for n in nodes:
                        if n.node_id != node_id and n.node_id not in seen_ids:
                            seen_ids.add(n.node_id)
                            next_ids.add(n.node_id)
            current_ids = next_ids

        result: list[GraphNode] = []
        for nid in seen_ids:
            node = await self.get_node(nid)
            if node:
                result.append(node)
        return result

    async def get_all_nodes(
        self, limit: int | None = None
    ) -> list[GraphNode]:
        q = "MATCH (n:Entity) RETURN n"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        return await self._query_nodes(q)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[GraphNode]:
        scored = await self.semantic_search_with_scores(
            vector, top_k, namespace=namespace
        )
        return [n for n, _ in scored]

    async def semantic_search_with_scores(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[tuple[GraphNode, float]]:
        # Filter candidate nodes by namespace at DB level when provided
        if namespace is not None:
            all_nodes = await self.find_nodes_by_namespace(namespace, limit=10000)
        else:
            all_nodes = await self.get_all_nodes()

        if not vector:
            return [(n, 0.0) for n in all_nodes[:top_k]]

        query_vec = np.array(vector, dtype=np.float32)
        scored: list[tuple[float, GraphNode]] = []
        for node in all_nodes:
            if not node.embedding:
                continue
            node_vec = np.array(node.embedding, dtype=np.float32)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
            if denom == 0:
                continue
            sim = float(np.dot(query_vec, node_vec) / denom)
            scored.append((sim, node))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [(n, s) for s, n in scored[:top_k]]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    async def get_relations_for_node(self, node_id: str) -> list[dict]:
        result = await self._execute(
            "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
            "WHERE a.node_id = $id "
            "RETURN r.predicate, r.confidence, r.date, "
            "       r.contested, b.name, b.node_id",
            {"id": node_id},
        )
        relations: list[dict] = []
        while result.has_next():
            row = result.get_next()
            relations.append(
                {
                    "predicate": row[0],
                    "confidence": row[1],
                    "date": row[2],
                    "contested": row[3],
                    "target_name": row[4],
                    "target_node_id": row[5],
                }
            )
        return relations

    # ------------------------------------------------------------------
    # Ontology
    # ------------------------------------------------------------------

    async def register_node_type(self, node_type: str) -> None:
        key = f"node_type:{node_type}"
        existing = await self._scalar(
            "MATCH (m:Meta) WHERE m.key = $key RETURN m.key", {"key": key}
        )
        if existing is None:
            await self._execute(
                "CREATE (m:Meta {key: $key, value: $val})",
                {"key": key, "val": node_type},
            )

    async def get_registered_node_types(self) -> list[str]:
        return await self._column(
            "MATCH (m:Meta) WHERE m.key STARTS WITH 'node_type:' RETURN m.value"
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> GraphStats:
        node_count = await self._scalar(
            "MATCH (n:Entity) RETURN count(n)"
        )
        rel_count = await self._scalar(
            "MATCH ()-[r:Relation]->() RETURN count(r)"
        )
        ns_rows = await self._column(
            "MATCH (n:Entity) RETURN DISTINCT n.namespace"
        )
        type_rows = await self._column(
            "MATCH (n:Entity) RETURN DISTINCT n.node_type"
        )
        return GraphStats(
            total_nodes=int(node_count or 0),
            total_relations=int(rel_count or 0),
            namespaces=ns_rows,
            node_types=type_rows,
        )

    async def count_relations_in_namespace(self, namespace: str) -> int:
        """Single aggregate query — O(1) round-trip regardless of graph size."""
        count = await self._scalar(
            "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
            "WHERE a.namespace = $ns AND b.namespace = $ns "
            "RETURN count(r)",
            {"ns": namespace},
        )
        return int(count or 0)

    # ------------------------------------------------------------------
    # Transactions (simplified — Kùzu auto-commits)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def begin_transaction(self):
        self._in_transaction = True
        try:
            yield
        finally:
            self._in_transaction = False

    async def commit(self) -> None:
        self._in_transaction = False

    async def rollback(self) -> None:
        self._in_transaction = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute(self, query: str, params: dict | None = None):
        def _run():
            return self._conn.execute(query, params or {})

        return await asyncio.to_thread(_run)

    async def _try_execute(
        self, query: str, *, schema: bool = False
    ) -> None:
        try:
            await self._execute(query)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "already has" in msg:
                return
            kind = "Schema" if schema else "Query"
            raise GraphSchemaError(f"{kind} error: {exc}") from exc

    async def _query_nodes(
        self, query: str, params: dict | None = None
    ) -> list[GraphNode]:
        result = await self._execute(query, params)
        nodes: list[GraphNode] = []
        while result.has_next():
            row = result.get_next()
            raw = row[0]
            nodes.append(self._to_graph_node(raw))
        return nodes

    async def _scalar(self, query: str, params: dict | None = None):
        result = await self._execute(query, params)
        if result.has_next():
            return result.get_next()[0]
        return None

    async def _column(
        self, query: str, params: dict | None = None
    ) -> list[str]:
        result = await self._execute(query, params)
        values: list[str] = []
        while result.has_next():
            values.append(str(result.get_next()[0]))
        return values

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_graph_node(raw: dict) -> GraphNode:
        embedding = raw.get("embedding", [])
        meta_str = raw.get("metadata", "{}")
        created = raw.get("created_at", "")
        accessed = raw.get("last_accessed", "")

        return GraphNode(
            node_id=raw["node_id"],
            name=raw["name"],
            node_type=raw["node_type"],
            aliases=raw.get("aliases") or [],
            namespace=raw.get("namespace", "general"),
            confidence=raw.get("confidence", 1.0),
            temperature=raw.get("temperature", 0.5),
            embedding=embedding if embedding else None,
            source=raw.get("source") or None,
            source_authority=raw.get("source_authority", "slm_single"),
            verified_by=raw.get("verified_by") or [],
            uncertain=bool(raw.get("uncertain", False)),
            created_at=_parse_dt(created),
            last_accessed=_parse_dt(accessed),
            metadata=json.loads(meta_str) if isinstance(meta_str, str) else {},
        )


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(UTC)
