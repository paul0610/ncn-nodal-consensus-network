"""SingleWriter — the ONLY process that writes to the graph.

Models in the swarm NEVER write directly.
The consensus engine NEVER writes directly.
ONLY this writer writes.

Now supports **Source Authority**: high-authority sources can correct
low-authority nodes in-place, like a human brain learning from experts.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from core.config_loader import Config
from core.models import (
    Claim,
    CorrectionEvent,
    GraphNode,
    GraphRelation,
    SourceAuthority,
    authority_rank,
)
from graph.base import GraphPort

if TYPE_CHECKING:
    from ontology.agent import OntologyAgent
    from retrieval.embedder import Embedder


class SingleWriter:
    """Single-Writer / Multiple-Reader pattern for the knowledge graph.

    Accepts optional *ontology_agent* (for type classification) and
    *embedder* (for generating node embeddings).  When omitted the writer
    falls back to ``node_type="concepto"`` and ``embedding=None``.
    """

    def __init__(
        self,
        graph: GraphPort,
        config: Config,
        ontology_agent: OntologyAgent | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.ontology_agent = ontology_agent
        self.embedder = embedder
        self._write_lock = asyncio.Lock()
        self._corrections: list[CorrectionEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write_verified_claims(
        self,
        claims: list[Claim],
        namespace: str | None = None,
    ) -> list[str]:
        """Write verified claims to the graph. Returns created node_ids.

        If *namespace* is provided, all new nodes are created in that namespace.
        If None, falls back to ``config.graph.default_namespace``.
        """
        ns = namespace or self.config.graph.default_namespace
        async with self._write_lock:
            self._corrections = []
            created_node_ids: list[str] = []
            async with self.graph.begin_transaction():
                try:
                    for claim in claims:
                        subject_id = await self._upsert_entity(
                            claim.subject, claim, namespace=ns
                        )
                        object_id = await self._upsert_entity(
                            claim.object, claim, namespace=ns
                        )
                        await self._upsert_relation(
                            subject_id, object_id, claim
                        )
                        # Detect and correct conflicts
                        await self._detect_and_correct(
                            subject_id, claim
                        )
                        created_node_ids.extend([subject_id, object_id])
                    await self.graph.commit()
                except Exception as exc:
                    await self.graph.rollback()
                    logger.error(f"Write failed, rolled back: {exc}")
                    raise
            return list(set(created_node_ids))

    def get_and_clear_corrections(self) -> list[CorrectionEvent]:
        """Return accumulated corrections and clear the buffer."""
        corrections = list(self._corrections)
        self._corrections = []
        return corrections

    async def write_node_direct(self, node: GraphNode) -> str:
        """Write a node directly, bypassing the claim/consensus pipeline.

        Used by the agent module for skills, action logs, and memory.
        Still protected by the write lock.
        """
        async with self._write_lock:
            if self.embedder and not node.embedding:
                try:
                    node.embedding = await self.embedder.embed(node.name)
                except Exception as exc:
                    logger.warning(f"Embedding failed for '{node.name}': {exc}")
            return await self.graph.create_node(node)

    async def update_node_direct(self, node_id: str, updates: dict) -> None:
        """Update a node directly (lock-protected)."""
        async with self._write_lock:
            await self.graph.update_node(node_id, updates)

    async def update_temperatures(self, node_ids: list[str]) -> None:
        """Heat nodes that were consulted in this session."""
        async with self._write_lock:
            for node_id in node_ids:
                node = await self.graph.get_node(node_id)
                if node:
                    new_temp = min(
                        1.0,
                        node.temperature + self.config.graph.temperature_boost,
                    )
                    await self.graph.update_node(
                        node_id, {"temperature": new_temp}
                    )

    async def cool_down_all_nodes(self) -> None:
        """Decay temperatures at the end of a session (Pareto of the graph)."""
        async with self._write_lock:
            all_nodes = await self.graph.get_all_nodes()
            for node in all_nodes:
                new_temp = max(
                    0.0,
                    node.temperature - self.config.graph.temperature_decay,
                )
                await self.graph.update_node(
                    node.node_id, {"temperature": new_temp}
                )

    # ------------------------------------------------------------------
    # Entity upsert with authority
    # ------------------------------------------------------------------

    async def _upsert_entity(
        self,
        entity_name: str,
        claim: Claim,
        namespace: str | None = None,
    ) -> str:
        # Scope the lookup by namespace so that the same entity name can
        # coexist in isolated domains (e.g. "water" in survival vs cooking
        # must be distinct nodes, otherwise the first one wins forever).
        # If namespace is None (legacy callers), fall back to global name.
        if namespace is not None:
            existing = await self.graph.find_node_by_name_in_namespace(
                entity_name, namespace
            )
        else:
            existing = await self.graph.find_node_by_name(entity_name)
        claim_auth = claim.source_authority.value
        auth_weight = self._get_authority_weight(claim_auth)

        if existing:
            existing_rank = authority_rank(existing.source_authority)
            new_rank = authority_rank(claim_auth)

            # Higher authority → bigger confidence boost + upgrade authority
            if new_rank < existing_rank:  # lower rank number = higher authority
                gap = existing_rank - new_rank
                boost = (
                    self.config.graph.confidence_boost
                    + gap * self.config.source_authority.confidence_bonus_on_high_authority
                )
                new_auth = claim_auth
            else:
                boost = self.config.graph.confidence_boost
                new_auth = existing.source_authority

            new_confidence = min(1.0, existing.confidence + boost)
            await self.graph.update_node(
                existing.node_id,
                {
                    "confidence": new_confidence,
                    "source_authority": new_auth,
                    "last_accessed": datetime.now(UTC).isoformat(),
                },
            )
            return existing.node_id

        # --- Create new node ---
        node_type = "concepto"
        if self.ontology_agent:
            try:
                node_type = await self.ontology_agent.classify(entity_name, claim)
            except Exception as exc:
                logger.warning(f"Ontology classification failed for '{entity_name}': {exc}")

        embedding: list[float] | None = None
        if self.embedder:
            try:
                embedding = await self.embedder.embed(entity_name)
            except Exception as exc:
                logger.warning(f"Embedding failed for '{entity_name}': {exc}")

        # Authority-weighted initial confidence
        sa = self.config.source_authority
        if sa.enabled:
            initial_conf = claim.confidence * auth_weight
        else:
            initial_conf = claim.confidence

        node = GraphNode(
            name=entity_name,
            node_type=node_type,
            namespace=namespace or self.config.graph.default_namespace,
            confidence=initial_conf,
            embedding=embedding,
            source=claim.source_text,
            source_authority=claim_auth,
            verified_by=[claim_auth],
        )
        return await self.graph.create_node(node)

    async def _upsert_relation(
        self, source_id: str, target_id: str, claim: Claim
    ) -> None:
        relation = GraphRelation(
            source_node_id=source_id,
            target_node_id=target_id,
            predicate=claim.predicate,
            confidence=claim.confidence,
            temporal=claim.temporal,
            date=claim.date,
            source=claim.source_text,
            source_authority=claim.source_authority.value,
        )
        await self.graph.create_relation(relation)

    # ------------------------------------------------------------------
    # Conflict detection and correction
    # ------------------------------------------------------------------

    async def _detect_and_correct(
        self, subject_node_id: str, claim: Claim
    ) -> None:
        """Reduce confidence of conflicting low-authority nodes."""
        sa = self.config.source_authority
        if not sa.enabled or not sa.correction_enabled:
            return

        new_rank = authority_rank(claim.source_authority.value)
        # Only high-authority claims trigger corrections
        if new_rank > 2:  # consensus or lower don't correct
            return

        relations = await self.graph.get_relations_for_node(subject_node_id)
        for rel in relations:
            # Same predicate but different target → potential conflict
            if rel["predicate"] == claim.predicate:
                target_name = rel.get("target_name", "")
                if target_name and target_name != claim.object:
                    target_id = rel.get("target_node_id", "")
                    target_node = await self.graph.get_node(target_id) if target_id else None
                    if not target_node:
                        continue

                    existing_rank = authority_rank(target_node.source_authority)
                    gap = existing_rank - new_rank

                    if gap >= sa.authority_gap_for_correction:
                        old_conf = target_node.confidence
                        new_conf = max(0.05, old_conf - sa.confidence_penalty_on_correction)
                        is_uncertain = new_conf < self.config.graph.uncertain_threshold

                        await self.graph.update_node(
                            target_node.node_id,
                            {
                                "confidence": new_conf,
                                "uncertain": is_uncertain,
                            },
                        )

                        event = CorrectionEvent(
                            corrected_node_id=target_node.node_id,
                            corrected_node_name=target_node.name,
                            old_confidence=old_conf,
                            new_confidence=new_conf,
                            correcting_authority=claim.source_authority.value,
                            corrected_authority=target_node.source_authority,
                            correcting_claim_id=claim.claim_id,
                            reason=f"Conflict: '{claim.subject} → {claim.predicate}' "
                                   f"target '{claim.object}' vs '{target_name}'",
                        )
                        self._corrections.append(event)
                        logger.info(
                            f"CORRECTION: '{target_node.name}' conf "
                            f"{old_conf:.2f}→{new_conf:.2f} "
                            f"(auth: {target_node.source_authority}→corrected by {claim.source_authority.value})"
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_authority_weight(self, auth: str) -> float:
        sa = self.config.source_authority
        if not sa.enabled:
            return 1.0
        return sa.weights.get(auth, 0.30)
