"""OntologyAgent — auto-extensible ontology by consensus.

Classifies entities into types.  When no existing type fits, proposes a
new one validated by a mini-consensus of 3 models.

Academic concept: "self-extending ontology by consensus" — one of NCN's
original contributions.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, NodeRole
from graph.base import GraphPort
from ontology.classifier import classify_with_existing
from ontology.deduplicator import SemanticDeduplicator
from swarm.pool import SwarmPool


class OntologyAgent:
    """Classify entities and auto-extend the ontology when needed."""

    def __init__(
        self,
        config: Config,
        swarm: SwarmPool,
        graph: GraphPort,
        embedder=None,
    ) -> None:
        self.config = config
        self.swarm = swarm
        self.graph = graph
        self.deduplicator = SemanticDeduplicator(config, embedder=embedder)
        self._known_types: set[str] = set(config.graph.node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(self, entity_name: str, context: Claim) -> str:
        """Return the ontology type for *entity_name*.

        If no known type fits with sufficient confidence and
        ``ontology.auto_extend`` is enabled, a new type is proposed,
        deduplicated, and validated by mini-consensus before being
        registered.
        """
        best_type, confidence = await classify_with_existing(
            entity_name, context, self._known_types, self.swarm
        )

        if confidence >= self.config.ontology.min_confidence:
            return best_type

        if self.config.ontology.auto_extend:
            return await self._propose_new_type(entity_name, context)

        return "concepto"

    # ------------------------------------------------------------------
    # Proposal + validation
    # ------------------------------------------------------------------

    async def _propose_new_type(
        self, entity_name: str, context: Claim
    ) -> str:
        # Respect custom-type limit
        base_count = len(self.config.graph.node_types)
        custom_count = len(self._known_types) - base_count
        if custom_count >= self.config.ontology.max_custom_types:
            logger.warning("Custom type limit reached — using 'concepto'")
            return "concepto"

        # Ask a model to propose a type
        prompt = (
            f'La entidad "{entity_name}" no encaja en estos tipos existentes: '
            f"{sorted(self._known_types)}\n"
            f"Contexto de la afirmación: {context.predicate} → {context.object}\n\n"
            "Propón UN SOLO tipo de nodo nuevo que describa mejor esta entidad.\n"
            "El tipo debe ser:\n"
            "- En singular\n"
            "- En minúsculas con guiones bajos\n"
            "- Específico pero reutilizable\n\n"
            "Responde SOLO el nombre del tipo, sin explicación.\n"
            'Ejemplo: "tecnica_biologica" o "algoritmo_computacional"'
        )

        responses = await self.swarm.run_role(NodeRole.EXTRACTOR, prompt)
        if not responses:
            return "concepto"

        proposed = _extract_type_name(responses[0].content)

        # Semantic deduplication
        deduplicated = await self.deduplicator.deduplicate(
            proposed, self._known_types
        )
        if deduplicated != proposed:
            logger.info(f"Type '{proposed}' deduplicated to '{deduplicated}'")
            return deduplicated

        # Mini-consensus validation
        if await self._validate_new_type(proposed, entity_name):
            self._known_types.add(proposed)
            logger.info(f"New ontology type created: '{proposed}'")
            await self.graph.register_node_type(proposed)
            return proposed

        return "concepto"

    async def _validate_new_type(
        self, proposed: str, entity_name: str
    ) -> bool:
        """Validate a proposed type with a mini-consensus (N models)."""
        n = self.config.ontology.validators
        prompt = (
            f'¿Es "{proposed}" un buen tipo de nodo para la entidad "{entity_name}"?\n'
            f"Tipos existentes: {sorted(self._known_types)}\n\n"
            "Responde con JSON: "
            '{"approve": true/false, "confidence": 0.0-1.0}'
        )
        responses = await self.swarm.run_role(NodeRole.CRITIC, prompt)
        if not responses:
            return False

        approvals = 0
        for resp in responses[:n]:
            approve, _ = _parse_approval(resp.content)
            if approve:
                approvals += 1

        required = (n // 2) + 1  # simple majority
        return approvals >= required


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_TYPE_RE = re.compile(r"[a-záéíóúñü][a-záéíóúñü0-9_]*")
_JSON_OBJ = re.compile(r"\{[\s\S]*?\}")


def _extract_type_name(text: str) -> str:
    """Pull a clean snake_case type name from noisy LLM output."""
    cleaned = text.strip().strip('"').strip("'").strip().lower()
    cleaned = re.sub(r"[`\"']", "", cleaned)
    matches = _TYPE_RE.findall(cleaned)
    if not matches:
        return "concepto"
    # Prefer snake_case matches (actual type names) over plain words
    snake = [m for m in matches if "_" in m]
    if snake:
        return snake[0]
    return max(matches, key=len)


def _parse_approval(text: str) -> tuple[bool, float]:
    """Parse an approve/reject vote from critic output."""
    try:
        m = _JSON_OBJ.search(text)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("approve", False)), float(
                data.get("confidence", 0.5)
            )
    except Exception:
        pass
    # Heuristic fallback
    low = text.lower()
    if "true" in low or "approve" in low:
        return True, 0.5
    return False, 0.5
