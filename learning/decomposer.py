"""QueryDecomposer — breaks a broad topic into specific search queries.

Like Genspark: the LLM decides WHAT to search, not the user.
Uses the PLANNER role or EXTRACTOR as fallback.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import NodeRole
from swarm.pool import SwarmPool


class QueryDecomposer:
    """Decompose a broad topic into targeted search sub-queries."""

    def __init__(self, config: Config, swarm: SwarmPool) -> None:
        self.config = config
        self.swarm = swarm

    async def decompose(self, topic: str, count: int | None = None) -> list[str]:
        """Break *topic* into specific search queries.

        Returns up to *count* sub-queries (default from config).
        """
        n = count or self.config.continuous_learning.sub_queries_per_topic
        prompt = (
            f"TEMA A INVESTIGAR: {topic}\n\n"
            f"Genera exactamente {n} consultas de busqueda web especificas "
            "para aprender sobre este tema en profundidad.\n\n"
            "REGLAS:\n"
            "- Cada consulta debe ser especifica y buscar un aspecto diferente del tema\n"
            "- Las consultas deben estar en el mismo idioma que el tema\n"
            "- Cada consulta debe ser corta (5-10 palabras)\n"
            "- Cubre diferentes sub-temas: definiciones, historia, tipos, aplicaciones\n"
            "- Responde UNICAMENTE en JSON valido\n\n"
            "FORMATO:\n"
            '{"queries": ["consulta 1", "consulta 2", ...]}'
        )

        # Try PLANNER first, fallback to EXTRACTOR
        responses = await self.swarm.run_role(NodeRole.PLANNER, prompt)
        if not responses:
            responses = await self.swarm.run_role(NodeRole.EXTRACTOR, prompt)
        if not responses:
            # Last resort: generate simple queries from topic
            return self._fallback_queries(topic, n)

        return self._parse_queries(responses[0].content, topic, n)

    def _parse_queries(
        self, text: str, topic: str, max_count: int
    ) -> list[str]:
        """Parse sub-queries from LLM response."""
        try:
            data = _extract_json(text)
            queries = data.get("queries", [])
            if queries and isinstance(queries, list):
                return [str(q).strip() for q in queries[:max_count] if q]
        except Exception as exc:
            logger.warning(f"Decomposer parse error: {exc}")

        return self._fallback_queries(topic, max_count)

    @staticmethod
    def _fallback_queries(topic: str, count: int) -> list[str]:
        """Generate simple queries when LLM fails."""
        prefixes = [
            "que es", "historia de", "tipos de",
            "caracteristicas de", "como funciona",
            "aplicaciones de", "ejemplos de",
            "importancia de", "origen de", "definicion de",
        ]
        return [f"{p} {topic}" for p in prefixes[:count]]


# ------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
    raw = m.group(1) if m and m.lastindex else (m.group(0) if m else text)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(raw)
