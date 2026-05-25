"""SemanticDeduplicator — prevents redundant ontology types.

Uses embedding cosine-similarity to decide whether a proposed new type
is semantically equivalent to an already-known type.  If the similarity
exceeds ``ontology.dedup_threshold`` the existing type is reused.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import numpy as np

from core.config_loader import Config

if TYPE_CHECKING:
    from retrieval.embedder import Embedder


class SemanticDeduplicator:
    """Deduplicate proposed ontology types against known types."""

    def __init__(self, config: Config, embedder: Embedder | None = None) -> None:
        self.config = config
        self._threshold = config.ontology.dedup_threshold
        self._embedder = embedder

    async def deduplicate(
        self, proposed: str, known_types: set[str]
    ) -> str:
        """Return *proposed* if unique, else the matching existing type."""
        if not known_types:
            return proposed

        proposed_norm = _normalise_type(proposed)

        # Fast exact/substring check first
        for kt in known_types:
            if proposed_norm == _normalise_type(kt):
                return kt

        # Semantic similarity via embeddings (only if embedder available)
        if self._embedder is None:
            return proposed

        proposed_vec = np.array(await self._embedder.embed(proposed_norm), dtype=np.float32)

        best_type = proposed
        best_sim = 0.0

        for kt in known_types:
            kt_vec = np.array(await self._embedder.embed(_normalise_type(kt)), dtype=np.float32)
            sim = _cosine(proposed_vec, kt_vec)
            if sim > best_sim:
                best_sim = sim
                best_type = kt

        if best_sim >= self._threshold:
            return best_type

        return proposed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_UNDER = re.compile(r"[_\-]+")


def _normalise_type(name: str) -> str:
    """Lowercase, collapse underscores/hyphens to space."""
    return _UNDER.sub(" ", name.strip().lower())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
