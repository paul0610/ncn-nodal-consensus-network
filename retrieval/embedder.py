"""Embedder — generates vector embeddings via sentence-transformers.

Model is loaded lazily so import cost is near-zero until first use.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.config_loader import Config


class Embedder:
    """Wraps sentence-transformers for async embedding generation."""

    def __init__(self, config: Config) -> None:
        self._model_name = config.retrieval.model
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        self._load_model()
        vector = await asyncio.to_thread(
            self._model.encode, text, normalize_embeddings=True
        )
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one call (more efficient)."""
        if not texts:
            return []
        self._load_model()
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model ready")
