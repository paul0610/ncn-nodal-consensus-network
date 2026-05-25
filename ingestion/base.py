"""IngestorPort — abstract interface for all ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import RawContent


class IngestorPort(ABC):
    """Port that every ingestor must implement.

    The returned ``RawContent`` objects are later fed into the consensus
    pipeline for verification before being written to the graph.
    """

    @abstractmethod
    async def ingest(self, source: Any) -> list[RawContent]:
        """Ingest from *source* and return raw content chunks."""
