"""TextIngestor — ingests plain text supplied by the user."""

from __future__ import annotations

from core.config_loader import Config
from core.models import RawContent
from ingestion.base import IngestorPort


class TextIngestor(IngestorPort):
    """Split user-supplied text into paragraph-level chunks."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def ingest(self, source: str) -> list[RawContent]:
        """*source* is a plain text string."""
        if not source or not source.strip():
            return []

        paragraphs = [p.strip() for p in source.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [source.strip()]

        return [
            RawContent(
                text=para,
                source="user:text",
                source_type="text",
            )
            for para in paragraphs
        ]
