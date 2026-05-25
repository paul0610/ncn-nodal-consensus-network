"""PDFIngestor — extracts text from PDF files via PyMuPDF (fitz).

Architecture mandate: use PyMuPDF — NOT PyPDF2 or pdfplumber.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from core.config_loader import Config
from core.exceptions import NCNError
from core.models import RawContent
from ingestion.base import IngestorPort


class PDFIngestor(IngestorPort):
    """Extract page-level text chunks from a PDF file."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def ingest(self, source: str) -> list[RawContent]:
        """*source* is a file-system path to a PDF."""
        path = Path(source)
        if not path.exists():
            raise NCNError(f"PDF not found: {source}")

        max_mb = self.config.ingestion.pdf.max_size_mb
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            raise NCNError(
                f"PDF too large ({size_mb:.1f} MB > {max_mb} MB limit)"
            )

        return await asyncio.to_thread(self._extract, str(path))

    @staticmethod
    def _extract(pdf_path: str) -> list[RawContent]:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        contents: list[RawContent] = []
        try:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    contents.append(
                        RawContent(
                            text=text.strip(),
                            source=f"pdf:{pdf_path}:page{page_num + 1}",
                            source_type="pdf",
                        )
                    )
        finally:
            doc.close()

        logger.info(f"PDF ingested: {pdf_path} → {len(contents)} pages")
        return contents
