"""Ingestor factory — returns the right IngestorPort for a source type."""

from __future__ import annotations

from core.config_loader import Config
from core.exceptions import ConfigValidationError
from ingestion.base import IngestorPort
from ingestion.docx_ingestor import DocxIngestor
from ingestion.epub_ingestor import EpubIngestor
from ingestion.markdown_ingestor import MarkdownIngestor
from ingestion.pdf_ingestor import PDFIngestor
from ingestion.text_ingestor import TextIngestor
from ingestion.txt_ingestor import TXTIngestor
from ingestion.web_ingestor import WebIngestor


_REGISTRY: dict[str, type[IngestorPort]] = {
    # Raw text from user (paragraph split)
    "text": TextIngestor,
    # Plain text file on disk
    "txt": TXTIngestor,
    # PDF file (PyMuPDF)
    "pdf": PDFIngestor,
    # Word documents
    "docx": DocxIngestor,
    # EPUB books
    "epub": EpubIngestor,
    # Markdown (header-aware chunking)
    "markdown": MarkdownIngestor,
    "md": MarkdownIngestor,  # alias
    # Web search (DuckDuckGo + Wikipedia)
    "web": WebIngestor,
}


def create_ingestor(source_type: str, config: Config) -> IngestorPort:
    """Instantiate an ingestor by ``source_type``.

    Recognized types: ``text``, ``txt``, ``pdf``, ``docx``, ``epub``,
    ``markdown`` (alias ``md``), ``web``.
    """
    cls = _REGISTRY.get(source_type)
    if cls is None:
        raise ConfigValidationError(
            f"Ingestor type '{source_type}' not recognised. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return cls(config)


def supported_ingestor_types() -> list[str]:
    """Return all source_type values accepted by ``create_ingestor``."""
    return sorted(_REGISTRY.keys())
