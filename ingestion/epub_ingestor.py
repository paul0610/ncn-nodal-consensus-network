"""EpubIngestor — ingest EPUB books via ebooklib.

Extracts chapter-level text from an EPUB file by walking all HTML documents
in the spine. Each chapter becomes one or more RawContent chunks, so the
consensus engine can process them as natural units.

Chapter detection: we treat each EPUB item of type ITEM_DOCUMENT as a
chapter and strip its HTML markup via BeautifulSoup. This is robust across
most EPUBs whether they use <h1> chapter headers or not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from core.config_loader import Config
from core.exceptions import NCNError
from core.models import RawContent
from ingestion.base import IngestorPort


_MAX_FILE_SIZE_MB = 100  # books can be larger
_CHUNK_TARGET_CHARS = 2000


class EpubIngestor(IngestorPort):
    """Extract chapter-level text chunks from an EPUB file."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def ingest(self, source: str) -> list[RawContent]:
        path = Path(source)
        if not path.exists():
            raise NCNError(f"EPUB file not found: {source}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_FILE_SIZE_MB:
            raise NCNError(
                f"EPUB file too large ({size_mb:.1f} MB > {_MAX_FILE_SIZE_MB} MB)"
            )

        return await asyncio.to_thread(self._extract, str(path))

    @staticmethod
    def _extract(epub_path: str) -> list[RawContent]:
        try:
            from ebooklib import epub, ITEM_DOCUMENT
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise NCNError(
                "ebooklib not installed. Run: pip install ebooklib"
            ) from exc

        try:
            book = epub.read_epub(epub_path, options={"ignore_ncx": True})
        except Exception as exc:
            raise NCNError(f"Cannot open EPUB '{epub_path}': {exc}") from exc

        name = Path(epub_path).name
        contents: list[RawContent] = []
        chapter_idx = 0

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            try:
                html = item.get_content().decode("utf-8", errors="replace")
            except Exception:
                continue

            soup = BeautifulSoup(html, "html.parser")
            # Strip non-content tags
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n\n", strip=True)
            # Collapse excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            chapter_text = "\n\n".join(lines)

            if not chapter_text or len(chapter_text) < 50:
                continue  # skip empty TOC-like pages

            chapter_idx += 1
            chunks = _paragraph_chunks(chapter_text, _CHUNK_TARGET_CHARS)
            for j, chunk in enumerate(chunks):
                contents.append(
                    RawContent(
                        text=chunk,
                        source=f"epub:{name}:chapter{chapter_idx}:chunk{j + 1}",
                        source_type="epub",
                    )
                )

        logger.info(
            f"EPUB ingested: {epub_path} → "
            f"{chapter_idx} chapters, {len(contents)} chunks"
        )
        return contents


def _paragraph_chunks(text: str, target_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text.strip()]
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > target_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks
