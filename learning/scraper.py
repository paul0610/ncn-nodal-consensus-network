"""PageScraper — fetches and extracts full page content from URLs.

Auto-detects the source type via HEAD request (Content-Type) or URL
extension, and routes to the appropriate ingestor:

  HTML  → extract via BeautifulSoup (in-process)
  PDF   → download → PDFIngestor (PyMuPDF)
  DOCX  → download → DocxIngestor
  EPUB  → download → EpubIngestor
  TXT   → download → TXTIngestor
  MD    → download → MarkdownIngestor

This means that when the learning pipeline navigates the web and finds a
PDF or EPUB link, it processes that file properly instead of discarding it.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

from core.config_loader import Config
from core.models import RawContent
from ingestion.detect import detect_source_type


_USER_AGENT = "NCN/0.1 (Nodal Consensus Network; research project)"

# Source types that require downloading the whole file then running a
# dedicated ingestor on the local path. HTML is handled in-process.
_BINARY_SOURCE_TYPES = {"pdf", "docx", "epub", "txt", "markdown"}


class PageScraper:
    """Fetch full page content from URLs with rate limiting + format routing."""

    def __init__(self, config: Config) -> None:
        self.config = config
        cl = config.continuous_learning
        self._rate_limit = cl.rate_limit_seconds
        self._timeout = cl.page_timeout
        self._max_chars = cl.max_page_chars
        self._chunk_size = 500

    async def scrape_urls(self, urls: list[str]) -> list[RawContent]:
        """Fetch and clean multiple URLs with rate limiting between requests."""
        results: list[RawContent] = []
        for url in urls:
            try:
                chunks = await self.scrape_one(url)
                results.extend(chunks)
            except Exception as exc:
                logger.warning(f"Scrape failed for {url}: {exc}")

            if self._rate_limit > 0:
                await asyncio.sleep(self._rate_limit)

        return results

    async def scrape_one(self, url: str) -> list[RawContent]:
        """Fetch a URL, auto-detecting format and routing to the right ingestor."""
        try:
            # Detect type via HEAD request (avoids downloading unwanted binaries)
            source_type = await self._detect_remote_type(url)

            if source_type in _BINARY_SOURCE_TYPES:
                return await self._download_and_ingest(url, source_type)

            # HTML fallback: fetch and clean in-process
            content = await self._fetch_and_clean_html(url)
            if content:
                return self._chunk_html_text(content, url)
        except Exception as exc:
            logger.warning(f"Scrape failed for {url}: {exc}")
        return []

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    async def _detect_remote_type(self, url: str) -> str:
        """HEAD request → Content-Type → source_type, with URL fallback."""
        content_type: str | None = None
        try:
            headers = {"User-Agent": _USER_AGENT}
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.head(url, headers=headers, timeout=10)
                content_type = resp.headers.get("content-type", "")
        except Exception as exc:
            logger.debug(f"HEAD failed for {url}: {exc} — using URL fallback")

        return detect_source_type(url, content_type)

    # ------------------------------------------------------------------
    # HTML path
    # ------------------------------------------------------------------

    async def _fetch_and_clean_html(self, url: str) -> str:
        """Fetch a URL as HTML and return cleaned text."""
        headers = {"User-Agent": _USER_AGENT}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            html = resp.text

        cleaned = _clean_html(html)
        return cleaned[: self._max_chars] if cleaned else ""

    def _chunk_html_text(self, text: str, url: str) -> list[RawContent]:
        """Split long HTML-extracted text into chunks."""
        if len(text) <= self._chunk_size:
            return [
                RawContent(text=text, source=f"page:{url}", source_type="web_page")
            ]

        chunks: list[RawContent] = []
        paragraphs = text.split("\n\n")
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > self._chunk_size and current:
                chunks.append(
                    RawContent(
                        text=current.strip(),
                        source=f"page:{url}:chunk{len(chunks) + 1}",
                        source_type="web_page",
                    )
                )
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para

        if current.strip():
            chunks.append(
                RawContent(
                    text=current.strip(),
                    source=f"page:{url}:chunk{len(chunks) + 1}",
                    source_type="web_page",
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Binary path — download then ingest
    # ------------------------------------------------------------------

    async def _download_and_ingest(
        self, url: str, source_type: str
    ) -> list[RawContent]:
        """Download the URL to a temp file and run the matching ingestor."""
        # Pick an appropriate extension for the temp file
        ext_map = {
            "pdf": ".pdf",
            "docx": ".docx",
            "epub": ".epub",
            "txt": ".txt",
            "markdown": ".md",
        }
        suffix = ext_map.get(source_type, "")

        headers = {"User-Agent": _USER_AGENT}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.content

        # Write to a temp file so the ingestor can read from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            # Lazy import to avoid circular deps
            from ingestion.factory import create_ingestor

            ingestor = create_ingestor(source_type, self.config)
            chunks = await ingestor.ingest(tmp_path)

            # Rewrite source so downstream knows this came from a remote URL,
            # not a local file
            for chunk in chunks:
                chunk.source = f"{source_type}:{url}"
                chunk.source_type = source_type

            logger.info(
                f"Scraped {source_type.upper()}: {url} -> {len(chunks)} chunks"
            )
            return chunks
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTML cleaning helper
# ---------------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    """Strip HTML to readable text — same logic as web_ingestor."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n\n".join(lines)
