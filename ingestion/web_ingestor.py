"""WebIngestor — on-demand web search via DuckDuckGo + Wikipedia.

Content is cleaned of HTML before being returned as RawContent.
Results are cached in-memory (TTL configurable).
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from loguru import logger

from core.config_loader import Config
from core.models import RawContent
from ingestion.base import IngestorPort


class WebIngestor(IngestorPort):
    """Search the web and return cleaned text chunks."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache: dict[str, tuple[float, list[RawContent]]] = {}

    # ------------------------------------------------------------------
    # IngestorPort
    # ------------------------------------------------------------------

    async def ingest(self, source: str) -> list[RawContent]:
        """*source* is a search query string."""
        return await self.search(source)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str) -> list[RawContent]:
        """Search enabled sources, with optional caching."""
        inet = self.config.ingestion.internet

        if inet.cache_enabled:
            cached = self._get_cached(query)
            if cached is not None:
                return cached

        results = await self._do_search(query)

        if inet.cache_enabled:
            self._put_cache(query, results)

        return results

    # ------------------------------------------------------------------
    # Search dispatchers
    # ------------------------------------------------------------------

    async def _do_search(self, query: str) -> list[RawContent]:
        results: list[RawContent] = []
        for src in self.config.ingestion.internet.sources:
            if not src.enabled:
                continue
            try:
                if src.type == "web_search":
                    results.extend(await self._search_ddg(query))
                elif src.type == "wikipedia":
                    langs = src.languages or ["en"]
                    results.extend(await self._search_wikipedia(query, langs))
            except Exception as exc:
                logger.warning(f"Web source '{src.type}' failed: {exc}")
        return results

    # ------------------------------------------------------------------
    # DuckDuckGo
    # ------------------------------------------------------------------

    async def _search_ddg(self, query: str, max_results: int = 5) -> list[RawContent]:
        def _run():
            from ddgs import DDGS

            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await asyncio.to_thread(_run)
        return [
            RawContent(
                text=_clean_html(r.get("body", "")),
                source=f"web:{r.get('href', '')}",
                source_type="web_search",
            )
            for r in raw
            if r.get("body")
        ]

    # ------------------------------------------------------------------
    # Wikipedia
    # ------------------------------------------------------------------

    async def _search_wikipedia(
        self, query: str, languages: list[str]
    ) -> list[RawContent]:
        import httpx

        results: list[RawContent] = []
        for lang in languages:
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
            }
            try:
                headers = {"User-Agent": "NCN/0.1 (Nodal Consensus Network; research project)"}
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, params=params, headers=headers, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                for item in data.get("query", {}).get("search", []):
                    snippet = _clean_html(item.get("snippet", ""))
                    title = item.get("title", "")
                    if snippet:
                        results.append(
                            RawContent(
                                text=f"{title}: {snippet}",
                                source=f"wikipedia:{lang}:{title}",
                                source_type="wikipedia",
                            )
                        )
            except Exception as exc:
                logger.warning(f"Wikipedia ({lang}) search failed: {exc}")
        return results

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cached(self, query: str) -> list[RawContent] | None:
        entry = self._cache.get(self._cache_key(query))
        if entry is None:
            return None
        ts, results = entry
        ttl = self.config.ingestion.internet.cache_ttl_minutes * 60
        if time.time() - ts > ttl:
            return None
        return results

    def _put_cache(self, query: str, results: list[RawContent]) -> None:
        self._cache[self._cache_key(query)] = (time.time(), results)


# ------------------------------------------------------------------
# HTML cleaning
# ------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    """Strip HTML tags, scripts, nav — keep only readable text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)
