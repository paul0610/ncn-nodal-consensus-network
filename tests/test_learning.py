"""Phase 3 — Continuous Learning tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from core.config_loader import (
    Config,
    ContinuousLearningConfig,
    GraphConfig,
    KuzuConfig,
    ProviderConfig,
    SwarmConfig,
    SwarmNodeConfig,
    ConsensusConfig,
)
from core.models import NodeRole, RawContent
from graph.kuzu_adapter import KuzuAdapter
from graph.writer import SingleWriter
from learning.decomposer import QueryDecomposer
from learning.pipeline import LearningPipeline, _extract_urls
from learning.scraper import PageScraper, _clean_html
from consensus.engine import ConsensusEngine
from ingestion.web_ingestor import WebIngestor
from providers.ollama import OllamaProvider
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Aries", "predicate": "es", "object": "signo de fuego",
     "confidence": 0.9, "source_text": "Aries es un signo de fuego."},
]})
CRITIC_OK = json.dumps({"vote": True, "confidence": 0.85, "reason": "ok"})


def _cfg(tmp_path) -> Config:
    return Config(
        graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "learn_g"))),
        swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="planner"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="extractor"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="synthesizer"),
            ],
        ),
        consensus=ConsensusConfig(validators_per_claim=1, tenth_man_enabled=False),
        continuous_learning=ContinuousLearningConfig(
            sub_queries_per_topic=3,
            max_pages_per_query=2,
            max_claims_per_session=50,
            rate_limit_seconds=0,  # no delay in tests
            scrape_full_pages=False,  # skip real scraping in tests
        ),
    )


def _mock_pool(pool):
    for n in pool.nodes:
        n.provider = OllamaProvider(ProviderConfig(base_url="http://x"))
        if n.role == NodeRole.PLANNER:
            n.provider._post = AsyncMock(return_value={"response": "{}", "eval_count": 1})
        elif n.role == NodeRole.EXTRACTOR:
            n.provider._post = AsyncMock(return_value={"response": CLAIMS_JSON, "eval_count": 1})
        elif n.role == NodeRole.CRITIC:
            n.provider._post = AsyncMock(return_value={"response": CRITIC_OK, "eval_count": 1})
        else:
            n.provider._post = AsyncMock(return_value={"response": "Summary.", "eval_count": 1})


# ------------------------------------------------------------------
# QueryDecomposer
# ------------------------------------------------------------------

class TestQueryDecomposer:
    async def test_decompose_with_llm(self, tmp_path):
        cfg = _cfg(tmp_path)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        queries_json = json.dumps({"queries": [
            "signos del zodiaco caracteristicas",
            "elementos astrologia fuego tierra",
            "casas astrologicas significado",
        ]})
        for n in pool.nodes:
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(
                    return_value={"response": queries_json, "eval_count": 1}
                )

        decomposer = QueryDecomposer(cfg, pool)
        queries = await decomposer.decompose("astrologia")

        assert len(queries) == 3
        assert "signos" in queries[0].lower()

    async def test_fallback_on_bad_response(self, tmp_path):
        cfg = _cfg(tmp_path)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        # Planner returns garbage
        for n in pool.nodes:
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(
                    return_value={"response": "not json at all", "eval_count": 1}
                )

        decomposer = QueryDecomposer(cfg, pool)
        queries = await decomposer.decompose("astrologia")

        # Should fall back to generated queries
        assert len(queries) == 3
        assert any("astrologia" in q for q in queries)

    def test_fallback_queries(self):
        queries = QueryDecomposer._fallback_queries("astrologia", 3)
        assert len(queries) == 3
        assert all("astrologia" in q for q in queries)


# ------------------------------------------------------------------
# PageScraper
# ------------------------------------------------------------------

class TestPageScraper:
    def test_clean_html(self):
        html = "<html><nav>Menu</nav><p>Content here.</p><script>x()</script></html>"
        cleaned = _clean_html(html)
        assert "Content here" in cleaned
        assert "Menu" not in cleaned
        assert "x()" not in cleaned

    def test_chunk_text(self, tmp_path):
        cfg = _cfg(tmp_path)
        scraper = PageScraper(cfg)
        scraper._chunk_size = 50

        long_text = "First paragraph about topic A.\n\nSecond paragraph about topic B.\n\nThird paragraph about C."
        chunks = scraper._chunk_html_text(long_text, "http://example.com")

        assert len(chunks) >= 2
        assert all(c.source_type == "web_page" for c in chunks)

    def test_short_text_single_chunk(self, tmp_path):
        cfg = _cfg(tmp_path)
        scraper = PageScraper(cfg)

        chunks = scraper._chunk_html_text("Short text.", "http://x.com")
        assert len(chunks) == 1


# ------------------------------------------------------------------
# URL extraction
# ------------------------------------------------------------------

class TestExtractUrls:
    def test_extract_web_urls(self):
        snippets = [
            RawContent(text="text", source="web:https://example.com/page1", source_type="web_search"),
            RawContent(text="text", source="web:https://example.com/page2", source_type="web_search"),
            RawContent(text="text", source="wikipedia:en:Topic", source_type="wikipedia"),
        ]
        urls = _extract_urls(snippets, max_urls=5)
        assert len(urls) == 2
        assert urls[0] == "https://example.com/page1"

    def test_max_urls_limit(self):
        snippets = [
            RawContent(text="t", source=f"web:https://example.com/{i}", source_type="web_search")
            for i in range(10)
        ]
        urls = _extract_urls(snippets, max_urls=3)
        assert len(urls) == 3


# ------------------------------------------------------------------
# LearningPipeline (mocked web)
# ------------------------------------------------------------------

class TestLearningPipeline:
    async def test_full_pipeline(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        # Mock decomposer response
        queries_json = json.dumps({"queries": ["aries caracteristicas", "tauro caracteristicas"]})
        for n in pool.nodes:
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(
                    return_value={"response": queries_json, "eval_count": 1}
                )

        engine = ConsensusEngine(cfg)
        web_ingestor = WebIngestor(cfg)

        # Mock web search to return fake snippets
        mock_snippets = [
            RawContent(text="Aries es un signo de fuego.", source="web:https://astro.com/aries", source_type="web_search"),
        ]

        pipeline = LearningPipeline(cfg, pool, engine, writer, web_ingestor)

        with patch.object(pipeline.web_ingestor, "search", new_callable=AsyncMock, return_value=mock_snippets):
            result = await pipeline.learn("astrologia")

        assert result["topic"] == "astrologia"
        assert result["sub_queries"] >= 2
        assert result["claims_verified"] >= 0
        assert result["nodes_created"] >= 0

        await adapter.close()

    async def test_max_claims_limit(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "continuous_learning": ContinuousLearningConfig(
                max_claims_per_session=1,
                sub_queries_per_topic=3,
                scrape_full_pages=False,
                rate_limit_seconds=0,
            ),
        })
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        queries_json = json.dumps({"queries": ["q1", "q2", "q3"]})
        for n in pool.nodes:
            if n.role == NodeRole.PLANNER:
                n.provider._post = AsyncMock(
                    return_value={"response": queries_json, "eval_count": 1}
                )

        engine = ConsensusEngine(cfg)
        web = WebIngestor(cfg)

        mock_snippets = [RawContent(text="Fact.", source="web:http://x", source_type="web_search")]
        pipeline = LearningPipeline(cfg, pool, engine, writer, web)

        with patch.object(pipeline.web_ingestor, "search", new_callable=AsyncMock, return_value=mock_snippets):
            result = await pipeline.learn("topic")

        # Should stop early due to max_claims_per_session=1
        assert result["claims_verified"] <= 1

        await adapter.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

class TestCLI:
    def test_learn_help(self):
        from interface.cli.app import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["learn", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output.lower()
