"""Sprint 6 checkpoint — ingestion tests.

CHECKPOINT: ingest a 10-page PDF → claims in the graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.config_loader import (
    Config,
    ConsensusConfig,
    GraphConfig,
    IngestionConfig,
    InternetConfig,
    InternetSourceConfig,
    KuzuConfig,
    PDFConfig,
    ProviderConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.exceptions import NCNError
from core.models import ClaimStatus, RawContent
from ingestion.factory import create_ingestor
from ingestion.pdf_ingestor import PDFIngestor
from ingestion.text_ingestor import TextIngestor
from ingestion.web_ingestor import WebIngestor, _clean_html


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _config(**overrides) -> Config:
    return Config(**overrides)


def _make_pdf(path: Path, pages: int = 10) -> Path:
    """Create a real multi-page PDF using PyMuPDF."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        text = (
            f"Page {i + 1} of the test document.\n\n"
            f"Python was created by Guido van Rossum in 1991.\n"
            f"It is a high-level programming language used for AI, web development, "
            f"and scientific computing.\n"
            f"Section {i + 1}: additional factual content for testing ingestion."
        )
        page.insert_text((72, 72), text, fontsize=12)
    pdf_path = path / "test_document.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ------------------------------------------------------------------
# TextIngestor
# ------------------------------------------------------------------

class TestTextIngestor:
    async def test_single_paragraph(self):
        ing = TextIngestor(_config())
        results = await ing.ingest("Python is a programming language.")
        assert len(results) == 1
        assert results[0].text == "Python is a programming language."
        assert results[0].source_type == "text"

    async def test_multiple_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        ing = TextIngestor(_config())
        results = await ing.ingest(text)
        assert len(results) == 3
        assert results[1].text == "Second paragraph."

    async def test_empty_text(self):
        ing = TextIngestor(_config())
        assert await ing.ingest("") == []
        assert await ing.ingest("   ") == []

    async def test_source_metadata(self):
        ing = TextIngestor(_config())
        results = await ing.ingest("hello")
        assert results[0].source == "user:text"


# ------------------------------------------------------------------
# PDFIngestor
# ------------------------------------------------------------------

class TestPDFIngestor:
    async def test_ingest_pdf(self, tmp_path):
        pdf_path = _make_pdf(tmp_path, pages=3)
        ing = PDFIngestor(_config())
        results = await ing.ingest(str(pdf_path))

        assert len(results) == 3
        assert "Page 1" in results[0].text
        assert results[0].source_type == "pdf"
        assert "page1" in results[0].source

    async def test_ingest_10_page_pdf(self, tmp_path):
        pdf_path = _make_pdf(tmp_path, pages=10)
        ing = PDFIngestor(_config())
        results = await ing.ingest(str(pdf_path))

        assert len(results) == 10
        for i, r in enumerate(results):
            assert f"Page {i + 1}" in r.text
            assert r.source_type == "pdf"

    async def test_pdf_not_found(self):
        ing = PDFIngestor(_config())
        with pytest.raises(NCNError, match="not found"):
            await ing.ingest("/nonexistent/path.pdf")

    async def test_pdf_too_large(self, tmp_path):
        pdf_path = _make_pdf(tmp_path, pages=1)
        cfg = _config(ingestion=IngestionConfig(pdf=PDFConfig(max_size_mb=0)))
        ing = PDFIngestor(cfg)
        with pytest.raises(NCNError, match="too large"):
            await ing.ingest(str(pdf_path))


# ------------------------------------------------------------------
# WebIngestor
# ------------------------------------------------------------------

class TestWebIngestor:
    async def test_search_with_mock_ddg(self):
        cfg = _config(ingestion=IngestionConfig(
            internet=InternetConfig(
                sources=[InternetSourceConfig(type="web_search", enabled=True)],
                cache_enabled=False,
            )
        ))
        ing = WebIngestor(cfg)

        mock_results = [
            {"body": "Python is great.", "href": "https://example.com/1"},
            {"body": "AI is the future.", "href": "https://example.com/2"},
        ]
        with patch.object(ing, "_search_ddg", new_callable=AsyncMock, return_value=[
            RawContent(text="Python is great.", source="web:https://example.com/1", source_type="web_search"),
            RawContent(text="AI is the future.", source="web:https://example.com/2", source_type="web_search"),
        ]):
            results = await ing.search("python programming")

        assert len(results) == 2
        assert results[0].source_type == "web_search"

    async def test_search_with_mock_wikipedia(self):
        cfg = _config(ingestion=IngestionConfig(
            internet=InternetConfig(
                sources=[InternetSourceConfig(type="wikipedia", enabled=True, languages=["en"])],
                cache_enabled=False,
            )
        ))
        ing = WebIngestor(cfg)

        with patch.object(ing, "_search_wikipedia", new_callable=AsyncMock, return_value=[
            RawContent(text="Python: A language.", source="wikipedia:en:Python", source_type="wikipedia"),
        ]):
            results = await ing.search("python")

        assert len(results) == 1
        assert results[0].source_type == "wikipedia"

    async def test_cache_hit(self):
        cfg = _config(ingestion=IngestionConfig(
            internet=InternetConfig(
                sources=[InternetSourceConfig(type="web_search", enabled=True)],
                cache_enabled=True,
                cache_ttl_minutes=60,
            )
        ))
        ing = WebIngestor(cfg)

        call_count = 0
        original = [RawContent(text="cached result", source="web:x", source_type="web_search")]

        async def mock_search(q):
            nonlocal call_count
            call_count += 1
            return original

        with patch.object(ing, "_do_search", side_effect=mock_search):
            r1 = await ing.search("test query")
            r2 = await ing.search("test query")

        assert call_count == 1  # second call used cache
        assert r1 == r2

    async def test_disabled_source_skipped(self):
        cfg = _config(ingestion=IngestionConfig(
            internet=InternetConfig(
                sources=[InternetSourceConfig(type="web_search", enabled=False)],
                cache_enabled=False,
            )
        ))
        ing = WebIngestor(cfg)
        results = await ing.search("anything")
        assert results == []


# ------------------------------------------------------------------
# HTML cleaning
# ------------------------------------------------------------------

class TestHTMLCleaning:
    def test_strips_tags(self):
        assert _clean_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_removes_scripts(self):
        html = '<p>Text</p><script>alert(1)</script>'
        assert "alert" not in _clean_html(html)

    def test_removes_nav(self):
        html = '<nav>Menu</nav><main>Content</main>'
        cleaned = _clean_html(html)
        assert "Menu" not in cleaned
        assert "Content" in cleaned


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

class TestFactory:
    def test_create_text(self):
        ing = create_ingestor("text", _config())
        assert isinstance(ing, TextIngestor)

    def test_create_pdf(self):
        ing = create_ingestor("pdf", _config())
        assert isinstance(ing, PDFIngestor)

    def test_create_web(self):
        ing = create_ingestor("web", _config())
        assert isinstance(ing, WebIngestor)

    def test_unknown_raises(self):
        with pytest.raises(Exception, match="not recognised"):
            create_ingestor("unknown", _config())


# ------------------------------------------------------------------
# CHECKPOINT: 10-page PDF → claims in the graph
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_pdf_to_claims_in_graph(self, tmp_path):
        """Full checkpoint: ingest a 10-page PDF, run consensus, write to graph."""
        from consensus.engine import ConsensusEngine
        from graph.kuzu_adapter import KuzuAdapter
        from graph.writer import SingleWriter
        from providers.ollama import OllamaProvider
        from swarm.pool import SwarmPool

        # --- config ---
        cfg = Config(
            graph=GraphConfig(
                adapter="kuzu",
                kuzu=KuzuConfig(path=str(tmp_path / "ckpt_graph")),
            ),
            swarm=SwarmConfig(
                max_concurrent=10,
                request_timeout=30,
                nodes=[
                    SwarmNodeConfig(provider="ollama", model="m", count=2, role="extractor"),
                    SwarmNodeConfig(provider="ollama", model="m", count=2, role="critic"),
                    SwarmNodeConfig(provider="ollama", model="m", count=1, role="synthesizer"),
                ],
            ),
            consensus=ConsensusConfig(validators_per_claim=2),
        )

        # --- graph ---
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # --- swarm (mocked) ---
        pool = SwarmPool(cfg)

        claims_json = json.dumps({"claims": [
            {"subject": "Python", "predicate": "created_by", "object": "Guido van Rossum",
             "confidence": 0.95, "source_text": "Python was created by Guido van Rossum."},
            {"subject": "Python", "predicate": "is_a", "object": "programming language",
             "confidence": 0.92, "source_text": "high-level programming language"},
        ]})
        critic_json = json.dumps({"vote": True, "confidence": 0.88, "reason": "supported by text"})
        synth_text = "Python is a programming language created by Guido van Rossum."

        for node in pool.nodes:
            from unittest.mock import AsyncMock as AM
            node.provider = OllamaProvider(
                ProviderConfig(base_url="http://localhost:11434")
            )
            if node.role.value == "extractor":
                node.provider._post = AM(return_value={"response": claims_json, "eval_count": 1})
            elif node.role.value == "critic":
                node.provider._post = AM(return_value={"response": critic_json, "eval_count": 1})
            else:
                node.provider._post = AM(return_value={"response": synth_text, "eval_count": 1})

        # --- ingest PDF ---
        pdf_path = _make_pdf(tmp_path, pages=10)
        ingestor = PDFIngestor(cfg)
        pages = await ingestor.ingest(str(pdf_path))
        assert len(pages) == 10

        # --- run consensus on first page ---
        from core.models import GraphContext

        engine = ConsensusEngine(cfg)
        context = GraphContext(serialized_context=pages[0].text)
        result = await engine.run(
            "¿Qué es Python?", context, pool
        )

        assert len(result.verified_claims) > 0

        # --- write verified claims to graph ---
        node_ids = await writer.write_verified_claims(result.verified_claims)
        assert len(node_ids) > 0

        # --- verify graph state ---
        stats = await adapter.get_stats()
        assert stats.total_nodes > 0
        assert stats.total_relations > 0

        python_node = await adapter.find_node_by_name("Python")
        assert python_node is not None

        guido_node = await adapter.find_node_by_name("Guido van Rossum")
        assert guido_node is not None

        await adapter.close()
