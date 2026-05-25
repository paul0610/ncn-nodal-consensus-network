"""Sprint 8 checkpoint — Orchestrator end-to-end tests.

CHECKPOINT: query → graph → swarm → consensus → response (full pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config_loader import (
    Config,
    ConsensusConfig,
    GraphConfig,
    IngestionConfig,
    InternetConfig,
    InternetSourceConfig,
    KuzuConfig,
    ProviderConfig,
    ReputationConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.models import (
    ClaimStatus,
    GraphContext,
    GraphNode,
    NodeRole,
    QueryResponse,
    RawContent,
)
from core.orchestrator import Orchestrator
from graph.kuzu_adapter import KuzuAdapter
from providers.ollama import OllamaProvider
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Python", "predicate": "created_by", "object": "Guido",
     "confidence": 0.95, "source_text": "Python was created by Guido."},
    {"subject": "Python", "predicate": "is_a", "object": "language",
     "confidence": 0.92, "source_text": "Python is a language."},
]})
CRITIC_OK = json.dumps({"vote": True, "confidence": 0.88, "reason": "supported"})
SYNTH_TEXT = "Python is a programming language created by Guido van Rossum."


def _cfg(tmp_path: Path) -> Config:
    return Config(
        graph=GraphConfig(
            adapter="kuzu",
            kuzu=KuzuConfig(path=str(tmp_path / "orch_graph")),
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
        consensus=ConsensusConfig(
            validators_per_claim=2,
            tenth_man_enabled=False,
        ),
        reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
        ),
        ingestion=IngestionConfig(
            internet=InternetConfig(search_mode="never"),
        ),
    )


def _mock_swarm(pool: SwarmPool) -> None:
    """Replace every provider in *pool* with a role-appropriate mock."""
    for node in pool.nodes:
        node.provider = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
        if node.role == NodeRole.EXTRACTOR:
            node.provider._post = AsyncMock(
                return_value={"response": CLAIMS_JSON, "eval_count": 1}
            )
        elif node.role == NodeRole.CRITIC:
            node.provider._post = AsyncMock(
                return_value={"response": CRITIC_OK, "eval_count": 1}
            )
        elif node.role == NodeRole.SYNTHESIZER:
            node.provider._post = AsyncMock(
                return_value={"response": SYNTH_TEXT, "eval_count": 1}
            )
        else:
            node.provider._post = AsyncMock(
                return_value={"response": "{}", "eval_count": 1}
            )


class MockEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 10

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class TestLifecycle:
    async def test_start_stop(self, tmp_path):
        orch = Orchestrator(_cfg(tmp_path), embedder=MockEmbedder())
        await orch.start()
        assert orch._started
        assert orch.graph_reader is not None
        assert orch.graph_writer is not None
        assert orch.retriever is not None
        await orch.stop()
        assert not orch._started

    async def test_not_started_raises(self, tmp_path):
        orch = Orchestrator(_cfg(tmp_path), embedder=MockEmbedder())
        with pytest.raises(RuntimeError, match="not started"):
            await orch.process_query("hello")

    async def test_double_start_is_safe(self, tmp_path):
        orch = Orchestrator(_cfg(tmp_path), embedder=MockEmbedder())
        await orch.start()
        await orch.start()  # should not raise
        await orch.stop()


# ------------------------------------------------------------------
# Web-search decision logic
# ------------------------------------------------------------------

class TestWebSearchDecision:
    def test_never(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(search_mode="never"))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(avg_confidence=0.0, nodes=[])
        assert orch._should_search_web("anything", ctx) is False

    def test_always(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(search_mode="always"))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(avg_confidence=1.0, nodes=[GraphNode(name="x", node_type="t")] * 5)
        assert orch._should_search_web("hello", ctx) is True

    def test_auto_low_confidence(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(
                search_mode="auto", graph_confidence_threshold=0.60
            ))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(avg_confidence=0.3, nodes=[GraphNode(name="a", node_type="t")] * 5)
        assert orch._should_search_web("safe query", ctx) is True

    def test_auto_low_coverage(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(search_mode="auto"))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(avg_confidence=0.9, nodes=[])  # 0 nodes < 3
        assert orch._should_search_web("safe query", ctx) is True

    def test_auto_temporal_keywords(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(search_mode="auto"))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(avg_confidence=0.9, nodes=[GraphNode(name="a", node_type="t")] * 5)
        assert orch._should_search_web("what happened today?", ctx) is True

    def test_auto_no_trigger(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(
                search_mode="auto", graph_confidence_threshold=0.60
            ))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(
            avg_confidence=0.9,
            nodes=[GraphNode(name=f"n{i}", node_type="t") for i in range(5)],
            max_relevance=0.9,  # high relevance → no web search trigger
        )
        assert orch._should_search_web("what is Python?", ctx) is False

    def test_auto_low_relevance(self, tmp_path):
        """Relevance Gate: low max_relevance should trigger web search."""
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "ingestion": IngestionConfig(internet=InternetConfig(
                search_mode="auto", graph_confidence_threshold=0.60
            ))
        })
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        ctx = GraphContext(
            avg_confidence=0.9,
            nodes=[GraphNode(name=f"n{i}", node_type="t") for i in range(5)],
            max_relevance=0.1,  # below relevance_threshold (0.35)
        )
        assert orch._should_search_web("what is Python?", ctx) is True


# ------------------------------------------------------------------
# Merge contexts
# ------------------------------------------------------------------

class TestMergeContexts:
    def test_merge(self):
        g = GraphContext(serialized_context="• A → rel → B", nodes=[], nodes_used=["id1"])
        web = [
            RawContent(text="Web fact 1", source="web:x", source_type="web"),
            RawContent(text="Web fact 2", source="web:y", source_type="web"),
        ]
        merged = Orchestrator._merge_contexts(g, web)
        assert "A → rel → B" in merged.serialized_context
        assert "[web] Web fact 1" in merged.serialized_context
        assert merged.nodes_used == ["id1"]


# ------------------------------------------------------------------
# process_query — full E2E
# ------------------------------------------------------------------

class TestProcessQuery:
    async def test_query_e2e(self, tmp_path):
        """CHECKPOINT: query → graph → swarm → consensus → response."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()

        _mock_swarm(orch.swarm_pool)

        # Pre-populate the graph so the Relevance Gate has nodes to return
        await orch.process_ingest(
            "Python was created by Guido van Rossum.", source_type="text"
        )

        resp = await orch.process_query("What is Python?")

        assert isinstance(resp, QueryResponse)
        assert len(resp.answer) > 0
        assert len(resp.claims) > 0
        assert all(c.status == ClaimStatus.VERIFIED for c in resp.claims)
        assert len(resp.nodes_created) > 0

        # Verify graph was updated
        stats = await orch.graph.get_stats()
        assert stats.total_nodes > 0
        assert stats.total_relations > 0

        python_node = await orch.graph.find_node_by_name("Python")
        assert python_node is not None

        await orch.stop()

    async def test_no_context_admits_ignorance(self, tmp_path):
        """Relevance Gate: empty graph + no web → admit ignorance instead of hallucinating."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        # No pre-population → empty graph
        resp = await orch.process_query("What is something obscure?")

        assert isinstance(resp, QueryResponse)
        assert resp.claims == []
        assert "no tengo informacion verificada" in resp.answer.lower()

        await orch.stop()

    async def test_two_queries_reinforce(self, tmp_path):
        """Second query about same topic should reinforce existing nodes."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        # Pre-populate with Python-related data
        await orch.process_ingest(
            "Python was created by Guido van Rossum.", source_type="text"
        )

        await orch.process_query("What is Python?")
        py_before = await orch.graph.find_node_by_name("Python")
        conf_before = py_before.confidence

        await orch.process_query("Tell me about Python")
        py_after = await orch.graph.find_node_by_name("Python")

        assert py_after.confidence >= conf_before

        await orch.stop()


# ------------------------------------------------------------------
# Namespace isolation — domain-scoped ingestion and retrieval
# ------------------------------------------------------------------

class TestNamespaceIsolation:
    """Verify that namespace-scoped ingestion and retrieval stay isolated."""

    async def test_ingest_writes_to_specified_namespace(self, tmp_path):
        """Nodes created by ingest should land in the given namespace."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        await orch.process_ingest(
            "Python is a programming language.",
            source_type="text",
            namespace="tech",
        )

        tech_nodes = await orch.graph.find_nodes_by_namespace("tech", limit=100)
        assert len(tech_nodes) > 0
        assert all(n.namespace == "tech" for n in tech_nodes)

        await orch.stop()

    async def test_two_namespaces_stay_isolated(self, tmp_path):
        """Ingest in namespace A, query in namespace B → no A content leaks."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        # Populate two separate namespaces
        await orch.process_ingest(
            "Python is a language.", source_type="text", namespace="tech",
        )
        await orch.process_ingest(
            "Salt is a mineral.", source_type="text", namespace="food",
        )

        # Query scoped to "tech" → should only see tech nodes
        resp = await orch.process_chat("what is Python?", namespace="tech")
        tech_node_ids = resp.nodes_used

        food_nodes = await orch.graph.find_nodes_by_namespace("food", limit=100)
        food_ids = {n.node_id for n in food_nodes}

        # No food-namespace node should appear in tech query results
        assert not set(tech_node_ids) & food_ids

        await orch.stop()

    async def test_same_entity_in_two_namespaces_gets_duplicated(self, tmp_path):
        """Same entity name in different namespaces -> two distinct nodes.

        Critical for domain isolation: ``water`` in survival must not be
        the same node as ``water`` in cooking, or the first-ingested
        namespace effectively owns it forever.
        """
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        await orch.process_ingest(
            "Python is a language.", source_type="text", namespace="tech",
        )
        await orch.process_ingest(
            "Python is a language.", source_type="text", namespace="biology",
        )

        tech_python = await orch.graph.find_node_by_name_in_namespace(
            "Python", "tech"
        )
        bio_python = await orch.graph.find_node_by_name_in_namespace(
            "Python", "biology"
        )

        assert tech_python is not None, "Python missing in 'tech' namespace"
        assert bio_python is not None, "Python missing in 'biology' namespace"
        assert tech_python.node_id != bio_python.node_id, (
            "Both namespaces returned the same node — isolation failed"
        )

        await orch.stop()

    async def test_query_without_namespace_sees_everything(self, tmp_path):
        """When no namespace is specified, retrieval is global (backward compat)."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        await orch.process_ingest(
            "Python is great.", source_type="text", namespace="tech",
        )
        await orch.process_ingest(
            "Salt is savoury.", source_type="text", namespace="food",
        )

        # No namespace arg → can retrieve from both
        resp = await orch.process_chat("tell me something")
        # MockEmbedder returns same vector for all, so both namespaces pass the gate
        assert isinstance(resp, QueryResponse)

        await orch.stop()


# ------------------------------------------------------------------
# process_chat — lightweight single-model mode
# ------------------------------------------------------------------

class TestProcessChat:
    async def test_chat_returns_answer(self, tmp_path):
        """Chat mode returns an answer from a single model."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        resp = await orch.process_chat("como estas?")
        assert isinstance(resp, QueryResponse)
        assert len(resp.answer) > 0

        await orch.stop()

    async def test_chat_uses_graph_context(self, tmp_path):
        """Chat mode retrieves from the graph and passes context to the model."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        # Pre-populate graph
        await orch.process_ingest("Python was created by Guido.", source_type="text")

        resp = await orch.process_chat("que es Python?")
        assert isinstance(resp, QueryResponse)
        assert len(resp.answer) > 0
        # Chat uses graph nodes but creates none
        assert len(resp.nodes_used) > 0

        await orch.stop()

    async def test_chat_no_graph_writes(self, tmp_path):
        """Chat mode does NOT create new nodes in the graph."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        stats_before = await orch.graph.get_stats()

        await orch.process_chat("hablame de algo interesante")

        stats_after = await orch.graph.get_stats()
        assert stats_after.total_nodes == stats_before.total_nodes

        await orch.stop()

    async def test_chat_no_claims(self, tmp_path):
        """Chat mode returns empty claims (no consensus verification)."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        resp = await orch.process_chat("que es la inteligencia artificial?")
        assert resp.claims == []
        assert resp.nodes_created == []

        await orch.stop()

    async def test_chat_greeting_blocked(self, tmp_path):
        """Greetings are still caught by the Query Router in chat mode."""
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()

        resp = await orch.process_chat("Holi")
        assert "Hola" in resp.answer or "hola" in resp.answer.lower()
        assert resp.claims == []

        await orch.stop()


# ------------------------------------------------------------------
# process_ingest
# ------------------------------------------------------------------

class TestProcessIngest:
    async def test_ingest_text(self, tmp_path):
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        summary = await orch.process_ingest(
            "Python was created by Guido van Rossum in 1991.\n\n"
            "It is widely used for AI and web development.",
            source_type="text",
        )

        assert summary["chunks"] == 2
        assert summary["claims_verified"] > 0
        assert summary["nodes_created"] > 0

        stats = await orch.graph.get_stats()
        assert stats.total_nodes > 0

        await orch.stop()

    async def test_ingest_pdf(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1}: Python is great.", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        summary = await orch.process_ingest(str(pdf_path), source_type="pdf")

        assert summary["chunks"] == 3
        assert summary["claims_verified"] > 0

        await orch.stop()


# ------------------------------------------------------------------
# CHECKPOINT: full end-to-end
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_full_end_to_end(self, tmp_path):
        """
        Complete system flow:
        1. Ingest text → claims verified → nodes in graph
        2. Query about ingested topic → retrieves from graph → consensus → answer
        """
        cfg = _cfg(tmp_path)
        orch = Orchestrator(cfg, embedder=MockEmbedder())
        await orch.start()
        _mock_swarm(orch.swarm_pool)

        # --- INGEST ---
        ingest_result = await orch.process_ingest(
            "Albert Einstein developed the theory of relativity in 1905. "
            "He was born in Ulm, Germany.",
            source_type="text",
        )
        assert ingest_result["claims_verified"] > 0

        # --- QUERY ---
        resp = await orch.process_query(
            "¿Quién desarrolló la relatividad?"
        )

        assert isinstance(resp, QueryResponse)
        assert resp.answer != ""
        assert len(resp.claims) > 0

        # Graph should have grown
        stats = await orch.graph.get_stats()
        assert stats.total_nodes > 0
        assert stats.total_relations > 0

        await orch.stop()
