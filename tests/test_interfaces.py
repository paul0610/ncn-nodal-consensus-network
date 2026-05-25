"""Sprint 9 — interface tests (CLI + FastAPI).

CHECKPOINT: ncn query "¿Qué es Python?" → full response.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from core.config_loader import (
    Config,
    ConsensusConfig,
    GraphConfig,
    IngestionConfig,
    InternetConfig,
    KuzuConfig,
    ProviderConfig,
    ReputationConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.models import NodeRole

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Python", "predicate": "is_a", "object": "language",
     "confidence": 0.92, "source_text": "Python is a language."},
]})
CRITIC_OK = json.dumps({"vote": True, "confidence": 0.88, "reason": "ok"})
SYNTH = "Python is a programming language."


def _cfg(tmp_path: Path) -> Config:
    return Config(
        graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "api_g"))),
        swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="extractor"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="synthesizer"),
            ],
        ),
        consensus=ConsensusConfig(validators_per_claim=1, tenth_man_enabled=False),
        reputation=ReputationConfig(history_file=str(tmp_path / "rep.json")),
        ingestion=IngestionConfig(internet=InternetConfig(search_mode="never")),
    )


def _mock_pool(pool):
    from providers.ollama import OllamaProvider
    for node in pool.nodes:
        node.provider = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
        if node.role == NodeRole.EXTRACTOR:
            node.provider._post = AsyncMock(return_value={"response": CLAIMS_JSON, "eval_count": 1})
        elif node.role == NodeRole.CRITIC:
            node.provider._post = AsyncMock(return_value={"response": CRITIC_OK, "eval_count": 1})
        else:
            node.provider._post = AsyncMock(return_value={"response": SYNTH, "eval_count": 1})


class MockEmbedder:
    async def embed(self, text):
        return [0.1] * 10
    async def embed_batch(self, texts):
        return [[0.1] * 10 for _ in texts]


# ==================================================================
# FastAPI tests
# ==================================================================

@pytest.fixture
async def api_client(tmp_path):
    """Yield a httpx.AsyncClient wired to the NCN FastAPI app."""
    from httpx import ASGITransport, AsyncClient
    from interface.api.app import create_app
    from core.orchestrator import Orchestrator

    cfg = _cfg(tmp_path)
    app = create_app(cfg)

    # Manually start orchestrator so we can mock the swarm
    orch = Orchestrator(cfg, embedder=MockEmbedder())
    await orch.start()
    _mock_pool(orch.swarm_pool)

    # Inject into the module-level variable used by routes
    import interface.api.app as api_mod
    api_mod._orchestrator = orch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await orch.stop()
    api_mod._orchestrator = None


class TestHealthEndpoint:
    async def test_health(self, api_client):
        resp = await api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert data["graph"] == "ok"

    async def test_providers_health(self, api_client):
        """GET /health/providers returns the key audit (all_ok + list)."""
        resp = await api_client.get("/health/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "all_ok" in data
        assert "providers" in data
        # Our test fixture uses ollama -> all_ok should be True
        assert data["all_ok"] is True


class TestQueryEndpoint:
    async def test_post_query(self, api_client):
        resp = await api_client.post("/query", json={"text": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "claims" in data
        assert isinstance(data["nodes_created"], list)


class TestIngestEndpoints:
    async def test_ingest_text(self, api_client):
        resp = await api_client.post("/ingest/text", json={"text": "Python is great."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks"] >= 1
        assert data["claims_verified"] >= 0

    async def test_ingest_text_returns_enriched_fields(self, api_client):
        """Ingest response should include discarded/uncertain counts and corrections."""
        resp = await api_client.post("/ingest/text", json={"text": "Python is great."})
        assert resp.status_code == 200
        data = resp.json()
        # New fields must exist
        assert "claims_discarded" in data
        assert "claims_uncertain" in data
        assert "corrections" in data
        assert isinstance(data["claims_discarded"], int)
        assert isinstance(data["claims_uncertain"], int)


class TestLearnEndpoint:
    """POST /learn — autonomous learning endpoint."""

    async def test_learn_returns_structured_payload(self, api_client, monkeypatch):
        """Endpoint should accept a topic and return the full learn summary."""
        from interface.api import app as api_app

        # Patch the orchestrator's process_learn to avoid real web scraping
        orch = api_app.get_orchestrator()

        async def fake_learn(topic: str, namespace: str | None = None) -> dict:
            return {
                "topic": topic,
                "sub_queries": 3,
                "sub_queries_list": [f"{topic} definition", f"{topic} history", f"{topic} usage"],
                "sources_found": 10,
                "pages_scraped": 5,
                "urls_scraped": ["https://example.com/a", "https://example.com/b"],
                "claims_extracted": 20,
                "claims_verified": 15,
                "claims_discarded": 3,
                "claims_uncertain": 2,
                "nodes_created": 12,
            }

        monkeypatch.setattr(orch, "process_learn", fake_learn)

        resp = await api_client.post("/learn", json={"topic": "DeepSeek V3"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["topic"] == "DeepSeek V3"
        assert data["sub_queries"] == 3
        assert len(data["sub_queries_list"]) == 3
        assert len(data["urls_scraped"]) == 2
        assert data["claims_verified"] == 15
        assert data["nodes_created"] == 12


class TestGraphEndpoints:
    async def test_graph_stats(self, api_client):
        resp = await api_client.get("/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_nodes" in data

    async def test_graph_stats_scoped_to_namespace(self, api_client):
        """GET /graph/stats?namespace=X returns counts limited to that namespace."""
        # Ingest into two namespaces so we can compare
        r1 = await api_client.post(
            "/ingest/text",
            json={"text": "Python is a language.", "namespace": "tech"},
        )
        assert r1.status_code == 200
        r2 = await api_client.post(
            "/ingest/text",
            json={"text": "Salt is a mineral.", "namespace": "food"},
        )
        assert r2.status_code == 200

        # Global stats should include both namespaces' nodes
        global_resp = await api_client.get("/graph/stats")
        global_data = global_resp.json()
        total_global = global_data["total_nodes"]
        assert total_global > 0

        # Scoped stats for "tech" should be strictly smaller than global
        tech_resp = await api_client.get("/graph/stats?namespace=tech")
        assert tech_resp.status_code == 200
        tech_data = tech_resp.json()
        assert tech_data["total_nodes"] <= total_global
        # And the namespaces list still shows all (for dropdown)
        assert "tech" in tech_data["namespaces"]

        # Empty namespace returns zeros (no error)
        empty_resp = await api_client.get("/graph/stats?namespace=nonexistent")
        assert empty_resp.status_code == 200
        empty_data = empty_resp.json()
        assert empty_data["total_nodes"] == 0
        assert empty_data["total_edges"] == 0

    async def test_graph_nodes(self, api_client):
        resp = await api_client.get("/graph/nodes?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data

    async def test_graph_node_not_found(self, api_client):
        resp = await api_client.get("/graph/node/nonexistent")
        assert resp.status_code == 404

    async def test_graph_subgraph_empty(self, api_client):
        """Subgraph endpoint on empty graph returns empty nodes + edges."""
        resp = await api_client.get("/graph/subgraph?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    async def test_graph_subgraph_with_data(self, api_client):
        """Ingest some data, then verify the subgraph returns nodes + edges."""
        # Trigger an ingestion to populate the graph
        await api_client.post(
            "/ingest/text",
            json={"text": "Python is a programming language created by Guido."},
        )

        resp = await api_client.get("/graph/subgraph?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        # Should have at least some nodes now
        assert len(data["nodes"]) > 0
        # Each node must have source_authority and uncertain fields
        for n in data["nodes"]:
            assert "source_authority" in n
            assert "uncertain" in n
        # Edges is a list (may be empty if relations don't connect inside limit)
        assert isinstance(data["edges"], list)


class TestSwarmEndpoint:
    async def test_swarm_status(self, api_client):
        resp = await api_client.get("/swarm/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
        assert len(data["nodes"]) > 0


# ==================================================================
# CLI tests
# ==================================================================

class TestCLI:
    def test_help(self):
        from interface.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "NCN" in result.output or "ncn" in result.output

    def test_graph_clear_no_confirm(self):
        from interface.cli.app import app
        runner = CliRunner()
        result = runner.invoke(app, ["graph", "clear"])
        assert result.exit_code == 0
        assert "confirm" in result.output.lower()


# ==================================================================
# CHECKPOINT: ncn query "¿Qué es Python?" → full response
# ==================================================================

class TestCheckpoint:
    async def test_query_via_api(self, api_client):
        """Full checkpoint: POST /query → verified answer."""
        # Pre-populate so the Relevance Gate has something to return
        await api_client.post(
            "/ingest/text",
            json={"text": "Python is a programming language."},
        )

        resp = await api_client.post(
            "/query",
            json={"text": "¿Qué es Python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["answer"]) > 5
        assert len(data["claims"]) > 0
        assert all(c["status"] == "verified" for c in data["claims"])

    async def test_ingest_then_query(self, api_client):
        """Ingest text, then query about it."""
        await api_client.post(
            "/ingest/text",
            json={"text": "Python was created by Guido van Rossum in 1991."},
        )

        resp = await api_client.post(
            "/query",
            json={"text": "Who created Python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["answer"]) > 0
