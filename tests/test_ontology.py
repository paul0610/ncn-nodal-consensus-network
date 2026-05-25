"""Sprint 7 checkpoint — ontology auto-extension tests.

CHECKPOINT: entity without a known type → new type proposed and validated.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core.config_loader import (
    Config,
    GraphConfig,
    KuzuConfig,
    OntologyConfig,
    ProviderConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.models import Claim, NodeRole
from graph.kuzu_adapter import KuzuAdapter
from ontology.agent import OntologyAgent, _extract_type_name
from ontology.classifier import _parse_classification, classify_with_existing
from ontology.deduplicator import SemanticDeduplicator, _normalise_type
from providers.ollama import OllamaProvider
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mock_provider(reply: str = "concepto") -> OllamaProvider:
    p = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
    p._post = AsyncMock(return_value={"response": reply, "eval_count": 1})
    return p


def _swarm_cfg(
    extractors: int = 2, critics: int = 3
) -> Config:
    return Config(
        swarm=SwarmConfig(
            max_concurrent=10,
            request_timeout=30,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=extractors, role="extractor"),
                SwarmNodeConfig(provider="ollama", model="m", count=critics, role="critic"),
            ],
        ),
        ontology=OntologyConfig(
            auto_extend=True,
            min_confidence=0.70,
            validators=3,
            max_custom_types=50,
            dedup_threshold=0.85,
        ),
    )


def _make_pool(cfg: Config, extractor_reply: str = "", critic_reply: str = "") -> SwarmPool:
    pool = SwarmPool(cfg)
    for node in pool.nodes:
        node.provider = _mock_provider()
        if node.role == NodeRole.EXTRACTOR and extractor_reply:
            node.provider._post = AsyncMock(
                return_value={"response": extractor_reply, "eval_count": 1}
            )
        elif node.role == NodeRole.CRITIC and critic_reply:
            node.provider._post = AsyncMock(
                return_value={"response": critic_reply, "eval_count": 1}
            )
    return pool


def _claim(subj: str = "X", pred: str = "is", obj: str = "Y") -> Claim:
    return Claim(subject=subj, predicate=pred, object=obj, confidence=0.8, source_model="e")


# ------------------------------------------------------------------
# Deduplicator — normalise helper
# ------------------------------------------------------------------

class TestDeduplicatorHelpers:
    def test_normalise(self):
        assert _normalise_type("Técnica_Biológica") == "técnica biológica"
        assert _normalise_type("algoritmo-computacional") == "algoritmo computacional"

    def test_exact_match(self):
        """Exact normalised match returns the known type."""
        from ontology.deduplicator import SemanticDeduplicator

        class FakeDedup(SemanticDeduplicator):
            def _load_model(self):
                pass
            async def _embed(self, text):
                import numpy as np
                return np.zeros(10)

        cfg = Config(ontology=OntologyConfig(dedup_threshold=0.85))
        d = FakeDedup(cfg)
        loop_result = None

        async def _run():
            return await d.deduplicate("persona", {"persona", "concepto"})

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "persona"


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class TestClassifier:
    def test_parse_classification_exact(self):
        text = '{"type": "persona", "confidence": 0.9}'
        t, c = _parse_classification(text, {"persona", "concepto"})
        assert t == "persona"
        assert c == pytest.approx(0.9)

    def test_parse_classification_unknown_type(self):
        text = '{"type": "alien", "confidence": 0.8}'
        t, c = _parse_classification(text, {"persona", "concepto"})
        assert t == "concepto"

    def test_parse_classification_bad_json(self):
        t, c = _parse_classification("not json", {"persona"})
        assert t == "concepto"
        assert c == 0.0

    async def test_classify_with_existing(self):
        cfg = _swarm_cfg(extractors=1)
        reply = json.dumps({"type": "persona", "confidence": 0.85})
        pool = _make_pool(cfg, extractor_reply=reply)

        t, c = await classify_with_existing(
            "Einstein", _claim("Einstein", "developed", "Relativity"),
            {"persona", "concepto", "evento"}, pool,
        )
        assert t == "persona"
        assert c > 0.5


# ------------------------------------------------------------------
# _extract_type_name
# ------------------------------------------------------------------

class TestExtractTypeName:
    def test_clean_type(self):
        assert _extract_type_name("tecnica_biologica") == "tecnica_biologica"

    def test_quoted(self):
        assert _extract_type_name('"algoritmo_computacional"') == "algoritmo_computacional"

    def test_noisy(self):
        assert _extract_type_name('The type is: `lenguaje_programacion`') == "lenguaje_programacion"

    def test_fallback(self):
        assert _extract_type_name("!!!") == "concepto"


# ------------------------------------------------------------------
# OntologyAgent — full flow (mocked)
# ------------------------------------------------------------------

class TestOntologyAgent:
    async def test_classify_existing_type(self, tmp_path):
        cfg = _swarm_cfg()
        cfg = cfg.model_copy(update={
            "graph": GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "g1")))
        })

        reply = json.dumps({"type": "persona", "confidence": 0.90})
        pool = _make_pool(cfg, extractor_reply=reply)

        adapter = KuzuAdapter(cfg)
        await adapter.initialize()

        agent = OntologyAgent(cfg, pool, adapter)
        result = await agent.classify("Einstein", _claim("Einstein", "born_in", "Ulm"))

        assert result == "persona"
        await adapter.close()

    async def test_propose_new_type(self, tmp_path):
        """CHECKPOINT: entity with no known type → new type proposed + validated."""
        cfg = _swarm_cfg(extractors=2, critics=3)
        cfg = cfg.model_copy(update={
            "graph": GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "g2")))
        })

        # Classifier returns low confidence → triggers proposal
        classify_reply = json.dumps({"type": "concepto", "confidence": 0.30})
        # Extractor proposes a new type
        proposal_reply = "lenguaje_programacion"
        # Critics approve
        approve_reply = json.dumps({"approve": True, "confidence": 0.85})

        pool = SwarmPool(cfg)
        call_count = {"extractor": 0}

        for node in pool.nodes:
            node.provider = _mock_provider()
            if node.role == NodeRole.EXTRACTOR:
                async def ext_side_effect(*a, **kw):
                    call_count["extractor"] += 1
                    # First call: classification → low conf
                    # Second call: proposal → new type name
                    if call_count["extractor"] <= 2:  # 2 extractors for classify
                        return type("R", (), {"content": classify_reply, "json": lambda s: json.loads(classify_reply)})()
                    return type("R", (), {"content": proposal_reply, "json": lambda s: {}})()

                # Simpler: just always return the proposal (classify will fail, agent falls through)
                node.provider._post = AsyncMock(
                    return_value={"response": proposal_reply, "eval_count": 1}
                )
            elif node.role == NodeRole.CRITIC:
                node.provider._post = AsyncMock(
                    return_value={"response": approve_reply, "eval_count": 1}
                )

        adapter = KuzuAdapter(cfg)
        await adapter.initialize()

        # Use a mock deduplicator to skip real embedding model
        agent = OntologyAgent(cfg, pool, adapter)

        class MockDedup:
            async def deduplicate(self, proposed, known):
                return proposed  # always treat as new
        agent.deduplicator = MockDedup()

        result = await agent.classify(
            "Python",
            _claim("Python", "is_a", "language"),
        )

        assert result == "lenguaje_programacion"
        assert "lenguaje_programacion" in agent._known_types

        # Verify persisted in graph
        registered = await adapter.get_registered_node_types()
        assert "lenguaje_programacion" in registered

        await adapter.close()

    async def test_max_custom_types_respected(self, tmp_path):
        cfg = Config(
            graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "g3"))),
            swarm=SwarmConfig(
                max_concurrent=5, request_timeout=30,
                nodes=[
                    SwarmNodeConfig(provider="ollama", model="m", count=1, role="extractor"),
                    SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic"),
                ],
            ),
            ontology=OntologyConfig(auto_extend=True, max_custom_types=0),
        )
        pool = _make_pool(cfg, extractor_reply="new_type")

        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        agent = OntologyAgent(cfg, pool, adapter)

        # With max_custom_types=0, should fallback to "concepto"
        result = await agent.classify("X", _claim())
        assert result == "concepto"
        await adapter.close()

    async def test_auto_extend_disabled(self, tmp_path):
        cfg = Config(
            graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "g4"))),
            swarm=SwarmConfig(
                max_concurrent=5, request_timeout=30,
                nodes=[
                    SwarmNodeConfig(provider="ollama", model="m", count=1, role="extractor"),
                ],
            ),
            ontology=OntologyConfig(auto_extend=False),
        )
        # Classifier returns low confidence
        reply = json.dumps({"type": "concepto", "confidence": 0.20})
        pool = _make_pool(cfg, extractor_reply=reply)

        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        agent = OntologyAgent(cfg, pool, adapter)

        result = await agent.classify("StrangeEntity", _claim())
        assert result == "concepto"
        await adapter.close()


# ------------------------------------------------------------------
# Graph integration — register_node_type
# ------------------------------------------------------------------

class TestGraphNodeTypeRegistration:
    async def test_register_and_get(self, tmp_path):
        cfg = Config(graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "g5"))))
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()

        await adapter.register_node_type("tecnica_biologica")
        await adapter.register_node_type("algoritmo_ml")
        # Duplicate should be idempotent
        await adapter.register_node_type("tecnica_biologica")

        types = await adapter.get_registered_node_types()
        assert "tecnica_biologica" in types
        assert "algoritmo_ml" in types
        assert len(types) == 2

        await adapter.close()


# ------------------------------------------------------------------
# CHECKPOINT: unknown entity → new type proposed and validated
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_unknown_entity_gets_new_type(self, tmp_path):
        """
        Full flow: 'CRISPR' has no matching type → agent proposes
        'tecnica_biologica' → critics approve → type is registered.
        """
        cfg = _swarm_cfg(extractors=1, critics=3)
        cfg = cfg.model_copy(update={
            "graph": GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "ckpt")))
        })

        pool = SwarmPool(cfg)
        for node in pool.nodes:
            node.provider = _mock_provider()
            if node.role == NodeRole.EXTRACTOR:
                node.provider._post = AsyncMock(
                    return_value={"response": "tecnica_biologica", "eval_count": 1}
                )
            elif node.role == NodeRole.CRITIC:
                node.provider._post = AsyncMock(
                    return_value={
                        "response": json.dumps({"approve": True, "confidence": 0.9}),
                        "eval_count": 1,
                    }
                )

        adapter = KuzuAdapter(cfg)
        await adapter.initialize()

        agent = OntologyAgent(cfg, pool, adapter)

        class MockDedup:
            async def deduplicate(self, proposed, known):
                return proposed

        agent.deduplicator = MockDedup()

        result = await agent.classify(
            "CRISPR",
            _claim("CRISPR", "edits", "genes"),
        )

        assert result == "tecnica_biologica"
        assert "tecnica_biologica" in agent._known_types

        types = await adapter.get_registered_node_types()
        assert "tecnica_biologica" in types

        await adapter.close()
