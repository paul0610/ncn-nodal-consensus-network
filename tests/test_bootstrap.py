"""Sprint 10 — bootstrap teacher-student tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from bootstrap.pipeline import BootstrapPipeline
from bootstrap.teacher import QAPair, TeacherModel, _parse_qa
from core.config_loader import (
    Config,
    ConsensusConfig,
    GraphConfig,
    KnowledgeBootstrapConfig,
    KuzuConfig,
    ProviderConfig,
    ReputationConfig,
    SwarmConfig,
    SwarmNodeConfig,
    TeacherConfig,
)
from core.exceptions import TermsNotAcceptedError
from core.models import BootstrapResult, NodeRole
from consensus.engine import ConsensusEngine
from graph.kuzu_adapter import KuzuAdapter
from graph.writer import SingleWriter
from providers.ollama import OllamaProvider
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Python", "predicate": "is_a", "object": "language",
     "confidence": 0.9, "source_text": "Python is a language."},
]})
CRITIC_OK = json.dumps({"vote": True, "confidence": 0.85, "reason": "ok"})
SYNTH = "Python is a language."

QA_JSON = json.dumps({"qa_pairs": [
    {"question": "What is Python?", "answer": "Python is a programming language."},
    {"question": "Who created Python?", "answer": "Guido van Rossum."},
]})


def _cfg(tmp_path, terms: bool = True) -> Config:
    return Config(
        graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "bs_g"))),
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
        knowledge_bootstrap=KnowledgeBootstrapConfig(
            enabled=True,
            terms_accepted=terms,
            teacher=TeacherConfig(provider="ollama", model="m"),
            topics=["Python"],
            questions_per_topic=2,
        ),
    )


def _mock_pool(pool):
    for node in pool.nodes:
        node.provider = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
        if node.role == NodeRole.EXTRACTOR:
            node.provider._post = AsyncMock(return_value={"response": CLAIMS_JSON, "eval_count": 1})
        elif node.role == NodeRole.CRITIC:
            node.provider._post = AsyncMock(return_value={"response": CRITIC_OK, "eval_count": 1})
        else:
            node.provider._post = AsyncMock(return_value={"response": SYNTH, "eval_count": 1})


# ------------------------------------------------------------------
# Teacher parsing
# ------------------------------------------------------------------

class TestTeacherParsing:
    def test_parse_qa(self):
        pairs = _parse_qa(QA_JSON)
        assert len(pairs) == 2
        assert pairs[0].question == "What is Python?"

    def test_parse_qa_bad_json(self):
        assert _parse_qa("not json") == []

    def test_parse_qa_fenced(self):
        text = f"```json\n{QA_JSON}\n```"
        assert len(_parse_qa(text)) == 2


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class TestBootstrapPipeline:
    async def test_terms_not_accepted(self, tmp_path):
        cfg = _cfg(tmp_path, terms=False)
        pool = SwarmPool(cfg)
        _mock_pool(pool)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        engine = ConsensusEngine(cfg)

        pipeline = BootstrapPipeline(cfg, engine, writer, pool)
        with pytest.raises(TermsNotAcceptedError):
            await pipeline.run()
        await adapter.close()

    async def test_no_topics(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg = cfg.model_copy(update={
            "knowledge_bootstrap": KnowledgeBootstrapConfig(
                terms_accepted=True, topics=[],
                teacher=TeacherConfig(provider="ollama", model="m"),
            )
        })
        pool = SwarmPool(cfg)
        _mock_pool(pool)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        engine = ConsensusEngine(cfg)

        pipeline = BootstrapPipeline(cfg, engine, writer, pool)
        result = await pipeline.run()
        assert result.topics_processed == 0
        await adapter.close()

    async def test_full_bootstrap(self, tmp_path):
        """Bootstrap pipeline: teacher → consensus → graph."""
        cfg = _cfg(tmp_path)
        pool = SwarmPool(cfg)
        _mock_pool(pool)

        # Teacher also uses the same mocked provider
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)
        engine = ConsensusEngine(cfg)

        pipeline = BootstrapPipeline(cfg, engine, writer, pool)

        # Mock the teacher to return Q&A pairs
        mock_teacher = TeacherModel(cfg)
        mock_teacher._provider = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
        mock_teacher._provider._post = AsyncMock(
            return_value={"response": QA_JSON, "eval_count": 1}
        )

        # Monkey-patch the pipeline to use our mock teacher
        original_run = pipeline.run

        async def patched_run(topics=None):
            # Replace TeacherModel construction
            import bootstrap.pipeline as bp_mod
            original_teacher_cls = bp_mod.TeacherModel

            class MockTeacherCls:
                def __init__(self, *a, **kw):
                    pass
                async def generate_qa(self, topic, count=20):
                    return _parse_qa(QA_JSON)

            bp_mod.TeacherModel = MockTeacherCls
            try:
                return await original_run(topics)
            finally:
                bp_mod.TeacherModel = original_teacher_cls

        result = await patched_run()

        assert isinstance(result, BootstrapResult)
        assert result.topics_processed == 1
        assert result.claims_stored > 0

        stats = await adapter.get_stats()
        assert stats.total_nodes > 0

        await adapter.close()
