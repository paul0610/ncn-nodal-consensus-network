"""Sprint 4 checkpoint — swarm tests.

CHECKPOINT: 10 nodes in parallel answering the same prompt, measure latency.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from core.config_loader import (
    Config,
    ProviderConfig,
    ProvidersConfig,
    SwarmConfig,
    SwarmNodeConfig,
    ReputationConfig,
)
from core.models import ModelResponse, NodeResponse, NodeRole
from providers.ollama import OllamaProvider
from swarm.node import SwarmNode
from swarm.pool import SwarmPool
from swarm.reputation import ReputationSystem
from swarm.roles import ROLE_SYSTEM_PROMPTS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mock_provider(name: str = "mock") -> OllamaProvider:
    """OllamaProvider with _post mocked so no network needed."""
    p = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
    p._post = AsyncMock(return_value={"response": "mock reply", "eval_count": 5})
    return p


def _swarm_config(node_count: int = 10) -> Config:
    """Config with *node_count* extractor nodes (all Ollama-mocked)."""
    return Config(
        swarm=SwarmConfig(
            max_concurrent=10,
            request_timeout=30,
            temperature=0.7,
            nodes=[
                SwarmNodeConfig(
                    provider="ollama",
                    model="mock-model",
                    count=node_count,
                    role="extractor",
                    temperature=0.8,
                ),
            ],
        ),
        reputation=ReputationConfig(
            history_file="data/test_reputation.json",
        ),
    )


# ------------------------------------------------------------------
# Roles
# ------------------------------------------------------------------

class TestRoles:
    def test_all_roles_have_prompts(self):
        for role in NodeRole:
            assert role in ROLE_SYSTEM_PROMPTS
            assert len(ROLE_SYSTEM_PROMPTS[role]) > 50

    def test_extractor_prompt_mentions_json(self):
        assert "JSON" in ROLE_SYSTEM_PROMPTS[NodeRole.EXTRACTOR]

    def test_synthesizer_prompt_no_json(self):
        prompt = ROLE_SYSTEM_PROMPTS[NodeRole.SYNTHESIZER]
        assert "NO generes JSON" in prompt


# ------------------------------------------------------------------
# SwarmNode
# ------------------------------------------------------------------

class TestSwarmNode:
    def test_dataclass_fields(self):
        p = _mock_provider()
        node = SwarmNode(
            node_id="n-0",
            provider=p,
            model="llama3.2:1b",
            role=NodeRole.EXTRACTOR,
            temperature=0.8,
        )
        assert node.node_id == "n-0"
        assert node.role == NodeRole.EXTRACTOR
        assert node.reputation_score == 1.0  # default


# ------------------------------------------------------------------
# ReputationSystem
# ------------------------------------------------------------------

class TestReputationSystem:
    def test_new_node_score(self, tmp_path):
        cfg = Config(reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
        ))
        rep = ReputationSystem(cfg)
        assert rep.get_score("unknown-node") == 0.5

    async def test_update_increases_score(self, tmp_path):
        cfg = Config(reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
        ))
        rep = ReputationSystem(cfg)
        await rep.update({"node-a": 0.9, "node-b": 0.3})

        assert rep.get_score("node-a") > rep.get_score("node-b")

    async def test_decay_inactive(self, tmp_path):
        cfg = Config(reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
            decay_per_session=0.05,
        ))
        rep = ReputationSystem(cfg)
        await rep.update({"active": 0.8, "lazy": 0.8})
        score_before = rep.get_score("lazy")

        # Only 'active' participates in the next session
        await rep.update({"active": 0.9})
        assert rep.get_score("lazy") < score_before

    async def test_history_persistence(self, tmp_path):
        path = str(tmp_path / "rep.json")
        cfg = Config(reputation=ReputationConfig(history_file=path))

        rep1 = ReputationSystem(cfg)
        await rep1.update({"node-x": 0.95})

        rep2 = ReputationSystem(cfg)  # reload from disk
        assert rep2.get_score("node-x") > 0.5

    def test_elite_threshold(self, tmp_path):
        cfg = Config(reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
            elite_ratio=0.20,
        ))
        rep = ReputationSystem(cfg)
        rep.history = {
            "a": [0.95], "b": [0.90], "c": [0.80],
            "d": [0.70], "e": [0.60],
        }
        p = _mock_provider()
        nodes = [
            SwarmNode(node_id=nid, provider=p, model="m", role=NodeRole.EXTRACTOR, temperature=0.7)
            for nid in ("a", "b", "c", "d", "e")
        ]
        threshold = rep.get_elite_threshold(nodes)
        # Top 20 % of 5 = 1 node → threshold should be around 0.95
        assert threshold >= 0.9

    def test_score_clamped(self, tmp_path):
        cfg = Config(reputation=ReputationConfig(
            history_file=str(tmp_path / "rep.json"),
        ))
        rep = ReputationSystem(cfg)
        rep.history = {"bad": [-1.0, -2.0]}
        assert rep.get_score("bad") == 0.1  # floor

        rep.history = {"good": [5.0, 5.0]}
        assert rep.get_score("good") == 1.0  # ceiling


# ------------------------------------------------------------------
# SwarmPool — mocked providers
# ------------------------------------------------------------------

class TestSwarmPool:
    def test_build_nodes(self):
        cfg = _swarm_config(node_count=4)
        pool = SwarmPool(cfg)
        assert len(pool.nodes) == 4
        assert all(n.role == NodeRole.EXTRACTOR for n in pool.nodes)

    async def test_run_role(self):
        cfg = _swarm_config(node_count=3)
        pool = SwarmPool(cfg)
        for node in pool.nodes:
            node.provider._post = AsyncMock(
                return_value={"response": f"reply-{node.node_id}", "eval_count": 1}
            )

        results = await pool.run_role(NodeRole.EXTRACTOR, "test prompt")
        assert len(results) == 3
        assert all(isinstance(r, NodeResponse) for r in results)

    async def test_run_role_tolerates_failures(self):
        cfg = _swarm_config(node_count=4)
        pool = SwarmPool(cfg)
        for i, node in enumerate(pool.nodes):
            node.provider = _mock_provider()  # separate instance per node
            if i == 0:
                node.provider._post = AsyncMock(side_effect=RuntimeError("boom"))

        results = await pool.run_role(NodeRole.EXTRACTOR, "test")
        assert len(results) == 3  # 1 failed, 3 succeeded

    async def test_run_role_empty(self):
        cfg = _swarm_config(node_count=3)
        pool = SwarmPool(cfg)
        results = await pool.run_role(NodeRole.JUDGE, "test")
        assert results == []  # no judge nodes configured

    async def test_run_all(self):
        cfg = _swarm_config(node_count=5)
        pool = SwarmPool(cfg)
        for node in pool.nodes:
            node.provider._post = AsyncMock(
                return_value={"response": "ok", "eval_count": 1}
            )

        results = await pool.run_all("test prompt")
        assert len(results) == 5

    def test_get_node(self):
        cfg = _swarm_config(node_count=2)
        pool = SwarmPool(cfg)
        first = pool.nodes[0]
        found = pool.get_node(first.node_id)
        assert found is first
        assert pool.get_node("nonexistent") is None

    async def test_semaphore_limits_concurrency(self):
        cfg = Config(
            swarm=SwarmConfig(
                max_concurrent=2,
                request_timeout=30,
                temperature=0.7,
                nodes=[
                    SwarmNodeConfig(
                        provider="ollama", model="m", count=5,
                        role="extractor", temperature=0.7,
                    ),
                ],
            ),
        )
        pool = SwarmPool(cfg)

        concurrent = 0
        max_concurrent = 0

        async def slow_reply(*_args, **_kwargs):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return {"response": "ok", "eval_count": 1}

        for node in pool.nodes:
            node.provider._post = slow_reply

        await pool.run_role(NodeRole.EXTRACTOR, "test")
        assert max_concurrent <= 2


# ------------------------------------------------------------------
# CHECKPOINT — 10 nodes in parallel, measure latency
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_10_nodes_parallel(self):
        """10 mocked nodes answer in parallel — verify results + latency."""
        cfg = _swarm_config(node_count=10)
        pool = SwarmPool(cfg)
        assert len(pool.nodes) == 10

        counter = 0

        async def fake_reply(*_args, **_kwargs):
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.02)  # simulate ~20ms latency
            return {"response": f"answer-{counter}", "eval_count": 10}

        for node in pool.nodes:
            node.provider._post = fake_reply

        t0 = time.perf_counter()
        results = await pool.run_role(NodeRole.EXTRACTOR, "What is Python?")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == 10
        assert all(r.content.startswith("answer-") for r in results)
        # With 10 concurrent (semaphore=10), all should run in parallel
        # Expect ~20–80 ms, NOT 10 × 20 = 200 ms
        assert elapsed_ms < 500

    async def test_10_nodes_different_content(self):
        """Each node can return different content."""
        cfg = _swarm_config(node_count=10)
        pool = SwarmPool(cfg)

        for i, node in enumerate(pool.nodes):
            node.provider = _mock_provider()  # separate instance per node
            reply = f"unique-{i}"
            node.provider._post = AsyncMock(
                return_value={"response": reply, "eval_count": 1}
            )

        results = await pool.run_role(NodeRole.EXTRACTOR, "prompt")
        contents = {r.content for r in results}
        assert len(contents) == 10  # all different


def _ollama_reachable() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
class TestSwarmIntegration:
    async def test_real_swarm_pool(self):
        """Build a real SwarmPool from default config and run one prompt."""
        cfg = Config()
        pool = SwarmPool(cfg)
        if not pool.nodes:
            pytest.skip("No swarm nodes configured or Ollama models not pulled")

        results = await pool.run_role(
            NodeRole.EXTRACTOR,
            "Responde solo con la palabra 'hola'.",
        )
        assert len(results) >= 1
        assert all(len(r.content) > 0 for r in results)
