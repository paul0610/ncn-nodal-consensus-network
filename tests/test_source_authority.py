"""Source Authority system tests.

Verifies the trust hierarchy, authority-weighted confidence,
conflict detection, correction mechanism, and backward compatibility.
"""

from __future__ import annotations

import pytest

from core.config_loader import (
    Config,
    GraphConfig,
    KuzuConfig,
    SourceAuthorityConfig,
)
from core.models import (
    Claim,
    ClaimStatus,
    CorrectionEvent,
    GraphNode,
    SourceAuthority,
    Vote,
    authority_rank,
)
from consensus.aggregator import compute_weighted_score
from graph.kuzu_adapter import KuzuAdapter
from graph.writer import SingleWriter
from providers.ollama import OllamaProvider
from core.config_loader import ProviderConfig, SwarmConfig, SwarmNodeConfig
from core.models import NodeRole
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cfg(tmp_path, **overrides) -> Config:
    base = {
        "graph": GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "sa_graph"))),
    }
    base.update(overrides)
    return Config(**base)


def _claim(
    subj="A", pred="rel", obj="B", conf=0.8,
    auth=SourceAuthority.SLM_SINGLE,
) -> Claim:
    return Claim(
        subject=subj, predicate=pred, object=obj,
        confidence=conf, source_model="test",
        source_authority=auth,
    )


# ------------------------------------------------------------------
# SourceAuthority enum
# ------------------------------------------------------------------

class TestSourceAuthorityEnum:
    def test_values(self):
        assert SourceAuthority.USER_DIRECT.value == "user_direct"
        assert SourceAuthority.SLM_SINGLE.value == "slm_single"

    def test_rank_order(self):
        assert authority_rank("user_direct") < authority_rank("slm_single")
        assert authority_rank("coach_model") < authority_rank("web_raw")

    def test_rank_unknown(self):
        assert authority_rank("unknown") == 5  # lowest


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

class TestAuthorityConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.source_authority.enabled is True
        assert cfg.source_authority.weights["user_direct"] == 1.0
        assert cfg.source_authority.weights["slm_single"] == 0.30

    def test_source_type_mapping(self):
        cfg = Config()
        assert cfg.source_authority.source_type_authority["text"] == "user_direct"
        assert cfg.source_authority.source_type_authority["web_search"] == "web_raw"


# ------------------------------------------------------------------
# Authority-weighted confidence in writer
# ------------------------------------------------------------------

class TestAuthorityWeightedWrites:
    async def test_user_direct_high_confidence(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        claim = _claim(auth=SourceAuthority.USER_DIRECT, conf=0.8)
        await writer.write_verified_claims([claim])

        node = await adapter.find_node_by_name("A")
        assert node is not None
        # user_direct weight = 1.0, so conf = 0.8 * 1.0 = 0.8
        assert node.confidence == pytest.approx(0.8, abs=0.01)
        assert node.source_authority == "user_direct"
        await adapter.close()

    async def test_slm_single_low_confidence(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        claim = _claim(auth=SourceAuthority.SLM_SINGLE, conf=0.8)
        await writer.write_verified_claims([claim])

        node = await adapter.find_node_by_name("A")
        assert node is not None
        # slm_single weight = 0.30, so conf = 0.8 * 0.30 = 0.24
        assert node.confidence == pytest.approx(0.24, abs=0.01)
        assert node.source_authority == "slm_single"
        await adapter.close()

    async def test_higher_authority_upgrades_node(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # First: low authority
        c1 = _claim(subj="Python", pred="is", obj="lang", auth=SourceAuthority.SLM_SINGLE, conf=0.5)
        await writer.write_verified_claims([c1])
        py1 = await adapter.find_node_by_name("Python")
        assert py1.source_authority == "slm_single"

        # Second: high authority on same entity
        c2 = _claim(subj="Python", pred="created_by", obj="Guido", auth=SourceAuthority.USER_DIRECT, conf=0.9)
        await writer.write_verified_claims([c2])
        py2 = await adapter.find_node_by_name("Python")
        assert py2.source_authority == "user_direct"  # upgraded
        assert py2.confidence > py1.confidence  # boosted

        await adapter.close()


# ------------------------------------------------------------------
# Conflict detection and correction
# ------------------------------------------------------------------

class TestConflictCorrection:
    async def test_high_auth_corrects_low_auth(self, tmp_path):
        """USER_DIRECT claim corrects conflicting SLM_SINGLE node."""
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # Bad claim (low authority)
        bad = _claim(subj="Hyde", pred="created", obj="Python",
                     auth=SourceAuthority.SLM_SINGLE, conf=0.9)
        await writer.write_verified_claims([bad])

        bad_node = await adapter.find_node_by_name("Python")
        assert bad_node is not None
        old_conf = bad_node.confidence

        # Correct claim (high authority)
        good = _claim(subj="Hyde", pred="created", obj="HydeMagne nickname",
                      auth=SourceAuthority.USER_DIRECT, conf=0.95)
        await writer.write_verified_claims([good])

        corrections = writer.get_and_clear_corrections()
        assert len(corrections) >= 1

        # The bad node should have reduced confidence
        bad_after = await adapter.find_node_by_name("Python")
        assert bad_after.confidence < old_conf

        await adapter.close()

    async def test_same_authority_no_correction(self, tmp_path):
        """Claims at the same authority level don't trigger corrections."""
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        c1 = _claim(subj="X", pred="is", obj="A", auth=SourceAuthority.CONSENSUS, conf=0.8)
        await writer.write_verified_claims([c1])

        c2 = _claim(subj="X", pred="is", obj="B", auth=SourceAuthority.CONSENSUS, conf=0.8)
        await writer.write_verified_claims([c2])

        corrections = writer.get_and_clear_corrections()
        assert len(corrections) == 0
        await adapter.close()

    async def test_low_authority_no_correction(self, tmp_path):
        """Low-authority claims (rank > 2) never trigger corrections."""
        cfg = _cfg(tmp_path, source_authority=SourceAuthorityConfig(
            authority_gap_for_correction=2
        ))
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # web_raw (rank 4) then web_verified (rank 3) — low authority can't correct
        c1 = _claim(subj="X", pred="is", obj="A", auth=SourceAuthority.WEB_RAW, conf=0.7)
        await writer.write_verified_claims([c1])

        c2 = _claim(subj="X", pred="is", obj="B", auth=SourceAuthority.WEB_VERIFIED, conf=0.8)
        await writer.write_verified_claims([c2])

        corrections = writer.get_and_clear_corrections()
        assert len(corrections) == 0
        await adapter.close()

    async def test_correction_sets_uncertain(self, tmp_path):
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # Low authority node with low confidence
        bad = _claim(subj="Y", pred="is", obj="wrong",
                     auth=SourceAuthority.SLM_SINGLE, conf=0.4)
        await writer.write_verified_claims([bad])

        # High authority correction
        good = _claim(subj="Y", pred="is", obj="correct",
                      auth=SourceAuthority.USER_DIRECT, conf=0.9)
        await writer.write_verified_claims([good])

        wrong_node = await adapter.find_node_by_name("wrong")
        assert wrong_node is not None
        assert wrong_node.uncertain is True
        await adapter.close()


# ------------------------------------------------------------------
# Authority bias in consensus aggregator
# ------------------------------------------------------------------

class TestAuthorityBiasInConsensus:
    def test_user_direct_gets_bonus(self):
        from unittest.mock import AsyncMock
        pool = SwarmPool(Config(swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[SwarmNodeConfig(provider="ollama", model="m", count=2, role="critic")],
        )))
        for n in pool.nodes:
            n.provider = OllamaProvider(ProviderConfig(base_url="http://x"))

        critics = [n for n in pool.nodes if n.role == NodeRole.CRITIC]
        votes = [
            Vote(node_id=critics[0].node_id, claim_id="c", vote=True, confidence=0.6),
            Vote(node_id=critics[1].node_id, claim_id="c", vote=False, confidence=0.6),
        ]

        # Without authority — should be ~0.5
        score_no_auth = compute_weighted_score(votes, pool, Config(), claim_authority=None)

        # With USER_DIRECT authority — should get bonus
        score_with_auth = compute_weighted_score(
            votes, pool, Config(), claim_authority=SourceAuthority.USER_DIRECT
        )
        assert score_with_auth > score_no_auth

    def test_slm_single_no_bonus(self):
        pool = SwarmPool(Config(swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic")],
        )))
        for n in pool.nodes:
            n.provider = OllamaProvider(ProviderConfig(base_url="http://x"))

        votes = [
            Vote(node_id=pool.nodes[0].node_id, claim_id="c", vote=True, confidence=0.5),
        ]
        score_no = compute_weighted_score(votes, pool, Config(), claim_authority=None)
        score_slm = compute_weighted_score(
            votes, pool, Config(), claim_authority=SourceAuthority.SLM_SINGLE
        )
        # SLM weight 0.30 < 0.5 → no bonus → same score
        assert score_slm == pytest.approx(score_no, abs=0.01)


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_claim_default_authority(self):
        c = Claim(subject="A", predicate="r", object="B", confidence=0.5, source_model="m")
        assert c.source_authority == SourceAuthority.SLM_SINGLE

    def test_graphnode_default_authority(self):
        n = GraphNode(name="X", node_type="concepto")
        assert n.source_authority == "slm_single"

    def test_config_without_authority_section(self):
        cfg = Config()
        assert cfg.source_authority.enabled is True
        assert cfg.source_authority.weights["user_direct"] == 1.0


# ------------------------------------------------------------------
# End-to-end: write low-auth, contradict with high-auth, verify
# ------------------------------------------------------------------

class TestEndToEndCorrection:
    async def test_full_correction_flow(self, tmp_path):
        """
        1. Write SLM claim: 'Hyde → created → Python'
        2. Write USER_DIRECT claim: 'Hyde → created → HydeMagne nickname'
        3. Verify 'Python' node confidence was reduced
        4. Verify correction event was logged
        """
        cfg = _cfg(tmp_path)
        adapter = KuzuAdapter(cfg)
        await adapter.initialize()
        writer = SingleWriter(adapter, cfg)

        # Step 1: bad SLM claim
        bad_claim = _claim(
            subj="Hyde", pred="created", obj="Python",
            auth=SourceAuthority.SLM_SINGLE, conf=0.9,
        )
        await writer.write_verified_claims([bad_claim])

        python_before = await adapter.find_node_by_name("Python")
        assert python_before is not None

        # Step 2: authoritative correction
        good_claim = _claim(
            subj="Hyde", pred="created", obj="HydeMagne",
            auth=SourceAuthority.USER_DIRECT, conf=0.95,
        )
        await writer.write_verified_claims([good_claim])

        # Step 3: verify confidence reduced
        python_after = await adapter.find_node_by_name("Python")
        assert python_after.confidence < python_before.confidence

        # Step 4: verify correction event
        corrections = writer.get_and_clear_corrections()
        assert len(corrections) >= 1
        assert any("Python" in c.corrected_node_name for c in corrections)

        # The new correct node should exist with high confidence
        hydemagne = await adapter.find_node_by_name("HydeMagne")
        assert hydemagne is not None
        assert hydemagne.source_authority == "user_direct"

        await adapter.close()
