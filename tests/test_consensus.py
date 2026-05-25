"""Sprint 5 checkpoint — consensus engine tests.

CHECKPOINT: text → claims → consensus → verified claims with scores.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core.config_loader import (
    Config,
    ConsensusConfig,
    ProviderConfig,
    ReputationConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.models import (
    Claim,
    ClaimStatus,
    ConsensusResult,
    GraphContext,
    NodeResponse,
    NodeRole,
    Vote,
)
from consensus.aggregator import compute_performances, compute_weighted_score, decide_status
from consensus.engine import ConsensusEngine
from consensus.extractor import build_extractor_prompt, parse_claims
from consensus.judge import build_judge_prompt, invoke_judge
from consensus.tenth_man import (
    build_tenth_man_prompt,
    parse_counter_claims,
    quick_consensus_preview,
    should_trigger,
)
from consensus.validator import build_critic_prompt, select_validators
from providers.ollama import OllamaProvider
from swarm.node import SwarmNode
from swarm.pool import SwarmPool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mock_provider() -> OllamaProvider:
    p = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
    p._post = AsyncMock(return_value={"response": "{}", "eval_count": 1})
    return p


def _make_swarm(
    extractors: int = 3,
    critics: int = 3,
    tenth_man: int = 1,
    judges: int = 1,
    synthesizers: int = 1,
) -> SwarmPool:
    """Build a SwarmPool with mocked providers — no real HTTP."""
    cfg = Config(
        swarm=SwarmConfig(
            max_concurrent=20,
            request_timeout=30,
            temperature=0.7,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=extractors, role="extractor"),
                SwarmNodeConfig(provider="ollama", model="m", count=critics, role="critic"),
                SwarmNodeConfig(provider="ollama", model="m", count=tenth_man, role="tenth_man"),
                SwarmNodeConfig(provider="ollama", model="m", count=judges, role="judge"),
                SwarmNodeConfig(provider="ollama", model="m", count=synthesizers, role="synthesizer"),
            ],
        ),
    )
    pool = SwarmPool(cfg)
    for node in pool.nodes:
        node.provider = _mock_provider()
    return pool


def _mock_extractor_response(node_id: str, claims_json: list[dict]) -> NodeResponse:
    return NodeResponse(
        node_id=node_id,
        role="extractor",
        content=json.dumps({"claims": claims_json}),
        model="m",
        provider="ollama",
    )


def _set_role_reply(pool: SwarmPool, role: NodeRole, reply: str) -> None:
    """Set all nodes of *role* to return *reply*."""
    for node in pool.nodes:
        if node.role == role:
            node.provider._post = AsyncMock(
                return_value={"response": reply, "eval_count": 1}
            )


SAMPLE_CLAIMS_JSON = [
    {
        "subject": "Python",
        "predicate": "is",
        "object": "programming language",
        "confidence": 0.95,
        "temporal": False,
        "source_text": "Python is a programming language.",
    },
    {
        "subject": "Python",
        "predicate": "created_by",
        "object": "Guido van Rossum",
        "confidence": 0.90,
        "temporal": False,
        "source_text": "Created by Guido van Rossum.",
    },
]


# ------------------------------------------------------------------
# Extractor
# ------------------------------------------------------------------

class TestExtractor:
    def test_build_prompt(self):
        ctx = GraphContext(serialized_context="• A → rel → B")
        prompt = build_extractor_prompt("What is A?", ctx)
        assert "What is A?" in prompt
        assert "A → rel → B" in prompt

    def test_parse_claims_json(self):
        resp = _mock_extractor_response("ext-0", SAMPLE_CLAIMS_JSON)
        claims = parse_claims([resp])
        assert len(claims) == 2
        assert claims[0].subject == "Python"
        assert claims[0].source_model == "ext-0"

    def test_parse_claims_fenced_json(self):
        content = "Here are claims:\n```json\n" + json.dumps({"claims": SAMPLE_CLAIMS_JSON}) + "\n```"
        resp = NodeResponse(node_id="e", role="extractor", content=content, model="m", provider="p")
        claims = parse_claims([resp])
        assert len(claims) == 2

    def test_parse_claims_bad_json(self):
        resp = NodeResponse(node_id="e", role="extractor", content="not json at all", model="m", provider="p")
        claims = parse_claims([resp])
        assert claims == []

    def test_parse_claims_multiple_responses(self):
        r1 = _mock_extractor_response("e1", SAMPLE_CLAIMS_JSON[:1])
        r2 = _mock_extractor_response("e2", SAMPLE_CLAIMS_JSON[1:])
        claims = parse_claims([r1, r2])
        assert len(claims) == 2


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------

class TestValidator:
    def test_build_critic_prompt(self):
        claim = Claim(subject="A", predicate="is", object="B", confidence=0.8, source_model="e")
        ctx = GraphContext(serialized_context="context text")
        prompt = build_critic_prompt("query", ctx, claim)
        assert "A" in prompt and "B" in prompt

    def test_select_validators(self):
        pool = _make_swarm(critics=5)
        validators = select_validators(pool, k=3, exclude_model="ext-0")
        assert len(validators) == 3
        assert all(v.role == NodeRole.CRITIC for v in validators)

    def test_select_validators_capped_by_critics(self):
        """If fewer critics than k, only available critics are returned."""
        pool = _make_swarm(critics=1, extractors=5)
        validators = select_validators(pool, k=3, exclude_model="")
        assert len(validators) == 1
        assert validators[0].role == NodeRole.CRITIC


# ------------------------------------------------------------------
# Aggregator
# ------------------------------------------------------------------

class TestAggregator:
    def test_all_positive_votes(self):
        pool = _make_swarm(critics=3)
        votes = [
            Vote(node_id=n.node_id, claim_id="c1", vote=True, confidence=0.9)
            for n in pool.nodes if n.role == NodeRole.CRITIC
        ]
        score = compute_weighted_score(votes, pool, Config())
        assert score > 0.8

    def test_all_negative_votes(self):
        pool = _make_swarm(critics=3)
        votes = [
            Vote(node_id=n.node_id, claim_id="c1", vote=False, confidence=0.9)
            for n in pool.nodes if n.role == NodeRole.CRITIC
        ]
        score = compute_weighted_score(votes, pool, Config())
        assert score < 0.1

    def test_empty_votes(self):
        """Empty votes = no validators configured → auto-pass (trust extractor)."""
        pool = _make_swarm()
        assert compute_weighted_score([], pool, Config()) == 1.0

    def test_decide_verified(self):
        cfg = Config(consensus=ConsensusConfig(verification_threshold=0.66))
        assert decide_status(0.80, cfg) == ClaimStatus.VERIFIED

    def test_decide_discarded(self):
        cfg = Config(consensus=ConsensusConfig(discard_threshold=0.33))
        assert decide_status(0.20, cfg) == ClaimStatus.DISCARDED

    def test_decide_uncertain(self):
        cfg = Config(consensus=ConsensusConfig(
            verification_threshold=0.66, discard_threshold=0.33
        ))
        assert decide_status(0.50, cfg) == ClaimStatus.UNCERTAIN

    def test_compute_performances(self):
        claims = [
            Claim(subject="a", predicate="r", object="b", confidence=0.9,
                  source_model="m1", status=ClaimStatus.VERIFIED),
            Claim(subject="c", predicate="r", object="d", confidence=0.8,
                  source_model="m1", status=ClaimStatus.DISCARDED),
            Claim(subject="e", predicate="r", object="f", confidence=0.7,
                  source_model="m2", status=ClaimStatus.VERIFIED),
        ]
        perf = compute_performances(claims)
        assert perf["m1"] == pytest.approx(0.5)  # 1 verified / 2 total
        assert perf["m2"] == pytest.approx(1.0)  # 1 / 1

    def test_elite_bonus(self):
        """Node with high reputation gets a multiplied vote weight."""
        pool = _make_swarm(critics=2)
        critics = [n for n in pool.nodes if n.role == NodeRole.CRITIC]
        critics[0].reputation_score = 0.99  # elite
        critics[1].reputation_score = 0.50

        cfg = Config(reputation=ReputationConfig(
            enabled=True,
            elite_ratio=0.50,
            elite_vote_multiplier=3.0,
        ))

        votes = [
            Vote(node_id=critics[0].node_id, claim_id="c", vote=True, confidence=0.9),
            Vote(node_id=critics[1].node_id, claim_id="c", vote=False, confidence=0.9),
        ]
        score = compute_weighted_score(votes, pool, cfg)
        # Elite's True vote should dominate
        assert score > 0.6


# ------------------------------------------------------------------
# Tenth Man
# ------------------------------------------------------------------

class TestTenthMan:
    def test_should_trigger_high_consensus(self):
        claims = [Claim(subject="a", predicate="r", object="b", confidence=0.95, source_model="e")]
        cfg = Config(consensus=ConsensusConfig(
            tenth_man_enabled=True, tenth_man_triggers_above=0.80
        ))
        assert should_trigger(claims, cfg) is True

    def test_should_not_trigger_low_consensus(self):
        claims = [Claim(subject="a", predicate="r", object="b", confidence=0.60, source_model="e")]
        cfg = Config(consensus=ConsensusConfig(
            tenth_man_enabled=True, tenth_man_triggers_above=0.80
        ))
        assert should_trigger(claims, cfg) is False

    def test_should_not_trigger_disabled(self):
        claims = [Claim(subject="a", predicate="r", object="b", confidence=0.99, source_model="e")]
        cfg = Config(consensus=ConsensusConfig(tenth_man_enabled=False))
        assert should_trigger(claims, cfg) is False

    def test_parse_counter_claims(self):
        content = json.dumps({
            "refutation_possible": True,
            "counter_claims": [
                {"subject": "X", "predicate": "not", "object": "Y", "confidence": 0.6}
            ],
            "reasoning": "because reasons",
        })
        resp = NodeResponse(node_id="tm", role="tenth_man", content=content, model="m", provider="p")
        cc = parse_counter_claims([resp])
        assert len(cc) == 1
        assert cc[0].is_counter_claim is True

    def test_parse_no_refutation(self):
        content = json.dumps({"refutation_possible": False, "counter_claims": [], "reasoning": ""})
        resp = NodeResponse(node_id="tm", role="tenth_man", content=content, model="m", provider="p")
        assert parse_counter_claims([resp]) == []

    def test_quick_consensus_preview(self):
        claims = [
            Claim(subject="a", predicate="r", object="b", confidence=0.9, source_model="e"),
            Claim(subject="c", predicate="r", object="d", confidence=0.8, source_model="e"),
        ]
        assert quick_consensus_preview(claims) == pytest.approx(0.85)


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------

class TestJudge:
    async def test_invoke_judge_verified(self):
        pool = _make_swarm(judges=1)
        _set_role_reply(pool, NodeRole.JUDGE, json.dumps({
            "winner": "A", "confidence": 0.85, "reasoning": "clear"
        }))
        ctx = GraphContext()
        claim = Claim(subject="X", predicate="is", object="Y", confidence=0.5, source_model="e")
        status = await invoke_judge("q", ctx, claim, pool, Config())
        assert status == ClaimStatus.VERIFIED

    async def test_invoke_judge_discarded(self):
        pool = _make_swarm(judges=1)
        _set_role_reply(pool, NodeRole.JUDGE, json.dumps({
            "winner": "B", "confidence": 0.80, "reasoning": "not supported"
        }))
        ctx = GraphContext()
        claim = Claim(subject="X", predicate="is", object="Y", confidence=0.5, source_model="e")
        status = await invoke_judge("q", ctx, claim, pool, Config())
        assert status == ClaimStatus.DISCARDED

    async def test_invoke_judge_no_judges(self):
        pool = _make_swarm(judges=0)
        ctx = GraphContext()
        claim = Claim(subject="X", predicate="is", object="Y", confidence=0.5, source_model="e")
        status = await invoke_judge("q", ctx, claim, pool, Config())
        assert status == ClaimStatus.UNCERTAIN


# ------------------------------------------------------------------
# ConsensusEngine — full pipeline (mocked)
# ------------------------------------------------------------------

class TestConsensusEngine:
    async def test_full_pipeline(self):
        """text → claims → consensus → verified claims with scores."""
        pool = _make_swarm(extractors=2, critics=3, synthesizers=1)

        # Extractors return valid claims
        extractor_reply = json.dumps({"claims": SAMPLE_CLAIMS_JSON})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)

        # Critics all vote True with high confidence
        critic_reply = json.dumps({"vote": True, "confidence": 0.9, "reason": "supported"})
        _set_role_reply(pool, NodeRole.CRITIC, critic_reply)

        # Synthesizer produces answer
        _set_role_reply(pool, NodeRole.SYNTHESIZER, "Python is a programming language created by Guido.")

        engine = ConsensusEngine(Config(consensus=ConsensusConfig(validators_per_claim=3)))
        ctx = GraphContext(serialized_context="Python info here")
        result = await engine.run("What is Python?", ctx, pool)

        assert isinstance(result, ConsensusResult)
        assert len(result.verified_claims) > 0
        assert result.synthesized_answer != ""
        assert result.processing_time_ms > 0

    async def test_pipeline_with_discards(self):
        """Critics reject claims → they should be discarded."""
        pool = _make_swarm(extractors=1, critics=3, synthesizers=1)

        extractor_reply = json.dumps({"claims": SAMPLE_CLAIMS_JSON})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)

        # Critics all vote False
        critic_reply = json.dumps({"vote": False, "confidence": 0.9, "reason": "not supported"})
        _set_role_reply(pool, NodeRole.CRITIC, critic_reply)

        _set_role_reply(pool, NodeRole.SYNTHESIZER, "No verified info.")

        engine = ConsensusEngine(Config())
        result = await engine.run("query", GraphContext(), pool)

        assert len(result.discarded_claims) > 0
        assert len(result.verified_claims) == 0

    async def test_pipeline_no_extractors(self):
        pool = _make_swarm(extractors=0)
        engine = ConsensusEngine(Config())
        result = await engine.run("query", GraphContext(), pool)
        assert result.synthesized_answer != ""
        assert len(result.verified_claims) == 0

    async def test_pipeline_with_tenth_man(self):
        pool = _make_swarm(extractors=1, critics=2, tenth_man=1, synthesizers=1)

        extractor_reply = json.dumps({"claims": [
            {"subject": "A", "predicate": "is", "object": "B", "confidence": 0.95}
        ]})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)

        tenth_man_reply = json.dumps({
            "refutation_possible": True,
            "counter_claims": [
                {"subject": "A", "predicate": "is_not", "object": "B", "confidence": 0.7}
            ],
            "reasoning": "counter evidence",
        })
        _set_role_reply(pool, NodeRole.TENTH_MAN, tenth_man_reply)

        critic_reply = json.dumps({"vote": True, "confidence": 0.8, "reason": "ok"})
        _set_role_reply(pool, NodeRole.CRITIC, critic_reply)
        _set_role_reply(pool, NodeRole.SYNTHESIZER, "answer")

        cfg = Config(consensus=ConsensusConfig(
            tenth_man_enabled=True, tenth_man_triggers_above=0.80
        ))
        engine = ConsensusEngine(cfg)
        result = await engine.run("query", GraphContext(), pool)

        # Both original and counter-claims should exist
        all_claims = result.verified_claims + result.uncertain_claims + result.discarded_claims
        assert len(all_claims) >= 2

    async def test_pipeline_with_judge(self):
        """Uncertain claims get resolved by judge when auto_judge is on."""
        pool = _make_swarm(extractors=1, critics=2, judges=1, synthesizers=1)

        extractor_reply = json.dumps({"claims": [SAMPLE_CLAIMS_JSON[0]]})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)

        # Mixed votes → uncertain
        critics = [n for n in pool.nodes if n.role == NodeRole.CRITIC]
        critics[0].provider._post = AsyncMock(
            return_value={"response": json.dumps({"vote": True, "confidence": 0.6}), "eval_count": 1}
        )
        critics[1].provider._post = AsyncMock(
            return_value={"response": json.dumps({"vote": False, "confidence": 0.6}), "eval_count": 1}
        )

        # Judge resolves as A (verified)
        _set_role_reply(pool, NodeRole.JUDGE, json.dumps({
            "winner": "A", "confidence": 0.8, "reasoning": "clear"
        }))
        _set_role_reply(pool, NodeRole.SYNTHESIZER, "final answer")

        cfg = Config(consensus=ConsensusConfig(
            auto_judge_uncertain=True,
            verification_threshold=0.66,
            discard_threshold=0.33,
            validators_per_claim=2,
        ))
        engine = ConsensusEngine(cfg)
        result = await engine.run("query", GraphContext(), pool)

        # Judge should have promoted the uncertain claim to verified
        assert len(result.verified_claims) >= 1

    async def test_model_performances_tracked(self):
        pool = _make_swarm(extractors=2, critics=2, synthesizers=1)

        extractor_reply = json.dumps({"claims": SAMPLE_CLAIMS_JSON})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)
        critic_reply = json.dumps({"vote": True, "confidence": 0.9, "reason": "ok"})
        _set_role_reply(pool, NodeRole.CRITIC, critic_reply)
        _set_role_reply(pool, NodeRole.SYNTHESIZER, "answer")

        engine = ConsensusEngine(Config())
        result = await engine.run("query", GraphContext(), pool)

        assert isinstance(result.model_performances, dict)


# ------------------------------------------------------------------
# CHECKPOINT: text → claims → consensus → verified claims with scores
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_end_to_end_consensus(self):
        """Full checkpoint: text produces claims that survive consensus."""
        pool = _make_swarm(extractors=3, critics=4, synthesizers=1)

        claims_data = [
            {"subject": "Einstein", "predicate": "developed", "object": "Relativity",
             "confidence": 0.95, "source_text": "Einstein developed the theory of relativity."},
            {"subject": "Relativity", "predicate": "published_in", "object": "1905",
             "confidence": 0.90, "temporal": True, "date": "1905",
             "source_text": "Published in 1905."},
            {"subject": "Newton", "predicate": "influenced", "object": "Einstein",
             "confidence": 0.80, "source_text": "Newton's work influenced Einstein."},
        ]
        extractor_reply = json.dumps({"claims": claims_data})
        _set_role_reply(pool, NodeRole.EXTRACTOR, extractor_reply)

        # Strong positive consensus
        critic_reply = json.dumps({"vote": True, "confidence": 0.88, "reason": "well supported"})
        _set_role_reply(pool, NodeRole.CRITIC, critic_reply)

        _set_role_reply(pool, NodeRole.SYNTHESIZER,
            "Einstein developed the theory of relativity, published in 1905.")

        engine = ConsensusEngine(Config())
        context = GraphContext(
            serialized_context="• Einstein → developed → Relativity (conf: 0.94, año: 1905)"
        )
        result = await engine.run(
            "¿Quién desarrolló la relatividad?", context, pool
        )

        # Verified claims with scores
        assert len(result.verified_claims) >= 1
        for claim in result.verified_claims:
            assert claim.status == ClaimStatus.VERIFIED
            assert claim.confidence > 0

        # Synthesis produced
        assert len(result.synthesized_answer) > 10

        # Timing tracked
        assert result.processing_time_ms > 0

        # Session id present
        assert result.session_id != ""
