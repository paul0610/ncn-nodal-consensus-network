"""Aggregation — Phase 3 of the consensus pipeline.

Computes reputation-weighted scores and decides claim status.
"""

from __future__ import annotations

from core.config_loader import Config
from core.models import Claim, ClaimStatus, SourceAuthority, Vote
from swarm.pool import SwarmPool


def compute_weighted_score(
    votes: list[Vote],
    swarm: SwarmPool,
    config: Config,
    claim_authority: SourceAuthority | None = None,
) -> float:
    """Reputation-weighted vote score implementing the Pareto principle.

    The top 20 % of models (by reputation) get an *elite_vote_multiplier*
    boost on their vote weight.

    When *claim_authority* is provided and source authority is enabled,
    high-authority claims receive a small score bonus (up to +0.15 for
    ``USER_DIRECT``), making them more likely to survive consensus.
    """
    if not votes:
        # No validators configured → auto-pass (trust the extractor)
        return 1.0

    elite_cutoff = _elite_cutoff(swarm, config)

    weighted_sum = 0.0
    weight_total = 0.0

    for vote in votes:
        node = swarm.get_node(vote.node_id)
        reputation = node.reputation_score if node else 1.0

        if config.reputation.enabled and reputation >= elite_cutoff:
            reputation *= config.reputation.elite_vote_multiplier

        weight = reputation * vote.confidence
        weighted_sum += weight * (1.0 if vote.vote else 0.0)
        weight_total += weight

    base_score = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Authority bias: high-authority claims get a score bonus
    if claim_authority and config.source_authority.enabled:
        auth_weight = config.source_authority.weights.get(
            claim_authority.value, 0.30
        )
        authority_bias = max(0.0, (auth_weight - 0.5) * 0.3)
        return min(1.0, base_score + authority_bias)

    return base_score


def decide_status(score: float, config: Config) -> ClaimStatus:
    """Map a weighted score to a ``ClaimStatus``."""
    if score >= config.consensus.verification_threshold:
        return ClaimStatus.VERIFIED
    if score <= config.consensus.discard_threshold:
        return ClaimStatus.DISCARDED
    return ClaimStatus.UNCERTAIN


def compute_performances(
    claims: list[Claim],
    source_field: str = "source_model",
) -> dict[str, float]:
    """Compute per-model performance: verified / total claims ratio."""
    totals: dict[str, int] = {}
    verified: dict[str, int] = {}

    for claim in claims:
        src = getattr(claim, source_field, "") or ""
        if not src:
            continue
        totals[src] = totals.get(src, 0) + 1
        if claim.status == ClaimStatus.VERIFIED:
            verified[src] = verified.get(src, 0) + 1

    return {
        nid: verified.get(nid, 0) / total
        for nid, total in totals.items()
        if total > 0
    }


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _elite_cutoff(swarm: SwarmPool, config: Config) -> float:
    """Reputation score above which a node is *elite*."""
    if not config.reputation.enabled or not swarm.nodes:
        return float("inf")
    reps = sorted(
        [n.reputation_score for n in swarm.nodes], reverse=True
    )
    idx = max(0, int(len(reps) * config.reputation.elite_ratio) - 1)
    return reps[idx]
