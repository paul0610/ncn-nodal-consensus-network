"""ConsensusEngine — orchestrates the 4-phase consensus pipeline.

Phase 1: Extraction   — extractors produce claims in parallel
Phase 2: Validation   — NxK cross-validation by critics
Phase 3: Aggregation  — reputation-weighted scoring + judge tie-break
Phase 4: Synthesis    — synthesiser produces natural-language answer
"""

from __future__ import annotations

import time

from loguru import logger

from core.config_loader import Config
from core.models import (
    Claim,
    ClaimStatus,
    ConsensusResult,
    GraphContext,
    NodeRole,
)
from consensus.aggregator import compute_performances, compute_weighted_score, decide_status
from consensus.extractor import build_extractor_prompt, parse_claims
from consensus.judge import invoke_judge
from consensus.tenth_man import run_tenth_man, should_trigger
from consensus.validator import run_validations
from swarm.pool import SwarmPool


class ConsensusEngine:
    """Coordinate the full consensus pipeline."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def run(
        self,
        query: str,
        context: GraphContext,
        swarm: SwarmPool,
        *,
        skip_tenth_man: bool = False,
    ) -> ConsensusResult:
        t0 = time.perf_counter()

        # === PHASE 1: EXTRACTION ===
        logger.info("Consensus Phase 1 — extraction")
        extractor_prompt = build_extractor_prompt(query, context)
        extractor_responses = await swarm.run_role(
            NodeRole.EXTRACTOR, extractor_prompt
        )
        raw_claims = parse_claims(extractor_responses)
        all_claims = _deduplicate_claims(raw_claims)
        logger.info(
            f"  extracted {len(raw_claims)} claims from {len(extractor_responses)} extractors"
            f" → {len(all_claims)} after dedup"
        )

        # Tenth man (if enabled, consensus is high, and NOT in ingest mode)
        if not skip_tenth_man and should_trigger(all_claims, self.config):
            logger.info("Consensus — tenth man triggered")
            counter_claims = await run_tenth_man(
                query, context, all_claims, swarm
            )
            all_claims = all_claims + counter_claims
            logger.info(f"  +{len(counter_claims)} counter-claims")

        if not all_claims:
            return self._empty_result(t0)

        # === PHASE 2: CROSS-VALIDATION (NxK) ===
        logger.info("Consensus Phase 2 — validation")
        votes_by_claim = await run_validations(
            all_claims, query, context, swarm, self.config
        )

        # === PHASE 3: AGGREGATION ===
        logger.info("Consensus Phase 3 — aggregation")
        for claim in all_claims:
            votes = votes_by_claim.get(claim.claim_id, [])
            score = compute_weighted_score(
                votes, swarm, self.config,
                claim_authority=claim.source_authority,
            )
            claim.status = decide_status(score, self.config)

            # Optional judge tie-break
            if (
                claim.status == ClaimStatus.UNCERTAIN
                and self.config.consensus.auto_judge_uncertain
            ):
                claim.status = await invoke_judge(
                    query, context, claim, swarm, self.config
                )

        verified = [c for c in all_claims if c.status == ClaimStatus.VERIFIED]
        uncertain = [c for c in all_claims if c.status == ClaimStatus.UNCERTAIN]
        discarded = [c for c in all_claims if c.status == ClaimStatus.DISCARDED]
        logger.info(
            f"  verified={len(verified)} uncertain={len(uncertain)} discarded={len(discarded)}"
        )

        # === PHASE 4: SYNTHESIS ===
        logger.info("Consensus Phase 4 — synthesis")
        synthesized = await self._synthesize(query, verified, swarm)

        elapsed = (time.perf_counter() - t0) * 1000
        return ConsensusResult(
            verified_claims=verified,
            uncertain_claims=uncertain,
            discarded_claims=discarded,
            synthesized_answer=synthesized,
            model_performances=compute_performances(all_claims),
            processing_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Synthesis helper
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        verified_claims: list[Claim],
        swarm: SwarmPool,
    ) -> str:
        if not verified_claims:
            return "No se verificaron claims suficientes para generar una respuesta."

        claims_text = "\n".join(
            f"• {c.subject} → {c.predicate} → {c.object} (conf: {c.confidence:.2f})"
            for c in verified_claims
        )
        prompt = (
            f"PREGUNTA DEL USUARIO: {query}\n\n"
            f"CLAIMS VERIFICADOS POR CONSENSO:\n{claims_text}\n\n"
            "Genera una respuesta clara y natural basada SOLO en estos claims."
        )
        responses = await swarm.run_role(NodeRole.SYNTHESIZER, prompt)
        if not responses:
            return "No hay sintetizadores disponibles."
        # Pick the longest response (simple heuristic for best synthesis)
        return max(responses, key=lambda r: len(r.content)).content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, t0: float) -> ConsensusResult:
        return ConsensusResult(
            synthesized_answer="No se extrajeron claims del texto.",
            processing_time_ms=(time.perf_counter() - t0) * 1000,
        )


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Remove near-duplicate claims, keeping the one with highest confidence.

    Two claims are considered duplicates when their (subject, predicate, object)
    triple matches after lowercasing and stripping whitespace.
    """
    seen: dict[tuple[str, str, str], Claim] = {}
    for claim in claims:
        key = (
            claim.subject.strip().lower(),
            claim.predicate.strip().lower(),
            claim.object.strip().lower(),
        )
        existing = seen.get(key)
        if existing is None or claim.confidence > existing.confidence:
            seen[key] = claim
    return list(seen.values())
