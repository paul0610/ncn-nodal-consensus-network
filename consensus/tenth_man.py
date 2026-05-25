"""Tenth-man protocol — devil's advocate for epistemic diversity.

Triggered when emerging consensus is too high (> threshold).
Generates counter-claims that enrich the claim pool before validation.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, ClaimStatus, GraphContext, NodeResponse, NodeRole
from swarm.pool import SwarmPool


def should_trigger(
    claims: list[Claim],
    config: Config,
) -> bool:
    """Return ``True`` when the quick consensus preview exceeds the trigger threshold."""
    if not config.consensus.tenth_man_enabled or not claims:
        return False
    preview = quick_consensus_preview(claims)
    return preview > config.consensus.tenth_man_triggers_above


def quick_consensus_preview(claims: list[Claim]) -> float:
    """Average confidence of all pending claims — a fast proxy for consensus level."""
    if not claims:
        return 0.0
    return sum(c.confidence for c in claims) / len(claims)


def build_tenth_man_prompt(
    query: str,
    context: GraphContext,
    claims: list[Claim],
) -> str:
    """Build the user-prompt sent to tenth-man nodes."""
    claims_text = "\n".join(
        f"  - [{c.claim_id[:8]}] {c.subject} → {c.predicate} → {c.object} "
        f"(conf: {c.confidence:.2f})"
        for c in claims
    )
    return (
        f"PREGUNTA ORIGINAL: {query}\n\n"
        f"CONTEXTO:\n{context.serialized_context}\n\n"
        f"CLAIMS DEL ENJAMBRE (consenso emergente):\n{claims_text}\n\n"
        "Tu trabajo es REFUTAR estos claims. Busca evidencia contraria."
    )


async def run_tenth_man(
    query: str,
    context: GraphContext,
    claims: list[Claim],
    swarm: SwarmPool,
) -> list[Claim]:
    """Invoke tenth-man nodes and return counter-claims."""
    prompt = build_tenth_man_prompt(query, context, claims)
    responses = await swarm.run_role(NodeRole.TENTH_MAN, prompt)
    return parse_counter_claims(responses)


def parse_counter_claims(responses: list[NodeResponse]) -> list[Claim]:
    """Parse counter-claims from tenth-man LLM output."""
    counter: list[Claim] = []
    for resp in responses:
        try:
            data = _extract_json(resp.content)
            if not data.get("refutation_possible", False):
                continue
            for rc in data.get("counter_claims", []):
                counter.append(
                    Claim(
                        subject=str(rc.get("subject", "")),
                        predicate=str(rc.get("predicate", "")),
                        object=str(rc.get("object", "")),
                        confidence=float(rc.get("confidence", 0.5)),
                        source_model=resp.node_id,
                        status=ClaimStatus.PENDING,
                        is_counter_claim=True,
                        challenges_claim_id=rc.get("challenges_claim_id"),
                    )
                )
        except Exception as exc:
            logger.warning(f"Tenth-man parse error ({resp.node_id}): {exc}")
    return counter


# ------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
    raw = m.group(1) if m and m.lastindex else (m.group(0) if m else text)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)  # fix trailing commas
    return json.loads(raw)
