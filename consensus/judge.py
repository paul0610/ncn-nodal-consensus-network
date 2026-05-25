"""Judge — tie-breaker arbiter for uncertain claims.

Invoked when a claim's weighted score falls between the verification
and discard thresholds and ``auto_judge_uncertain`` is enabled.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, ClaimStatus, GraphContext, NodeRole
from swarm.pool import SwarmPool


def build_judge_prompt(
    query: str,
    context: GraphContext,
    claim: Claim,
) -> str:
    """Build the user-prompt sent to judge nodes."""
    return (
        f"PREGUNTA ORIGINAL: {query}\n\n"
        f"CONTEXTO:\n{context.serialized_context}\n\n"
        f"CLAIM EN DISPUTA:\n"
        f"  Hipótesis A (a favor): {claim.subject} → {claim.predicate} → {claim.object} "
        f"(conf extractor: {claim.confidence:.2f})\n"
        f"  Hipótesis B (en contra): el claim es incorrecto o no está respaldado\n\n"
        "Emite tu veredicto final en JSON."
    )


async def invoke_judge(
    query: str,
    context: GraphContext,
    claim: Claim,
    swarm: SwarmPool,
    config: Config,
) -> ClaimStatus:
    """Run judge nodes on one uncertain claim. Returns final status."""
    prompt = build_judge_prompt(query, context, claim)
    responses = await swarm.run_role(NodeRole.JUDGE, prompt)

    if not responses:
        logger.warning("No judge responses — claim stays uncertain")
        return ClaimStatus.UNCERTAIN

    # Use the first judge response
    verdict = _parse_judgment(responses[0].content)
    if verdict == "A":
        return ClaimStatus.VERIFIED
    if verdict == "B":
        return ClaimStatus.DISCARDED
    return ClaimStatus.UNCERTAIN


# ------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _parse_judgment(text: str) -> str:
    """Extract the ``winner`` field from judge LLM output."""
    try:
        m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
        raw = m.group(1) if m and m.lastindex else (m.group(0) if m else text)
        data = json.loads(raw)
        return str(data.get("winner", "")).upper()
    except Exception:
        return ""
