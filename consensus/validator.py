"""Cross-validation — Phase 2 of the consensus pipeline.

Each claim is validated by K randomly selected critic nodes (NxK, not NxN).
Validation calls go through the swarm's semaphore to avoid overloading
Ollama with too many concurrent requests.
"""

from __future__ import annotations

import asyncio
import json
import random
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, GraphContext, NodeResponse, NodeRole, Vote
from swarm.node import SwarmNode
from swarm.pool import SwarmPool
from swarm.roles import ROLE_SYSTEM_PROMPTS


def build_critic_prompt(
    query: str, context: GraphContext, claim: Claim
) -> str:
    """Build the user-prompt sent to a critic node for one claim.

    Uses Evidence-Gated verification: the critic must check that the
    ``source_text`` actually supports the subject→predicate→object triple,
    not just whether the claim "seems correct".
    """
    source = claim.source_text or "N/A"
    return (
        f"PREGUNTA ORIGINAL: {query}\n\n"
        f"CONTEXTO:\n{context.serialized_context}\n\n"
        f"CLAIM A VALIDAR:\n"
        f"  Sujeto: {claim.subject}\n"
        f"  Predicado: {claim.predicate}\n"
        f"  Objeto: {claim.object}\n"
        f"  Texto fuente citado: \"{source}\"\n\n"
        "INSTRUCCIONES DE VERIFICACIÓN:\n"
        "1. ¿El texto fuente citado REALMENTE dice que "
        f"'{claim.subject}' → '{claim.predicate}' → '{claim.object}'?\n"
        "2. Si el texto fuente es 'N/A' o no respalda la tripleta, vota FALSE.\n"
        "3. Si el texto fuente SÍ respalda la tripleta, vota TRUE.\n\n"
        "Responde ÚNICAMENTE en JSON: "
        '{\"vote\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}'
    )


def select_validators(
    swarm: SwarmPool,
    k: int,
    exclude_model: str = "",
) -> list[SwarmNode]:
    """Pick up to *k* critic nodes, excluding the claim's source model.

    Only uses actual critic-role nodes — never drafts extractors or other
    roles, since they use the wrong model and add unnecessary Ollama load.
    """
    critics = [
        n
        for n in swarm.nodes
        if n.role == NodeRole.CRITIC and n.node_id != exclude_model
    ]
    return random.sample(critics, min(k, len(critics)))


async def run_validations(
    claims: list[Claim],
    query: str,
    context: GraphContext,
    swarm: SwarmPool,
    config: Config,
) -> dict[str, list[Vote]]:
    """Validate all claims in parallel (NxK). Returns ``{claim_id: [Vote]}``.

    All calls are throttled through the swarm's semaphore to prevent
    overwhelming Ollama.
    """
    k = config.consensus.validators_per_claim
    tasks: list[asyncio.Task] = []
    task_meta: list[tuple[str, str]] = []  # (claim_id, validator_node_id)

    for claim in claims:
        validators = select_validators(swarm, k, exclude_model=claim.source_model)
        prompt = build_critic_prompt(query, context, claim)
        sys_prompt = ROLE_SYSTEM_PROMPTS.get(NodeRole.CRITIC)

        for validator in validators:
            coro = _validate_one(
                validator, prompt, sys_prompt, swarm.config, swarm.semaphore
            )
            tasks.append(asyncio.ensure_future(coro))
            task_meta.append((claim.claim_id, validator.node_id))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    votes: dict[str, list[Vote]] = {c.claim_id: [] for c in claims}
    for (claim_id, node_id), result in zip(task_meta, raw_results):
        if isinstance(result, Vote):
            votes[claim_id].append(result)
        elif isinstance(result, Exception):
            logger.warning(f"Validator {node_id} failed for {claim_id[:8]}: {result}")

    return votes


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

async def _validate_one(
    node: SwarmNode,
    prompt: str,
    system_prompt: str | None,
    config,
    semaphore: asyncio.Semaphore,
) -> Vote:
    """Run a single validation call, throttled by *semaphore*."""
    async with semaphore:
        response = await node.provider.complete(
            prompt=prompt,
            model=node.model,
            temperature=node.temperature,
            timeout=config.swarm.request_timeout,
            system_prompt=system_prompt,
        )
    return _parse_vote(response.content, node.node_id, "")


def _parse_vote(text: str, node_id: str, claim_id: str) -> Vote:
    """Parse a Vote from critic LLM output."""
    try:
        data = _extract_json(text)
        return Vote(
            node_id=node_id,
            claim_id=claim_id,
            vote=bool(data.get("vote", False)),
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason"),
        )
    except Exception:
        return Vote(
            node_id=node_id,
            claim_id=claim_id,
            vote=False,
            confidence=0.3,
            reason="parse_error",
        )


# ------------------------------------------------------------------
# Robust JSON extraction
# ------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction tolerant of SLM quirks."""
    m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
    raw = m.group(1) if m and m.lastindex else (m.group(0) if m else text)
    return json.loads(_fix_json(raw))


def _fix_json(text: str) -> str:
    """Fix common JSON errors from small language models."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove trailing content after last } or ]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in "}]":
            text = text[: i + 1]
            break
    return text
