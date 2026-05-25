"""Classifier — classifies entities into existing ontology types.

Uses a small LLM call to decide the best-fitting type among the
currently known types.  Returns ``(type_name, confidence)``.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, NodeRole
from swarm.pool import SwarmPool


async def classify_with_existing(
    entity_name: str,
    context: Claim,
    known_types: set[str],
    swarm: SwarmPool,
) -> tuple[str, float]:
    """Ask an extractor to pick the best existing type for *entity_name*.

    Returns ``(best_type, confidence)``.  Falls back to
    ``("concepto", 0.0)`` when the model can't decide.
    """
    if not known_types:
        return "concepto", 0.0

    types_list = ", ".join(sorted(known_types))
    prompt = (
        f'Clasifica la entidad "{entity_name}" en UNO de estos tipos existentes:\n'
        f"  [{types_list}]\n\n"
        f"Contexto: {entity_name} → {context.predicate} → {context.object}\n\n"
        "Responde ÚNICAMENTE con JSON:\n"
        '{"type": "<tipo elegido>", "confidence": <0.0-1.0>}'
    )

    responses = await swarm.run_role(NodeRole.EXTRACTOR, prompt)
    if not responses:
        return "concepto", 0.0

    return _parse_classification(responses[0].content, known_types)


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*?\}")


def _parse_classification(
    text: str, known_types: set[str]
) -> tuple[str, float]:
    if not text or not text.strip():
        return "concepto", 0.0

    # Try JSON parsing first
    try:
        m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
        if m:
            raw = m.group(1) if m.lastindex else m.group(0)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)  # fix trailing commas
            data = json.loads(raw)
            chosen = str(data.get("type", "concepto")).strip().lower()
            conf = float(data.get("confidence", 0.0))

            if chosen in known_types:
                return chosen, conf

            for kt in known_types:
                if kt in chosen or chosen in kt:
                    return kt, conf * 0.9
    except Exception:
        pass

    # Heuristic fallback: look for a known type mentioned in the text
    text_lower = text.lower()
    for kt in known_types:
        if kt in text_lower:
            return kt, 0.5

    return "concepto", 0.0
