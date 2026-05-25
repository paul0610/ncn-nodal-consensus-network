"""Claim extraction — Phase 1 of the consensus pipeline.

Parses structured JSON claims from extractor-role LLM responses.
Tolerant of the malformed JSON that small models often produce
(trailing commas, truncated arrays, fenced code blocks).
"""

from __future__ import annotations

import json
import re

from loguru import logger

from core.config_loader import Config
from core.models import Claim, ClaimStatus, GraphContext, NodeResponse


def build_extractor_prompt(query: str, context: GraphContext) -> str:
    """Build the user-prompt sent to extractor nodes."""
    parts = [f"PREGUNTA DEL USUARIO: {query}"]
    if context.serialized_context:
        parts.append(f"\nCONTEXTO DEL GRAFO:\n{context.serialized_context}")
    parts.append(
        "\nExtrae los claims factuales relevantes del texto y contexto anteriores. "
        "Máximo 5 claims, los más importantes. "
        "Responde ÚNICAMENTE con JSON válido, sin texto antes ni después."
    )
    return "\n".join(parts)


def parse_claims(responses: list[NodeResponse]) -> list[Claim]:
    """Parse ``Claim`` objects from raw extractor responses."""
    claims: list[Claim] = []
    for resp in responses:
        try:
            data = _extract_json(resp.content)
            raw_claims = data.get("claims", [])
            for rc in raw_claims:
                subject = str(rc.get("subject", "")).strip()
                predicate = str(rc.get("predicate", "")).strip()
                obj = str(rc.get("object", "")).strip()
                if not subject or not predicate or not obj:
                    continue
                claims.append(
                    Claim(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=_safe_float(rc.get("confidence"), 0.5),
                        temporal=bool(rc.get("temporal", False)),
                        date=rc.get("date") if rc.get("date") else None,
                        source_text=rc.get("source_text"),
                        source_model=resp.node_id,
                        status=ClaimStatus.PENDING,
                    )
                )
        except Exception as exc:
            logger.warning(
                f"Could not parse claims from {resp.node_id}: {exc}"
            )
    return claims


# ------------------------------------------------------------------
# Robust JSON extraction
# ------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction tolerant of SLM quirks."""
    # Try fenced code block first
    m = _JSON_BLOCK.search(text)
    if m:
        return json.loads(_fix_json(m.group(1)))
    # Try raw JSON object
    m = _JSON_OBJ.search(text)
    if m:
        return json.loads(_fix_json(m.group(0)))
    # Last resort
    return json.loads(_fix_json(text))


def _fix_json(text: str) -> str:
    """Fix common JSON errors from small language models."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Try to close truncated arrays/objects
    opens = text.count("{") + text.count("[")
    closes = text.count("}") + text.count("]")
    if opens > closes:
        # Find the outermost structure and try to close it
        bracket_count = text.count("[") - text.count("]")
        brace_count = text.count("{") - text.count("}")
        text += "]" * max(0, bracket_count)
        text += "}" * max(0, brace_count)
    # Remove trailing content after last } or ]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in "}]":
            text = text[: i + 1]
            break
    return text


def _safe_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
