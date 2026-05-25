"""NodeSerializer — converts graph nodes to numbered CLAIM lines.

Output format (one claim per line, globally numbered so the SLM can
reference them explicitly — e.g. "segun CLAIM #3..."):

    CLAIM #1: Einstein → desarrolló → Teoría de la Relatividad [conf:0.94, año:1905]
    CLAIM #2: Teoría de la Relatividad → publicada_en → Annalen der Physik [conf:0.88]
    CLAIM #3: Cooking [concepto] [conf:0.90]

This structured, citable format lets small models (llama 1b/3b) anchor
their answer to specific claims instead of hallucinating. The extractive
CHAT_SYSTEM_PROMPT assumes this exact numbering to work.
"""

from __future__ import annotations

from core.config_loader import Config
from core.models import GraphNode
from graph.base import GraphPort


# ``"CLAIM #123: "`` prefix is ~12 chars ≈ 4 tokens. Budget accounting
# adds this per-claim so token limits stay accurate after numbering.
_CLAIM_PREFIX_TOKENS = 4


class NodeSerializer:
    """Serialize graph nodes into budget-constrained numbered CLAIM lines."""

    def __init__(self, graph: GraphPort, config: Config) -> None:
        self.graph = graph
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply_budget(
        self,
        nodes: list[GraphNode],
        budget: int | None = None,
    ) -> list[GraphNode]:
        """Select as many *nodes* as fit within *budget* tokens."""
        budget = budget if budget is not None else self.config.retrieval.token_budget
        selected: list[GraphNode] = []
        total_tokens = 0

        for node in nodes:
            raw_lines = await self._render_node_claims(node)
            body_tokens = _estimate_tokens("\n".join(raw_lines))
            prefix_tokens = _CLAIM_PREFIX_TOKENS * len(raw_lines)
            node_tokens = body_tokens + prefix_tokens
            if total_tokens + node_tokens > budget:
                break
            selected.append(node)
            total_tokens += node_tokens

        return selected

    async def serialize(self, nodes: list[GraphNode]) -> str:
        """Render *nodes* (already budget-filtered) to numbered CLAIM lines."""
        lines: list[str] = []
        counter = 1
        for node in nodes:
            for raw in await self._render_node_claims(node):
                lines.append(f"CLAIM #{counter}: {raw}")
                counter += 1
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _render_node_claims(self, node: GraphNode) -> list[str]:
        """Return one raw claim line per relation (or a single node-typed
        line if the node has no relations). CLAIM numbering is applied
        later in :meth:`serialize`, so lines here have no prefix."""
        relations = await self.graph.get_relations_for_node(node.node_id)

        auth = node.source_authority if node.source_authority != "slm_single" else ""

        if not relations:
            meta = [f"conf:{node.confidence:.2f}"]
            if auth:
                meta.append(f"auth:{auth}")
            return [
                f"{node.name} [{node.node_type}] [{', '.join(meta)}]"
            ]

        lines: list[str] = []
        for rel in relations:
            core = f"{node.name} → {rel['predicate']} → {rel['target_name']}"
            meta: list[str] = []
            if rel.get("confidence") is not None:
                meta.append(f"conf:{rel['confidence']:.2f}")
            date = rel.get("date")
            if date and date.strip():
                meta.append(f"año:{date}")
            if rel.get("contested"):
                meta.append("contested")
            if auth:
                meta.append(f"auth:{auth}")
            if meta:
                core += f" [{', '.join(meta)}]"
            lines.append(core)
        return lines


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token (good enough for budget gating)."""
    return len(text) // 4 + 1
