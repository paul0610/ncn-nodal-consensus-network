"""SwarmNode — a single node in the swarm (model + role)."""

from __future__ import annotations

from dataclasses import dataclass, field

from providers.base import ModelProviderPort
from core.models import NodeRole


@dataclass
class SwarmNode:
    """One model instance with an assigned role inside the swarm."""

    node_id: str
    provider: ModelProviderPort
    model: str
    role: NodeRole
    temperature: float
    reputation_score: float = field(default=1.0)
