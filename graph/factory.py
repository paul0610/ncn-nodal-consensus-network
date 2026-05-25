"""Graph adapter factory — returns the right GraphPort implementation."""

from __future__ import annotations

from core.config_loader import Config
from core.exceptions import ConfigValidationError
from graph.base import GraphPort


async def create_graph_adapter(config: Config) -> GraphPort:
    """Instantiate and initialise the adapter declared in *config.graph.adapter*."""
    name = config.graph.adapter.lower()

    if name == "kuzu":
        from graph.kuzu_adapter import KuzuAdapter

        adapter = KuzuAdapter(config)
    else:
        raise ConfigValidationError(
            f"Graph adapter '{name}' not recognised. Available: kuzu"
        )

    await adapter.initialize()
    return adapter
