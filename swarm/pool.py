"""SwarmPool — manages N swarm nodes with asyncio concurrency control."""

from __future__ import annotations

import asyncio

from loguru import logger

from core.config_loader import Config
from core.models import NodeResponse
from providers.factory import ProviderFactory
from swarm.node import SwarmNode
from swarm.roles import ROLE_SYSTEM_PROMPTS, NodeRole


class SwarmPool:
    """Run model nodes in parallel, throttled by a semaphore."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.nodes: list[SwarmNode] = self._build_nodes(config)
        self.semaphore = asyncio.Semaphore(config.swarm.max_concurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_role(
        self,
        role: NodeRole,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[NodeResponse]:
        """Launch all nodes with *role* in parallel and collect results."""
        role_nodes = [n for n in self.nodes if n.role == role]
        if not role_nodes:
            logger.warning(f"No nodes with role {role.value}")
            return []

        sys_prompt = system_prompt or ROLE_SYSTEM_PROMPTS.get(role)
        tasks = [self._run_node(node, prompt, sys_prompt) for node in role_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful: list[NodeResponse] = []
        for r in results:
            if isinstance(r, NodeResponse):
                successful.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Swarm node failed: {r}")
        return successful

    async def run_all(self, prompt: str) -> list[NodeResponse]:
        """Launch **every** node (all roles) in parallel."""
        tasks = [
            self._run_node(node, prompt, ROLE_SYSTEM_PROMPTS.get(node.role))
            for node in self.nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, NodeResponse)]

    def get_node(self, node_id: str) -> SwarmNode | None:
        """Look up a node by id."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_node(
        self,
        node: SwarmNode,
        prompt: str,
        system_prompt: str | None = None,
    ) -> NodeResponse:
        async with self.semaphore:
            try:
                response = await node.provider.complete(
                    prompt=prompt,
                    model=node.model,
                    temperature=node.temperature,
                    timeout=self.config.swarm.request_timeout,
                    system_prompt=system_prompt,
                )
                return NodeResponse(
                    node_id=node.node_id,
                    role=node.role.value,
                    content=response.content,
                    model=node.model,
                    provider=node.provider.name,
                )
            except Exception as exc:
                logger.error(f"Node {node.node_id} failed: {exc}")
                raise

    def _build_nodes(self, config: Config) -> list[SwarmNode]:
        """Construct nodes from *config.yaml* swarm section."""
        nodes: list[SwarmNode] = []
        factory = ProviderFactory(config)
        for nc in config.swarm.nodes:
            provider = factory.create(nc.provider)
            for i in range(nc.count):
                nodes.append(
                    SwarmNode(
                        node_id=f"{nc.model}-{nc.role}-{i}",
                        provider=provider,
                        model=nc.model,
                        role=NodeRole(nc.role),
                        temperature=nc.temperature or config.swarm.temperature,
                    )
                )
        return nodes
