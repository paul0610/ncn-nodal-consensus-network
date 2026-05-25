"""Re-embed all graph nodes with the current embedding model.

Use this after changing the embedding model in config.yaml to migrate
existing node embeddings to the new vector space.

Usage:
    python -m scripts.reembed_nodes [--dry-run]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loguru import logger

from core.config_loader import Config, load_config
from graph.kuzu_adapter import KuzuAdapter
from retrieval.embedder import Embedder


async def main(dry_run: bool = False) -> None:
    config = (
        load_config("config.yaml") if Path("config.yaml").exists() else Config()
    )

    print(f"\n== RE-EMBEDDING MIGRATION ==")
    print(f"  model: {config.retrieval.model}")
    print(f"  dry_run: {dry_run}")

    graph = KuzuAdapter(config)
    await graph.initialize()
    embedder = Embedder(config)

    all_nodes = await graph.get_all_nodes(limit=5000)
    print(f"  nodes to re-embed: {len(all_nodes)}")

    if not all_nodes:
        print("  No nodes found — nothing to do.")
        await graph.close()
        return

    # Embed all node names in batch
    names = [n.name for n in all_nodes]
    print(f"\n  Embedding {len(names)} texts with '{config.retrieval.model}'...")
    vectors = await embedder.embed_batch(names)

    if len(vectors) != len(all_nodes):
        print(f"  ERROR: expected {len(all_nodes)} vectors, got {len(vectors)}")
        await graph.close()
        return

    print(f"  Got {len(vectors)} vectors (dim={len(vectors[0])})")

    if dry_run:
        print("\n  DRY RUN — no changes written. Remove --dry-run to apply.")
        await graph.close()
        return

    # Write new embeddings back to graph
    updated = 0
    failed = 0
    for node, vector in zip(all_nodes, vectors):
        try:
            await graph.update_node(node.node_id, {"embedding": vector})
            updated += 1
        except Exception as exc:
            logger.warning(f"Failed to update {node.node_id} ({node.name}): {exc}")
            failed += 1

    print(f"\n  UPDATED: {updated}")
    if failed:
        print(f"  FAILED:  {failed}")

    await graph.close()
    print(f"\n  Migration complete! All nodes now use '{config.retrieval.model}' embeddings.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry))
