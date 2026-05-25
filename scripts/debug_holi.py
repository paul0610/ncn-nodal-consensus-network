"""Debug script: inspect Holi nodes and diagnose cross-contamination.

Usage:
    python -m scripts.debug_holi
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

from core.config_loader import Config, load_config
from graph.kuzu_adapter import KuzuAdapter
from retrieval.embedder import Embedder


async def main() -> None:
    config = (
        load_config("config.yaml") if Path("config.yaml").exists() else Config()
    )
    graph = KuzuAdapter(config)
    await graph.initialize()

    all_nodes = await graph.get_all_nodes(limit=1000)
    print(f"\n== GRAPH TOTALS ==")
    print(f"  total nodes: {len(all_nodes)}")

    # By authority
    auth_count = Counter(n.source_authority for n in all_nodes)
    print(f"\n== BY AUTHORITY ==")
    for auth, cnt in auth_count.most_common():
        print(f"  {auth:14}: {cnt}")

    # By namespace
    ns_count = Counter(n.namespace for n in all_nodes)
    print(f"\n== BY NAMESPACE ==")
    for ns, cnt in ns_count.most_common():
        print(f"  {ns:20}: {cnt}")

    # Holi nodes
    holi_nodes = [n for n in all_nodes if "holi" in n.name.lower()]
    print(f"\n== HOLI-RELATED NODES ({len(holi_nodes)}) ==")
    for n in holi_nodes:
        print(
            f"  [{n.source_authority:14}] {n.name!r:40} "
            f"type={n.node_type}  ns={n.namespace}  conf={n.confidence:.2f}"
        )
        if n.source:
            safe = n.source[:80].encode("ascii", errors="replace").decode()
            print(f"      source: {safe}")

    # Now the key experiment: compute cosine similarity between
    # "ornostocaustico" and all Holi nodes
    print(f"\n== RELEVANCE TEST: 'ornostocaustico' vs Holi nodes ==")
    embedder = Embedder(config)
    query_vec = await embedder.embed("Dime que es un ornostocaustico")

    import numpy as np
    qv = np.array(query_vec, dtype=np.float32)
    qv_norm = np.linalg.norm(qv)

    scored: list[tuple[float, str, str]] = []
    for n in all_nodes:
        if not n.embedding:
            continue
        nv = np.array(n.embedding, dtype=np.float32)
        denom = qv_norm * np.linalg.norm(nv)
        if denom == 0:
            continue
        sim = float(np.dot(qv, nv) / denom)
        scored.append((sim, n.name, n.source_authority))

    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"\n  TOP 10 most similar nodes to 'ornostocaustico':")
    for sim, name, auth in scored[:10]:
        safe_name = name.encode("ascii", errors="replace").decode()
        flag = "  << holi" if "holi" in name.lower() else ""
        print(f"    {sim:.4f}  [{auth:13}] {safe_name}{flag}")

    threshold = config.retrieval.relevance_threshold
    passed = [s for s in scored if s[0] >= threshold]
    print(f"\n  Relevance threshold: {threshold}")
    print(f"  Nodes above threshold: {len(passed)}/{len(scored)}")

    await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
