"""Export a namespace-scoped subgraph from Kuzu to a mobile-friendly SQLite DB.

Produces an output bundle designed for React Native / llama.rn integration:

    output_dir/
      ├── graph.sqlite         # nodes + edges (flat schema, query-friendly)
      ├── embeddings.bin       # contiguous float32 matrix of node embeddings
      ├── manifest.json        # metadata: namespace, counts, schema version
      └── README.md            # how to consume this bundle

Usage:
    python -m scripts.export_namespace --namespace survival --output app_bundle/
    python -m scripts.export_namespace --namespace general --output everything/
    python -m scripts.export_namespace --list                 # list namespaces only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import struct
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.config_loader import Config, load_config
from graph.kuzu_adapter import KuzuAdapter


SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# SQLite schema (mobile-friendly, no Kuzu-specifics)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id          TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    node_type        TEXT NOT NULL,
    namespace        TEXT NOT NULL,
    confidence       REAL NOT NULL,
    temperature      REAL NOT NULL,
    embedding_idx    INTEGER NOT NULL,
    source           TEXT,
    source_authority TEXT,
    uncertain        INTEGER NOT NULL DEFAULT 0,
    metadata_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_namespace ON nodes(namespace);

CREATE TABLE IF NOT EXISTS edges (
    source_id        TEXT NOT NULL,
    target_id        TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    confidence       REAL NOT NULL,
    date             TEXT,
    contested        INTEGER NOT NULL DEFAULT 0,
    source_authority TEXT,
    PRIMARY KEY (source_id, target_id, predicate),
    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_predicate ON edges(predicate);
"""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

async def list_namespaces(graph: KuzuAdapter) -> None:
    all_nodes = await graph.get_all_nodes(limit=10000)
    ns_count = Counter(n.namespace for n in all_nodes)
    print(f"\n== Namespaces in graph ({len(all_nodes)} total nodes) ==")
    for ns, count in ns_count.most_common():
        print(f"  {ns:20} {count:6} nodes")
    print()


async def export_namespace(
    graph: KuzuAdapter,
    namespace: str,
    output_dir: Path,
) -> dict:
    """Export a single namespace to the SQLite + embeddings bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch nodes in namespace
    if namespace == "*":
        nodes = await graph.get_all_nodes(limit=100000)
    else:
        nodes = await graph.find_nodes_by_namespace(namespace, limit=100000)

    if not nodes:
        print(f"  No nodes found in namespace '{namespace}' — nothing to export.")
        return {}

    print(f"  {len(nodes)} nodes in namespace '{namespace}'")

    # 2. Determine embedding dimension (from first node with embedding)
    embedding_dim = 0
    for n in nodes:
        if n.embedding:
            embedding_dim = len(n.embedding)
            break
    if embedding_dim == 0:
        print("  WARNING: no embeddings found — nodes will have empty vectors.")

    # 3. Write SQLite schema
    sqlite_path = output_dir / "graph.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(sqlite_path)
    conn.executescript(SCHEMA_SQL)

    # 4. Build embedding matrix + insert nodes
    embeddings: list[np.ndarray] = []
    node_ids_in_scope: set[str] = set()

    for idx, node in enumerate(nodes):
        if node.embedding and len(node.embedding) == embedding_dim:
            vec = np.array(node.embedding, dtype=np.float32)
        else:
            vec = np.zeros(embedding_dim, dtype=np.float32) if embedding_dim else np.zeros(1, dtype=np.float32)
        embeddings.append(vec)
        node_ids_in_scope.add(node.node_id)

        conn.execute(
            """INSERT INTO nodes
               (node_id, name, node_type, namespace, confidence, temperature,
                embedding_idx, source, source_authority, uncertain, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.node_id,
                node.name,
                node.node_type,
                node.namespace,
                node.confidence,
                node.temperature,
                idx,
                node.source,
                node.source_authority,
                1 if node.uncertain else 0,
                json.dumps(node.metadata) if node.metadata else None,
            ),
        )

    # 5. Fetch relations — only keep those where BOTH endpoints are in scope
    edge_count = 0
    for node in nodes:
        try:
            relations = await graph.get_relations_for_node(node.node_id)
        except Exception as exc:
            print(f"    WARNING: failed to load relations for {node.name}: {exc}")
            continue

        for rel in relations:
            target_id = rel.get("target_node_id")
            if not target_id or target_id not in node_ids_in_scope:
                continue  # cross-namespace edge, skip

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO edges
                       (source_id, target_id, predicate, confidence, date,
                        contested, source_authority)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        node.node_id,
                        target_id,
                        rel.get("predicate", ""),
                        rel.get("confidence", 1.0),
                        rel.get("date"),
                        1 if rel.get("contested") else 0,
                        rel.get("source_authority", ""),
                    ),
                )
                edge_count += 1
            except Exception as exc:
                print(f"    WARNING: edge insert failed: {exc}")

    conn.commit()
    conn.close()

    # 6. Dump embeddings as packed float32 binary
    embeddings_path = output_dir / "embeddings.bin"
    if embeddings:
        matrix = np.stack(embeddings).astype(np.float32)
        matrix.tofile(embeddings_path)
    else:
        embeddings_path.write_bytes(b"")

    # 7. Manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "embedding_file": "embeddings.bin",
        "sqlite_file": "graph.sqlite",
        "exported_at": datetime.now(UTC).isoformat(),
        "node_types": sorted({n.node_type for n in nodes}),
        "source_authorities": dict(Counter(n.source_authority for n in nodes)),
        "avg_confidence": (
            sum(n.confidence for n in nodes) / len(nodes) if nodes else 0.0
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 8. README with consumption instructions
    readme_path = output_dir / "README.md"
    readme_path.write_text(_build_readme(manifest), encoding="utf-8")

    print(f"\n  [OK] Exported to {output_dir}")
    print(f"    - {sqlite_path.name}: {len(nodes)} nodes, {edge_count} edges")
    print(f"    - {embeddings_path.name}: {len(embeddings)} x {embedding_dim} float32")
    print(f"    - {manifest_path.name}")
    print(f"    - {readme_path.name}")

    return manifest


def _build_readme(manifest: dict) -> str:
    return f"""# NCN Namespace Export Bundle

**Namespace:** `{manifest['namespace']}`
**Exported:** {manifest['exported_at']}
**Schema version:** {manifest['schema_version']}

## Contents

| File | Purpose |
|---|---|
| `graph.sqlite` | Nodes + edges (SQLite 3, query with any SQL library) |
| `embeddings.bin` | Contiguous `float32` matrix of node embeddings |
| `manifest.json` | Metadata: counts, dimensions, authorities |

## Stats

- **Nodes:** {manifest['node_count']}
- **Edges:** {manifest['edge_count']}
- **Embedding dim:** {manifest['embedding_dim']} (float32)
- **Avg confidence:** {manifest['avg_confidence']:.2f}
- **Node types:** {', '.join(manifest['node_types'])}

## SQLite schema

```sql
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    name TEXT,
    node_type TEXT,
    namespace TEXT,
    confidence REAL,
    temperature REAL,
    embedding_idx INTEGER,     -- row index into embeddings.bin
    source TEXT,
    source_authority TEXT,
    uncertain INTEGER,
    metadata_json TEXT
);

CREATE TABLE edges (
    source_id TEXT,
    target_id TEXT,
    predicate TEXT,
    confidence REAL,
    date TEXT,
    contested INTEGER,
    source_authority TEXT,
    PRIMARY KEY (source_id, target_id, predicate)
);
```

## Reading embeddings (Python example)

```python
import numpy as np
import json

with open("manifest.json") as f:
    manifest = json.load(f)

dim = manifest["embedding_dim"]
emb = np.fromfile("embeddings.bin", dtype=np.float32).reshape(-1, dim)

# Row N of emb corresponds to the node with embedding_idx = N in SQLite
```

## Reading embeddings (React Native / Kotlin — pseudocode)

```kotlin
// Load binary file into ByteBuffer, read as float array of shape [nodeCount, dim]
val bytes = assets.open("embeddings.bin").readBytes()
val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
val floats = FloatArray(bytes.size / 4)
for (i in floats.indices) floats[i] = buffer.float

// Retrieve embedding for a given node:
// offset = embedding_idx * dim
// vector = floats.sliceArray(offset until offset + dim)
```

## Typical retrieval flow on mobile

```
user query
  → embed query (ONNX Runtime with MiniLM)
  → cosine similarity vs embeddings matrix
  → top-k node_ids
  → SQL: SELECT * FROM nodes WHERE node_id IN (...)
         SELECT * FROM edges WHERE source_id IN (...)
  → build context string
  → pass to local SLM (llama.rn)
  → response
```
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main(args) -> None:
    config = (
        load_config("config.yaml") if Path("config.yaml").exists() else Config()
    )
    graph = KuzuAdapter(config)
    await graph.initialize()

    if args.list:
        await list_namespaces(graph)
        await graph.close()
        return

    if not args.namespace or not args.output:
        print("ERROR: --namespace and --output are required (or use --list)")
        await graph.close()
        return

    output_dir = Path(args.output)
    print(f"\n== Exporting namespace '{args.namespace}' -> {output_dir} ==")
    await export_namespace(graph, args.namespace, output_dir)
    await graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=str, help="Namespace to export (use '*' for all)")
    parser.add_argument("--output", type=str, help="Output directory for the bundle")
    parser.add_argument("--list", action="store_true", help="List existing namespaces and exit")
    args = parser.parse_args()
    asyncio.run(main(args))
