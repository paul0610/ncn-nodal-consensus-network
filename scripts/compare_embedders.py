"""Compare multilingual embedders on noise floor vs signal strength.

Tests 3 candidate models against real texts from the graph:
  1. all-MiniLM-L6-v2              (current, English-first)
  2. paraphrase-multilingual-MiniLM-L12-v2  (multilingual drop-in, 384d)
  3. intfloat/multilingual-e5-small (SOTA multilingual, 384d)

For each model it measures:
  - Noise floor:  max similarity of a nonsense query vs all texts
  - Signal:       max similarity of a real query vs a known-relevant text
  - Discrimination gap: signal - noise_floor
                   (higher = better semantic discrimination in Spanish)

Usage:
    python -m scripts.compare_embedders

Notes:
  - First run downloads ~500 MB of models (cached by sentence-transformers).
  - Read-only access to the graph, no changes made.
  - e5 models require "query: " / "passage: " prefixes — handled automatically.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config_loader import Config, load_config
from graph.kuzu_adapter import KuzuAdapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS: list[dict[str, Any]] = [
    {
        "name": "all-MiniLM-L6-v2",
        "alias": "current",
        "prefix_query": "",
        "prefix_doc": "",
    },
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "alias": "mpnet-multi",
        "prefix_query": "",
        "prefix_doc": "",
    },
    {
        "name": "intfloat/multilingual-e5-small",
        "alias": "e5-small",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
    },
]

# Queries to test the embedder with
NOISE_QUERIES = [
    "Dime que es un ornostocaustico",
    "explica el zarcamulaxtico",
    "asdfghjkl",
]

SIGNAL_QUERIES = [
    ("que es DeepSeek V3", "deepseek"),      # expect match with DeepSeek content
    ("que es el festival Holi", "holi"),      # expect match with Holi content
    ("que es OpenClaw", "openclaw"),          # expect match with OpenClaw content
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed(model: SentenceTransformer, text: str, prefix: str = "") -> np.ndarray:
    vec = model.encode(prefix + text, convert_to_numpy=True, normalize_embeddings=False)
    return vec.astype(np.float32)


def pick_relevant_text(texts: list[str], keyword: str) -> str | None:
    """Return the first text that contains *keyword* (case-insensitive)."""
    keyword = keyword.lower()
    for t in texts:
        if keyword in t.lower():
            return t
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    random.seed(42)

    config = (
        load_config("config.yaml") if Path("config.yaml").exists() else Config()
    )
    graph = KuzuAdapter(config)
    await graph.initialize()

    all_nodes = await graph.get_all_nodes(limit=1000)
    print(f"\n== LOADED GRAPH ==")
    print(f"  {len(all_nodes)} nodes")

    # Collect diverse texts (node names) — cap at 80 for speed
    texts = [n.name for n in all_nodes if n.name]
    if len(texts) > 80:
        texts = random.sample(texts, 80)
    print(f"  sampled {len(texts)} texts for comparison")
    await graph.close()

    # Check that the relevant texts exist for signal queries
    print(f"\n== SIGNAL QUERY VALIDATION ==")
    valid_signal_queries: list[tuple[str, str]] = []
    for query, keyword in SIGNAL_QUERIES:
        target = pick_relevant_text(texts, keyword)
        if target is None:
            print(f"  WARNING: no text with '{keyword}' in sample — skipping '{query}'")
        else:
            print(f"  '{query}'  →  target: '{target[:60]}'")
            valid_signal_queries.append((query, keyword))

    # Compare each model
    print("\n" + "=" * 78)
    print("COMPARATIVE ANALYSIS")
    print("=" * 78)

    summary: list[dict[str, Any]] = []

    for model_cfg in MODELS:
        print(f"\n▼ MODEL: {model_cfg['name']}  (alias: {model_cfg['alias']})")
        print("-" * 78)

        try:
            model = SentenceTransformer(model_cfg["name"])
        except Exception as exc:
            print(f"  FAILED to load: {exc}")
            continue

        # Embed all sampled texts once
        doc_prefix = model_cfg["prefix_doc"]
        query_prefix = model_cfg["prefix_query"]
        text_vecs = np.stack([embed(model, t, doc_prefix) for t in texts])

        # ---- Noise floor ----
        noise_maxes: list[float] = []
        for nq in NOISE_QUERIES:
            qv = embed(model, nq, query_prefix)
            sims = [cos_sim(qv, tv) for tv in text_vecs]
            noise_maxes.append(max(sims))
            top3_idx = np.argsort(sims)[::-1][:3]
            print(
                f"  NOISE '{nq[:30]:30}' → max={max(sims):.3f}  "
                f"top3: {[round(sims[i], 3) for i in top3_idx]}"
            )
        avg_noise = sum(noise_maxes) / len(noise_maxes)

        # ---- Signal strength ----
        signal_maxes: list[float] = []
        for query, keyword in valid_signal_queries:
            target_text = pick_relevant_text(texts, keyword)
            if not target_text:
                continue
            qv = embed(model, query, query_prefix)
            tv = embed(model, target_text, doc_prefix)
            sim = cos_sim(qv, tv)
            signal_maxes.append(sim)
            print(
                f"  SIGNAL '{query[:30]:30}' → sim={sim:.3f}  "
                f"vs '{target_text[:40]}'"
            )
        avg_signal = sum(signal_maxes) / len(signal_maxes) if signal_maxes else 0.0

        gap = avg_signal - avg_noise
        print(f"  ▸ noise_floor_avg={avg_noise:.3f}")
        print(f"  ▸ signal_avg     ={avg_signal:.3f}")
        print(f"  ▸ discrimination_gap={gap:.3f}  (higher = better)")

        summary.append({
            "model": model_cfg["alias"],
            "noise": avg_noise,
            "signal": avg_signal,
            "gap": gap,
        })

    # ---- Summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"{'model':<25} {'noise_floor':>12} {'signal':>10} "
        f"{'gap':>8} {'verdict':>20}"
    )
    print("-" * 78)
    if summary:
        best_gap = max(s["gap"] for s in summary)
        for s in summary:
            verdict = "⭐ BEST" if s["gap"] == best_gap else ""
            print(
                f"{s['model']:<25} {s['noise']:>12.3f} {s['signal']:>10.3f} "
                f"{s['gap']:>8.3f} {verdict:>20}"
            )

    print("\n  Interpretation:")
    print("    - noise_floor: lower = less false positives for nonsense queries")
    print("    - signal:      higher = stronger match for real queries")
    print("    - gap:         higher = better semantic discrimination")


if __name__ == "__main__":
    asyncio.run(main())
