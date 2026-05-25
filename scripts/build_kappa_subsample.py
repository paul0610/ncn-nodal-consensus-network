"""Build the Cohen's kappa subsample (paper §3.7).

Select 10 queries (balanced across tiers + categories) × 5 SLMs × 2 conditions
(A graph, C strong baseline budget-400) = 100 cells.

Output: paper_evidence/kappa_validation/subsample.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper_evidence" / "kappa_validation" / "subsample.json"

MODELS = ["qwen2_5_0_5b", "gemma3_1b", "llama3_2_1b", "phi3_5", "llama3_2_3b"]

# 10 balanced queries with their (tier, expected, source dirs for A and C)
QUERY_SPEC = [
    # WHO Tier 4 (3 queries — factoid, factoid+parametric-risk, OOS)
    {
        "qkey": "Q3_EN_atomic_A",
        "tier": "tier_4_who",
        "scope": "factoid",
        "expected_short": "Airway",
        "a_source": "tier_4_baseline",
        "c_source": "tier_C_strong_baseline/tier_4_who",
    },
    {
        "qkey": "Q4_ES_atomic_C",
        "tier": "tier_4_who",
        "scope": "factoid+parametric-risk",
        "expected_short": "Circulation",
        "a_source": "tier_4_baseline",
        "c_source": "tier_C_strong_baseline/tier_4_who",
    },
    {
        "qkey": "Q6_ES_adversarial",
        "tier": "tier_4_who",
        "scope": "OOS",
        "expected_short": "ABSTAIN (OOS — World Cup not in corpus)",
        "a_source": "tier_4_baseline",
        "c_source": "tier_C_strong_baseline/tier_4_who",
    },
    # WHO Tier 5 (3 queries — M1 factoid, negation trap, OOS)
    {
        "qkey": "A2_EN_glucose_threshold",
        "tier": "tier_5_who",
        "scope": "M1 factoid",
        "expected_short": "<3.5 mmol/L",
        "a_source": "tier_5_adversarial",
        "c_source": "tier_C_strong_baseline/tier_5_who",
    },
    {
        "qkey": "A12_EN_trauma_airway_avoid",
        "tier": "tier_5_who",
        "scope": "negation trap",
        "expected_short": "head-tilt/chin-lift",
        "a_source": "tier_5_adversarial",
        "c_source": "tier_C_strong_baseline/tier_5_who",
    },
    {
        "qkey": "A18_ES_capital_australia",
        "tier": "tier_5_who",
        "scope": "OOS",
        "expected_short": "ABSTAIN (OOS — Australia not in corpus)",
        "a_source": "tier_5_adversarial",
        "c_source": "tier_C_strong_baseline/tier_5_who",
    },
    # FAA (4 queries — factoid clear, reranker paradox, specific, OOS)
    {
        "qkey": "F4_EN_food_max_days",
        "tier": "tier_4_5_faa",
        "scope": "M1 factoid",
        "expected_short": "30 days",
        "a_source": "tier_4_5_faa_combined",
        "c_source": "tier_C_strong_baseline/tier_4_5_faa",
    },
    {
        "qkey": "F8_EN_elt_test_time",
        "tier": "tier_4_5_faa",
        "scope": "reranker paradox (quiz chunk)",
        "expected_short": "first 5 minutes of every hour",
        "a_source": "tier_4_5_faa_combined",
        "c_source": "tier_C_strong_baseline/tier_4_5_faa",
    },
    {
        "qkey": "F11_EN_shock_feet_elevation",
        "tier": "tier_4_5_faa",
        "scope": "specific factoid",
        "expected_short": "12 inches",
        "a_source": "tier_4_5_faa_combined",
        "c_source": "tier_C_strong_baseline/tier_4_5_faa",
    },
    {
        "qkey": "F17_ES_internet",
        "tier": "tier_4_5_faa",
        "scope": "OOS",
        "expected_short": "ABSTAIN (OOS — Internet definition not in corpus)",
        "a_source": "tier_4_5_faa_combined",
        "c_source": "tier_C_strong_baseline/tier_4_5_faa",
    },
]


def truncate(s: str, n: int = 500) -> str:
    """Truncate long answers; preserve head + tail for context."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    head = s[: n - 100]
    tail = s[-50:]
    return f"{head}... [truncated total={len(s)} chars] ...{tail}"


def load_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    cells: list[dict] = []

    for q in QUERY_SPEC:
        for model in MODELS:
            # A graph
            a_path = REPO / "paper_evidence" / "iterations" / q["a_source"] / f"eval_{model}.json"
            a_data = load_eval(a_path)
            a_entry = a_data.get("A_graph", {}).get(q["qkey"])
            if a_entry:
                cells.append({
                    "cell_id": f"{q['qkey']}__{model}__A_graph",
                    "qkey": q["qkey"],
                    "scope": q["scope"],
                    "model": model,
                    "condition": "A_graph",
                    "query": a_entry.get("query", ""),
                    "expected_short": q["expected_short"],
                    "answer": truncate(a_entry.get("answer", "")),
                    "answer_chars": len(a_entry.get("answer", "")),
                })
            # C budget-400
            c_path = REPO / "paper_evidence" / "iterations" / q["c_source"] / f"eval_{model}.json"
            c_data = load_eval(c_path)
            c_entry = c_data.get("C_budget_400", {}).get(q["qkey"])
            if c_entry:
                cells.append({
                    "cell_id": f"{q['qkey']}__{model}__C_budget_400",
                    "qkey": q["qkey"],
                    "scope": q["scope"],
                    "model": model,
                    "condition": "C_budget_400",
                    "query": c_entry.get("query", ""),
                    "expected_short": q["expected_short"],
                    "answer": truncate(c_entry.get("answer", "")),
                    "answer_chars": len(c_entry.get("answer", "")),
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": len(QUERY_SPEC),
        "n_models": len(MODELS),
        "n_conditions": 2,
        "n_cells_expected": len(QUERY_SPEC) * len(MODELS) * 2,
        "n_cells_actual": len(cells),
        "label_categories": ["correct", "partial", "wrong", "abstain_correct"],
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO)}")
    print(f"  cells: {len(cells)}/{payload['n_cells_expected']}")
    if len(cells) < payload['n_cells_expected']:
        print(f"  WARN: {payload['n_cells_expected'] - len(cells)} cells missing")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
