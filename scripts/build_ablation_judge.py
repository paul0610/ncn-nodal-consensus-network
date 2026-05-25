"""Compact view for the reranker-ablation judge phase.

Per query: prints each model's answer in C1/C2/C3 in a single block,
truncated to 250 chars + length+latency metadata.
Also prints the C3 selected_sentence for inspection (key for paper).

Usage:
    python scripts/build_ablation_judge.py --out paper_evidence/iterations/tier_C_strong_baseline/ablation_reranker/judge_compact.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "ablation_reranker"

MODELS = ["qwen2_5_0_5b", "gemma3_1b", "llama3_2_1b", "phi3_5", "llama3_2_3b"]
SHORT = {
    "qwen2_5_0_5b": "qwen.5b",
    "gemma3_1b": "gemma1b",
    "llama3_2_1b": "llama1b",
    "phi3_5": "phi3.5",
    "llama3_2_3b": "llama3b",
}


def collapse(s: str, n: int = 250) -> str:
    if not s:
        return "<empty>"
    s = re.sub(r"\s+", " ", s).strip()
    total = len(s)
    chunk_count = len(re.findall(r"CHUNK\s*#\s*\d+", s))
    flag = ""
    if total > 5000:
        flag = f" [LOOP {total}c]"
    elif chunk_count >= 3:
        flag = f" [{chunk_count}xCHUNK]"
    if total > n:
        return f"{s[:n]}…{flag}"
    return f"{s}{flag}"


def fmt(entry: dict | None, score_key: str | None = None) -> str:
    if not entry:
        return "<missing>"
    chars = len(entry.get("answer", ""))
    lat = entry.get("latency_s", "?")
    score_str = ""
    if score_key and score_key in entry and entry[score_key] is not None:
        score_str = f" {score_key}={entry[score_key]}"
    return f"[{chars}c {lat}s{score_str}] {collapse(entry.get('answer', ''))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = {
        m: json.loads((DIR / f"eval_{m}.json").read_text(encoding="utf-8"))
        for m in MODELS
    }

    queries = list(data["qwen2_5_0_5b"]["C1_no_reranker"].keys())
    expected = {
        "A2_EN_glucose_threshold": "<3.5 mmol/L",
        "A4_EN_anaphylaxis_signs": "swelling/stridor/wheezing/hypotension",
        "A8_ES_postpartum_hemorrhage": "compresa empapada en <5 min, sangrado vaginal pesado",
        "A12_EN_trauma_airway_avoid": "head-tilt/chin-lift",
        "F8_EN_elt_test_time": "first 5 minutes of every hour",
        "F11_EN_shock_feet_elevation": "12 inches",
        "F12_ES_dehydration_fatal": "10% del peso corporal",
        "F3_EN_water_max_days": "3 days",
    }

    lines: list[str] = []
    lines.append("# Reranker ablation — compact judge input\n")
    lines.append(f"_5 SLMs × 8 queries × 3 conditions = 120 calls_\n")
    lines.append(
        "_C1 = dense top-5 (no reranker), C2 = rerank top-5 (P1.2 baseline), "
        "C3 = rerank top-1 → answer-sentence_\n"
    )

    for qkey in queries:
        q_text = data["qwen2_5_0_5b"]["C1_no_reranker"][qkey]["query"]
        scope = data["qwen2_5_0_5b"]["C1_no_reranker"][qkey]["scope"]
        lines.append(f"\n## {qkey}  [scope={scope}]")
        lines.append(f"**Q:** {q_text}")
        lines.append(f"**Expected:** {expected.get(qkey, '?')}\n")

        # show C2/C3 retrieval metadata once (same across models for the same query)
        c2_sample = data["qwen2_5_0_5b"]["C2_with_reranker"][qkey]
        c3_sample = data["qwen2_5_0_5b"]["C3_reranker_plus_sentence"][qkey]
        c1_sample = data["qwen2_5_0_5b"]["C1_no_reranker"][qkey]
        lines.append(
            f"_retrieval: C1 max_dense={c1_sample['max_dense_score']}  "
            f"C2 max_rerank={c2_sample['max_rerank_score']} chunks={c2_sample['n_chunks']}  "
            f"C3 top1_rerank={c3_sample['top1_rerank_score']} sent_chars={c3_sample['selected_sentence_chars']}_\n"
        )
        sent = c3_sample["selected_sentence"]
        sent_clean = re.sub(r"\s+", " ", sent).strip()
        if len(sent_clean) > 300:
            sent_clean = sent_clean[:300] + "…"
        lines.append(f"_C3 selected sentence:_ `{sent_clean}`\n")

        for m in MODELS:
            short = SHORT[m]
            d = data[m]
            c1 = d["C1_no_reranker"].get(qkey)
            c2 = d["C2_with_reranker"].get(qkey)
            c3 = d["C3_reranker_plus_sentence"].get(qkey)
            lines.append(f"- **{short}** C1: {fmt(c1)}")
            lines.append(f"- **{short}** C2: {fmt(c2)}")
            lines.append(f"- **{short}** C3: {fmt(c3)}")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote {args.out} "
        f"({args.out.stat().st_size//1024} KB, {len(lines)} lines)"
    )


if __name__ == "__main__":
    main()
