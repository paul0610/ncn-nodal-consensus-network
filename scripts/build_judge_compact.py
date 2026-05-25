"""Compact triple-comparison for Claude in-session judge.

Output: per query, a single ~80-line block with the 5 models stacked.
Each (model, condition) is one line:
    {model:10s} | A | <250 chars one-liner> [N chars, latency]

This is much smaller than build_judge_table.py output and fits in context.

Usage:
    python scripts/build_judge_compact.py <tier_key>

Tier keys: 4_who | 5_who | 4_5_faa
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

TIERS = {
    "4_who": (
        REPO / "paper_evidence" / "iterations" / "tier_4_baseline",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_who",
    ),
    "5_who": (
        REPO / "paper_evidence" / "iterations" / "tier_5_adversarial",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_5_who",
    ),
    "4_5_faa": (
        REPO / "paper_evidence" / "iterations" / "tier_4_5_faa_combined",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_5_faa",
    ),
}

MODELS = ["qwen2_5_0_5b", "gemma3_1b", "llama3_2_1b", "phi3_5", "llama3_2_3b"]
SHORT = {
    "qwen2_5_0_5b": "qwen.5b",
    "gemma3_1b": "gemma1b",
    "llama3_2_1b": "llama1b",
    "phi3_5": "phi3.5",
    "llama3_2_3b": "llama3b",
}


def collapse(s: str, n: int = 250) -> str:
    """Collapse multi-line answers into one line, truncate, mark pattern."""
    if not s:
        return "<empty>"
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    total = len(s)
    # detect copy patterns
    chunk_count = len(re.findall(r"CHUNK\s*#\s*\d+", s))
    has_repeat = bool(re.search(r"(.{20,})\1", s))
    flag = ""
    if total > 5000:
        flag = f" [LOOP {total}c]"
    elif chunk_count >= 3:
        flag = f" [{chunk_count}xCHUNK]"
    elif has_repeat:
        flag = " [repeat]"
    if total > n:
        return f"{s[:n]}…{flag}"
    return f"{s}{flag}"


def fmt_entry(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "<missing>"
    ans = entry.get("answer", "")
    chars = len(ans)
    lat = entry.get("latency_s", "?")
    body = collapse(ans)
    return f"[{chars}c, {lat}s] {body}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=list(TIERS.keys()))
    parser.add_argument("--out", type=Path, required=True, help="Output markdown path")
    args = parser.parse_args()

    ab_dir, c_dir = TIERS[args.tier]

    ab_data = {m: json.loads((ab_dir / f"eval_{m}.json").read_text(encoding="utf-8"))
               for m in MODELS}
    c_data = {m: json.loads((c_dir / f"eval_{m}.json").read_text(encoding="utf-8"))
              for m in MODELS}

    # query keys
    qwen_a = ab_data["qwen2_5_0_5b"]["A_graph"]

    lines: list[str] = []
    lines.append(f"# Compact judge input — tier {args.tier}\n")
    lines.append(f"_legend: M | A=graph | B=raw_md | Cf=fixed-5 | Cb=budget-400_\n")
    lines.append(f"_flags: [LOOP Nc]=>5000 chars,  [NxCHUNK]=>=3 'CHUNK #' tokens,  [repeat]=internal repetition_\n")

    for qkey, qdata in qwen_a.items():
        query = qdata.get("query", "")
        scope = qdata.get("scope", "")
        lines.append(f"\n## {qkey}  [scope={scope}]")
        lines.append(f"**Q:** {query}\n")

        c_qwen_fixed = c_data["qwen2_5_0_5b"]["C_fixed_k"].get(qkey, {})
        rerank = c_qwen_fixed.get("max_rerank_score")
        n_chunks_fixed = c_qwen_fixed.get("n_chunks", "?")
        c_qwen_budget = c_data["qwen2_5_0_5b"]["C_budget_400"].get(qkey, {})
        n_chunks_budget = c_qwen_budget.get("n_chunks", "?")
        lines.append(f"_rerank_top={rerank}  n_chunks fixed={n_chunks_fixed} budget={n_chunks_budget}_\n")

        for m in MODELS:
            short = SHORT[m]
            ab = ab_data[m]
            cd = c_data[m]
            a_e = ab.get("A_graph", {}).get(qkey)
            b_e = ab.get("B_raw_md", {}).get(qkey) or ab.get("B_raw_md_full", {}).get(qkey)
            cf_e = cd.get("C_fixed_k", {}).get(qkey)
            cb_e = cd.get("C_budget_400", {}).get(qkey)
            lines.append(f"- **{short}** A : {fmt_entry(a_e)}")
            lines.append(f"- **{short}** B : {fmt_entry(b_e)}")
            lines.append(f"- **{short}** Cf: {fmt_entry(cf_e)}")
            lines.append(f"- **{short}** Cb: {fmt_entry(cb_e)}")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size//1024} KB, {len(lines)} lines)")


if __name__ == "__main__":
    main()
