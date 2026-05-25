"""Build triple-comparison markdown for LLM-judge phase.

For each tier, for each query, emits one markdown section with:
    - A graph answer    (from tier_4_baseline / tier_5_adversarial / tier_4_5_faa_combined)
    - B raw_md answer   (same files, B_raw_md block)
    - C fixed-5 answer  (from tier_C_strong_baseline/<tier>/eval_<model>.json)
    - C budget-400 answer

Long answers (e.g. llama1b 100K-char loops) are truncated to first 600 chars +
last 200 chars with [TRUNCATED total=N chars] marker, so the judge can
inspect the loop signature without drowning.

Usage:
    python scripts/build_judge_table.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# (tier_C_dir, AB_dir, output_md)
TIER_PAIRS = [
    (
        "tier_4_who",
        REPO / "paper_evidence" / "iterations" / "tier_4_baseline",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_who",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_who" / "judge_input.md",
    ),
    (
        "tier_5_who",
        REPO / "paper_evidence" / "iterations" / "tier_5_adversarial",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_5_who",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_5_who" / "judge_input.md",
    ),
    (
        "tier_4_5_faa",
        REPO / "paper_evidence" / "iterations" / "tier_4_5_faa_combined",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_5_faa",
        REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "tier_4_5_faa" / "judge_input.md",
    ),
]

MODELS = ["qwen2_5_0_5b", "gemma3_1b", "llama3_2_1b", "phi3_5", "llama3_2_3b"]
PRETTY = {
    "qwen2_5_0_5b": "qwen 0.5b",
    "gemma3_1b": "gemma 1b",
    "llama3_2_1b": "llama 1b",
    "phi3_5": "phi3.5 (3.8b)",
    "llama3_2_3b": "llama 3b",
}


def truncate(text: str, head: int = 600, tail: int = 200) -> str:
    if not text:
        return "*<empty>*"
    n = len(text)
    if n <= head + tail + 50:
        return text
    return f"{text[:head]}\n\n... [TRUNCATED total={n} chars] ...\n\n{text[-tail:]}"


def fmt_block(title: str, entry: dict[str, Any] | None, extra_meta: str = "") -> str:
    if entry is None:
        return f"- **{title}:** *<missing>*"
    ans = entry.get("answer", "")
    chars = len(ans)
    lat = entry.get("latency_s", "?")
    meta = f"{chars} chars, {lat}s{extra_meta}"
    return f"- **{title}** ({meta}):\n```\n{truncate(ans)}\n```"


def build_query_section(qkey: str, query: str, scope: str, ab_data: dict, c_data: dict) -> str:
    out = [f"## {qkey}\n", f"**Query:** {query}", f"**Scope:** {scope or '(none)'}\n"]
    for m in MODELS:
        out.append(f"### {PRETTY[m]}")
        ab = ab_data.get(m, {})
        cd = c_data.get(m, {})

        a_entry = ab.get("A_graph", {}).get(qkey)
        b_entry_block = ab.get("B_raw_md", {}) or ab.get("B_raw_md_full", {})
        b_entry = b_entry_block.get(qkey)
        c_fixed_entry = cd.get("C_fixed_k", {}).get(qkey)
        c_budget_entry = cd.get("C_budget_400", {}).get(qkey)

        # extra metadata for C
        c_fixed_meta = ""
        if c_fixed_entry:
            n_chunks = c_fixed_entry.get("n_chunks", "?")
            ctx_tok = c_fixed_entry.get("context_tokens", "?")
            mrr = c_fixed_entry.get("max_rerank_score")
            c_fixed_meta = f", chunks={n_chunks} ctx={ctx_tok}tok rerank_top={mrr}"
        c_budget_meta = ""
        if c_budget_entry:
            n_chunks = c_budget_entry.get("n_chunks", "?")
            ctx_tok = c_budget_entry.get("context_tokens", "?")
            mrr = c_budget_entry.get("max_rerank_score")
            c_budget_meta = f", chunks={n_chunks} ctx={ctx_tok}tok rerank_top={mrr}"
        a_meta = ""
        if a_entry:
            relev = a_entry.get("max_relev")
            if relev is not None:
                a_meta = f", max_relev={relev}"

        out.append(fmt_block("A graph", a_entry, a_meta))
        out.append(fmt_block("B raw_md", b_entry))
        out.append(fmt_block("C fixed-5", c_fixed_entry, c_fixed_meta))
        out.append(fmt_block("C budget-400", c_budget_entry, c_budget_meta))
        out.append("")
    return "\n".join(out)


def load_eval_dir(d: Path) -> dict[str, dict[str, Any]]:
    """Load eval_<model>.json for each model into dict[model] -> data."""
    out = {}
    for m in MODELS:
        f = d / f"eval_{m}.json"
        if f.exists():
            out[m] = json.loads(f.read_text(encoding="utf-8"))
        else:
            out[m] = {}
    return out


def build_tier(tier_name: str, ab_dir: Path, c_dir: Path, out_md: Path) -> None:
    ab_data = load_eval_dir(ab_dir)
    c_data = load_eval_dir(c_dir)

    # query keys come from A_graph of any model that has them
    # use qwen as canonical (always has full set)
    qwen_ab = ab_data.get("qwen2_5_0_5b", {})
    a_block = qwen_ab.get("A_graph", {})

    sections = [f"# Triple comparison — {tier_name}\n",
                f"_A_dir: {ab_dir.relative_to(REPO)}_  ",
                f"_C_dir: {c_dir.relative_to(REPO)}_\n",
                "Long answers truncated as `head 600 + tail 200`. Inspect failures and copy-mode by reading the bracketed metadata.\n",
                "---\n"]

    for qkey, qdata in a_block.items():
        query = qdata.get("query", "")
        scope = qdata.get("scope", "")
        sections.append(build_query_section(qkey, query, scope, ab_data, c_data))
        sections.append("---\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(sections), encoding="utf-8")
    print(f"  -> {out_md.relative_to(REPO)}  ({out_md.stat().st_size//1024} KB, {len(a_block)} queries)")


def main():
    for entry in TIER_PAIRS:
        if len(entry) == 4:
            tier_name, ab_dir, c_dir, out_md = entry
        else:
            continue
        print(f"=== Building judge input for {tier_name}")
        build_tier(tier_name, ab_dir, c_dir, out_md)


if __name__ == "__main__":
    main()
