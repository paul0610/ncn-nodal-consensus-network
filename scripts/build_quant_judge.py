"""Compact view for the P1.3b quantization-ablation judge phase.

For each query, prints:
    - expected
    - rerank_top
    - per-variant: answer (truncated 250 chars), length, latency

Usage:
    python scripts/build_quant_judge.py --out <md path>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "paper_evidence" / "iterations" / "tier_C_strong_baseline" / "ablation_quantization"

VARIANTS = ["qwen2_5_0_5b_q4_k_m", "qwen2_5_0_5b_fp16", "gemma3_1b_q4_k_m", "gemma3_1b_fp16"]
SHORT = {
    "qwen2_5_0_5b_q4_k_m": "qwen.5b Q4",
    "qwen2_5_0_5b_fp16":   "qwen.5b FP16",
    "gemma3_1b_q4_k_m":    "gemma1b Q4",
    "gemma3_1b_fp16":      "gemma1b FP16",
}

EXPECTED = {
    "A2_EN_glucose_threshold": "<3.5 mmol/L",
    "A4_EN_anaphylaxis_signs": "swelling/stridor/wheezing/hypotension",
    "A8_ES_postpartum_hemorrhage": "compresa empapada en <5 min",
    "A12_EN_trauma_airway_avoid": "head-tilt/chin-lift",
    "F3_EN_water_max_days": "3 days",
    "F8_EN_elt_test_time": "first 5 minutes of every hour",
    "F11_EN_shock_feet_elevation": "12 inches",
    "F12_ES_dehydration_fatal": "10% del peso corporal",
    "F15_ES_capital_japan": "ABSTAIN (OOS)",
    "F16_EN_einstein": "ABSTAIN (OOS)",
    "F17_ES_internet": "ABSTAIN (OOS) — qwen Q4 looped 187K chars in P1.2",
}


def collapse(s: str, n: int = 250) -> str:
    if not s:
        return "<empty>"
    s = re.sub(r"\s+", " ", s).strip()
    total = len(s)
    flag = ""
    if total > 5000:
        flag = f" [LOOP {total}c]"
    elif total > 1000:
        flag = f" [{total}c]"
    chunk_count = len(re.findall(r"CHUNK\s*#\s*\d+", s))
    if chunk_count >= 3:
        flag += f" [{chunk_count}xCHUNK]"
    if total > n:
        return f"{s[:n]}…{flag}"
    return f"{s}{flag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = {
        v: json.loads((DIR / f"eval_{v}.json").read_text(encoding="utf-8"))
        for v in VARIANTS
    }

    queries = list(data["qwen2_5_0_5b_q4_k_m"]["C_rerank_budget400"].keys())

    lines: list[str] = []
    lines.append("# P1.3b Quantization Ablation — compact judge input\n")
    lines.append("_4 variants × 11 queries = 44 cells_\n")
    lines.append("_All in C_rerank_budget400 condition (same as P1.2 budget)._\n")
    lines.append("_seed=42, temperature=0.1, num_predict=200_\n")

    for qkey in queries:
        q_text = data["qwen2_5_0_5b_q4_k_m"]["C_rerank_budget400"][qkey]["query"]
        scope = data["qwen2_5_0_5b_q4_k_m"]["C_rerank_budget400"][qkey]["scope"]
        sample = data["qwen2_5_0_5b_q4_k_m"]["C_rerank_budget400"][qkey]
        lines.append(f"\n## {qkey}  [scope={scope}]")
        lines.append(f"**Q:** {q_text}")
        lines.append(f"**Expected:** {EXPECTED.get(qkey, '?')}")
        lines.append(
            f"_max_rerank={sample.get('max_rerank_score')}  "
            f"max_dense={sample.get('max_dense_score')}  "
            f"chunks={sample.get('n_chunks')}  ctx_tok={sample.get('context_tokens')}_\n"
        )

        for v in VARIANTS:
            short = SHORT[v]
            entry = data[v]["C_rerank_budget400"].get(qkey)
            if not entry:
                lines.append(f"- **{short}**: <missing>")
                continue
            chars = entry.get("answer_chars", len(entry.get("answer", "")))
            lat = entry.get("latency_s", "?")
            ans = collapse(entry.get("answer", ""))
            lines.append(f"- **{short}** [{chars}c {lat}s] {ans}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size//1024} KB, {len(lines)} lines)")


if __name__ == "__main__":
    main()
