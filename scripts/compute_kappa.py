"""Compute Cohen's kappa between Claude and DeepSeek labels (paper §3.7).

Reports:
- pairwise Cohen's kappa
- confusion matrix
- agreement rate
- interpretation per Landis & Koch (1977)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLAUDE_PATH = REPO / "paper_evidence" / "kappa_validation" / "claude_labels.json"
DEEPSEEK_PATH = REPO / "paper_evidence" / "kappa_validation" / "deepseek_labels.json"
OUT = REPO / "paper_evidence" / "kappa_validation" / "kappa_result.json"


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> tuple[float, float]:
    """Compute Cohen's kappa and raw agreement.

    Returns (kappa, raw_agreement) where:
      - kappa = (po - pe) / (1 - pe)
      - po = observed agreement
      - pe = expected agreement by chance
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"Length mismatch: {len(labels_a)} vs {len(labels_b)}")
    n = len(labels_a)
    if n == 0:
        return float("nan"), float("nan")
    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        confusion[cat_idx[a]][cat_idx[b]] += 1

    # Observed agreement
    po = sum(confusion[i][i] for i in range(k)) / n
    # Expected agreement by chance
    row_totals = [sum(row) for row in confusion]
    col_totals = [sum(confusion[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_totals[i] / n) * (col_totals[i] / n) for i in range(k))
    if abs(1 - pe) < 1e-9:
        return float("nan"), po
    kappa = (po - pe) / (1 - pe)
    return kappa, po


def confusion_matrix(labels_a: list[str], labels_b: list[str]) -> dict:
    categories = sorted(set(labels_a) | set(labels_b))
    matrix: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in categories} for c in categories}
    for a, b in zip(labels_a, labels_b):
        matrix[a][b] += 1
    return matrix


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    claude = json.loads(CLAUDE_PATH.read_text(encoding="utf-8"))
    deepseek = json.loads(DEEPSEEK_PATH.read_text(encoding="utf-8"))
    cl = claude["labels"]
    ds = deepseek["labels"]

    common = sorted(set(cl) & set(ds))
    missing_claude = sorted(set(ds) - set(cl))
    missing_deepseek = sorted(set(cl) - set(ds))

    # Filter out error/unparsed labels from DeepSeek
    valid_common = [c for c in common
                    if not ds[c].startswith(("ERROR", "UNPARSED"))]

    labels_a = [cl[c] for c in valid_common]
    labels_b = [ds[c] for c in valid_common]
    n = len(labels_a)
    kappa, po = cohen_kappa(labels_a, labels_b)

    matrix = confusion_matrix(labels_a, labels_b)
    dist_a = Counter(labels_a)
    dist_b = Counter(labels_b)

    print(f"=== Cohen's kappa: Claude (Opus 4.7) vs DeepSeek-Chat ===")
    print(f"N cells judged by both: {n}/{len(cl)}")
    if missing_claude:
        print(f"  missing in Claude: {len(missing_claude)}")
    if missing_deepseek:
        print(f"  missing in DeepSeek: {len(missing_deepseek)}")
    print()
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Raw agreement (po): {po:.4f} ({100*po:.1f}%)")
    print(f"Interpretation (Landis & Koch 1977): {interpret_kappa(kappa)}")
    print()
    print(f"Label distribution:")
    print(f"  {'category':20s} {'Claude':>8s} {'DeepSeek':>10s}")
    for cat in sorted(set(dist_a) | set(dist_b)):
        print(f"  {cat:20s} {dist_a[cat]:>8d} {dist_b[cat]:>10d}")
    print()
    print(f"Confusion matrix (rows=Claude, cols=DeepSeek):")
    cats = sorted(matrix.keys())
    header = "                  " + " ".join(f"{c[:12]:>12s}" for c in cats)
    print(header)
    for ra in cats:
        row = "  " + f"{ra[:16]:16s}  " + " ".join(f"{matrix[ra][rb]:>12d}" for rb in cats)
        print(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raters": ["claude_opus_4.7", "deepseek-chat"],
        "n_cells_judged_by_both": n,
        "n_cells_in_claude": len(cl),
        "n_cells_in_deepseek": len(ds),
        "missing_in_claude": missing_claude,
        "missing_in_deepseek": missing_deepseek,
        "cohen_kappa": round(kappa, 4),
        "raw_agreement_po": round(po, 4),
        "interpretation": interpret_kappa(kappa),
        "label_distribution": {
            "claude": dict(dist_a),
            "deepseek": dict(dist_b),
        },
        "confusion_matrix": matrix,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
