"""DeepSeek-Chat judges the 99 sub-sample cells (paper §3.7).

Input:  paper_evidence/kappa_validation/subsample.json
Output: paper_evidence/kappa_validation/deepseek_labels.json

Per cell, sends query+expected+answer to DeepSeek-Chat with a structured
classification prompt. Parses response into one of 4 categories matching
the Claude rater's labels.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
SUBSAMPLE = REPO / "paper_evidence" / "kappa_validation" / "subsample.json"
OUT = REPO / "paper_evidence" / "kappa_validation" / "deepseek_labels.json"
ENV_FILE = REPO / ".env"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
TIMEOUT_S = 60


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Given a question, the expected short "
    "answer, and a model's response, classify the response into exactly ONE "
    "of these four categories:\n\n"
    "1. correct — the response states the expected fact clearly. Citation "
    "markers (like 'segun CHUNK #1' or '[F2]') are acceptable. Minor "
    "stylistic noise is OK.\n\n"
    "2. partial — the response contains the relevant information but has "
    "minor error, ambiguity, off-target phrasing, or mixes correct info with "
    "noise/irrelevance.\n\n"
    "3. wrong — the response is incorrect, hallucinated, off-topic, or fails "
    "to respond (e.g. just a citation marker with no content). For non-OOS "
    "queries, saying 'no information' or 'I don't know' counts as WRONG "
    "because the information was present in the context.\n\n"
    "4. abstain_correct — ONLY applies when 'Expected' indicates OOS / "
    "ABSTAIN. The model correctly says 'GATE_CLOSED' or 'no tengo "
    "informacion verificada' or equivalent. For OOS queries, any attempt "
    "to give an answer (even citing a chunk) is WRONG.\n\n"
    "Return EXACTLY ONE word: correct, partial, wrong, or abstain_correct. "
    "No explanation. No other text."
)


def build_user_prompt(query: str, expected: str, answer: str) -> str:
    return (
        f"Query: {query}\n"
        f"Expected: {expected}\n"
        f"Model response: {answer}\n\n"
        f"Classify (one word only):"
    )


VALID_LABELS = {"correct", "partial", "wrong", "abstain_correct"}


def parse_label(raw: str) -> str:
    s = raw.strip().lower()
    # Strip punctuation and surrounding markdown
    s = re.sub(r"[^a-z_]", "", s)
    if s in VALID_LABELS:
        return s
    # Fuzzy: contains keyword
    for label in VALID_LABELS:
        if label in s:
            return label
    # Common DeepSeek alternative phrasings
    if "abstain" in s:
        return "abstain_correct"
    if "incorrect" in s or "fail" in s:
        return "wrong"
    return f"UNPARSED:{raw[:50]}"


def call_judge(query: str, expected: str, answer: str, api_key: str) -> tuple[str, dict]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_user_prompt(query, expected, answer)},
        ],
        "max_tokens": 20,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(DEEPSEEK_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return raw, usage


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not in env or .env", file=sys.stderr)
        return 1

    subsample = json.loads(SUBSAMPLE.read_text(encoding="utf-8"))
    cells = subsample["cells"]
    print(f"=== Judging {len(cells)} cells via DeepSeek-Chat")

    labels: dict[str, str] = {}
    raw_responses: dict[str, str] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failures: list[str] = []
    t0 = time.time()

    for i, cell in enumerate(cells, 1):
        cell_id = cell["cell_id"]
        try:
            raw, usage = call_judge(
                cell["query"], cell["expected_short"], cell["answer"], api_key
            )
            label = parse_label(raw)
            labels[cell_id] = label
            raw_responses[cell_id] = raw
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            if label.startswith("UNPARSED"):
                failures.append(f"{cell_id}: unparsed='{raw}'")
            if i % 10 == 0 or i == len(cells):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i:>3}/{len(cells)}] {cell_id[:60]:60s} -> {label}  ({rate:.1f}/s)")
        except Exception as exc:
            labels[cell_id] = f"ERROR:{type(exc).__name__}"
            failures.append(f"{cell_id}: {exc}")
            print(f"  [FAIL {i}] {cell_id}: {exc}", file=sys.stderr)

    elapsed = time.time() - t0
    # DeepSeek-Chat pricing (Feb 2024 pricing): $0.27/1M input, $1.10/1M output
    cost_input = total_prompt_tokens * 0.27e-6
    cost_output = total_completion_tokens * 1.10e-6
    total_cost = cost_input + cost_output

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rater": MODEL,
        "rating_method": "automated_via_deepseek_api",
        "n_cells": len(cells),
        "n_labels_assigned": len([v for v in labels.values() if not v.startswith(("UNPARSED", "ERROR"))]),
        "elapsed_s": round(elapsed, 1),
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": round(total_cost, 4),
        },
        "failures": failures,
        "labels": labels,
        "raw_responses": raw_responses,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    dist = Counter(labels.values())
    print(f"\n=== DONE in {elapsed:.1f}s ===")
    print(f"Labels distribution:")
    for k, v in dist.most_common():
        print(f"  {k}: {v}")
    print(f"\nCost: ${total_cost:.4f} ({total_prompt_tokens} in + {total_completion_tokens} out)")
    print(f"-> wrote {OUT.relative_to(REPO)}")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for f in failures[:5]:
            print(f"  {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
