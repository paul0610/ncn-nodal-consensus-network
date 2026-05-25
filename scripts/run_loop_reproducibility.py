"""Loop reproducibility study (P1.3b extension).

Tests whether qwen 2.5:0.5b Q4_K_M reliably enters generation loops on
F17 OOS "Que es Internet?" under sampling non-determinism (no fixed seed
or different seeds), or whether the 187,500-char loop observed in P1.2
was a one-off pathological event.

Setup
-----
- Model: qwen2.5:0.5b (Q4_K_M, 397 MB)
- Query: F17_ES_internet (the OOS query that produced the 187K loop in P1.2)
- Retrieval: same as P1.2 (rerank top-20 → top-5 → matched-budget 400)
- Sampling: temperature=0.1, num_predict=200, vary seed across 20 values

Hypothesis
----------
H0: 187K loop was sampling-seed dependent. With seed=42 it produces 18 chars.
    Other seeds may or may not loop.
H1: Most seeds produce normal (≤500 char) abstain responses; loops are rare
    (e.g. <10% of seeds).

Output
------
paper_evidence/iterations/tier_C_strong_baseline/loop_reproducibility/
    eval_qwen_q4_loops.json — per-seed answer length, latency, full answer
    summary.md — distribution + verdict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD_DIR = REPO_ROOT / "paper_evidence" / "iterations" / "source_md"
OUTPUT_DIR = (
    REPO_ROOT
    / "paper_evidence"
    / "iterations"
    / "tier_C_strong_baseline"
    / "loop_reproducibility"
)

EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME = "BAAI/bge-reranker-large"
MODEL_TAG = "qwen2.5:0.5b"
QUERY = "Que es Internet?"
QUERY_KEY = "F17_ES_internet"
SOURCE_MD = "faa_survival_full.md"

CHUNK_SIZE_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 64
TOP_K_DENSE = 20
MATCHED_TOKEN_BUDGET = 400

OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 240  # higher than P1.2 (180s) to give loops a chance to complete or be detected
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 200

# 20 seeds — including 0 (often default) and 42 (P1.3b deterministic)
# plus a spread. None = use ollama default (no seed param).
SEEDS_TO_TEST: list[int | None] = [
    None,            # default ollama random behaviour
    0, 1, 2, 3, 7,
    13, 17, 23, 31,
    42,              # P1.3b reference
    99, 100, 123, 256,
    420, 777, 1024, 2026,
    9999,
]


SYSTEM_PROMPT = (
    "Eres un asistente EXTRACTIVO. Tu UNICA fuente de informacion son los\n"
    "CHUNKS DEL DOCUMENTO que se te proporcionan a continuacion de la\n"
    "pregunta.\n\n"
    "PROCESO OBLIGATORIO:\n"
    "1. Lee CADA chunk del contexto proporcionado.\n"
    "2. Identifica los chunks que responden DIRECTAMENTE a la pregunta.\n"
    "3. Si NINGUN chunk responde la pregunta, responde literalmente:\n"
    "   'No tengo informacion verificada sobre ese tema en mi base de\n"
    "   conocimiento.'\n"
    "4. Si hay chunks que responden, CITALOS referenciando su numero\n"
    "   (ej. 'segun CHUNK #3 ...') y usa su contenido sin parafrasear\n"
    "   ni alterar el significado.\n"
    "5. Responde en el mismo idioma que la pregunta del usuario.\n\n"
    "REGLAS ABSOLUTAS:\n"
    "- NO uses conocimiento externo al contexto proporcionado.\n"
    "- NO inventes informacion, ni siquiera para 'completar' una respuesta\n"
    "  parcial.\n"
    "- NO extrapoles, ni mezcles chunks de distintos temas para cubrir\n"
    "  huecos.\n"
    "- Si el contexto es parcial o ambiguo, di que es parcial — no lo\n"
    "  rellenes con conocimiento previo.\n"
    "- NO generes JSON. Responde en lenguaje natural directamente."
)


def build_user_prompt(query: str, context: str) -> str:
    return (
        f"CHUNKS DEL DOCUMENTO (unica fuente permitida):\n"
        f"{context}\n\n"
        f"PREGUNTA: {query}\n\n"
        f"Recuerda: usa UNICAMENTE los chunks anteriores. "
        f"Si ningun chunk responde la pregunta, responde literalmente "
        f"'No tengo informacion verificada sobre ese tema en mi base "
        f"de conocimiento.' Cita los chunks por su numero "
        f"(ej. 'segun CHUNK #2...')."
    )


CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(text: str, size_tokens: int = CHUNK_SIZE_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    words = text.split()
    if not words:
        return []
    size_chars = size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    chunks: list[str] = []
    i = 0
    while i < len(words):
        cur_chars = 0
        j = i
        while j < len(words) and cur_chars < size_chars:
            cur_chars += len(words[j]) + 1
            j += 1
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(words):
            break
        back_chars = 0
        k = j
        while k > i and back_chars < overlap_chars:
            k -= 1
            back_chars += len(words[k]) + 1
        i = k if k > i else j
    return chunks


def build_context(embedder: SentenceTransformer, reranker: CrossEncoder) -> tuple[str, dict]:
    """Build the same matched-budget-400 context used in P1.2 for F17."""
    text = (SOURCE_MD_DIR / SOURCE_MD).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    embeddings = embedder.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    q_emb = embedder.encode([QUERY], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
    sims = embeddings @ q_emb
    top_idx = np.argsort(-sims)[:TOP_K_DENSE]
    top_chunks = [chunks[i] for i in top_idx]
    pairs = [[QUERY, c] for c in top_chunks]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(top_chunks, rerank_scores, top_idx, [estimate_tokens(c) for c in top_chunks]),
        key=lambda t: t[1], reverse=True,
    )
    selected = [ranked[0]]
    total = ranked[0][3]
    for chunk, score, ci, ntok in ranked[1:]:
        if total + ntok > MATCHED_TOKEN_BUDGET:
            continue
        selected.append((chunk, score, ci, ntok))
        total += ntok
    ctx_lines = []
    for i, (c, _, _, _) in enumerate(selected):
        ctx_lines.append(f"CHUNK #{i+1}: {c}")
    return (
        "\n\n".join(ctx_lines),
        {
            "n_chunks": len(selected),
            "context_tokens": sum(s[3] for s in selected),
            "max_rerank_score": float(selected[0][1]),
            "chunk_indices": [int(s[2]) for s in selected],
        },
    )


def call_ollama(seed: int | None, system_prompt: str, user_prompt: str) -> tuple[str, float]:
    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "num_predict": OLLAMA_NUM_PREDICT,
    }
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": MODEL_TAG,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
    }
    t0 = time.time()
    with httpx.Client(timeout=OLLAMA_TIMEOUT_S) as client:
        r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    elapsed = time.time() - t0
    answer = data.get("message", {}).get("content", "").strip()
    return answer, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds only")
    args = parser.parse_args()

    seeds = SEEDS_TO_TEST[:2] if args.smoke else SEEDS_TO_TEST
    print(f"=== Testing {len(seeds)} seeds on {MODEL_TAG} / {QUERY_KEY}")

    print(f"=== Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    print(f"=== Loading reranker: {RERANKER_NAME}")
    reranker = CrossEncoder(RERANKER_NAME)

    print(f"=== Building context (same as P1.2)")
    context, ctx_meta = build_context(embedder, reranker)
    print(
        f"  context: {ctx_meta['n_chunks']} chunks, "
        f"{ctx_meta['context_tokens']} tokens, "
        f"max_rerank={ctx_meta['max_rerank_score']:.3f}"
    )

    user_prompt = build_user_prompt(QUERY, context)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for seed in seeds:
        seed_label = "DEFAULT" if seed is None else str(seed)
        try:
            ans, lat = call_ollama(seed, SYSTEM_PROMPT, user_prompt)
            chars = len(ans)
            looped = chars > 5000
            entry = {
                "seed": seed,
                "seed_label": seed_label,
                "answer_chars": chars,
                "answer_tokens_est": estimate_tokens(ans),
                "latency_s": round(lat, 2),
                "looped_>5000c": looped,
                "answer_first_300": ans[:300],
                "answer_last_200": ans[-200:] if chars > 500 else "",
                "full_answer": ans if chars <= 2000 else None,  # truncate huge answers
            }
            results.append(entry)
            flag = " 🚨 LOOP" if looped else ""
            print(f"  seed={seed_label:>8s}  chars={chars:>7d}  lat={lat:>5.1f}s{flag}")
        except Exception as exc:
            results.append({
                "seed": seed,
                "seed_label": seed_label,
                "error": str(exc),
            })
            print(f"  seed={seed_label}  ERROR: {exc}", file=sys.stderr)

    # Aggregate stats
    chars_arr = [r.get("answer_chars", 0) for r in results if "answer_chars" in r]
    n_loops = sum(1 for r in results if r.get("looped_>5000c"))
    payload = {
        "model": MODEL_TAG,
        "query_key": QUERY_KEY,
        "query": QUERY,
        "n_seeds_tested": len(seeds),
        "config": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
            "timeout_s": OLLAMA_TIMEOUT_S,
            "ollama_seeds": [str(s) if s is not None else "DEFAULT" for s in seeds],
        },
        "context_meta": ctx_meta,
        "stats": {
            "n_loops_>5000c": n_loops,
            "loop_rate_pct": round(100 * n_loops / max(1, len(chars_arr)), 1),
            "min_chars": min(chars_arr) if chars_arr else None,
            "max_chars": max(chars_arr) if chars_arr else None,
            "median_chars": int(np.median(chars_arr)) if chars_arr else None,
            "mean_chars": int(np.mean(chars_arr)) if chars_arr else None,
        },
        "per_seed_results": results,
    }
    out_file = OUTPUT_DIR / "eval_qwen_q4_loops.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== DONE ===")
    print(f"Loops (>5000 chars): {n_loops}/{len(chars_arr)} = {payload['stats']['loop_rate_pct']}%")
    print(f"Char range: min={payload['stats']['min_chars']}  median={payload['stats']['median_chars']}  max={payload['stats']['max_chars']}")
    print(f"-> wrote {out_file.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
