"""Quantization ablation (P1.3b) — FP16 vs Q4_K_M for sub-1B SLMs.

Tests whether the generation loops, copy mode, and citation fabrication
observed in P1.2 with Q4_K_M-quantized SLMs (qwen 0.5b 187K chars on F17,
gemma 1b "Sydney/Kyoto" parametric on OOS) are artefacts of weight
quantization, or inherent to the model.

Setup
-----
4 model variants:
    qwen2.5:0.5b              (Q4_K_M, 397 MB)
    qwen2.5:0.5b-instruct-fp16 (994 MB)
    gemma3:1b                 (Q4_K_M, 815 MB)
    gemma3:1b-it-fp16         (2.0 GB)

11 queries (8 diagnostic from P1.3a + 3 OOS where Q4 produced extreme
loops in P1.2):
    A2/A4/A8/A12  (WHO BEC)
    F3/F8/F11/F12 (FAA)
    F15/F16/F17   (OOS — Japan/Einstein/Internet, where qwen Q4 looped)

1 retrieval condition: C strong baseline (rerank top-20→top-5 → budget 400).
Same as P1.2 budget condition — the one where loops were observed.

Total: 11 × 4 = 44 LLM calls (~6-10 min).
Deterministic: temperature=0.1, seed=42 for reproducibility across runs.

Usage
-----
    python scripts/run_quantization_ablation.py --all-models
    python scripts/run_quantization_ablation.py --model qwen2.5:0.5b --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
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
    / "ablation_quantization"
)

EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME = "BAAI/bge-reranker-large"

CHUNK_SIZE_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 64
TOP_K_DENSE = 20
TOP_K_FINAL = 5
MATCHED_TOKEN_BUDGET = 400

OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 180
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 200
OLLAMA_SEED = 42

# 4 model variants — eval_key uses the FULL ollama tag with safe chars
MODELS: dict[str, str] = {
    "qwen2_5_0_5b_q4_k_m": "qwen2.5:0.5b",
    "qwen2_5_0_5b_fp16": "qwen2.5:0.5b-instruct-fp16",
    "gemma3_1b_q4_k_m": "gemma3:1b",
    "gemma3_1b_fp16": "gemma3:1b-it-fp16",
}

QUERIES: dict[str, dict[str, str]] = {
    # --- WHO BEC diagnostic ---
    "A2_EN_glucose_threshold": {
        "query": "What is the threshold for hypoglycaemia in mmol/L?",
        "scope": "M1",
        "source_md": "who_bec_full.md",
        "expected_short": "<3.5 mmol/L",
    },
    "A4_EN_anaphylaxis_signs": {
        "query": "What are signs of anaphylaxis airway obstruction?",
        "scope": "M1",
        "source_md": "who_bec_full.md",
        "expected_short": "swelling/stridor/wheezing/hypotension",
    },
    "A8_ES_postpartum_hemorrhage": {
        "query": "Cuales son los signos de hemorragia postparto?",
        "scope": "M2-6",
        "source_md": "who_bec_full.md",
        "expected_short": "compresa empapada en <5 min, sangrado vaginal pesado",
    },
    "A12_EN_trauma_airway_avoid": {
        "query": "What airway manoeuvre should NOT be used in trauma?",
        "scope": "M1",
        "source_md": "who_bec_full.md",
        "expected_short": "head-tilt/chin-lift",
    },
    # --- FAA diagnostic ---
    "F3_EN_water_max_days": {
        "query": "How long can a person survive without water in extreme conditions?",
        "scope": "M1",
        "source_md": "faa_survival_full.md",
        "expected_short": "3 days",
    },
    "F8_EN_elt_test_time": {
        "query": "When should you test your ELT?",
        "scope": "specific",
        "source_md": "faa_survival_full.md",
        "expected_short": "first 5 minutes of every hour",
    },
    "F11_EN_shock_feet_elevation": {
        "query": "How many inches should feet be elevated for shock treatment?",
        "scope": "specific",
        "source_md": "faa_survival_full.md",
        "expected_short": "12 inches",
    },
    "F12_ES_dehydration_fatal": {
        "query": "Que porcentaje de deshidratacion es generalmente fatal?",
        "scope": "specific",
        "source_md": "faa_survival_full.md",
        "expected_short": "10% del peso corporal",
    },
    # --- OOS (where Q4 caused extreme loops/parametric in P1.2) ---
    "F15_ES_capital_japan": {
        "query": "Cual es la capital de Japon?",
        "scope": "OOS",
        "source_md": "faa_survival_full.md",
        "expected_short": "ABSTAIN (gemma Q4 said 'Kyoto' wrong)",
    },
    "F16_EN_einstein": {
        "query": "Who developed the theory of relativity?",
        "scope": "OOS",
        "source_md": "faa_survival_full.md",
        "expected_short": "ABSTAIN (qwen Q4 generated 1.5K chars from GPS chunk)",
    },
    "F17_ES_internet": {
        "query": "Que es Internet?",
        "scope": "OOS",
        "source_md": "faa_survival_full.md",
        "expected_short": "ABSTAIN (qwen Q4 generated 187,500 chars LOOP)",
    },
}


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


def chunk_text(
    text: str,
    size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
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


@dataclass
class CorpusIndex:
    chunks: list[str]
    embeddings: np.ndarray
    md_name: str


_INDEX_CACHE: dict[str, CorpusIndex] = {}


def get_or_build_index(md_name: str, embedder: SentenceTransformer) -> CorpusIndex:
    if md_name in _INDEX_CACHE:
        return _INDEX_CACHE[md_name]
    text = (SOURCE_MD_DIR / md_name).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    embeddings = embedder.encode(
        chunks, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    idx = CorpusIndex(chunks=chunks, embeddings=embeddings, md_name=md_name)
    _INDEX_CACHE[md_name] = idx
    print(f"  [{md_name}] {len(chunks)} chunks indexed")
    return idx


@dataclass
class Cand:
    text: str
    dense_score: float
    rerank_score: float | None
    chunk_idx: int
    n_tokens: int


def dense_top_k(query: str, index: CorpusIndex, embedder: SentenceTransformer, k: int) -> list[Cand]:
    q_emb = embedder.encode(
        [query], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )[0]
    sims = index.embeddings @ q_emb
    top_idx = np.argsort(-sims)[:k]
    return [
        Cand(
            text=index.chunks[i],
            dense_score=float(sims[i]),
            rerank_score=None,
            chunk_idx=int(i),
            n_tokens=estimate_tokens(index.chunks[i]),
        )
        for i in top_idx
    ]


def apply_reranker(cands: list[Cand], query: str, reranker: CrossEncoder) -> list[Cand]:
    pairs = [[query, c.text] for c in cands]
    scores = reranker.predict(pairs, show_progress_bar=False)
    out = [
        Cand(
            text=c.text, dense_score=c.dense_score,
            rerank_score=float(s), chunk_idx=c.chunk_idx, n_tokens=c.n_tokens,
        )
        for c, s in zip(cands, scores)
    ]
    out.sort(key=lambda c: (c.rerank_score or 0.0), reverse=True)
    return out


def select_for_budget(cands: list[Cand], budget: int = MATCHED_TOKEN_BUDGET) -> list[Cand]:
    if not cands:
        return []
    selected = [cands[0]]
    total = cands[0].n_tokens
    for c in cands[1:]:
        if total + c.n_tokens > budget:
            continue
        selected.append(c)
        total += c.n_tokens
    return selected


def format_chunks(chunks: list[Cand]) -> str:
    return "\n\n".join(f"CHUNK #{i+1}: {c.text}" for i, c in enumerate(chunks))


def context_C(query: str, index: CorpusIndex, embedder: SentenceTransformer, reranker: CrossEncoder) -> tuple[str, list[Cand]]:
    cands = dense_top_k(query, index, embedder, k=TOP_K_DENSE)
    reranked = apply_reranker(cands, query, reranker)
    cands_budget = select_for_budget(reranked, MATCHED_TOKEN_BUDGET)
    return format_chunks(cands_budget), cands_budget


def call_ollama(model_tag: str, system_prompt: str, user_prompt: str) -> tuple[str, float]:
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
            "seed": OLLAMA_SEED,
        },
    }
    t0 = time.time()
    with httpx.Client(timeout=OLLAMA_TIMEOUT_S) as client:
        r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    elapsed = time.time() - t0
    answer = data.get("message", {}).get("content", "").strip()
    return answer, elapsed


@dataclass
class Stats:
    n_calls: int = 0
    total_lat: float = 0.0
    failures: list[str] = field(default_factory=list)


def run_one_model(
    eval_key: str, model_tag: str,
    embedder: SentenceTransformer, reranker: CrossEncoder,
    smoke: bool = False,
) -> dict:
    results: dict[str, dict[str, Any]] = {"C_rerank_budget400": {}}
    stats = Stats()
    qiter = list(QUERIES.items())
    if smoke:
        qiter = qiter[:1]

    for qkey, qcfg in qiter:
        query = qcfg["query"]
        scope = qcfg["scope"]
        md = qcfg["source_md"]
        idx = get_or_build_index(md, embedder)

        try:
            ctx, cands = context_C(query, idx, embedder, reranker)
            ans, lat = call_ollama(model_tag, SYSTEM_PROMPT, build_user_prompt(query, ctx))
            results["C_rerank_budget400"][qkey] = {
                "query": query, "scope": scope,
                "answer": ans,
                "answer_chars": len(ans),
                "latency_s": round(lat, 2),
                "n_chunks": len(cands),
                "context_tokens": sum(c.n_tokens for c in cands),
                "max_rerank_score": (
                    round(cands[0].rerank_score, 3) if cands else None
                ),
                "max_dense_score": round(
                    max((c.dense_score for c in cands), default=0.0), 3
                ),
                "chunk_indices": [c.chunk_idx for c in cands],
            }
            stats.n_calls += 1
            stats.total_lat += lat
            print(
                f"    [{eval_key}] {qkey:30s} "
                f"lat={lat:5.1f}s  ans_chars={len(ans):>6}  "
                f"n_chunks={len(cands)}"
            )
        except Exception as exc:
            stats.failures.append(f"{qkey}: {exc}")
            print(f"    [FAIL] {qkey}: {exc}", file=sys.stderr)

    return {
        "tier": "ablation_quantization",
        "model": model_tag,
        "model_eval_key": eval_key,
        "config": {
            "embedder": EMBEDDER_NAME,
            "reranker": RERANKER_NAME,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "top_k_dense": TOP_K_DENSE,
            "top_k_final": TOP_K_FINAL,
            "matched_token_budget": MATCHED_TOKEN_BUDGET,
            "ollama_temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
            "ollama_seed": OLLAMA_SEED,
        },
        "stats": {
            "n_calls": stats.n_calls,
            "total_latency_s": round(stats.total_lat, 2),
            "avg_latency_s": (
                round(stats.total_lat / stats.n_calls, 2) if stats.n_calls else 0.0
            ),
            "failures": stats.failures,
        },
        **results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Single ollama tag")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.all_models:
        models = list(MODELS.items())
    elif args.model:
        match = [(k, v) for k, v in MODELS.items() if v == args.model]
        if not match:
            parser.error(
                f"Unknown model {args.model}. Known: {list(MODELS.values())}"
            )
        models = match
    else:
        parser.error("Need --model or --all-models")

    print(f"=== Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    print(f"=== Loading reranker: {RERANKER_NAME}")
    reranker = CrossEncoder(RERANKER_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_calls = 0
    grand_lat = 0.0
    grand_fails: list[str] = []

    for eval_key, tag in models:
        print(f"--- Running {tag}")
        out = run_one_model(eval_key, tag, embedder, reranker, smoke=args.smoke)
        out_file = OUTPUT_DIR / f"eval_{eval_key}.json"
        out_file.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  -> wrote {out_file.relative_to(REPO_ROOT)}")
        grand_calls += out["stats"]["n_calls"]
        grand_lat += out["stats"]["total_latency_s"]
        grand_fails.extend(out["stats"]["failures"])

    print("\n=== DONE ===")
    print(f"Total calls: {grand_calls}")
    print(f"Total latency: {grand_lat:.1f}s ({grand_lat/60:.1f} min)")
    if grand_fails:
        print(f"FAILURES ({len(grand_fails)}):")
        for f in grand_fails:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
