"""Strong MD-RAG baseline (condition C) — chunked retrieval with reranker.

Pipeline per query:
    1. Chunk the source .md (384-token windows, 64-token overlap).
    2. Dense retrieve top-20 with paraphrase-multilingual-MiniLM-L12-v2.
    3. Cross-encoder rerank with BAAI/bge-reranker-large -> top-5.
    4. Build two contexts:
         - C_fixed_k    : top-5 chunks verbatim
         - C_budget_400 : greedy fill <= 400 tokens (matched to graph)
    5. Prompt SLM with structurally identical instructions to graph A.
       (only difference: "CHUNK #n" instead of "CLAIM #n")

Queries are pulled from the existing tier eval JSONs (A_graph block) so
condition C uses the exact same query set as A and B with no drift.

Usage:
    python scripts/run_eval_baseline_C.py --tier 4_who --model qwen2.5:0.5b
    python scripts/run_eval_baseline_C.py --tier 4_who --all-models
    python scripts/run_eval_baseline_C.py --all
    python scripts/run_eval_baseline_C.py --tier 4_who --smoke    # 1 query, 1 model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Quiet down noisy loggers from sentence-transformers / huggingface
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402


# =============================================================================
# Config
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD_DIR = REPO_ROOT / "paper_evidence" / "iterations" / "source_md"
OUTPUT_DIR = REPO_ROOT / "paper_evidence" / "iterations" / "tier_C_strong_baseline"

EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME = "BAAI/bge-reranker-large"

CHUNK_SIZE_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 64
TOP_K_RETRIEVE = 20
TOP_K_FIXED = 5
MATCHED_TOKEN_BUDGET = 400

OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 180
OLLAMA_TEMPERATURE = 0.1  # deterministic-ish, mirrors grafo answer mode

# eval-key (filename safe) -> ollama tag
MODELS: dict[str, str] = {
    "qwen2_5_0_5b": "qwen2.5:0.5b",
    "gemma3_1b": "gemma3:1b",
    "llama3_2_1b": "llama3.2:1b",
    "phi3_5": "phi3.5:latest",
    "llama3_2_3b": "llama3.2:3b",
}

# Tier definitions: source corpus + which prior eval JSON to read queries from
TIERS: dict[str, dict[str, Any]] = {
    "4_who": {
        "source_md": SOURCE_MD_DIR / "who_bec_full.md",
        "queries_from": (
            REPO_ROOT
            / "paper_evidence"
            / "iterations"
            / "tier_4_baseline"
            / "eval_qwen2_5_0_5b.json"
        ),
        "out_dir": OUTPUT_DIR / "tier_4_who",
        "expected_n_queries": 7,
    },
    "5_who": {
        "source_md": SOURCE_MD_DIR / "who_bec_full.md",
        "queries_from": (
            REPO_ROOT
            / "paper_evidence"
            / "iterations"
            / "tier_5_adversarial"
            / "eval_qwen2_5_0_5b.json"
        ),
        "out_dir": OUTPUT_DIR / "tier_5_who",
        "expected_n_queries": 20,
    },
    "4_5_faa": {
        "source_md": SOURCE_MD_DIR / "faa_survival_full.md",
        "queries_from": (
            REPO_ROOT
            / "paper_evidence"
            / "iterations"
            / "tier_4_5_faa_combined"
            / "eval_qwen2_5_0_5b.json"
        ),
        "out_dir": OUTPUT_DIR / "tier_4_5_faa",
        "expected_n_queries": 17,
    },
}


# =============================================================================
# Prompt — structurally identical to graph CHAT_SYSTEM_PROMPT
# (only "CLAIM #n" -> "CHUNK #n")
# =============================================================================

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
    """Mirror of process_chat prompt construction (orchestrator.py:288-297)."""
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


# =============================================================================
# Tokenization & chunking
# =============================================================================

# Char-based approximation: ~4 chars per token (matches retrieval/serializer.py
# and is consistent with the budget accounting used for the graph condition).
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedy whitespace-aware chunker.

    Walks word-by-word, accumulating until the running token estimate hits
    ``size_tokens``. Then steps back ``overlap_tokens`` worth of words to
    start the next window — gives natural overlap without splitting mid-word.
    """
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
            cur_chars += len(words[j]) + 1  # +1 for the space
            j += 1
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(words):
            break
        # step back by overlap to start next window
        back_chars = 0
        k = j
        while k > i and back_chars < overlap_chars:
            k -= 1
            back_chars += len(words[k]) + 1
        i = k if k > i else j  # always make forward progress
    return chunks


# =============================================================================
# Index building & retrieval
# =============================================================================

@dataclass
class CorpusIndex:
    chunks: list[str]
    embeddings: np.ndarray
    source_md_path: Path
    chunk_size_tokens: int
    overlap_tokens: int
    embedder_name: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "source_md": str(self.source_md_path.name),
            "n_chunks": len(self.chunks),
            "chunk_size_tokens": self.chunk_size_tokens,
            "overlap_tokens": self.overlap_tokens,
            "embedder": self.embedder_name,
            "avg_chars_per_chunk": float(np.mean([len(c) for c in self.chunks])),
        }


def build_index(source_md: Path, embedder: SentenceTransformer) -> CorpusIndex:
    text = source_md.read_text(encoding="utf-8")
    chunks = chunk_text(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    print(f"  [{source_md.name}] {len(chunks)} chunks")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return CorpusIndex(
        chunks=chunks,
        embeddings=embeddings,
        source_md_path=source_md,
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS,
        embedder_name=EMBEDDER_NAME,
    )


@dataclass
class RetrievedChunk:
    text: str
    dense_score: float
    rerank_score: float
    chunk_idx: int
    n_tokens: int


def retrieve_and_rerank(
    query: str,
    index: CorpusIndex,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_final: int = TOP_K_FIXED,
) -> list[RetrievedChunk]:
    """Dense top-K retrieval -> cross-encoder rerank -> top-N final."""
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    sims = index.embeddings @ q_emb  # cosine sim (both normalized)

    top_idx = np.argsort(-sims)[:top_k_retrieve]
    top_chunks = [index.chunks[i] for i in top_idx]
    top_dense = [float(sims[i]) for i in top_idx]

    pairs = [[query, c] for c in top_chunks]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)

    candidates: list[RetrievedChunk] = []
    for chunk_text_, dense_s, rs, ci in zip(top_chunks, top_dense, rerank_scores, top_idx):
        candidates.append(
            RetrievedChunk(
                text=chunk_text_,
                dense_score=dense_s,
                rerank_score=float(rs),
                chunk_idx=int(ci),
                n_tokens=estimate_tokens(chunk_text_),
            )
        )
    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_k_final] if top_k_final else candidates


def select_for_budget(
    candidates: list[RetrievedChunk],
    budget_tokens: int = MATCHED_TOKEN_BUDGET,
) -> list[RetrievedChunk]:
    """Greedy fill in rerank-score order, no truncation, do not exceed budget.

    Includes the top-1 chunk even if it alone exceeds the budget — otherwise
    the budget condition could return an empty context for queries whose
    best chunk happens to be longer than 400 tokens.
    """
    if not candidates:
        return []
    selected: list[RetrievedChunk] = [candidates[0]]
    total = candidates[0].n_tokens
    for c in candidates[1:]:
        if total + c.n_tokens > budget_tokens:
            continue
        selected.append(c)
        total += c.n_tokens
    return selected


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """CHUNK #n: <text>  (one block per chunk, blank-line separated)."""
    return "\n\n".join(
        f"CHUNK #{i+1}: {c.text}" for i, c in enumerate(chunks)
    )


# =============================================================================
# Ollama client
# =============================================================================

def call_ollama(
    model_tag: str, system_prompt: str, user_prompt: str
) -> tuple[str, float]:
    """Single non-streaming Ollama /api/chat call. Returns (answer, latency_s)."""
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    t0 = time.time()
    with httpx.Client(timeout=OLLAMA_TIMEOUT_S) as client:
        r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    elapsed = time.time() - t0
    answer = data.get("message", {}).get("content", "").strip()
    return answer, elapsed


# =============================================================================
# Query loading from existing eval JSONs
# =============================================================================

def load_queries(tier_cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Read the A_graph block of an existing eval JSON to get query+scope.

    Returns dict[query_key] -> {"query": str, "scope": str} preserving the
    insertion order of the source file.
    """
    src = tier_cfg["queries_from"]
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    a_graph = data["A_graph"]
    out: dict[str, dict[str, str]] = {}
    for k, v in a_graph.items():
        out[k] = {"query": v["query"], "scope": v.get("scope", "")}
    expected = tier_cfg["expected_n_queries"]
    if len(out) != expected:
        raise RuntimeError(
            f"Expected {expected} queries in {src}, got {len(out)}"
        )
    return out


# =============================================================================
# Main runner
# =============================================================================

@dataclass
class RunStats:
    n_calls: int = 0
    total_latency_s: float = 0.0
    failures: list[str] = field(default_factory=list)


def run_one_model_one_tier(
    tier_key: str,
    model_eval_key: str,
    model_tag: str,
    queries: dict[str, dict[str, str]],
    index: CorpusIndex,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    out_dir: Path,
    smoke: bool = False,
) -> RunStats:
    """Run all queries for (tier, model) -> write eval_<model>.json."""
    stats = RunStats()
    results: dict[str, dict[str, Any]] = {"C_fixed_k": {}, "C_budget_400": {}}

    qiter = list(queries.items())
    if smoke:
        qiter = qiter[:1]

    for qkey, qcfg in qiter:
        query = qcfg["query"]
        scope = qcfg["scope"]
        try:
            t_retrieve_0 = time.time()
            ranked = retrieve_and_rerank(
                query, index, embedder, reranker,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_final=0,  # keep all 20 for budget mode
            )
            retrieve_latency = time.time() - t_retrieve_0

            fixed = ranked[:TOP_K_FIXED]
            budget = select_for_budget(ranked, MATCHED_TOKEN_BUDGET)

            # ----- C_fixed_k -----
            ctx_fixed = format_chunks(fixed)
            user_p_fixed = build_user_prompt(query, ctx_fixed)
            ans_fixed, lat_fixed = call_ollama(model_tag, SYSTEM_PROMPT, user_p_fixed)
            stats.n_calls += 1
            stats.total_latency_s += lat_fixed
            results["C_fixed_k"][qkey] = {
                "query": query,
                "scope": scope,
                "context_type": "chunked_md_top5",
                "answer": ans_fixed,
                "latency_s": round(lat_fixed, 2),
                "retrieve_latency_s": round(retrieve_latency, 2),
                "n_chunks": len(fixed),
                "context_tokens": sum(c.n_tokens for c in fixed),
                "max_rerank_score": (
                    round(fixed[0].rerank_score, 3) if fixed else None
                ),
                "max_dense_score": (
                    round(max(c.dense_score for c in fixed), 3) if fixed else None
                ),
                "chunk_indices": [c.chunk_idx for c in fixed],
            }

            # ----- C_budget_400 -----
            ctx_budget = format_chunks(budget)
            user_p_budget = build_user_prompt(query, ctx_budget)
            ans_budget, lat_budget = call_ollama(model_tag, SYSTEM_PROMPT, user_p_budget)
            stats.n_calls += 1
            stats.total_latency_s += lat_budget
            results["C_budget_400"][qkey] = {
                "query": query,
                "scope": scope,
                "context_type": "chunked_md_budget400",
                "answer": ans_budget,
                "latency_s": round(lat_budget, 2),
                "n_chunks": len(budget),
                "context_tokens": sum(c.n_tokens for c in budget),
                "max_rerank_score": (
                    round(budget[0].rerank_score, 3) if budget else None
                ),
                "max_dense_score": (
                    round(max(c.dense_score for c in budget), 3) if budget else None
                ),
                "chunk_indices": [c.chunk_idx for c in budget],
            }

            print(
                f"    [{model_eval_key}] {qkey:30s} "
                f"fixed={lat_fixed:5.1f}s budget={lat_budget:5.1f}s "
                f"({len(fixed)}/{len(budget)} chunks)"
            )
        except Exception as exc:  # noqa: BLE001
            stats.failures.append(f"{qkey}: {exc}")
            print(f"    [FAIL] {qkey}: {exc}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{model_eval_key}.json"
    payload = {
        "tier": tier_key,
        "model": model_tag,
        "model_eval_key": model_eval_key,
        "config": {
            "embedder": EMBEDDER_NAME,
            "reranker": RERANKER_NAME,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_k_fixed": TOP_K_FIXED,
            "matched_token_budget": MATCHED_TOKEN_BUDGET,
            "ollama_temperature": OLLAMA_TEMPERATURE,
        },
        "index": index.to_summary(),
        "stats": {
            "n_calls": stats.n_calls,
            "total_latency_s": round(stats.total_latency_s, 2),
            "avg_latency_s": (
                round(stats.total_latency_s / stats.n_calls, 2)
                if stats.n_calls else 0
            ),
            "failures": stats.failures,
        },
        "C_fixed_k": results["C_fixed_k"],
        "C_budget_400": results["C_budget_400"],
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {out_file.relative_to(REPO_ROOT)}")
    return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=list(TIERS.keys()), help="Single tier to run")
    parser.add_argument("--model", help="Single ollama model tag (e.g. qwen2.5:0.5b)")
    parser.add_argument("--all-models", action="store_true", help="Run all models for the chosen tier")
    parser.add_argument("--all", action="store_true", help="Run every tier x every model")
    parser.add_argument("--smoke", action="store_true", help="Only run the first query (for sanity checks)")
    args = parser.parse_args()

    if not (args.all or args.tier):
        parser.error("Need --tier or --all")

    tiers_to_run = list(TIERS.keys()) if args.all else [args.tier]

    if args.all or args.all_models:
        models_to_run = list(MODELS.items())
    elif args.model:
        # find by tag
        match = [(k, v) for k, v in MODELS.items() if v == args.model]
        if not match:
            parser.error(f"Unknown model {args.model}. Known: {list(MODELS.values())}")
        models_to_run = match
    else:
        parser.error("Need --model or --all-models or --all")

    print(f"=== Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    print(f"=== Loading reranker: {RERANKER_NAME}")
    reranker = CrossEncoder(RERANKER_NAME)

    # Build per-tier corpus indices once (reused across models)
    indices: dict[str, CorpusIndex] = {}
    for tk in tiers_to_run:
        cfg = TIERS[tk]
        print(f"=== Building index for tier {tk}")
        indices[tk] = build_index(cfg["source_md"], embedder)

    grand_calls = 0
    grand_latency = 0.0
    grand_failures: list[str] = []

    for tk in tiers_to_run:
        cfg = TIERS[tk]
        queries = load_queries(cfg)
        print(f"=== Tier {tk}: {len(queries)} queries from {cfg['queries_from'].name}")
        for eval_key, tag in models_to_run:
            print(f"--- Running {tag} on tier {tk}")
            stats = run_one_model_one_tier(
                tier_key=tk,
                model_eval_key=eval_key,
                model_tag=tag,
                queries=queries,
                index=indices[tk],
                embedder=embedder,
                reranker=reranker,
                out_dir=cfg["out_dir"],
                smoke=args.smoke,
            )
            grand_calls += stats.n_calls
            grand_latency += stats.total_latency_s
            grand_failures.extend(stats.failures)

    print("\n=== DONE ===")
    print(f"Total LLM calls: {grand_calls}")
    print(f"Total latency:   {grand_latency:.1f}s ({grand_latency/60:.1f} min)")
    if grand_failures:
        print(f"FAILURES ({len(grand_failures)}):")
        for f in grand_failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
