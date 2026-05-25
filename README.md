# NCN — Nodal Consensus Network

A verification-first knowledge graph ingestion architecture for small language models. NCN extracts atomic, verified claims from technical PDFs through a multi-role LLM swarm (extractor, critics, optional tenth-man, judge, synthesiser), stores them in a property graph with attached source-authority and confidence metadata, and exposes a relevance-gated retrieval surface that lets sub-1B quantised SLMs answer faithfully on long technical reference documents.

## Why

Conventional chunked retrieval-augmented generation (RAG) is unreliable at the sub-1B parameter scale on long technical PDFs, even with strong cross-encoder reranking. The form of the evidence presented to a small answerer matters at least as much as the precision of the retriever. NCN replaces "chunks of text" with "verified atomic claims" as the unit of evidence, and adds an explicit relevance gate that short-circuits LLM invocation when no relevant claim is found.

On a 44-query diagnostic evaluation across two domains (WHO Basic Emergency Care medical guidelines + FAA Survival manual) and five small language models, verified-graph retrieval improves correctness and abstention behaviour over a strong chunked-RAG baseline (dense retrieval + `bge-reranker-large` + matched token budget) by **39 percentage points absolute** overall (69% vs 30%), and by **+64 to +68 percentage points** for sub-1B Q4-quantised models (qwen 2.5:0.5b, gemma 3:1b). Out-of-scope abstention reaches 100% with the relevance gate vs 60-62% for the strong chunked baseline. A companion paper (in preparation for arXiv) reports the full experimental protocol and ablations.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a detailed walkthrough. At a glance:

- `core/` — orchestrator, models, query router
- `swarm/` — multi-role LLM pool (extractor, critics, tenth-man, judge, synthesiser)
- `consensus/` — verification engine (claim voting, evidence gate)
- `graph/` — property-graph adapters (Kuzu default; FalkorDB/Neo4j optional), reader/writer, namespace isolation
- `retrieval/` — embedder (paraphrase-multilingual-MiniLM-L12-v2), searcher, claim serialiser, relevance gate
- `ingestion/` — source ingestors (text, PDF, web, docx, epub) with chunking and consensus-per-chunk
- `ontology/` — auto-extending ontology agent (predicate vocabulary, namespaces)
- `learning/` — autonomous-learning pipeline (decompose → search → scrape → verify → store)
- `bootstrap/` — teacher-student bootstrap (DeepSeek-as-teacher seeded claims)
- `providers/` — LLM provider abstractions (DeepSeek, OpenAI, Anthropic, Qwen, Together, Ollama, custom)
- `tests/` — pytest suite

## Setup

Requires Python 3.11+ and access to one or more LLM providers. The default production configuration uses DeepSeek for the swarm roles and the Ollama runtime for local SLM inference.

```bash
# Clone
git clone https://github.com/paul0610/ncn-nodal-consensus-network.git
cd ncn-nodal-consensus-network

# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # if you intend to run tests

# Configure API keys via environment variables (see config.yaml for the full list)
export DEEPSEEK_API_KEY=...
# Optional:
# export OPENAI_API_KEY=...
# export ANTHROPIC_API_KEY=...
# export QWEN_API_KEY=...
# export TOGETHER_API_KEY=...

# Local SLM inference (optional, for evaluation scripts)
# Install Ollama: https://ollama.com
ollama pull qwen2.5:0.5b
ollama pull gemma3:1b
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull phi3.5
```

Configuration lives in `config.yaml`. All sensitive values are loaded from environment variables via `${VAR}` substitution; no credentials are committed to the repository.

## Reproducing the paper's evaluation

The `scripts/` directory contains the runners used in the companion paper.

| Script | Experiment |
|---|---|
| `run_eval_baseline_C.py` | Strong chunked-RAG baseline (Condition C: dense + reranker + matched budget). Used for the cross-tier comparison. |
| `run_reranker_ablation.py` | Reranker ablation: dense-only vs reranker vs reranker+sentence on 8 diagnostic queries. |
| `run_quantization_ablation.py` | Quantisation ablation: Q4_K_M vs FP16 for qwen 0.5b and gemma 1b. |
| `run_loop_reproducibility.py` | Loop reproducibility study (20 seeds × qwen Q4 × F17 OOS). |
| `run_oos_parametric_validation.py` | OOS parametric trap validation under deterministic seed (raw-markdown condition). |
| `build_judge_table.py`, `build_judge_compact.py` | Render eval outputs into a human-readable comparison table. |
| `build_ablation_judge.py`, `build_quant_judge.py` | Compact views for the ablation phases. |
| `build_kappa_subsample.py` | Sample 99 cells (10 queries × 5 models × 2 conditions) for inter-rater agreement. |
| `run_deepseek_kappa_judge.py` | Independent cross-vendor judge via DeepSeek-Chat API. |
| `compute_kappa.py` | Cohen's κ + confusion matrix between two raters. |

Each runner reads from existing evaluation JSONs or builds them from scratch. See the script docstrings for usage.

## Citation

A pre-print of the companion paper is forthcoming on arXiv. Until then, please reference this repository directly. A versioned snapshot of the release accompanying the arXiv pre-print is planned for deposit on Zenodo (DOI to follow).

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)).

Note that the source documents used in the paper's evaluation have their own upstream licenses (WHO Basic Emergency Care: CC-BY-NC-SA 3.0 IGO; FAA Survival manual: US-federal public domain) and are NOT distributed in this repository. The evaluation scripts assume these documents have been obtained from their respective publishers.

## Acknowledgements

This work was developed iteratively with the assistance of Claude Opus 4.7 (Anthropic) as a research collaborator across multiple sessions. External review of intermediate drafts was provided through Genspark. The author bears sole responsibility for any errors of interpretation or implementation.

## Contact

Paul Florencio Rojas Quispe — `paulrojas0610@gmail.com`
