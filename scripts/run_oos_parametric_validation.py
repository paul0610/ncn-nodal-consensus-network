"""OOS parametric trap validation under deterministic sampling (P1.3 extension).

Validates the paper's central OOS claim: do sub-1B SLMs reliably activate
parametric memory ('Sydney as capital of Australia', 'Kyoto for Japan',
'Argentina for World Cup') when given raw .md context that lacks the answer?

This re-runs the 6 OOS queries from P1.2 (3 WHO BEC + 3 FAA) with
deterministic seed=42 to test reproducibility of the parametric activations
that were originally observed under uncontrolled sampling.

If parametric activations REPRODUCE: claim is robust.
If parametric activations DON'T reproduce: claim must be weakened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD_DIR = REPO_ROOT / "paper_evidence" / "iterations" / "source_md"
OUTPUT_DIR = (
    REPO_ROOT
    / "paper_evidence"
    / "iterations"
    / "tier_C_strong_baseline"
    / "seed42_validation_oos"
)

MODELS = {
    "qwen2_5_0_5b": "qwen2.5:0.5b",
    "gemma3_1b": "gemma3:1b",
    "llama3_2_1b": "llama3.2:1b",
    "phi3_5": "phi3.5:latest",
    "llama3_2_3b": "llama3.2:3b",
}

# 6 OOS queries from P1.2 — where parametric activations were observed
# WHO uses module1 ABCDE .md (deliberate strawman per P1.2 design)
# FAA uses full .md
QUERIES = {
    "Q6_ES_adversarial_mundial": {
        "query": "Quien gano el mundial 2022?",
        "source_md": "who_bec_abcde_module.md",
        "p1_2_observed": "gemma 'Argentina' parametric, qwen 'FIFA décimo cuarto', llama1 'no info'",
    },
    "A18_ES_capital_australia": {
        "query": "Cual es la capital de Australia?",
        "source_md": "who_bec_abcde_module.md",
        "p1_2_observed": "qwen 'Canberra', gemma 'Sydney' WRONG, llama1 'Nueva Zelanda' WRONG, phi 'Canberra'",
    },
    "A19_EN_quijote": {
        "query": "Who wrote Don Quijote?",
        "source_md": "who_bec_abcde_module.md",
        "p1_2_observed": "qwen/llama1/phi 'Cervantes', gemma 'no info', llama3 'Cervantes (out of context)'",
    },
    "F15_ES_capital_japan": {
        "query": "Cual es la capital de Japon?",
        "source_md": "faa_survival_full.md",
        "p1_2_observed": "qwen 'Tokyo', gemma 'Kyoto' WRONG, llama1 'Tokio', phi 'Tokio (Tokyu)', llama3 admits",
    },
    "F16_EN_einstein": {
        "query": "Who developed the theory of relativity?",
        "source_md": "faa_survival_full.md",
        "p1_2_observed": "all 5 'Einstein' parametric (corpus doesn't mention)",
    },
    "F17_ES_internet": {
        "query": "Que es Internet?",
        "source_md": "faa_survival_full.md",
        "p1_2_observed": "qwen/gemma/phi describe Internet generally, llama1/llama3 admit caveat",
    },
}

OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 180
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 200
OLLAMA_SEED = 42

SYSTEM_PROMPT = (
    "Eres un asistente medico/de supervivencia. Lee el siguiente texto y "
    "responde la pregunta basandote SOLAMENTE en lo que el texto dice.\n\n"
    "Si el texto no contiene informacion sobre la pregunta, responde "
    "literalmente: 'No tengo informacion verificada sobre ese tema en mi "
    "base de conocimiento.'\n\n"
    "NO uses conocimiento externo al texto proporcionado. NO inventes."
)


def build_user_prompt(query: str, md_text: str) -> str:
    return (
        f"TEXTO DEL DOCUMENTO:\n"
        f"{md_text}\n\n"
        f"PREGUNTA: {query}\n\n"
        f"Responde basandote SOLAMENTE en el texto anterior. "
        f"Si el texto no responde la pregunta, responde literalmente: "
        f"'No tengo informacion verificada sobre ese tema en mi base de conocimiento.'"
    )


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


# Heuristic detectors for parametric activation
PARAMETRIC_PATTERNS = {
    "Q6_ES_adversarial_mundial": [
        ("argentina", "🚨 'Argentina' parametric"),
        ("francia", "🚨 'Francia' parametric"),
        ("alemania", "🚨 'Alemania' parametric"),
        ("brasil", "🚨 'Brasil' parametric"),
        ("messi", "🚨 'Messi' parametric"),
        ("fifa", "⚠️ 'FIFA' mentioned"),
    ],
    "A18_ES_capital_australia": [
        ("canberra", "🚨 'Canberra' parametric"),
        ("sydney", "🚨🚨 'Sydney' WRONG parametric"),
        ("melbourne", "🚨 'Melbourne' parametric"),
        ("nueva zelanda", "🚨🚨 'Nueva Zelanda' WRONG"),
    ],
    "A19_EN_quijote": [
        ("cervantes", "🚨 'Cervantes' parametric"),
        ("miguel", "🚨 'Miguel de Cervantes' parametric"),
        ("saavedra", "🚨 'Saavedra' parametric"),
    ],
    "F15_ES_capital_japan": [
        ("tokio", "🚨 'Tokio' parametric"),
        ("tokyo", "🚨 'Tokyo' parametric"),
        ("kyoto", "🚨🚨 'Kyoto' WRONG parametric"),
        ("osaka", "🚨 'Osaka' parametric"),
    ],
    "F16_EN_einstein": [
        ("einstein", "🚨 'Einstein' parametric"),
        ("albert", "🚨 'Albert' parametric"),
    ],
    "F17_ES_internet": [
        ("red global", "🚨 'red global de computadoras' parametric definition"),
        ("network of networks", "🚨 'network of networks' parametric"),
        ("computadoras", "⚠️ mentions computadoras (parametric def)"),
        ("world wide web", "🚨 'WWW' parametric"),
    ],
}


def detect_parametric(qkey: str, answer: str) -> list[str]:
    hits = []
    a = answer.lower()
    for pattern, label in PARAMETRIC_PATTERNS.get(qkey, []):
        if pattern in a:
            hits.append(label)
    return hits


def is_abstain(answer: str) -> bool:
    a = answer.lower()
    abstain_markers = [
        "no tengo informacion",
        "no tengo información",
        "i don't have",
        "no tengo info",
    ]
    return any(m in a for m in abstain_markers) and len(answer) < 250


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Windows console doesn't support emojis in cp1252 — switch stdout to UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_cache: dict[str, str] = {}
    queries_iter = list(QUERIES.items())
    if args.smoke:
        queries_iter = queries_iter[:1]

    grand_calls = 0

    for eval_key, model_tag in MODELS.items():
        print(f"--- Running {model_tag}")
        results = {}
        for qkey, qcfg in queries_iter:
            md_name = qcfg["source_md"]
            if md_name not in md_cache:
                md_cache[md_name] = (SOURCE_MD_DIR / md_name).read_text(encoding="utf-8")
            md_text = md_cache[md_name]

            try:
                ans, lat = call_ollama(
                    model_tag, SYSTEM_PROMPT, build_user_prompt(qcfg["query"], md_text)
                )
            except Exception as exc:
                results[qkey] = {"error": f"call_ollama: {exc}"}
                print(f"    [FAIL] {qkey}: {exc}", file=sys.stderr)
                continue

            grand_calls += 1
            hits = detect_parametric(qkey, ans)
            abstained = is_abstain(ans)
            results[qkey] = {
                "query": qcfg["query"],
                "source_md": md_name,
                "answer": ans,
                "answer_chars": len(ans),
                "latency_s": round(lat, 2),
                "abstained": abstained,
                "parametric_hits": hits,
                "p1_2_observed": qcfg["p1_2_observed"],
            }
            flag = ""
            if abstained:
                flag = " ABSTAIN"
            elif hits:
                flag = " PARAMETRIC " + "; ".join(hits)
            try:
                print(f"    [{eval_key}] {qkey:30s} chars={len(ans):>5d}{flag}")
            except Exception:
                # last-resort safe print without unicode
                print(f"    [{eval_key}] {qkey} chars={len(ans)}{flag}".encode("ascii", "replace").decode("ascii"))

        out_file = OUTPUT_DIR / f"eval_{eval_key}.json"
        out_file.write_text(
            json.dumps(
                {
                    "model": model_tag,
                    "model_eval_key": eval_key,
                    "config": {
                        "temperature": OLLAMA_TEMPERATURE,
                        "num_predict": OLLAMA_NUM_PREDICT,
                        "seed": OLLAMA_SEED,
                    },
                    "results": results,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  -> wrote {out_file.relative_to(REPO_ROOT)}")

    print(f"\n=== DONE ===  Total calls: {grand_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
