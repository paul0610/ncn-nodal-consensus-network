# NCN — Nodal Consensus Network
## Complete Architecture Document
### Version 1.0 — Full implementation context

> _A more detailed Spanish version of this document is preserved at the end of the file. The English version below is a functional summary intended for international readers; refer to the Spanish version for the full module-by-module specification._

---

> **NOTE FOR IMPLEMENTERS:**
> This document captures the complete design of the NCN system. Each architectural decision was made deliberately. Do not change any design decision without explicit user confirmation. Implement exactly what is described here. When something is unclear, ask before assuming. The implementation order is in Section 21.

---

## TABLE OF CONTENTS

1. Project Vision and Context
2. Technical Objectives and Differentiators
3. Technology Stack
4. Folder Structure
5. General Architecture
6. Module 1 — User Interface (CLI + FastAPI)
7. Module 2 — Orchestrator
8. Module 3 — Model Swarm (SwarmPool)
9. Module 4 — Model Providers (ModelProviderPort)
10. Module 5 — Consensus Engine
11. Module 6 — Retrieval Module
12. Module 7 — Ingestion Module
13. Module 8 — Knowledge Graph (GraphPort)
14. Module 9 — Auto-extensible Ontology Module
15. Module 10 — Reputation System (Pareto)
16. Module 11 — Internet Handling
17. Module 12 — Knowledge Bootstrap (Teacher-Student)
18. Complete Configuration (`config.yaml`)
19. Open Hooks for Phase 2 and Phase 3
20. Design Decisions and Rationale
21. Implementation Order
22. Metrics and Benchmarks for the Paper

---

## 1. PROJECT VISION AND CONTEXT

### 1.1 What is NCN?

NCN (Nodal Consensus Network) is a distributed AI system that replaces the monolithic-large-model paradigm with a *swarm* of small models operating over a living, verified knowledge graph.

Rather than a single large LLM that receives all context as plain text, NCN uses:

- N small models running in parallel (the swarm)
- A graph database as persistent, structured memory
- A consensus protocol that verifies which information is reliable before persisting it
- A system of differentiated roles (extractor, critic, tenth-man, judge, synthesiser)

### 1.2 The Problem NCN Solves

Current agentic systems (e.g., OpenClaw-style frameworks) commonly store skills, instructions, and context as `.md` files. This means every model call injects thousands of plain-text tokens into the prompt, which produces:

- High compute cost (tokens = time + money)
- Attention degradation in long contexts ("lost in the middle")
- Static skills that do not learn or evolve
- Unverified memory (any text can enter)

NCN replaces plain-text injection with surgical queries against the graph:

```
Current systems:
prompt = question + skill1.md (2,000 tokens) + skill2.md (1,500 tokens) + context (3,000 tokens)
total: ~6,500 tokens per call

NCN:
prompt = question + 8 relevant graph nodes (~200 tokens)
total: ~200 tokens per call — 97% reduction
```

### 1.3 Project Roadmap

```
PHASE 1 (current implementation) — The Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Local SLM swarm via Ollama
- Evolving knowledge graph (Kuzu DB)
- Consensus protocol with differentiated roles
- Ingestion: text, PDFs, on-demand internet
- Interface: CLI + FastAPI
- Reputation system (Pareto)
- Auto-extensible ontology
- Multi-provider support (local + cloud)

PHASE 2 (future) — The Agent (improved OpenClaw)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Skills stored as graph nodes (not .md)
- Real persistent memory across sessions via graph
- Action execution (OpenClaw-style)
- Dynamic role assignment by reputation
- Fully local, no required external dependencies

PHASE 3 (future) — Continuous Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Periodic crawler over configured sources
- Autonomous learning without user intervention
- Automatic 24/7 graph expansion
```

### 1.4 Target Audience

- Developers who want a local, private, efficient AI system
- Researchers interested in multi-model consensus
- Organisations with sensitive data unable to use cloud APIs
- Power users who want maximum configurability

---

## 2. TECHNICAL OBJECTIVES AND DIFFERENTIATORS

### 2.1 Differentiators vs Existing Projects

| Feature | OpenClaw | MiroFish | NCN |
|---|---|---|---|
| Primary objective | Automate tasks | Predict social behaviour | Verify and store knowledge |
| Memory | Local files (`.md`) | Zep Cloud (external, paid) | Local evolving graph |
| Verification | None | None | Multi-model consensus |
| Local / private | Yes | No (requires APIs) | Yes, 100% |
| Ontology | Fixed | Fixed | Auto-extensible |
| Provider-agnostic | No | No | Yes |
| Differentiated roles | No | No | Yes (5 roles) |
| Tenth-man dissent | No | No | Yes |
| Reputation system | No | No | Yes (Pareto) |

### 2.2 Technical Contributions for the Paper

1. **Provider-agnostic swarm**: a swarm that mixes local and cloud models under the same consensus protocol.
2. **Consensus-driven auto-extensible ontology**: the system creates its own node types and namespaces.
3. **Tenth-Man Protocol**: explicit dissent role that forces epistemic diversity inside the swarm.
4. **Pareto reputation**: per-model reputation tracked across sessions to weight votes.
5. **Source-authority hierarchy**: six-level trust scale (`user_direct`, `coach_model`, `consensus`, `web_verified`, `web_raw`, `slm_single`) that constrains how confident the system may be in any single claim.
6. **Relevance gate**: queries with cosine similarity below threshold short-circuit before reaching the LLM, eliminating one entire class of parametric hallucination.

---

## 3–22. MODULE DETAILS

The remaining sections (technology stack, folder layout, all twelve modules, full `config.yaml`, design rationale, implementation order, and paper-grade benchmarks) are provided in the **Spanish version below**. The Spanish version is the canonical, line-by-line specification used during implementation.

**Brief module map for the English reader:**

| Module | Location | Responsibility |
|---|---|---|
| 1. UI | `interface/` (CLI + FastAPI; **not** included in this public repo release) | Operator entry points |
| 2. Orchestrator | `core/orchestrator.py` | Coordinates all modules; routes queries |
| 3. Swarm | `swarm/` | Pool of role-tagged LLM nodes |
| 4. Providers | `providers/` | Adapters for DeepSeek, OpenAI, Anthropic, Qwen, Together, Ollama, custom |
| 5. Consensus | `consensus/` | Per-claim verification + aggregation + tenth-man |
| 6. Retrieval | `retrieval/` | Embedder, searcher, claim serialiser, relevance gate |
| 7. Ingestion | `ingestion/` | text / PDF / docx / epub / web ingestors |
| 8. Graph | `graph/` | Kuzu adapter (default); FalkorDB / Neo4j optional |
| 9. Ontology | `ontology/` | Auto-extending predicate vocabulary + namespaces |
| 10. Reputation | `swarm/reputation.py` | Pareto-weighted vote scoring |
| 11. Internet | `ingestion/web_ingestor.py` + `learning/scraper.py` | Web search + scraping with verification |
| 12. Bootstrap | `bootstrap/` | Teacher-student knowledge seeding |

For module-by-module API contracts, file-level pseudocode, configuration options, and reasoning behind each design choice, see the Spanish version starting at the next page break.

---

---

# 📚 SPANISH VERSION — Versión Completa en Español

_The original, full-detail Spanish specification follows. It is preserved verbatim as the canonical reference._

---

# NCN — Nodal Consensus Network
## Documento de Arquitectura Completo
### Versión 1.0 — Contexto completo para implementación con Claude Code

---

> **INSTRUCCIONES PARA CLAUDE CODE:**
> Este documento contiene el diseño completo del sistema NCN. Cada decisión de arquitectura fue tomada deliberadamente. No cambies ninguna decisión de diseño sin indicación explícita del usuario. Implementa exactamente lo que está aquí descrito. Cuando algo no esté claro, pregunta antes de asumir. El orden de implementación está en la sección 12.

---

## TABLA DE CONTENIDOS

1. Visión y Contexto del Proyecto
2. Objetivos Técnicos y Diferenciadores
3. Stack Tecnológico
4. Estructura de Carpetas
5. Arquitectura General
6. Módulo 1 — Interfaz de Usuario (CLI + FastAPI)
7. Módulo 2 — Orquestador
8. Módulo 3 — Enjambre de Modelos (SwarmPool)
9. Módulo 4 — Proveedores de Modelos (ModelProviderPort)
10. Módulo 5 — Motor de Consenso
11. Módulo 6 — Módulo de Recuperación (Retrieval)
12. Módulo 7 — Módulo de Ingesta
13. Módulo 8 — Grafo de Conocimiento (GraphPort)
14. Módulo 9 — Módulo de Ontología Auto-extensible
15. Módulo 10 — Sistema de Reputación (Pareto)
16. Módulo 11 — Manejo de Internet
17. Módulo 12 — Bootstrap de Conocimiento (Teacher-Student)
18. Configuración Completa (config.yaml)
19. Puertos Abiertos para Fase 2 y Fase 3
20. Decisiones de Diseño y Justificaciones
21. Orden de Implementación
22. Métricas y Benchmarks para Paper

---

## 1. VISIÓN Y CONTEXTO DEL PROYECTO

### 1.1 ¿Qué es NCN?

NCN (Nodal Consensus Network) es un sistema de inteligencia artificial distribuida que reemplaza el paradigma del modelo monolítico grande por un enjambre de modelos pequeños que operan sobre una memoria de grafos viva y verificada.

En lugar de un solo LLM grande que recibe todo el contexto como texto plano, NCN usa:
- N modelos pequeños corriendo en paralelo (el enjambre)
- Una base de datos de grafos como memoria persistente y estructurada
- Un protocolo de consenso para verificar qué información es confiable antes de persistirla
- Un sistema de roles diferenciados (extractor, crítico, décimo hombre, juez, sintetizador)

### 1.2 El Problema que Resuelve

Los sistemas agénticos actuales como OpenClaw tienen un problema fundamental: usan archivos `.md` para almacenar skills, instrucciones y contexto. Esto significa que en cada llamada al modelo se inyectan miles de tokens de texto plano al prompt, lo que genera:

- Costo computacional alto (tokens = tiempo + dinero)
- Degradación de atención del modelo en contextos largos ("lost in the middle")
- Skills estáticas que no aprenden ni evolucionan
- Memoria no verificada (cualquier texto puede entrar)

NCN resuelve esto reemplazando el texto plano por consultas quirúrgicas al grafo:

```
Sistema actual:
Prompt = pregunta + skill1.md (2000 tokens) + skill2.md (1500 tokens) + contexto (3000 tokens)
Total: ~6500 tokens por llamada

NCN:
Prompt = pregunta + 8 nodos relevantes del grafo (~200 tokens)
Total: ~200 tokens por llamada — reducción del 97%
```

### 1.3 Hoja de Ruta del Proyecto

```
FASE 1 (implementación actual) — El Motor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Enjambre local de SLMs via Ollama
- Grafo de conocimiento evolutivo (Kùzu DB)
- Protocolo de consenso con roles diferenciados
- Ingesta: texto, PDFs, internet on-demand
- Interfaz: CLI + FastAPI
- Sistema de reputación (Pareto)
- Ontología auto-extensible
- Soporte multi-provider (local + cloud)

FASE 2 (futura) — El Agente (OpenClaw mejorado)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Skills almacenadas como nodos del grafo (no .md)
- Memoria persistente real entre sesiones via grafo
- Ejecución de acciones (como OpenClaw)
- Asignación dinámica de roles por reputación
- 100% local, sin dependencias externas obligatorias

FASE 3 (futura) — Aprendizaje Continuo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Crawler periódico de fuentes configuradas
- Aprendizaje autónomo sin intervención del usuario
- Expansión automática del grafo 24/7
```

### 1.4 Audiencia del Proyecto

- Desarrolladores que quieren un sistema de IA local, privado y eficiente
- Investigadores interesados en consenso multi-modelo
- Empresas con datos sensibles que no pueden usar APIs cloud
- Usuarios avanzados que quieren máxima configurabilidad

---

## 2. OBJETIVOS TÉCNICOS Y DIFERENCIADORES

### 2.1 Diferenciadores vs Proyectos Existentes

| Característica | OpenClaw | MiroFish | NCN |
|---|---|---|---|
| Objetivo principal | Automatizar tareas | Predecir comportamiento social | Verificar y almacenar conocimiento |
| Memoria | Archivos locales (.md) | Zep Cloud (externo, de pago) | Grafo local evolutivo |
| Verificación | Ninguna | Ninguna | Consenso multi-modelo |
| Local/Privado | Sí | No (requiere APIs) | Sí, 100% |
| Ontología | Fija | Fija | Auto-extensible |
| Provider-agnostic | No | No | Sí |
| Roles diferenciados | No | No | Sí (5 roles) |
| Décimo hombre | No | No | Sí |
| Sistema de reputación | No | No | Sí (Pareto) |

### 2.2 Contribuciones Técnicas para Paper

1. **Provider-agnostic swarm**: primer enjambre que mezcla modelos locales y cloud en el mismo protocolo de consenso
2. **Ontología auto-extensible por consenso**: el sistema crea sus propios tipos de nodos y namespaces
3. **Tenth-Man Protocol**: aplicación del principio del décimo hombre para forzar diversidad epistémica en el enjambre
4. **Reputation-weighted consensus**: sistema Pareto donde el 20% de modelos más precisos pesan el 80% del voto
5. **Graph-as-verified-memory**: el grafo no es un índice de búsqueda sino una memoria verificada por consenso
6. **Knowledge Graph Bootstrap via Teacher-Student Distillation**: enseñar al grafo usando modelos más grandes localmente

### 2.3 Métricas de Éxito

- Reducción de tokens por consulta vs sistema RAG tradicional (objetivo: >80% reducción)
- Tasa de alucinaciones del enjambre vs modelo solo (objetivo: <50% de la tasa individual)
- Precisión del consenso en benchmark factual (objetivo: >85%)
- Tiempo de respuesta con 10 nodos paralelos (objetivo: <5 segundos)

---

## 3. STACK TECNOLÓGICO

### 3.1 Decisiones de Stack (NO cambiar sin justificación explícita)

| Capa | Tecnología | Versión | Justificación |
|---|---|---|---|
| Lenguaje | Python | 3.11+ | Ecosistema IA, asyncio nativo |
| Inferencia local | Ollama | Latest | API REST simple, multi-modelo, logprobs |
| Base de datos de grafos | Kùzu DB | Latest | Embebido (sin servidor), ACID, Cypher, rápido |
| Embeddings | sentence-transformers | Latest | Local, CPU, ~80MB, all-MiniLM-L6-v2 |
| Orquestación | asyncio (stdlib) | Python 3.11+ | Nativo, sin dependencias extra |
| API REST | FastAPI | Latest | Async nativo, auto-docs OpenAPI |
| Config | PyYAML | Latest | Legible, comentable |
| PDF parsing | PyMuPDF (fitz) | Latest | Rápido, sin dependencias Java |
| Web search | duckduckgo-search | Latest | Sin API key requerida |
| HTTP async | httpx | Latest | Async nativo, compatible con todas las APIs |
| Validación | Pydantic v2 | Latest | Tipos, validación, serialización |
| Testing | pytest + pytest-asyncio | Latest | Estándar Python |
| CLI | Typer | Latest | Basado en Click, tipado |
| Logging | loguru | Latest | Simple, estructurado |

### 3.2 Librerías por Módulo

```
core/
  - asyncio (stdlib)
  - pydantic
  - loguru
  - pyyaml

providers/
  - httpx (para Ollama, OpenAI-compatible APIs)
  - anthropic (SDK oficial)

graph/
  - kuzu
  - sentence-transformers
  - numpy

ingestion/
  - PyMuPDF (fitz)
  - duckduckgo-search
  - httpx
  - beautifulsoup4

interface/
  - fastapi
  - uvicorn
  - typer
  - rich (para CLI bonito)
```

---

## 4. ESTRUCTURA DE CARPETAS

```
ncn/
├── main.py                          # punto de entrada principal
├── config.yaml                      # configuración del usuario
├── config.example.yaml              # config de ejemplo con comentarios
├── requirements.txt                 # dependencias Python
├── requirements-dev.txt             # dependencias de desarrollo
├── .env.example                     # variables de entorno de ejemplo
├── README.md                        # documentación pública
├── ARCHITECTURE.md                  # este documento
│
├── core/                            # núcleo del sistema
│   ├── __init__.py
│   ├── config_loader.py             # carga y valida config.yaml
│   ├── orchestrator.py              # coordina el flujo completo
│   ├── models.py                    # modelos Pydantic compartidos
│   └── exceptions.py               # excepciones personalizadas
│
├── providers/                       # adaptadores de modelos
│   ├── __init__.py
│   ├── base.py                      # ModelProviderPort (ABC)
│   ├── ollama.py                    # OllamaProvider
│   ├── openai_compatible.py         # OpenAICompatibleProvider (base)
│   ├── openai.py                    # OpenAIProvider
│   ├── anthropic.py                 # AnthropicProvider
│   ├── deepseek.py                  # DeepSeekProvider
│   ├── qwen.py                      # QwenProvider
│   ├── together.py                  # TogetherProvider
│   ├── custom.py                    # CustomProvider (URL configurable)
│   └── factory.py                   # crea el provider correcto según config
│
├── swarm/                           # enjambre de modelos
│   ├── __init__.py
│   ├── pool.py                      # SwarmPool — gestiona N nodos
│   ├── node.py                      # SwarmNode — un nodo del enjambre
│   ├── roles.py                     # definición de roles y sus prompts
│   └── reputation.py               # sistema de reputación (Pareto)
│
├── consensus/                       # motor de consenso
│   ├── __init__.py
│   ├── engine.py                    # ConsensusEngine — orquesta las fases
│   ├── extractor.py                 # fase 1: extracción de claims
│   ├── validator.py                 # fase 2: validación cruzada NxK
│   ├── aggregator.py               # fase 3: agregación y votación
│   ├── tenth_man.py                 # rol décimo hombre
│   └── judge.py                    # árbitro de empates
│
├── graph/                           # capa de grafo
│   ├── __init__.py
│   ├── base.py                      # GraphPort (ABC) — interfaz agnóstica
│   ├── kuzu_adapter.py             # implementación Kùzu DB
│   ├── falkordb_adapter.py         # implementación FalkorDB
│   ├── neo4j_adapter.py            # implementación Neo4j
│   ├── writer.py                    # SingleWriter — única escritura al grafo
│   ├── reader.py                    # lecturas del grafo
│   └── factory.py                   # crea el adapter correcto según config
│
├── retrieval/                       # recuperación de contexto
│   ├── __init__.py
│   ├── embedder.py                  # genera embeddings con sentence-transformers
│   ├── searcher.py                  # búsqueda semántica en el grafo
│   ├── expander.py                  # expansión de vecindario (1-2 hops)
│   └── serializer.py               # serializa nodos a bullet points
│
├── ingestion/                       # ingesta de información
│   ├── __init__.py
│   ├── base.py                      # IngestorPort (ABC)
│   ├── text_ingestor.py            # texto plano del usuario
│   ├── pdf_ingestor.py             # PDFs via PyMuPDF
│   ├── web_ingestor.py             # internet on-demand
│   ├── stack_overflow_ingestor.py  # Stack Overflow API
│   ├── arxiv_ingestor.py           # arXiv papers
│   ├── github_ingestor.py          # GitHub docs
│   ├── rss_ingestor.py             # feeds RSS
│   ├── documentation_ingestor.py   # docs oficiales
│   └── factory.py                   # crea ingestores según config
│
├── ontology/                        # ontología auto-extensible
│   ├── __init__.py
│   ├── agent.py                     # OntologyAgent
│   ├── classifier.py               # clasifica entidades en tipos
│   └── deduplicator.py            # evita tipos redundantes
│
├── bootstrap/                       # aprendizaje teacher-student
│   ├── __init__.py
│   ├── teacher.py                   # genera Q&A desde modelo teacher
│   └── pipeline.py                 # pipeline completo de bootstrap
│
├── interface/                       # interfaces de usuario
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── app.py                   # CLI con Typer
│   └── api/
│       ├── __init__.py
│       ├── app.py                   # FastAPI app
│       ├── routes/
│       │   ├── query.py             # POST /query
│       │   ├── ingest.py            # POST /ingest
│       │   ├── graph.py             # GET /graph/*
│       │   └── health.py           # GET /health
│       └── middleware.py           # auth, logging, CORS
│
├── data/                            # datos persistentes (gitignored)
│   ├── graph/                       # base de datos Kùzu
│   ├── reputation.json             # historial de reputación
│   ├── metrics.jsonl               # métricas por sesión
│   └── cache/                      # caché de búsquedas web
│
├── logs/                            # logs del sistema (gitignored)
│
├── configs/                         # configs de experimentos para paper
│   ├── default.yaml
│   ├── experiment_pareto.yaml
│   ├── experiment_consensus.yaml
│   ├── experiment_swarm_size.yaml
│   ├── ablation_no_tenth_man.yaml
│   └── ablation_no_reputation.yaml
│
└── tests/
    ├── __init__.py
    ├── test_consensus.py
    ├── test_graph.py
    ├── test_providers.py
    ├── test_retrieval.py
    └── test_orchestrator.py
```

---

## 5. ARQUITECTURA GENERAL

### 5.1 Flujo Completo de una Consulta

```
Usuario envía: "¿Qué es CRISPR y cómo funciona?"
                        │
                        ▼
              ┌─────────────────┐
              │   INTERFAZ      │
              │  CLI o FastAPI  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  ORQUESTADOR    │  ← coordina todo
              └────┬────┬───────┘
                   │    │
          ┌────────┘    └──────────┐
          ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│   RETRIEVAL      │    │   ¿Buscar en     │
│  Embedding query │    │   internet?      │
│  → Top-5 nodos  │    │  (modo auto)     │
│  → 1-2 hops     │    └────────┬─────────┘
│  → ~200 tokens  │             │ si aplica
└────────┬─────────┘             ▼
         │              ┌──────────────────┐
         │              │  WEB INGESTOR    │
         │              │  → extrae texto  │
         │              │  → verifica      │
         │              └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────────┐
              │    ENJAMBRE     │  asyncio.gather()
              │  (N modelos en  │
              │    paralelo)    │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │  MOTOR DE       │
              │  CONSENSO       │
              │                 │
              │  Fase 1: Extract│
              │  Fase 2: Validate│
              │  Fase 3: Agree  │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │  SINGLE WRITER  │  ← único proceso que escribe
              │  → grafo crece  │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │  SINTETIZADOR   │
              │  → respuesta    │
              │    natural      │
              └────────┬────────┘
                       │
                       ▼
              Respuesta al usuario
```

### 5.2 Principios de Diseño (NO violar)

1. **Single Responsibility**: cada módulo hace UNA sola cosa
2. **Port & Adapter pattern**: GraphPort, ModelProviderPort, IngestorPort son interfaces abstractas. Las implementaciones concretas están detrás de esos puertos
3. **Single Writer**: SOLO el SingleWriter escribe al grafo. Los modelos NUNCA escriben directamente
4. **Models are Stateless**: los modelos del enjambre no guardan estado. El estado vive en el grafo
5. **Config-driven**: TODOS los parámetros vienen de config.yaml. Sin hardcoding
6. **Async First**: todo el pipeline es async/await. Nunca blocking I/O en el camino crítico
7. **Fail Gracefully**: si un nodo del enjambre falla, el sistema continúa con los demás

---

## 6. MÓDULO 1 — INTERFAZ DE USUARIO

### 6.1 CLI (interface/cli/app.py)

Construida con Typer y Rich para output bonito.

**Comandos:**

```bash
# Consulta principal
ncn query "¿Qué es el machine learning?"

# Ingestar un PDF
ncn ingest pdf ./documento.pdf

# Ingestar texto
ncn ingest text "El fotón es una partícula sin masa..."

# Ver estado del grafo
ncn graph stats

# Ver nodos del grafo
ncn graph nodes --limit 20 --type concepto

# Ver reputación del enjambre
ncn swarm status

# Correr con config alternativa (para experimentos)
ncn query "pregunta" --config configs/experiment_pareto.yaml

# Limpiar grafo (requiere confirmación)
ncn graph clear --confirm

# Health check
ncn health
```

**Output en CLI:**
- Usar Rich para colores y tablas
- Mostrar progreso del enjambre en tiempo real (qué modelos están procesando)
- Mostrar claims extraídos y sus scores de consenso
- Mostrar nodos nuevos escritos al grafo

### 6.2 FastAPI (interface/api/)

**Endpoints:**

```
POST /query
  Body: { "text": "pregunta", "session_id": "opcional" }
  Response: { "answer": "...", "claims": [...], "nodes_used": [...], "nodes_created": [...] }

POST /ingest/text
  Body: { "text": "...", "namespace": "general" }
  Response: { "claims_extracted": N, "claims_verified": N, "nodes_created": N }

POST /ingest/pdf
  Body: multipart/form-data con archivo PDF
  Response: { "pages_processed": N, "claims_extracted": N, "nodes_created": N }

GET /graph/stats
  Response: { "total_nodes": N, "total_edges": N, "namespaces": [...], "node_types": [...] }

GET /graph/nodes
  Query params: type, namespace, limit, offset
  Response: { "nodes": [...], "total": N }

GET /graph/node/{node_id}
  Response: nodo completo con sus relaciones

GET /swarm/status
  Response: { "nodes": [...con reputación...], "total": N }

GET /health
  Response: { "status": "ok", "graph": "ok", "swarm": "ok", "ollama": "ok" }
```

**Autenticación:**
- Opcional via config (`api_key_required: true`)
- Header: `X-API-Key: valor`

---

## 7. MÓDULO 2 — ORQUESTADOR

### 7.1 Responsabilidades (core/orchestrator.py)

El Orquestador es el director de orquesta. Coordina todos los módulos pero NO implementa lógica de ninguno.

```python
class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.swarm_pool = SwarmPool(config)
        self.consensus_engine = ConsensusEngine(config)
        self.graph_reader = GraphReader(config)
        self.graph_writer = SingleWriter(config)
        self.retriever = Retriever(config)
        self.web_ingestor = WebIngestor(config)
        self.ontology_agent = OntologyAgent(config)
        self.reputation_system = ReputationSystem(config)

    async def process_query(self, query: str, session_id: str) -> QueryResponse:
        # 1. Recuperar contexto del grafo
        graph_context = await self.retriever.retrieve(query)
        
        # 2. Decidir si buscar en internet
        if await self._should_search_web(query, graph_context):
            web_context = await self.web_ingestor.search(query)
            context = self._merge_contexts(graph_context, web_context)
        else:
            context = graph_context
        
        # 3. Lanzar enjambre y obtener consenso
        result = await self.consensus_engine.run(query, context, self.swarm_pool)
        
        # 4. Actualizar grafo con claims verificados
        await self.graph_writer.write_verified_claims(result.verified_claims)
        
        # 5. Actualizar reputación de modelos
        await self.reputation_system.update(result.model_performances)
        
        # 6. Actualizar temperaturas del grafo
        await self.graph_writer.update_temperatures(graph_context.nodes_used)
        
        # 7. Retornar respuesta sintetizada
        return QueryResponse(
            answer=result.synthesized_answer,
            claims=result.verified_claims,
            nodes_used=graph_context.nodes_used,
            nodes_created=result.new_nodes
        )

    async def _should_search_web(self, query: str, context: GraphContext) -> bool:
        mode = self.config.internet.search_mode
        if mode == "never": return False
        if mode == "always": return True
        # mode == "auto"
        low_confidence = context.avg_confidence < self.config.internet.graph_confidence_threshold
        low_coverage = len(context.nodes) < 3
        temporal = self._has_temporal_keywords(query)
        return low_confidence or low_coverage or temporal

    def _has_temporal_keywords(self, query: str) -> bool:
        keywords = ["hoy", "actual", "último", "ahora", "reciente", "2025", "2026", "today", "current", "latest"]
        return any(kw in query.lower() for kw in keywords)
```

---

## 8. MÓDULO 3 — ENJAMBRE DE MODELOS (SwarmPool)

### 8.1 SwarmNode (swarm/node.py)

Un nodo es una instancia de un modelo con un rol asignado.

```python
@dataclass
class SwarmNode:
    node_id: str           # uuid único
    provider: ModelProviderPort
    model: str             # nombre del modelo
    role: NodeRole         # extractor | critic | tenth_man | judge | synthesizer
    temperature: float     # temperatura de sampling
    reputation_score: float = 1.0  # empieza neutral
```

### 8.2 SwarmPool (swarm/pool.py)

Gestiona todos los nodos y controla la concurrencia.

```python
class SwarmPool:
    def __init__(self, config: Config):
        self.nodes: list[SwarmNode] = self._build_nodes(config)
        self.semaphore = asyncio.Semaphore(config.swarm.max_concurrent)

    async def run_role(self, role: NodeRole, prompt: str) -> list[NodeResponse]:
        """Lanza en paralelo todos los nodos con el rol especificado."""
        nodes = [n for n in self.nodes if n.role == role]
        tasks = [self._run_node(node, prompt) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filtrar excepciones, loggear errores, retornar los que funcionaron
        return [r for r in results if isinstance(r, NodeResponse)]

    async def _run_node(self, node: SwarmNode, prompt: str) -> NodeResponse:
        async with self.semaphore:
            try:
                response = await node.provider.complete(
                    prompt=prompt,
                    model=node.model,
                    temperature=node.temperature,
                    timeout=self.config.swarm.request_timeout
                )
                return NodeResponse(
                    node_id=node.node_id,
                    role=node.role,
                    content=response.content,
                    model=node.model,
                    provider=node.provider.name
                )
            except Exception as e:
                logger.error(f"Node {node.node_id} failed: {e}")
                raise

    def _build_nodes(self, config: Config) -> list[SwarmNode]:
        """Construye los nodos según config.yaml."""
        nodes = []
        provider_factory = ProviderFactory(config)
        for node_config in config.swarm.nodes:
            provider = provider_factory.create(node_config.provider)
            for i in range(node_config.count):
                nodes.append(SwarmNode(
                    node_id=f"{node_config.model}-{node_config.role}-{i}",
                    provider=provider,
                    model=node_config.model,
                    role=NodeRole(node_config.role),
                    temperature=node_config.temperature or config.swarm.temperature
                ))
        return nodes
```

### 8.3 Roles (swarm/roles.py)

```python
class NodeRole(str, Enum):
    EXTRACTOR = "extractor"
    CRITIC = "critic"
    TENTH_MAN = "tenth_man"
    JUDGE = "judge"
    SYNTHESIZER = "synthesizer"

ROLE_SYSTEM_PROMPTS = {
    NodeRole.EXTRACTOR: """
Eres un extractor de conocimiento preciso. Tu única tarea es leer el texto 
y contexto del grafo proporcionados y extraer afirmaciones factuales 
estructuradas (claims).

REGLAS:
- Extrae SOLO hechos verificables, no opiniones
- Cada claim debe ser una tripleta: sujeto → predicado → objeto
- Asigna un nivel de confianza entre 0.0 y 1.0
- Responde ÚNICAMENTE en JSON válido, sin texto adicional

FORMATO DE RESPUESTA:
{
  "claims": [
    {
      "subject": "nombre de la entidad",
      "predicate": "relación o acción",
      "object": "entidad o valor relacionado",
      "confidence": 0.85,
      "temporal": false,
      "date": null,
      "source_text": "fragmento exacto del texto que justifica este claim"
    }
  ]
}
""",

    NodeRole.CRITIC: """
Eres un validador crítico y riguroso. Recibirás una afirmación factual (claim).
Tu única tarea es determinar si es consistente con el texto fuente proporcionado.

REGLAS:
- Vota TRUE solo si el claim está CLARAMENTE respaldado por el texto
- Vota FALSE si el claim contradice el texto, es ambiguo, o no está respaldado
- Asigna confianza a tu voto entre 0.0 y 1.0
- Sé estricto: la duda beneficia a FALSE
- Responde ÚNICAMENTE en JSON válido

FORMATO DE RESPUESTA:
{
  "vote": true,
  "confidence": 0.87,
  "reason": "explicación breve de una línea"
}
""",

    NodeRole.TENTH_MAN: """
Eres el Abogado del Diablo. El enjambre está convergiendo hacia estas conclusiones.
Tu trabajo NO es validar — tu trabajo es REFUTAR.

REGLAS:
- ASUME que el consenso está EQUIVOCADO
- Busca activamente evidencia contraria en el texto fuente
- Genera claims alternativos o contradictorios
- Si genuinamente no puedes refutar, indica "no_refutation_found"
- Responde ÚNICAMENTE en JSON válido

FORMATO DE RESPUESTA:
{
  "refutation_possible": true,
  "counter_claims": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "confidence": 0.70,
      "challenges_claim_id": "id del claim que refuta"
    }
  ],
  "reasoning": "por qué el consenso podría estar equivocado"
}
""",

    NodeRole.JUDGE: """
Eres el árbitro final e imparcial. Dos hipótesis están en disputa y no hay consenso.
Tu veredicto es definitivo.

REGLAS:
- Analiza AMBAS hipótesis con igual rigor
- Basa tu veredicto SOLO en el texto fuente, no en conocimiento previo
- Explica tu razonamiento brevemente
- Responde ÚNICAMENTE en JSON válido

FORMATO DE RESPUESTA:
{
  "winner": "A",
  "confidence": 0.78,
  "reasoning": "El texto en el párrafo 2 establece claramente que...",
  "dissenting_notes": "aunque B tiene mérito en el punto X"
}
""",

    NodeRole.SYNTHESIZER: """
Eres el sintetizador final. Los siguientes claims han sido VERIFICADOS por consenso 
del enjambre. Tu trabajo es generar una respuesta clara y natural para el usuario.

REGLAS ABSOLUTAS:
- SOLO usa información de los claims verificados proporcionados
- NO inventes ni agregues información que no esté en los claims
- Si los claims son insuficientes para responder completamente, dilo
- Responde en el mismo idioma que la pregunta original del usuario
- Sé claro, conciso y útil

NO generes JSON. Genera lenguaje natural directamente.
"""
}
```

---

## 9. MÓDULO 4 — PROVEEDORES DE MODELOS

### 9.1 Interfaz Base (providers/base.py)

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ModelResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: int | None = None
    latency_ms: float | None = None

class ModelProviderPort(ABC):
    """
    Puerto abstracto para proveedores de modelos.
    TODAS las implementaciones deben heredar de esta clase.
    El nombre del provider se usa para logging y métricas.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del provider (ej: 'ollama', 'openai')"""
        pass

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: int = 30,
        system_prompt: str | None = None
    ) -> ModelResponse:
        """Genera una completación. Lanza ModelProviderError si falla."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifica que el provider está disponible."""
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Lista modelos disponibles en este provider."""
        pass
```

### 9.2 Ollama Provider (providers/ollama.py)

```python
class OllamaProvider(ModelProviderPort):
    """
    Provider para Ollama local.
    Endpoint: http://localhost:11434 (configurable)
    No requiere API key.
    """
    
    @property
    def name(self) -> str:
        return "ollama"

    async def complete(self, prompt, model, temperature=0.7, timeout=30, system_prompt=None):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            data = response.json()
            return ModelResponse(
                content=data["response"],
                model=model,
                provider=self.name,
                tokens_used=data.get("eval_count")
            )
```

### 9.3 OpenAI-Compatible Provider (providers/openai_compatible.py)

Base para OpenAI, DeepSeek, Qwen, Together, y cualquier API compatible:

```python
class OpenAICompatibleProvider(ModelProviderPort):
    """
    Provider base para cualquier API compatible con OpenAI.
    DeepSeek, Qwen, Together, y modelos custom usan esta base.
    Solo cambia base_url y api_key.
    """
    
    async def complete(self, prompt, model, temperature=0.7, timeout=30, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout
            )
            data = response.json()
            return ModelResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.name,
                tokens_used=data.get("usage", {}).get("total_tokens")
            )
```

### 9.4 Provider Factory (providers/factory.py)

```python
PROVIDER_REGISTRY = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "together": TogetherProvider,
    "custom": CustomProvider,
}

class ProviderFactory:
    def create(self, provider_name: str) -> ModelProviderPort:
        cls = PROVIDER_REGISTRY.get(provider_name)
        if not cls:
            raise ValueError(f"Provider '{provider_name}' no reconocido. "
                           f"Disponibles: {list(PROVIDER_REGISTRY.keys())}")
        return cls(self.config.providers[provider_name])
```

---

## 10. MÓDULO 5 — MOTOR DE CONSENSO

### 10.1 Flujo del Consenso (consensus/engine.py)

```python
class ConsensusEngine:
    async def run(self, query: str, context: GraphContext, swarm: SwarmPool) -> ConsensusResult:
        
        # === FASE 1: EXTRACCIÓN ===
        # Todos los extractores corren en paralelo
        extractor_prompt = self._build_extractor_prompt(query, context)
        extractor_responses = await swarm.run_role(NodeRole.EXTRACTOR, extractor_prompt)
        all_claims = self._parse_claims(extractor_responses)
        
        # El décimo hombre también corre en paralelo si está habilitado
        # y si el consenso emergente es muy alto (>tenth_man_triggers_above)
        if self.config.consensus.tenth_man_enabled:
            consensus_preview = self._quick_consensus_preview(all_claims)
            if consensus_preview > self.config.consensus.tenth_man_triggers_above:
                tenth_man_prompt = self._build_tenth_man_prompt(query, context, all_claims)
                tenth_man_responses = await swarm.run_role(NodeRole.TENTH_MAN, tenth_man_prompt)
                counter_claims = self._parse_counter_claims(tenth_man_responses)
                all_claims = all_claims + counter_claims  # enriquece el pool
        
        # === FASE 2: VALIDACIÓN CRUZADA (NxK) ===
        # Cada claim es validado por K modelos aleatorios (no todos)
        k = self.config.consensus.validators_per_claim
        validation_tasks = []
        for claim in all_claims:
            validators = self._select_validators(swarm, k, exclude_model=claim.source_model)
            for validator in validators:
                prompt = self._build_critic_prompt(query, context, claim)
                validation_tasks.append((claim.claim_id, validator, prompt))
        
        # Lanzar todas las validaciones en paralelo
        validation_results = await self._run_validations(validation_tasks, swarm)
        
        # === FASE 3: AGREGACIÓN Y DECISIÓN ===
        for claim in all_claims:
            votes = validation_results[claim.claim_id]
            # Score ponderado por reputación del modelo votante
            score = self._compute_weighted_score(votes, swarm)
            
            if score >= self.config.consensus.verification_threshold:
                claim.status = ClaimStatus.VERIFIED
            elif score <= self.config.consensus.discard_threshold:
                claim.status = ClaimStatus.DISCARDED
            else:
                claim.status = ClaimStatus.UNCERTAIN
                # Activar juez si está configurado
                if self.config.consensus.auto_judge_uncertain:
                    judgment = await self._invoke_judge(query, context, claim, swarm)
                    claim.status = ClaimStatus.VERIFIED if judgment.winner == "A" else ClaimStatus.DISCARDED
        
        # === FASE 4: SÍNTESIS ===
        verified_claims = [c for c in all_claims if c.status == ClaimStatus.VERIFIED]
        synthesizer_prompt = self._build_synthesizer_prompt(query, verified_claims)
        synthesis_responses = await swarm.run_role(NodeRole.SYNTHESIZER, synthesizer_prompt)
        final_answer = self._select_best_synthesis(synthesis_responses)
        
        return ConsensusResult(
            verified_claims=verified_claims,
            uncertain_claims=[c for c in all_claims if c.status == ClaimStatus.UNCERTAIN],
            discarded_claims=[c for c in all_claims if c.status == ClaimStatus.DISCARDED],
            synthesized_answer=final_answer,
            model_performances=self._compute_performances(extractor_responses, validation_results)
        )

    def _compute_weighted_score(self, votes: list[Vote], swarm: SwarmPool) -> float:
        """
        Score ponderado por reputación del modelo votante.
        Implementa el principio Pareto: el top 20% pesa más.
        """
        if not votes:
            return 0.0
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for vote in votes:
            node = swarm.get_node(vote.node_id)
            reputation = node.reputation_score
            
            # Elite bonus (top 20%)
            if self.config.reputation.enabled:
                all_reputations = sorted([n.reputation_score for n in swarm.nodes], reverse=True)
                elite_cutoff = all_reputations[int(len(all_reputations) * self.config.reputation.elite_ratio)]
                if reputation >= elite_cutoff:
                    reputation *= self.config.reputation.elite_vote_multiplier
            
            weight = reputation * vote.confidence
            weighted_sum += weight * (1.0 if vote.vote else 0.0)
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0.0
```

### 10.2 Modelos Pydantic del Consenso (core/models.py)

```python
class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    predicate: str
    object: str
    confidence: float
    temporal: bool = False
    date: str | None = None
    source_text: str | None = None
    source_model: str
    status: ClaimStatus = ClaimStatus.PENDING
    is_counter_claim: bool = False      # True si viene del décimo hombre
    challenges_claim_id: str | None = None

class Vote(BaseModel):
    node_id: str
    claim_id: str
    vote: bool
    confidence: float
    reason: str | None = None

class ConsensusResult(BaseModel):
    verified_claims: list[Claim]
    uncertain_claims: list[Claim]
    discarded_claims: list[Claim]
    synthesized_answer: str
    model_performances: dict[str, float]  # node_id → performance score
    session_id: str
    processing_time_ms: float

class ClaimStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    DISCARDED = "discarded"
```

---

## 11. MÓDULO 6 — RECUPERACIÓN (Retrieval)

### 11.1 Flujo de Recuperación (retrieval/searcher.py)

```python
class Retriever:
    def __init__(self, config: Config):
        self.embedder = Embedder(config)        # sentence-transformers
        self.graph_reader = GraphReader(config)
        self.expander = NeighborhoodExpander(config)
        self.serializer = NodeSerializer(config)

    async def retrieve(self, query: str) -> GraphContext:
        # Paso 1: Embedding de la query
        query_vector = await self.embedder.embed(query)
        
        # Paso 2: Top-K nodos más similares semánticamente
        seed_nodes = await self.graph_reader.semantic_search(
            vector=query_vector,
            top_k=self.config.retrieval.top_k_nodes
        )
        
        # Paso 3: Expansión de vecindario (1-2 hops)
        expanded_nodes = await self.expander.expand(
            seed_nodes=seed_nodes,
            hops=self.config.retrieval.hop_depth
        )
        
        # Paso 4: Filtro por temperatura (Pareto del grafo)
        # Nodos más consultados tienen prioridad
        ranked_nodes = sorted(expanded_nodes, key=lambda n: n.temperature, reverse=True)
        
        # Paso 5: Aplicar presupuesto de tokens
        selected_nodes = self.serializer.apply_budget(
            nodes=ranked_nodes,
            budget=self.config.retrieval.token_budget
        )
        
        # Paso 6: Serializar como bullet points (NO narrativo)
        serialized = self.serializer.serialize(selected_nodes)
        
        avg_confidence = sum(n.confidence for n in selected_nodes) / len(selected_nodes) if selected_nodes else 0.0
        
        return GraphContext(
            nodes=selected_nodes,
            serialized_context=serialized,
            avg_confidence=avg_confidence,
            nodes_used=[n.node_id for n in selected_nodes]
        )
```

### 11.2 Serialización de Nodos (retrieval/serializer.py)

Los nodos se convierten a bullet points, NO a texto narrativo. Esto consume ~60% menos tokens y los modelos pequeños lo procesan mejor.

```python
def serialize(self, nodes: list[GraphNode]) -> str:
    """
    Convierte nodos del grafo a bullet points eficientes.
    
    Ejemplo output:
    • Einstein → desarrolló → Teoría de la Relatividad (conf: 0.94, año: 1905)
    • Teoría de la Relatividad → publicada_en → Annalen der Physik (conf: 0.88)
    • Lorentz → influyó_en → Einstein (conf: 0.76) [contested]
    """
    lines = []
    for node in nodes:
        for relation in node.relations:
            line = f"• {node.name} → {relation.predicate} → {relation.target}"
            meta = []
            if relation.confidence:
                meta.append(f"conf: {relation.confidence:.2f}")
            if relation.date:
                meta.append(f"año: {relation.date}")
            if relation.contested:
                meta.append("contested")
            if meta:
                line += f" ({', '.join(meta)})"
            lines.append(line)
    return "\n".join(lines)
```

---

## 12. MÓDULO 7 — INGESTA

### 12.1 Interfaz Base (ingestion/base.py)

```python
class IngestorPort(ABC):
    @abstractmethod
    async def ingest(self, source: Any) -> list[RawContent]:
        """
        Ingesta desde la fuente y retorna contenido crudo.
        El contenido crudo luego pasa por el pipeline de consenso.
        """
        pass
```

### 12.2 PDF Ingestor (ingestion/pdf_ingestor.py)

```python
class PDFIngestor(IngestorPort):
    """
    Usa PyMuPDF (fitz) para extracción.
    NO usa PyPDF2 ni pdfplumber.
    """
    async def ingest(self, pdf_path: str) -> list[RawContent]:
        import fitz
        doc = fitz.open(pdf_path)
        contents = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                contents.append(RawContent(
                    text=text,
                    source=f"pdf:{pdf_path}:page{page_num+1}",
                    source_type="pdf"
                ))
        return contents
```

### 12.3 Web Ingestor (ingestion/web_ingestor.py)

```python
class WebIngestor(IngestorPort):
    """
    Búsqueda web on-demand.
    Por defecto usa DuckDuckGo (sin API key).
    El contenido pasa por limpieza antes de ser procesado.
    """
    
    async def search(self, query: str) -> WebContext:
        if not self.config.internet.cache_enabled:
            return await self._do_search(query)
        
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached and not self._is_expired(cached):
            return cached
        
        result = await self._do_search(query)
        self._cache[cache_key] = result
        return result

    async def _do_search(self, query: str) -> WebContext:
        results = []
        
        for source_config in self.config.internet.sources:
            if not source_config.enabled:
                continue
            
            if source_config.type == "web_search":
                raw = await self._search_web(query, source_config)
            elif source_config.type == "wikipedia":
                raw = await self._search_wikipedia(query, source_config)
            elif source_config.type == "stackoverflow":
                raw = await self._search_stackoverflow(query, source_config)
            # ... etc
            
            results.extend(raw)
        
        cleaned = [self._clean_html(r) for r in results]
        return WebContext(contents=cleaned, query=query)

    def _clean_html(self, raw: str) -> str:
        """Elimina HTML, scripts, anuncios. Solo texto limpio."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
```

---

## 13. MÓDULO 8 — GRAFO DE CONOCIMIENTO

### 13.1 Interfaz Agnóstica (graph/base.py)

```python
class GraphPort(ABC):
    """
    Puerto abstracto para la base de datos de grafos.
    
    TODAS las implementaciones concretas (Kùzu, FalkorDB, Neo4j)
    deben implementar esta interfaz.
    
    El resto del sistema SOLO habla con GraphPort.
    NUNCA importes KùzuAdapter directamente desde fuera del módulo graph/.
    """

    @abstractmethod
    async def create_node(self, node: GraphNode) -> str:
        """Crea un nodo. Retorna node_id."""
        pass

    @abstractmethod
    async def create_relation(self, relation: GraphRelation) -> None:
        pass

    @abstractmethod
    async def update_node(self, node_id: str, updates: dict) -> None:
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        pass

    @abstractmethod
    async def get_neighbors(self, node_id: str, hops: int = 1) -> list[GraphNode]:
        pass

    @abstractmethod
    async def semantic_search(self, vector: list[float], top_k: int) -> list[GraphNode]:
        pass

    @abstractmethod
    async def get_stats(self) -> GraphStats:
        pass

    @abstractmethod
    async def begin_transaction(self):
        pass

    @abstractmethod
    async def commit(self):
        pass

    @abstractmethod
    async def rollback(self):
        pass
```

### 13.2 Estructura de Nodos

```python
class GraphNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    node_type: str                    # tipo del nodo (configurable)
    aliases: list[str] = []           # nombres alternativos
    namespace: str = "general"        # namespace del nodo
    confidence: float = 1.0          # qué tan verificado está
    temperature: float = 0.5          # qué tan frecuentemente se consulta
    embedding: list[float] | None = None  # vector para búsqueda semántica
    source: str | None = None         # de dónde viene (pdf:archivo.pdf, web:url, user)
    verified_by: list[str] = []       # ["consenso", "juez", "bootstrap"]
    uncertain: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = {}               # campo libre para extensibilidad

class GraphRelation(BaseModel):
    relation_id: str = Field(default_factory=lambda: str(uuid4()))
    source_node_id: str
    target_node_id: str
    predicate: str                    # el verbo de la relación
    weight: float = 1.0
    confidence: float = 1.0
    temporal: bool = False
    date: str | None = None
    contested: bool = False           # True si el décimo hombre lo disputó
    source: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### 13.3 Single Writer (graph/writer.py)

```python
class SingleWriter:
    """
    EL ÚNICO PROCESO QUE ESCRIBE AL GRAFO.
    
    Los modelos del enjambre NUNCA escriben directamente.
    El motor de consenso NUNCA escribe directamente.
    SOLO este writer escribe.
    
    Esto elimina race conditions por diseño.
    Patrón: Single-Writer / Multiple-Reader.
    """
    
    def __init__(self, graph: GraphPort, ontology_agent: OntologyAgent, embedder: Embedder):
        self.graph = graph
        self.ontology_agent = ontology_agent
        self.embedder = embedder
        self._write_lock = asyncio.Lock()  # lock adicional por seguridad

    async def write_verified_claims(self, claims: list[Claim]) -> list[str]:
        """Escribe claims verificados al grafo. Retorna node_ids creados."""
        async with self._write_lock:
            created_node_ids = []
            async with self.graph.begin_transaction():
                try:
                    for claim in claims:
                        subject_id = await self._upsert_entity(claim.subject, claim)
                        object_id = await self._upsert_entity(claim.object, claim)
                        await self._upsert_relation(subject_id, object_id, claim)
                        created_node_ids.extend([subject_id, object_id])
                    await self.graph.commit()
                except Exception as e:
                    await self.graph.rollback()
                    logger.error(f"Write failed, rolled back: {e}")
                    raise
            return list(set(created_node_ids))

    async def _upsert_entity(self, entity_name: str, claim: Claim) -> str:
        """
        Crea el nodo si no existe, o refuerza su confianza si ya existe.
        Implementa CREATE (nuevo) o REINFORCE (existente).
        También invoca OntologyAgent si el tipo no está definido.
        """
        existing = await self.graph.find_node_by_name(entity_name)
        
        if existing:
            # REINFORCE: el claim confirma algo que ya sabíamos
            await self.graph.update_node(existing.node_id, {
                "confidence": min(1.0, existing.confidence + self.config.graph.confidence_boost),
                "last_accessed": datetime.utcnow()
            })
            return existing.node_id
        else:
            # CREATE: entidad nueva
            node_type = await self.ontology_agent.classify(entity_name, claim)
            embedding = await self.embedder.embed(entity_name)
            node = GraphNode(
                name=entity_name,
                node_type=node_type,
                namespace=self.config.graph.default_namespace,
                confidence=claim.confidence,
                embedding=embedding,
                source=claim.source_text,
                verified_by=["consenso"]
            )
            return await self.graph.create_node(node)

    async def update_temperatures(self, node_ids: list[str]) -> None:
        """Calienta los nodos que fueron consultados en esta sesión."""
        async with self._write_lock:
            for node_id in node_ids:
                node = await self.graph.get_node(node_id)
                if node:
                    new_temp = min(1.0, node.temperature + self.config.graph.temperature_boost)
                    await self.graph.update_node(node_id, {"temperature": new_temp})

    async def cool_down_all_nodes(self) -> None:
        """
        Llamar al final de cada sesión.
        Implementa el decay de temperatura (Pareto del grafo).
        Nodos que no se consultan se enfrían y eventualmente se archivan.
        """
        async with self._write_lock:
            all_nodes = await self.graph.get_all_nodes()
            for node in all_nodes:
                new_temp = max(0.0, node.temperature - self.config.graph.temperature_decay)
                updates = {"temperature": new_temp}
                if new_temp <= self.config.graph.cold_node_threshold:
                    updates["archived"] = True  # nodo frío, se archiva
                await self.graph.update_node(node.node_id, updates)
```

---

## 14. MÓDULO 9 — ONTOLOGÍA AUTO-EXTENSIBLE

### 14.1 OntologyAgent (ontology/agent.py)

```python
class OntologyAgent:
    """
    Clasifica entidades en tipos de nodos.
    Si ningún tipo existente encaja, propone uno nuevo.
    Los nuevos tipos son verificados por consenso mini (3 modelos).
    
    Concepto académico: "ontología auto-extensible por consenso"
    Es una de las contribuciones originales del paper.
    """
    
    def __init__(self, config: Config, swarm: SwarmPool, graph: GraphPort):
        self.config = config
        self.swarm = swarm
        self.graph = graph
        self.deduplicator = SemanticDeduplicator(config)
        self._known_types: set[str] = set(config.graph.node_types)

    async def classify(self, entity_name: str, context: Claim) -> str:
        """
        Clasifica una entidad. Si no encaja en ningún tipo conocido,
        propone un tipo nuevo y lo valida por consenso.
        """
        # Intentar clasificar con tipos existentes
        best_type, confidence = await self._classify_with_existing(entity_name, context)
        
        if confidence >= 0.75:
            return best_type
        
        # No encaja bien → proponer tipo nuevo
        if self.config.ontology.auto_extend:
            return await self._propose_new_type(entity_name, context)
        
        return "concepto"  # fallback

    async def _propose_new_type(self, entity_name: str, context: Claim) -> str:
        """
        Propone un nuevo tipo de nodo.
        El tipo propuesto pasa por verificación de 3 modelos antes de crearse.
        """
        # Verificar límite de tipos custom
        custom_count = len(self._known_types) - len(self.config.graph.node_types)
        if custom_count >= self.config.ontology.max_custom_types:
            logger.warning("Límite de tipos custom alcanzado. Usando 'concepto'.")
            return "concepto"
        
        # Pedir a un modelo que proponga el tipo
        prompt = f"""
        La entidad "{entity_name}" no encaja en estos tipos existentes: {list(self._known_types)}
        Contexto de la afirmación: {context.predicate} → {context.object}
        
        Propón UN SOLO tipo de nodo nuevo que describa mejor esta entidad.
        El tipo debe ser:
        - En singular
        - En minúsculas con guiones bajos
        - Específico pero reutilizable
        
        Responde SOLO el nombre del tipo, sin explicación.
        Ejemplo: "tecnica_biologica" o "algoritmo_computacional"
        """
        
        responses = await self.swarm.run_role(NodeRole.EXTRACTOR, prompt)
        proposed_type = self._extract_type_name(responses[0].content if responses else "concepto")
        
        # Deduplicar semánticamente
        deduplicated = await self.deduplicator.deduplicate(proposed_type, self._known_types)
        if deduplicated != proposed_type:
            logger.info(f"Tipo '{proposed_type}' deduplicado a '{deduplicated}'")
            return deduplicated
        
        # Validar con consenso mini (3 modelos)
        if await self._validate_new_type(proposed_type, entity_name):
            self._known_types.add(proposed_type)
            logger.info(f"Nuevo tipo ontológico creado: '{proposed_type}'")
            # Persistir en el grafo como metadato
            await self.graph.register_node_type(proposed_type)
            return proposed_type
        
        return "concepto"  # fallback si no pasa validación
```

---

## 15. MÓDULO 10 — SISTEMA DE REPUTACIÓN (PARETO)

### 15.1 ReputationSystem (swarm/reputation.py)

```python
class ReputationSystem:
    """
    Implementa el principio Pareto en el enjambre:
    el 20% de modelos más precisos produce el 80% de claims útiles.
    
    La reputación se actualiza después de cada sesión:
    - Si los claims de un modelo sobreviven el consenso → sube reputación
    - Si sus claims son descartados → baja reputación
    - La reputación decae con el tiempo si el modelo no participa
    
    El score de reputación modifica el peso del voto en el consenso.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.history: dict[str, list[float]] = self._load_history()

    async def update(self, model_performances: dict[str, float]) -> None:
        """
        Actualiza la reputación basada en el performance de la sesión.
        model_performances: {node_id → ratio de claims verificados / total claims}
        """
        for node_id, performance in model_performances.items():
            if node_id not in self.history:
                self.history[node_id] = []
            self.history[node_id].append(performance)
            
            # Limitar historial a últimas 100 sesiones
            self.history[node_id] = self.history[node_id][-100:]
        
        # Aplicar decay a modelos que no participaron
        all_node_ids = set(self.history.keys())
        active_node_ids = set(model_performances.keys())
        inactive = all_node_ids - active_node_ids
        for node_id in inactive:
            self.history[node_id].append(
                self._current_score(node_id) - self.config.reputation.decay_per_session
            )
        
        self._save_history()

    def get_score(self, node_id: str) -> float:
        """Retorna el score de reputación actual del nodo (0.0 a 1.0)."""
        if node_id not in self.history or not self.history[node_id]:
            return 0.5  # neutral para nodos nuevos
        return max(0.1, min(1.0, self._current_score(node_id)))

    def _current_score(self, node_id: str) -> float:
        """Promedio ponderado exponencialmente (sesiones recientes pesan más)."""
        history = self.history[node_id]
        if not history:
            return 0.5
        weights = [0.9 ** i for i in range(len(history) - 1, -1, -1)]
        weighted_sum = sum(h * w for h, w in zip(history, weights))
        return weighted_sum / sum(weights)

    def get_elite_threshold(self, all_nodes: list[SwarmNode]) -> float:
        """Calcula el umbral para ser considerado 'elite' (top 20%)."""
        scores = sorted([self.get_score(n.node_id) for n in all_nodes], reverse=True)
        cutoff_idx = max(0, int(len(scores) * self.config.reputation.elite_ratio) - 1)
        return scores[cutoff_idx] if scores else 0.5
```

---

## 16. MÓDULO 11 — MANEJO DE INTERNET

### 16.1 Decisión de Búsqueda

El modo `auto` evalúa tres señales:

```python
async def should_search(self, query: str, graph_context: GraphContext) -> bool:
    if self.config.internet.search_mode == "never":
        return False
    if self.config.internet.search_mode == "always":
        return True
    
    # modo "auto" — tres señales
    low_confidence = (
        graph_context.avg_confidence < self.config.internet.graph_confidence_threshold
    )
    low_coverage = len(graph_context.nodes) < 3
    temporal = any(
        kw in query.lower() 
        for kw in ["hoy", "actual", "último", "ahora", "reciente", 
                   "today", "current", "latest", "2025", "2026"]
    )
    
    return low_confidence or low_coverage or temporal
```

### 16.2 Pipeline de Ingesta Web

El contenido de internet NO entra crudo al prompt. Pasa por el mismo proceso de verificación:

```
HTML crudo
    ↓ limpieza (BeautifulSoup)
Texto limpio
    ↓ extracción de claims (enjambre)
Claims crudos
    ↓ consenso mini (3 modelos)
Claims verificados
    ↓ al prompt Y al grafo (si confidence > 0.80)
```

---

## 17. MÓDULO 12 — BOOTSTRAP DE CONOCIMIENTO (TEACHER-STUDENT)

### 17.1 Conceptos Legales

**PERMITIDO (y recomendado):**
- Modelos open source locales como teacher (Llama 70B, Mistral Large, etc.)
- DeepSeek API (licencia MIT, términos más permisivos) con `terms_accepted: true`
- Qwen API (Apache 2.0)
- Together.ai con modelos open source hospedados
- Documentación oficial pública (docs.python.org, learn.microsoft.com, etc.)
- Stack Overflow API (oficial, gratuita)
- arXiv (API pública)
- GitHub READMEs y wikis públicas

**NO PERMITIDO:**
- OpenAI API (términos prohíben usar outputs para entrenar/alimentar otros sistemas)
- Anthropic API (misma restricción)
- Google Gemini API (misma restricción)

### 17.2 Pipeline Teacher-Student (bootstrap/pipeline.py)

```python
class BootstrapPipeline:
    """
    Enseña al grafo usando un modelo teacher más grande.
    El conocimiento pasa por consenso antes de entrar al grafo.
    
    Concepto: "Knowledge Graph Bootstrap via Teacher-Student Distillation"
    Nombre para el paper.
    """
    
    async def run(self, topics: list[str]) -> BootstrapResult:
        if not self.config.knowledge_bootstrap.terms_accepted:
            raise TermsNotAcceptedError(
                f"Debes aceptar los términos de {self.config.knowledge_bootstrap.teacher.provider} "
                f"antes de usar bootstrap. Configura terms_accepted: true en config.yaml "
                f"después de leer: {PROVIDER_TERMS_URLS[self.config.knowledge_bootstrap.teacher.provider]}"
            )
        
        total_claims = 0
        for topic in topics:
            qa_pairs = await self.teacher.generate_qa(
                topic=topic,
                count=self.config.knowledge_bootstrap.questions_per_topic
            )
            for qa in qa_pairs:
                # Verificar por consenso local antes de escribir al grafo
                result = await self.consensus_engine.run(
                    query=qa.question,
                    context=GraphContext(nodes=[], serialized_context=qa.answer)
                )
                if self.config.knowledge_bootstrap.store_verified_only:
                    verified = result.verified_claims
                else:
                    verified = result.verified_claims + result.uncertain_claims
                
                await self.graph_writer.write_verified_claims(verified)
                total_claims += len(verified)
        
        return BootstrapResult(topics_processed=len(topics), claims_stored=total_claims)
```

---

## 18. CONFIGURACIÓN COMPLETA (config.yaml)

```yaml
# ============================================================
# NCN — Nodal Consensus Network
# config.yaml — configuración completa del sistema
# ============================================================

system:
  name: "NCN"
  version: "0.1.0"
  log_level: "INFO"          # DEBUG | INFO | WARNING | ERROR
  log_file: "logs/ncn.log"
  data_dir: "data/"

interface:
  cli:
    enabled: true
    history_file: ".ncn_history"
  api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    cors_enabled: true
    cors_origins: ["*"]
    api_key_required: false

swarm:
  max_concurrent: 10
  request_timeout: 30
  retry_attempts: 2
  temperature: 0.7
  nodes:
    - provider: ollama
      model: "llama3.2:1b"
      count: 4
      role: extractor
      temperature: 0.8
    - provider: ollama
      model: "qwen2.5:0.5b"
      count: 3
      role: critic
      temperature: 0.3
    - provider: ollama
      model: "llama3.2:3b"
      count: 1
      role: tenth_man
      temperature: 0.9
    - provider: ollama
      model: "llama3.2:3b"
      count: 1
      role: judge
      temperature: 0.1
    - provider: ollama
      model: "llama3.2:3b"
      count: 1
      role: synthesizer
      temperature: 0.5
  # Ejemplos comentados de nodos cloud:
  # - provider: deepseek
  #   model: "deepseek-chat"
  #   count: 1
  #   role: judge
  #   api_key: ${DEEPSEEK_API_KEY}
  #   terms_accepted: false

providers:
  ollama:
    base_url: "http://localhost:11434"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: ${OPENAI_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: ${DEEPSEEK_API_KEY}
  qwen:
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: ${QWEN_API_KEY}
  together:
    base_url: "https://api.together.xyz/v1"
    api_key: ${TOGETHER_API_KEY}
  custom:
    base_url: ${CUSTOM_BASE_URL}
    api_key: ${CUSTOM_API_KEY}

consensus:
  verification_threshold: 0.66
  discard_threshold: 0.33
  validators_per_claim: 5
  auto_judge_uncertain: false
  tenth_man_enabled: true
  tenth_man_triggers_above: 0.80

reputation:
  enabled: true
  elite_ratio: 0.20
  elite_vote_multiplier: 2.0
  decay_per_session: 0.01
  history_file: "data/reputation.json"

graph:
  adapter: "kuzu"
  kuzu:
    path: "data/graph/"
  falkordb:
    host: "localhost"
    port: 6379
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: ${NEO4J_PASSWORD}
  custom:
    connection_string: ${CUSTOM_GRAPH_URL}
  node_types:
    - persona
    - concepto
    - evento
    - lugar
    - organizacion
  namespaces_enabled: true
  default_namespace: "general"
  cross_namespace_relations: true
  temperature_boost: 0.10
  temperature_decay: 0.05
  cold_node_threshold: 0.02
  confidence_boost: 0.05
  confidence_decay: 0.15
  uncertain_threshold: 0.30

ontology:
  auto_extend: true
  min_confidence: 0.70
  validators: 3
  max_custom_types: 100
  dedup_threshold: 0.85
  namespace:
    auto_create: true
    min_nodes_to_create: 5

retrieval:
  model: "all-MiniLM-L6-v2"
  top_k_nodes: 5
  hop_depth: 2
  token_budget: 800
  cache_enabled: true
  cache_ttl_minutes: 30

ingestion:
  pdf:
    enabled: true
    max_size_mb: 50
    extract_images: false
  internet:
    search_mode: "auto"
    graph_confidence_threshold: 0.60
    cache_enabled: true
    cache_ttl_minutes: 60
    sources:
      - type: web_search
        provider: "duckduckgo"
        enabled: true
      - type: wikipedia
        enabled: true
        languages: ["es", "en"]
      - type: stackoverflow
        enabled: false
        api_key: ${SO_API_KEY}
        tags: ["python", "java", "azure"]
        min_score: 50
      - type: arxiv
        enabled: false
        categories: ["cs.AI", "cs.LG", "cs.SE"]
      - type: github
        enabled: false
        api_key: ${GITHUB_TOKEN}
        content: ["README", "docs/"]
      - type: rss
        enabled: false
        feeds: []
      - type: documentation
        enabled: false
        urls: []

knowledge_bootstrap:
  enabled: false
  terms_accepted: false
  teacher:
    provider: deepseek
    model: "deepseek-chat"
    api_key: ${DEEPSEEK_API_KEY}
  topics: []
  questions_per_topic: 20
  store_verified_only: true

continuous_learning:
  enabled: false              # Fase 3 — NO implementar aún
  interval_hours: 24
  max_claims_per_session: 100
  sources: []

experimental:
  metrics_enabled: true
  metrics_file: "data/metrics.jsonl"
  ab_testing: false
  ab_config_b: null
```

---

## 19. PUERTOS ABIERTOS PARA FASE 2 Y FASE 3

### 19.1 Puertos para Fase 2 (Agente tipo OpenClaw mejorado)

Los siguientes elementos están diseñados para conectar con Fase 2 sin reescribir código:

**SwarmPort — Skills como grafos:**
- El `SwarmNode` tiene campo `role` que en Fase 2 puede ser dinámico
- Los skills de Fase 2 se almacenarán como nodos del grafo con `node_type: "skill"`
- El `Retriever` ya recupera nodos por tipo — solo agregar filtro `node_type == "skill"`
- El `ROLE_SYSTEM_PROMPTS` en `swarm/roles.py` se puede extender con roles de Fase 2

**GraphPort — Memoria de acciones:**
- La interfaz `GraphPort` ya soporta cualquier tipo de nodo
- En Fase 2 agregar `node_type: "action_log"` para memoria de acciones ejecutadas
- El `SingleWriter` no necesita cambios — solo nuevos tipos de nodos

**IngestorPort — Nuevas fuentes:**
- En Fase 2 agregar `ActionIngestor` que ingesta resultados de acciones ejecutadas
- La interfaz `IngestorPort` ya está preparada para nuevas implementaciones

### 19.2 Puertos para Fase 3 (Aprendizaje Continuo)

```python
# En continuous_learning/ (crear en Fase 3):
class ContinuousLearner:
    """
    Puerto abierto en Fase 1.
    En Fase 3 implementar este módulo.
    Se activa con continuous_learning.enabled: true en config.yaml.
    Corre en background como proceso independiente.
    Usa el mismo pipeline de ingesta y consenso de Fase 1.
    """
    pass
```

**Config ya preparada:**
```yaml
continuous_learning:
  enabled: false   # cambiar a true en Fase 3
  interval_hours: 24
  max_claims_per_session: 100
  sources: []
```

---

## 20. DECISIONES DE DISEÑO Y JUSTIFICACIONES

Estas decisiones fueron tomadas deliberadamente. NO cambiarlas sin discutirlo con el usuario.

| Decisión | Justificación | Alternativa rechazada |
|---|---|---|
| Kùzu como default para grafos | Embebido, sin servidor, ACID, Cypher nativo, muy rápido para grafos <1M nodos | Neo4j: demasiado pesado para MVP; FalkorDB: requiere Docker |
| asyncio nativo (no Celery, no Ray) | Suficiente para el MVP, sin dependencias extra, más fácil de debuggear | Ray: overkill; Celery: necesita Redis |
| sentence-transformers all-MiniLM-L6-v2 | 80MB, corre en CPU, suficientemente preciso para recuperación | OpenAI embeddings: costo, privacidad; BERT grande: lento en CPU |
| PyMuPDF para PDFs | Más rápido que alternatives, sin Java, bien mantenido | PyPDF2: lento; pdfplumber: overhead |
| DuckDuckGo como default para web | Sin API key, privacidad, suficiente para MVP | Google: requiere API key y pago; Bing: requiere API key |
| Single-writer para el grafo | Elimina race conditions por diseño. Simple > complejo | Locks por entidad: más complejo, mismo resultado |
| Pydantic v2 para modelos | Validación, tipado, serialización automática | dataclasses: menos validación; attrs: menos ecosistema |
| FastAPI para API | Async nativo, auto-docs, typing integrado | Django REST: síncrono; Flask: sin async |
| YAML para config | Legible, comentable, soportado nativamente | TOML: menos conocido; JSON: no soporta comentarios |
| NxK validación (no NxN) | NxN crece cuadráticamente. Con K=5 y N=10 son 50 llamadas vs 100 | NxN: inviable con muchos claims |
| Bullet points para serializar nodos | ~60% menos tokens que narrativo, modelos pequeños lo parsean mejor | Narrativo: más tokens, peor parseo por SLMs |
| Temperatura configurable por rol | Extractores necesitan variedad, jueces necesitan determinismo | Temperatura global: no permite optimización por rol |

---

## 21. ORDEN DE IMPLEMENTACIÓN

Implementar en este orden exacto. Cada paso tiene un entregable testeable.

### Sprint 1 — El Grafo (base de todo)
```
1. core/config_loader.py          → carga config.yaml con Pydantic
2. core/models.py                 → GraphNode, GraphRelation, Claim, etc.
3. core/exceptions.py             → excepciones personalizadas
4. graph/base.py                  → GraphPort (interfaz abstracta)
5. graph/kuzu_adapter.py          → implementación Kùzu
6. graph/factory.py               → factory de adapters
7. graph/writer.py                → SingleWriter (sin ontología aún)
8. graph/reader.py                → lecturas básicas

CHECKPOINT: pytest tests/test_graph.py → insertar 50 nodos, hacer queries, validar
```

### Sprint 2 — Embeddings y Recuperación
```
9.  retrieval/embedder.py         → sentence-transformers
10. retrieval/searcher.py         → búsqueda semántica en Kùzu
11. retrieval/expander.py         → expansión de vecindario
12. retrieval/serializer.py       → bullet points con presupuesto de tokens

CHECKPOINT: query → 5 nodos relevantes → serializado en <800 tokens
```

### Sprint 3 — Providers de Modelos
```
13. providers/base.py             → ModelProviderPort (interfaz)
14. providers/ollama.py           → OllamaProvider
15. providers/openai_compatible.py → base para APIs compatibles
16. providers/deepseek.py         → DeepSeekProvider
17. providers/qwen.py             → QwenProvider
18. providers/together.py         → TogetherProvider
19. providers/custom.py           → CustomProvider
20. providers/factory.py          → ProviderFactory

CHECKPOINT: llamar a Ollama con llama3.2:1b y obtener respuesta
```

### Sprint 4 — Enjambre
```
21. swarm/roles.py                → NodeRole, system prompts
22. swarm/node.py                 → SwarmNode
23. swarm/reputation.py           → ReputationSystem
24. swarm/pool.py                 → SwarmPool con asyncio

CHECKPOINT: 10 nodos en paralelo respondiendo el mismo prompt, medir latencia
```

### Sprint 5 — Motor de Consenso
```
25. consensus/extractor.py        → extracción de claims
26. consensus/validator.py        → validación cruzada NxK
27. consensus/aggregator.py       → agregación ponderada
28. consensus/tenth_man.py        → rol décimo hombre
29. consensus/judge.py            → árbitro de empates
30. consensus/engine.py           → orquesta las 4 fases

CHECKPOINT: texto → claims → consenso → claims verificados con scores
```

### Sprint 6 — Ingesta
```
31. ingestion/base.py             → IngestorPort
32. ingestion/text_ingestor.py    → texto plano
33. ingestion/pdf_ingestor.py     → PDFs con PyMuPDF
34. ingestion/web_ingestor.py     → DuckDuckGo + Wikipedia
35. ingestion/factory.py          → factory de ingestores

CHECKPOINT: ingestar un PDF de 10 páginas → claims en el grafo
```

### Sprint 7 — Ontología
```
36. ontology/deduplicator.py      → deduplicación semántica
37. ontology/classifier.py        → clasificación en tipos existentes
38. ontology/agent.py             → OntologyAgent completo

CHECKPOINT: entidad sin tipo conocido → tipo nuevo propuesto y validado
```

### Sprint 8 — Orquestador
```
39. core/orchestrator.py          → conecta todos los módulos

CHECKPOINT: flujo completo end-to-end: query → grafo → enjambre → consenso → respuesta
```

### Sprint 9 — Interfaces
```
40. interface/cli/app.py          → CLI con Typer
41. interface/api/app.py          → FastAPI
42. interface/api/routes/         → todos los endpoints
43. main.py                       → punto de entrada

CHECKPOINT: ncn query "¿Qué es Python?" → respuesta completa
```

### Sprint 10 — Bootstrap y Pulido
```
44. bootstrap/teacher.py          → teacher model Q&A
45. bootstrap/pipeline.py         → pipeline completo
46. tests/                        → suite completa de tests
47. configs/                      → configs de experimentos
48. README.md                     → documentación pública

CHECKPOINT: sistema completo funcionando, tests pasando, README claro
```

---

## 22. MÉTRICAS Y BENCHMARKS PARA PAPER

### 22.1 Métricas que Registrar por Sesión (data/metrics.jsonl)

```json
{
  "session_id": "uuid",
  "timestamp": "ISO-8601",
  "query": "texto de la query",
  "config_hash": "hash del config.yaml activo",
  
  "retrieval": {
    "nodes_found": 5,
    "avg_confidence": 0.82,
    "tokens_retrieved": 187,
    "retrieval_ms": 45
  },
  
  "web_search": {
    "triggered": true,
    "trigger_reason": "low_coverage",
    "results_found": 3,
    "web_ms": 820
  },
  
  "swarm": {
    "nodes_active": 10,
    "nodes_failed": 0,
    "parallel_ms": 2100
  },
  
  "consensus": {
    "claims_extracted": 23,
    "claims_verified": 14,
    "claims_uncertain": 3,
    "claims_discarded": 6,
    "tenth_man_triggered": true,
    "tenth_man_counter_claims": 2,
    "judge_invoked": false,
    "consensus_ms": 450
  },
  
  "graph": {
    "nodes_created": 8,
    "nodes_reinforced": 6,
    "nodes_disputed": 1
  },
  
  "reputation": {
    "top_performer": "llama3.2:1b-extractor-2",
    "top_score": 0.89,
    "bottom_performer": "qwen2.5:0.5b-critic-1",
    "bottom_score": 0.43
  },
  
  "total_tokens": 203,
  "total_ms": 3415,
  "answer_length_chars": 342
}
```

### 22.2 Experimentos para el Paper

**Experimento 1 — Eficiencia de tokens:**
Comparar tokens usados por NCN vs RAG tradicional vs prompt sin contexto.
Config: `configs/experiment_tokens.yaml`

**Experimento 2 — Ablation del décimo hombre:**
¿Cuánto mejora la precisión con el décimo hombre activado?
Config: `configs/ablation_no_tenth_man.yaml`

**Experimento 3 — Efecto del Pareto:**
¿Cómo afecta el `elite_ratio` a la precisión del consenso?
Config: `configs/experiment_pareto.yaml` (variar 0.10, 0.20, 0.30, 0.50)

**Experimento 4 — Tamaño del enjambre:**
¿A partir de qué N el retorno disminuye?
Config: `configs/experiment_swarm_size.yaml` (N=3, 5, 7, 10, 15)

**Experimento 5 — Threshold de consenso:**
¿Cuál es el `verification_threshold` óptimo?
Config: `configs/experiment_consensus.yaml` (0.50, 0.60, 0.66, 0.75, 0.80)

**Experimento 6 — Provider mixing:**
¿Mejora el consenso mezclando proveedores vs usando solo modelos homogéneos?
Config: `configs/experiment_mixing.yaml`

### 22.3 Benchmark Factual Sugerido

Para medir precisión del sistema de forma reproducible:

- **Dataset**: subset de TriviaQA o Natural Questions (públicos)
- **Dominio**: preguntas factuales con respuesta verificable
- **Métrica principal**: porcentaje de respuestas correctas
- **Métrica secundaria**: tokens usados por pregunta respondida correctamente
- **Baseline A**: modelo llama3.2:1b solo (sin enjambre, sin grafo)
- **Baseline B**: RAG tradicional (chunking + embedding + single model)
- **NCN**: sistema completo

El paper demuestra que NCN supera ambos baselines en precisión con menor consumo de tokens.

---

## RESUMEN EJECUTIVO PARA CLAUDE CODE

Este proyecto se llama NCN (Nodal Consensus Network). Es un enjambre de modelos de lenguaje pequeños que operan sobre una base de datos de grafos como memoria compartida y verificada.

**Los tres principios que no puedes violar:**
1. Solo el `SingleWriter` escribe al grafo — nadie más
2. Todo es configurable desde `config.yaml` — nada hardcodeado
3. Los puertos (GraphPort, ModelProviderPort, IngestorPort) son interfaces abstractas — las implementaciones van detrás de esos puertos

**El orden de implementación está en la Sección 21** — síguelo exactamente.

**Cuando tengas dudas** sobre una decisión de diseño, consulta la Sección 20 antes de asumir.

**El objetivo final** es un sistema que en Fase 2 reemplace a OpenClaw con una arquitectura de grafos en lugar de archivos `.md`, siendo más eficiente en tokens y con memoria verificada por consenso.

---

*Documento generado para uso exclusivo con Claude Code. Versión 1.0.*
*Proyecto: NCN — Nodal Consensus Network*
