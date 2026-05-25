"""Orchestrator — the conductor that coordinates every NCN module.

The Orchestrator does NOT implement logic — it delegates to each module
and wires them together.

Flows:
    process_query  → retrieve context → (optional web) → consensus → write → response
    process_ingest → ingest source → consensus per chunk → write to graph
"""

from __future__ import annotations

from loguru import logger

from core.config_loader import Config
from core.models import (
    Claim,
    GraphContext,
    QueryResponse,
    RawContent,
    SourceAuthority,
    TokenMetrics,
)
from core.query_router import QueryIntent, route_query
from swarm.roles import CHAT_SYSTEM_PROMPT
from consensus.engine import ConsensusEngine
from graph.base import GraphPort
from graph.factory import create_graph_adapter
from graph.reader import GraphReader
from graph.writer import SingleWriter
from ingestion.factory import create_ingestor
from ingestion.web_ingestor import WebIngestor
from ontology.agent import OntologyAgent
from retrieval.embedder import Embedder
from retrieval.searcher import Retriever
from swarm.pool import SwarmPool
from swarm.reputation import ReputationSystem


class Orchestrator:
    """Top-level coordinator for the NCN system."""

    def __init__(
        self,
        config: Config,
        *,
        graph: GraphPort | None = None,
        swarm_pool: SwarmPool | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config

        # --- core components (injected or created) ---
        self._graph = graph
        self.swarm_pool = swarm_pool or SwarmPool(config)
        self.embedder = embedder or Embedder(config)

        # --- modules (initialised in ``start()``) ---
        self.consensus_engine = ConsensusEngine(config)
        self.reputation_system = ReputationSystem(config)
        self.web_ingestor = WebIngestor(config)

        # These need the graph, so they're wired in ``start()``
        self.graph_reader: GraphReader | None = None
        self.graph_writer: SingleWriter | None = None
        self.ontology_agent: OntologyAgent | None = None
        self.retriever: Retriever | None = None

        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the graph and wire all modules together."""
        if self._started:
            return

        # Upfront sanity check: warn LOUDLY about missing API keys so the
        # user doesn't discover it 30 minutes into a bulk ingestion.
        self._audit_provider_keys()

        if self._graph is None:
            self._graph = await create_graph_adapter(self.config)

        self.ontology_agent = OntologyAgent(
            self.config, self.swarm_pool, self._graph,
            embedder=self.embedder,
        )
        self.graph_reader = GraphReader(self._graph, self.config)
        self.graph_writer = SingleWriter(
            self._graph, self.config,
            ontology_agent=self.ontology_agent,
            embedder=self.embedder,
        )
        self.retriever = Retriever(
            self._graph, self.config, embedder=self.embedder
        )
        self._started = True
        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Cool-down graph temperatures and release resources."""
        if not self._started:
            return
        try:
            await self.graph_writer.cool_down_all_nodes()
        except Exception as exc:
            logger.warning(f"Cool-down failed: {exc}")
        if self._graph:
            await self._graph.close()
        self._started = False
        logger.info("Orchestrator stopped")

    @property
    def graph(self) -> GraphPort:
        assert self._graph is not None, "Orchestrator not started"
        return self._graph

    # ------------------------------------------------------------------
    # Query flow
    # ------------------------------------------------------------------

    async def process_query(
        self,
        query: str,
        namespace: str | None = None,
    ) -> QueryResponse:
        """End-to-end: query → route → context → consensus → graph update → answer.

        If *namespace* is provided, retrieval is scoped to that namespace and
        any new claims produced are written to the same namespace.
        """
        self._ensure_started()

        # 0. Route query intent (greeting / too_short / url / normal)
        intent, extracted_url = route_query(query)

        if intent == QueryIntent.GREETING:
            return QueryResponse(
                answer="¡Hola! Soy NCN. ¿Qué te gustaría saber?",
                claims=[], nodes_used=[], nodes_created=[],
            )

        if intent == QueryIntent.TOO_SHORT:
            return QueryResponse(
                answer=(
                    "Tu consulta es muy corta para generar una respuesta "
                    "verificada. Por favor hazme una pregunta más específica."
                ),
                claims=[], nodes_used=[], nodes_created=[],
            )

        # 1. Retrieve context from the graph (namespace-scoped if provided)
        graph_context = await self.retriever.retrieve(query, namespace=namespace)
        logger.info(
            f"Retrieved {len(graph_context.nodes)} nodes "
            f"(avg_conf={graph_context.avg_confidence:.2f}, "
            f"max_relevance={graph_context.max_relevance:.2f})"
        )

        # 2. Context enrichment: URL scraping OR web search
        context = graph_context
        web_searched = False

        if intent == QueryIntent.URL_LEARN and extracted_url:
            # Direct scraping — user provided a URL explicitly
            logger.info(f"URL scraping: {extracted_url}")
            from learning.scraper import PageScraper
            scraper = PageScraper(self.config)
            scraped = await scraper.scrape_one(extracted_url)
            if scraped:
                context = self._merge_contexts(graph_context, scraped)
                web_searched = True
                logger.info(f"Scraped {len(scraped)} chunks from {extracted_url}")
            else:
                # Scraping failed — fall back to web search about the URL
                logger.warning(
                    f"Scraping failed for {extracted_url}, "
                    f"falling back to web search"
                )
                web_results = await self.web_ingestor.search(query)
                context = self._merge_contexts(graph_context, web_results)
                web_searched = bool(web_results)
        elif self._should_search_web(query, graph_context):
            logger.info("Web search triggered")
            web_results = await self.web_ingestor.search(query)
            context = self._merge_contexts(graph_context, web_results)
            web_searched = bool(web_results)

        # 2b. Relevance Gate short-circuit: if graph had nothing relevant
        # AND web/scrape didn't help, admit ignorance instead of hallucinating
        if not graph_context.nodes and not web_searched:
            logger.info("Relevance Gate: no context available → admitting ignorance")
            return QueryResponse(
                answer=(
                    "No tengo informacion verificada sobre este tema en mi "
                    "base de conocimiento. Puedes ingestar datos relevantes "
                    "con `ncn ingest text` o `ncn learn`, o habilitar la "
                    "busqueda en internet en config.yaml."
                ),
                claims=[],
                nodes_used=[],
                nodes_created=[],
                token_metrics=_compute_token_metrics(context),
            )

        # 3. Consensus
        result = await self.consensus_engine.run(
            query, context, self.swarm_pool
        )

        # 4. Stamp authority on verified claims (query-derived = CONSENSUS)
        _stamp_authority(result.verified_claims, SourceAuthority.CONSENSUS)

        # 5. Write verified claims to the graph (in the query's namespace)
        new_node_ids: list[str] = []
        if result.verified_claims:
            new_node_ids = await self.graph_writer.write_verified_claims(
                result.verified_claims, namespace=namespace
            )

        # 6. Update model reputation
        if result.model_performances:
            await self.reputation_system.update(result.model_performances)

        # 7. Heat consulted nodes
        if graph_context.nodes_used:
            await self.graph_writer.update_temperatures(
                graph_context.nodes_used
            )

        return QueryResponse(
            answer=result.synthesized_answer,
            claims=result.verified_claims,
            nodes_used=graph_context.nodes_used,
            nodes_created=new_node_ids,
            token_metrics=_compute_token_metrics(context),
        )

    # ------------------------------------------------------------------
    # Chat flow (lightweight — single model + graph context, no consensus)
    # ------------------------------------------------------------------

    async def process_chat(
        self,
        query: str,
        namespace: str | None = None,
    ) -> QueryResponse:
        """Chat mode: retrieve graph context → single LLM call → response.

        Unlike ``process_query``, this does NOT run consensus, web search,
        claim extraction, or graph writes. The model can use both graph
        context and its base knowledge to answer.

        If *namespace* is provided, retrieval is restricted to that namespace.
        """
        self._ensure_started()
        from core.models import NodeRole

        # 0. Route query intent (greetings/too_short still blocked)
        intent, _ = route_query(query)

        if intent == QueryIntent.GREETING:
            return QueryResponse(
                answer="¡Hola! Soy NCN. ¿Qué te gustaría saber?",
                claims=[], nodes_used=[], nodes_created=[],
            )
        if intent == QueryIntent.TOO_SHORT:
            return QueryResponse(
                answer=(
                    "Tu consulta es muy corta para generar una respuesta. "
                    "Por favor hazme una pregunta más específica."
                ),
                claims=[], nodes_used=[], nodes_created=[],
            )

        # 1. Retrieve graph context (namespace-scoped if provided)
        graph_context = await self.retriever.retrieve(query, namespace=namespace)
        logger.info(
            f"Chat: retrieved {len(graph_context.nodes)} nodes "
            f"(max_relevance={graph_context.max_relevance:.2f}, ns={namespace})"
        )

        # 2. Build prompt with graph context
        if graph_context.serialized_context:
            prompt = (
                f"CLAIMS VERIFICADOS DEL GRAFO (unica fuente permitida):\n"
                f"{graph_context.serialized_context}\n\n"
                f"PREGUNTA: {query}\n\n"
                f"Recuerda: usa UNICAMENTE los claims anteriores. "
                f"Si ningun claim responde la pregunta, responde literalmente "
                f"'No tengo informacion verificada sobre ese tema en mi base "
                f"de conocimiento.' Cita los claims por su numero "
                f"(ej. 'segun CLAIM #2...')."
            )
        else:
            prompt = (
                f"PREGUNTA: {query}\n\n"
                f"No hay claims verificados en el grafo para esta pregunta. "
                f"Responde literalmente: 'No tengo informacion verificada "
                f"sobre ese tema en mi base de conocimiento.'"
            )

        # 3. Get the first SYNTHESIZER node for the single LLM call
        synth_nodes = [
            n for n in self.swarm_pool.nodes
            if n.role == NodeRole.SYNTHESIZER
        ]
        if not synth_nodes:
            # Fallback: use any available node
            if not self.swarm_pool.nodes:
                return QueryResponse(
                    answer="No hay modelos configurados en el swarm.",
                    claims=[], nodes_used=[], nodes_created=[],
                )
            synth_nodes = [self.swarm_pool.nodes[0]]

        node = synth_nodes[0]

        # 4. Single LLM call (no consensus, no multi-model)
        try:
            response = await self.swarm_pool._run_node(
                node, prompt, CHAT_SYSTEM_PROMPT
            )
            answer = response.content
        except Exception as exc:
            logger.error(f"Chat LLM call failed: {exc}")
            answer = f"Error al consultar el modelo: {exc}"

        return QueryResponse(
            answer=answer,
            claims=[],
            nodes_used=graph_context.nodes_used,
            nodes_created=[],
            token_metrics=_compute_token_metrics(graph_context),
        )

    # ------------------------------------------------------------------
    # Ingest flow
    # ------------------------------------------------------------------

    async def process_ingest(
        self,
        source: str,
        source_type: str = "text",
        namespace: str | None = None,
    ) -> dict:
        """Ingest a source, verify via consensus, write to graph.

        If *namespace* is provided, all resulting nodes are stored in that
        namespace (enables domain isolation — e.g. "survival", "cooking").

        Returns a summary dict with chunks, claim counts, nodes_created, etc.
        """
        self._ensure_started()

        ingestor = create_ingestor(source_type, self.config)
        chunks: list[RawContent] = await ingestor.ingest(source)
        logger.info(f"Ingested {len(chunks)} chunks from {source_type}")

        base_auth = self._resolve_source_authority(source_type)
        return await self._process_chunks(chunks, base_auth, namespace)

    async def process_ingest_url(
        self,
        url: str,
        namespace: str | None = None,
    ) -> dict:
        """Download + auto-detect format + ingest a single URL.

        Uses PageScraper to determine the format (PDF/DOCX/EPUB/HTML/...)
        via HEAD request, then dispatches to the right ingestor internally.

        Claims ingested from URLs receive ``SourceAuthority.WEB_RAW`` and
        get promoted to WEB_VERIFIED after consensus verification.
        """
        self._ensure_started()

        from learning.scraper import PageScraper
        scraper = PageScraper(self.config)
        chunks = await scraper.scrape_one(url)
        logger.info(f"URL ingested: {url} -> {len(chunks)} chunks")

        if not chunks:
            return {
                "chunks": 0,
                "claims_extracted": 0,
                "claims_verified": 0,
                "claims_discarded": 0,
                "claims_uncertain": 0,
                "nodes_created": 0,
                "corrections": 0,
            }

        # All URL ingestion starts at WEB_RAW authority
        return await self._process_chunks(
            chunks, SourceAuthority.WEB_RAW, namespace
        )

    async def _process_chunks(
        self,
        chunks: list[RawContent],
        base_auth: SourceAuthority,
        namespace: str | None,
    ) -> dict:
        """Shared pipeline: consensus per chunk + write + reputation update.

        Factored out so ``process_ingest`` and ``process_ingest_url`` can
        share the verification/writing logic without duplication.
        """
        total_claims = 0
        total_verified = 0
        total_discarded = 0
        total_uncertain = 0
        total_corrections = 0
        all_node_ids: list[str] = []

        for chunk in chunks:
            context = GraphContext(serialized_context=chunk.text)
            result = await self.consensus_engine.run(
                chunk.text, context, self.swarm_pool,
                skip_tenth_man=True,  # ingest = trusted source, don't question
            )
            total_claims += (
                len(result.verified_claims)
                + len(result.uncertain_claims)
                + len(result.discarded_claims)
            )
            total_verified += len(result.verified_claims)
            total_discarded += len(result.discarded_claims)
            total_uncertain += len(result.uncertain_claims)

            # Stamp authority: promote after consensus verification
            promoted = _promote_authority(base_auth)
            _stamp_authority(result.verified_claims, promoted)

            if result.verified_claims:
                ids = await self.graph_writer.write_verified_claims(
                    result.verified_claims, namespace=namespace
                )
                all_node_ids.extend(ids)
                total_corrections += len(
                    self.graph_writer.get_and_clear_corrections()
                )

            if result.model_performances:
                await self.reputation_system.update(result.model_performances)

        return {
            "chunks": len(chunks),
            "claims_extracted": total_claims,
            "claims_verified": total_verified,
            "claims_discarded": total_discarded,
            "claims_uncertain": total_uncertain,
            "nodes_created": len(set(all_node_ids)),
            "corrections": total_corrections,
        }

    # ------------------------------------------------------------------
    # Web ingest flow
    # ------------------------------------------------------------------

    async def process_ingest_web(self, query: str) -> dict:
        """Search the web for *query*, verify results, write to graph."""
        self._ensure_started()

        web_results = await self.web_ingestor.search(query)
        if not web_results:
            return {"sources": 0, "claims_extracted": 0, "claims_verified": 0, "nodes_created": 0}

        logger.info(f"Web ingest: {len(web_results)} results for '{query}'")

        total_claims = 0
        total_verified = 0
        all_node_ids: list[str] = []

        for chunk in web_results:
            context = GraphContext(serialized_context=chunk.text)
            result = await self.consensus_engine.run(
                chunk.text, context, self.swarm_pool,
                skip_tenth_man=True,
            )
            total_claims += (
                len(result.verified_claims)
                + len(result.uncertain_claims)
                + len(result.discarded_claims)
            )

            promoted = _promote_authority(SourceAuthority.WEB_RAW)
            _stamp_authority(result.verified_claims, promoted)

            total_verified += len(result.verified_claims)
            if result.verified_claims:
                ids = await self.graph_writer.write_verified_claims(
                    result.verified_claims
                )
                all_node_ids.extend(ids)

        return {
            "sources": len(web_results),
            "claims_extracted": total_claims,
            "claims_verified": total_verified,
            "nodes_created": len(set(all_node_ids)),
        }

    # ------------------------------------------------------------------
    # Learn flow (Phase 3 — Genspark-style)
    # ------------------------------------------------------------------

    async def process_learn(
        self,
        topic: str,
        namespace: str | None = None,
    ) -> dict:
        """Autonomous learning: decompose → search → scrape → verify → store.

        If *namespace* is provided, all learned claims are stored in that
        namespace (keeps domain-scoped graphs isolated).
        """
        self._ensure_started()

        from learning.pipeline import LearningPipeline

        pipeline = LearningPipeline(
            config=self.config,
            swarm=self.swarm_pool,
            consensus_engine=self.consensus_engine,
            graph_writer=self.graph_writer,
            web_ingestor=self.web_ingestor,
        )
        return await pipeline.learn(topic, namespace=namespace)

    # ------------------------------------------------------------------
    # Bootstrap flow
    # ------------------------------------------------------------------

    async def process_bootstrap(
        self, topics: list[str] | None = None
    ) -> dict:
        """Run teacher-student bootstrap. Returns summary dict."""
        self._ensure_started()

        from bootstrap.pipeline import BootstrapPipeline

        pipeline = BootstrapPipeline(
            config=self.config,
            consensus_engine=self.consensus_engine,
            graph_writer=self.graph_writer,
            swarm_pool=self.swarm_pool,
        )
        result = await pipeline.run(topics)
        return {
            "topics_processed": result.topics_processed,
            "claims_stored": result.claims_stored,
        }

    # ------------------------------------------------------------------
    # Web-search decision
    # ------------------------------------------------------------------

    def _should_search_web(
        self, query: str, context: GraphContext
    ) -> bool:
        mode = self.config.internet.search_mode
        if mode == "never":
            return False
        if mode == "always":
            return True
        # mode == "auto" — any of these signals triggers web search
        threshold = self.config.internet.graph_confidence_threshold
        low_confidence = context.avg_confidence < threshold
        low_coverage = len(context.nodes) < 3
        # Relevance Gate: if the best retrieved node is semantically
        # far from the query, the graph likely has no info on this topic
        low_relevance = (
            context.max_relevance < self.config.retrieval.relevance_threshold
        )
        temporal = self._has_temporal_keywords(query)
        return low_confidence or low_coverage or low_relevance or temporal

    @staticmethod
    def _has_temporal_keywords(query: str) -> bool:
        keywords = [
            "hoy", "actual", "último", "ahora", "reciente",
            "2025", "2026", "today", "current", "latest",
        ]
        q = query.lower()
        return any(kw in q for kw in keywords)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_contexts(
        graph_ctx: GraphContext,
        web_results: list[RawContent],
    ) -> GraphContext:
        """Append web snippets to the serialised graph context."""
        web_text = "\n".join(
            f"• [web] {r.text}" for r in web_results if r.text
        )
        merged = graph_ctx.serialized_context
        if web_text:
            merged = f"{merged}\n\n{web_text}" if merged else web_text
        return GraphContext(
            nodes=graph_ctx.nodes,
            serialized_context=merged,
            avg_confidence=graph_ctx.avg_confidence,
            nodes_used=graph_ctx.nodes_used,
        )

    def _resolve_source_authority(self, source_type: str) -> SourceAuthority:
        """Map an ingestor source_type to a SourceAuthority."""
        sa = self.config.source_authority
        if not sa.enabled:
            return SourceAuthority.SLM_SINGLE
        auth_str = sa.source_type_authority.get(source_type, "slm_single")
        try:
            return SourceAuthority(auth_str)
        except ValueError:
            return SourceAuthority.SLM_SINGLE

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "Orchestrator not started — call ``await orchestrator.start()`` first"
            )

    def _audit_provider_keys(self) -> None:
        """Log loud warnings for any provider used by the swarm that is
        missing an API key. Called during ``start()`` so the user notices
        BEFORE running a 30-minute ingestion that will fail at every call."""
        from providers.factory import audit_provider_keys

        audit = audit_provider_keys(self.config)
        missing = [a for a in audit if not a["ok"]]
        if not missing:
            return

        logger.warning("=" * 70)
        logger.warning("⚠️  MISSING PROVIDER API KEYS — requests to these will 401")
        logger.warning("=" * 70)
        for item in missing:
            hint = item.get("env_var_hint") or "config.yaml providers section"
            roles = ", ".join(item["used_by_roles"])
            logger.warning(
                f"  ❌ {item['provider']}  →  used by roles: [{roles}]"
            )
            logger.warning(
                f"     Fix: set env var {hint}=... "
                f"(or edit providers.{item['provider']}.api_key in config.yaml)"
            )
        logger.warning("=" * 70)
        logger.warning(
            "Backend will start anyway, but these providers will fail at runtime."
        )
        logger.warning("=" * 70)


# ------------------------------------------------------------------
# Authority helpers
# ------------------------------------------------------------------

def _stamp_authority(claims: list[Claim], authority: SourceAuthority) -> None:
    """Set ``source_authority`` on every claim in the list."""
    for claim in claims:
        claim.source_authority = authority


def _promote_authority(base: SourceAuthority) -> SourceAuthority:
    """Promote authority after consensus verification."""
    promotions = {
        SourceAuthority.WEB_RAW: SourceAuthority.WEB_VERIFIED,
        SourceAuthority.SLM_SINGLE: SourceAuthority.CONSENSUS,
    }
    return promotions.get(base, base)


# ------------------------------------------------------------------
# Token metrics
# ------------------------------------------------------------------

# Typical .md-based system: 2 skills (~2000 tokens each) + context (~3000)
_ESTIMATED_MD_TOKENS = 6500


def _estimate_tokens(text: str) -> int:
    """~4 chars per token for mixed-language text."""
    return len(text) // 4 + 1 if text else 0


def _compute_token_metrics(context: GraphContext) -> TokenMetrics:
    ctx_text = context.serialized_context
    ctx_tokens = _estimate_tokens(ctx_text)
    ctx_chars = len(ctx_text)
    nodes = len(context.nodes)
    md_est = _ESTIMATED_MD_TOKENS
    reduction = max(0.0, (1 - ctx_tokens / md_est) * 100) if md_est else 0.0

    return TokenMetrics(
        context_tokens=ctx_tokens,
        context_chars=ctx_chars,
        nodes_retrieved=nodes,
        estimated_md_tokens=md_est,
        reduction_pct=round(reduction, 1),
    )
