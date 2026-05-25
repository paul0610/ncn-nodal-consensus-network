"""LearningPipeline — autonomous learning: decompose → search → scrape → verify → store.

Genspark-style learning: the system decides what to search, opens full pages,
verifies through consensus, and writes verified knowledge to the graph.
"""

from __future__ import annotations

from loguru import logger

from consensus.engine import ConsensusEngine
from core.config_loader import Config
from core.models import GraphContext, RawContent, SourceAuthority
from graph.writer import SingleWriter
from ingestion.web_ingestor import WebIngestor
from learning.decomposer import QueryDecomposer
from learning.scraper import PageScraper
from swarm.pool import SwarmPool


class LearningPipeline:
    """Autonomous topic learning: decompose → search → scrape → verify → store."""

    def __init__(
        self,
        config: Config,
        swarm: SwarmPool,
        consensus_engine: ConsensusEngine,
        graph_writer: SingleWriter,
        web_ingestor: WebIngestor,
    ) -> None:
        self.config = config
        self.swarm = swarm
        self.consensus_engine = consensus_engine
        self.graph_writer = graph_writer
        self.web_ingestor = web_ingestor
        self.decomposer = QueryDecomposer(config, swarm)
        self.scraper = PageScraper(config)

    async def learn(
        self,
        topic: str,
        namespace: str | None = None,
    ) -> dict:
        """Full learning cycle for one topic.

        If *namespace* is provided, all verified claims are written into
        that namespace (enables isolated domain-specific graphs).

        Returns summary with counts AND the actual sub-queries and URLs
        visited so the frontend can show the full trace.
        """
        cl = self.config.continuous_learning
        total_claims = 0
        total_verified = 0
        total_discarded = 0
        total_uncertain = 0
        total_sources = 0
        total_pages = 0
        all_node_ids: list[str] = []
        all_urls_scraped: list[str] = []

        # 1. Decompose topic into sub-queries
        logger.info(f"Learning: decomposing '{topic}'")
        sub_queries = await self.decomposer.decompose(topic)
        logger.info(f"  → {len(sub_queries)} sub-queries generated")

        # 2. For each sub-query: search → scrape → verify → store
        for i, query in enumerate(sub_queries):
            logger.info(f"  [{i+1}/{len(sub_queries)}] Searching: {query}")

            # Search (snippets from DuckDuckGo + Wikipedia)
            snippets = await self.web_ingestor.search(query)
            total_sources += len(snippets)

            # Collect URLs for full-page scraping
            all_chunks: list[RawContent] = list(snippets)

            if cl.scrape_full_pages:
                urls = _extract_urls(snippets, cl.max_pages_per_query)
                if urls:
                    logger.info(f"    Scraping {len(urls)} pages...")
                    pages = await self.scraper.scrape_urls(urls)
                    total_pages += len(pages)
                    all_urls_scraped.extend(urls)
                    all_chunks.extend(pages)

            # Verify each chunk through consensus
            for chunk in all_chunks:
                if total_verified >= cl.max_claims_per_session:
                    logger.info("  Max claims per session reached, stopping")
                    break

                context = GraphContext(serialized_context=chunk.text)
                result = await self.consensus_engine.run(
                    chunk.text, context, self.swarm,
                    skip_tenth_man=True,
                )

                claim_count = (
                    len(result.verified_claims)
                    + len(result.uncertain_claims)
                    + len(result.discarded_claims)
                )
                total_claims += claim_count
                total_discarded += len(result.discarded_claims)
                total_uncertain += len(result.uncertain_claims)

                # Stamp authority and write
                for claim in result.verified_claims:
                    claim.source_authority = SourceAuthority.WEB_VERIFIED

                total_verified += len(result.verified_claims)
                if result.verified_claims:
                    ids = await self.graph_writer.write_verified_claims(
                        result.verified_claims, namespace=namespace
                    )
                    all_node_ids.extend(ids)

            if total_verified >= cl.max_claims_per_session:
                break

        summary = {
            "topic": topic,
            "sub_queries": len(sub_queries),
            "sub_queries_list": sub_queries,
            "sources_found": total_sources,
            "pages_scraped": total_pages,
            "urls_scraped": list(dict.fromkeys(all_urls_scraped)),  # dedup preserving order
            "claims_extracted": total_claims,
            "claims_verified": total_verified,
            "claims_discarded": total_discarded,
            "claims_uncertain": total_uncertain,
            "nodes_created": len(set(all_node_ids)),
        }
        logger.info(
            f"Learning complete: {topic} → "
            f"{total_verified} claims verified, "
            f"{len(set(all_node_ids))} nodes created"
        )
        return summary


def _extract_urls(snippets: list[RawContent], max_urls: int) -> list[str]:
    """Extract URLs from search snippets for full-page scraping."""
    urls: list[str] = []
    for s in snippets:
        # source format: "web:https://..." or "wikipedia:en:Title"
        if s.source.startswith("web:"):
            url = s.source[4:]  # strip "web:" prefix
            if url.startswith("http"):
                urls.append(url)
        if len(urls) >= max_urls:
            break
    return urls
