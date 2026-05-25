"""BootstrapPipeline — Knowledge Graph Bootstrap via Teacher-Student Distillation.

Uses a large teacher model to generate Q&A pairs about configured topics.
Each answer is verified through the local consensus pipeline before
being written to the graph.

Academic concept: "Knowledge Graph Bootstrap via Teacher-Student Distillation"
"""

from __future__ import annotations

from loguru import logger

from bootstrap.teacher import PROVIDER_TERMS_URLS, TeacherModel
from consensus.engine import ConsensusEngine
from core.config_loader import Config
from core.exceptions import TermsNotAcceptedError
from core.models import BootstrapResult, GraphContext, SourceAuthority
from graph.writer import SingleWriter
from swarm.pool import SwarmPool


class BootstrapPipeline:
    """Teach the graph by distilling knowledge from a larger model."""

    def __init__(
        self,
        config: Config,
        consensus_engine: ConsensusEngine,
        graph_writer: SingleWriter,
        swarm_pool: SwarmPool,
    ) -> None:
        self.config = config
        self.consensus_engine = consensus_engine
        self.graph_writer = graph_writer
        self.swarm_pool = swarm_pool

    async def run(self, topics: list[str] | None = None) -> BootstrapResult:
        """Run the full bootstrap pipeline.

        *topics* defaults to ``config.knowledge_bootstrap.topics``.
        """
        bc = self.config.knowledge_bootstrap

        if not bc.terms_accepted:
            provider = bc.teacher.provider
            url = PROVIDER_TERMS_URLS.get(provider, "(see provider docs)")
            raise TermsNotAcceptedError(
                f"You must accept the terms of '{provider}' before using "
                f"bootstrap. Set terms_accepted: true in config.yaml after "
                f"reading: {url}"
            )

        topics = topics or bc.topics
        if not topics:
            logger.warning("No bootstrap topics configured")
            return BootstrapResult()

        teacher = TeacherModel(self.config)
        total_claims = 0

        for topic in topics:
            logger.info(f"Bootstrap topic: {topic}")
            qa_pairs = await teacher.generate_qa(
                topic=topic,
                count=bc.questions_per_topic,
            )
            logger.info(f"  generated {len(qa_pairs)} Q&A pairs")

            for qa in qa_pairs:
                context = GraphContext(
                    serialized_context=qa.answer,
                )
                result = await self.consensus_engine.run(
                    query=qa.question,
                    context=context,
                    swarm=self.swarm_pool,
                    skip_tenth_man=True,  # bootstrap = trusted teacher, absorb don't question
                )

                if bc.store_verified_only:
                    to_store = result.verified_claims
                else:
                    to_store = result.verified_claims + result.uncertain_claims

                # Teacher-generated knowledge = high authority
                for claim in to_store:
                    claim.source_authority = SourceAuthority.COACH_MODEL

                if to_store:
                    await self.graph_writer.write_verified_claims(to_store)
                    total_claims += len(to_store)

        logger.info(
            f"Bootstrap complete: {len(topics)} topics, {total_claims} claims stored"
        )
        return BootstrapResult(
            topics_processed=len(topics),
            claims_stored=total_claims,
        )
