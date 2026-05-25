"""Sprint 2 checkpoint — embeddings & retrieval tests.

CHECKPOINT: query → 5 relevant nodes → serialised in <800 tokens.
"""

from __future__ import annotations

import math

import pytest

from core.config_loader import Config, GraphConfig, KuzuConfig, RetrievalConfig
from core.models import GraphContext, GraphNode, GraphRelation
from graph.kuzu_adapter import KuzuAdapter
from retrieval.expander import NeighborhoodExpander
from retrieval.searcher import Retriever
from retrieval.serializer import NodeSerializer, _estimate_tokens


# ------------------------------------------------------------------
# Mock embedder (no model download needed)
# ------------------------------------------------------------------

class MockEmbedder:
    """Deterministic embedder for tests — maps known texts to known vectors."""

    def __init__(self, dim: int = 10) -> None:
        self._dim = dim
        self._registry: dict[str, list[float]] = {}

    def register(self, text: str, vector: list[float]) -> None:
        self._registry[text] = _normalise(vector)

    async def embed(self, text: str) -> list[float]:
        if text in self._registry:
            return self._registry[text]
        # fallback: deterministic hash-based vector
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 1) / 1.0 for i in range(self._dim)]
        return _normalise(vec)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _normalise(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def _basis(dim: int, index: int, weight: float = 1.0) -> list[float]:
    """One-hot-ish vector along *index*."""
    v = [0.0] * dim
    v[index % dim] = weight
    return v


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DIM = 10

SCIENCE_NODES = [
    ("Machine Learning",    _basis(DIM, 0)),
    ("Deep Learning",       _basis(DIM, 0, 0.9) ),
    ("Neural Network",      _basis(DIM, 0, 0.85)),
    ("Gradient Descent",    _basis(DIM, 0, 0.8) ),
    ("Backpropagation",     _basis(DIM, 0, 0.75)),
    ("Cooking",             _basis(DIM, 5)),
    ("Baking",              _basis(DIM, 5, 0.9)),
    ("Geography",           _basis(DIM, 7)),
]


@pytest.fixture
def config(tmp_path):
    return Config(
        graph=GraphConfig(
            adapter="kuzu",
            kuzu=KuzuConfig(path=str(tmp_path / "ret_graph")),
        ),
        retrieval=RetrievalConfig(
            top_k_nodes=5,
            hop_depth=1,
            token_budget=800,
        ),
    )


@pytest.fixture
async def adapter(config):
    a = KuzuAdapter(config)
    await a.initialize()
    yield a
    await a.close()


@pytest.fixture
async def seeded_adapter(adapter):
    """Adapter pre-loaded with SCIENCE_NODES + relations."""
    ids: dict[str, str] = {}
    for name, vec in SCIENCE_NODES:
        node = GraphNode(
            name=name,
            node_type="concepto",
            namespace="science",
            embedding=_normalise(vec),
            confidence=0.9,
            temperature=0.6,
        )
        ids[name] = await adapter.create_node(node)

    # Some relations among ML nodes
    rels = [
        ("Machine Learning",  "includes",   "Deep Learning"),
        ("Deep Learning",     "uses",       "Neural Network"),
        ("Neural Network",    "trained_by", "Backpropagation"),
        ("Backpropagation",   "computes",   "Gradient Descent"),
    ]
    for src, pred, tgt in rels:
        await adapter.create_relation(
            GraphRelation(
                source_node_id=ids[src],
                target_node_id=ids[tgt],
                predicate=pred,
                confidence=0.9,
            )
        )
    return adapter


@pytest.fixture
def mock_embedder():
    emb = MockEmbedder(dim=DIM)
    emb.register("machine learning concepts", _normalise(_basis(DIM, 0)))
    emb.register("recipes for dinner",        _normalise(_basis(DIM, 5)))
    return emb


# ------------------------------------------------------------------
# Embedder
# ------------------------------------------------------------------

class TestMockEmbedder:
    async def test_embed_returns_vector(self, mock_embedder):
        vec = await mock_embedder.embed("anything")
        assert isinstance(vec, list)
        assert len(vec) == DIM

    async def test_registered_text(self, mock_embedder):
        vec = await mock_embedder.embed("machine learning concepts")
        assert vec[0] > 0.9  # dominant dimension

    async def test_embed_batch(self, mock_embedder):
        vecs = await mock_embedder.embed_batch(["a", "b"])
        assert len(vecs) == 2


# ------------------------------------------------------------------
# Expander
# ------------------------------------------------------------------

class TestExpander:
    async def test_expand_adds_neighbors(self, seeded_adapter, config):
        expander = NeighborhoodExpander(seeded_adapter, config)

        ml = await seeded_adapter.find_node_by_name("Machine Learning")
        assert ml is not None
        seed = [ml]

        expanded = await expander.expand(seed, hops=1)
        names = {n.name for n in expanded}
        assert "Machine Learning" in names
        assert "Deep Learning" in names  # 1-hop neighbor

    async def test_expand_preserves_seeds(self, seeded_adapter, config):
        expander = NeighborhoodExpander(seeded_adapter, config)
        ml = await seeded_adapter.find_node_by_name("Machine Learning")
        expanded = await expander.expand([ml], hops=0)
        # With 0 hops, only seed should be returned
        assert len(expanded) == 1


# ------------------------------------------------------------------
# Serializer
# ------------------------------------------------------------------

class TestSerializer:
    async def test_claim_format(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        ml = await seeded_adapter.find_node_by_name("Machine Learning")
        text = await ser.serialize([ml])

        assert "CLAIM #" in text
        assert "Machine Learning" in text
        assert "→" in text
        assert "Deep Learning" in text

    async def test_claims_are_numbered_sequentially(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        ml = await seeded_adapter.find_node_by_name("Machine Learning")
        cooking = await seeded_adapter.find_node_by_name("Cooking")
        text = await ser.serialize([ml, cooking])

        # Global numbering must start at #1 and be contiguous across nodes
        lines = text.strip().split("\n")
        assert lines[0].startswith("CLAIM #1:")
        for i, line in enumerate(lines, start=1):
            assert line.startswith(f"CLAIM #{i}:")

    async def test_node_without_relations(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        cooking = await seeded_adapter.find_node_by_name("Cooking")
        text = await ser.serialize([cooking])

        assert "Cooking" in text
        assert "[concepto]" in text
        assert text.startswith("CLAIM #1:")

    async def test_budget_limits_nodes(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        all_nodes = await seeded_adapter.get_all_nodes()

        selected = await ser.apply_budget(all_nodes, budget=50)
        assert len(selected) < len(all_nodes)

    async def test_budget_large_enough(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        all_nodes = await seeded_adapter.get_all_nodes()

        selected = await ser.apply_budget(all_nodes, budget=10_000)
        assert len(selected) == len(all_nodes)

    async def test_confidence_in_output(self, seeded_adapter, config):
        ser = NodeSerializer(seeded_adapter, config)
        ml = await seeded_adapter.find_node_by_name("Machine Learning")
        text = await ser.serialize([ml])
        assert "conf:" in text


# ------------------------------------------------------------------
# Retriever (full pipeline)
# ------------------------------------------------------------------

class TestRetriever:
    async def test_retrieve_returns_context(
        self, seeded_adapter, config, mock_embedder
    ):
        ret = Retriever(seeded_adapter, config, embedder=mock_embedder)
        ctx = await ret.retrieve("machine learning concepts")

        assert isinstance(ctx, GraphContext)
        assert len(ctx.nodes) > 0
        assert ctx.serialized_context != ""
        assert ctx.avg_confidence > 0.0

    async def test_retrieve_finds_relevant_nodes(
        self, seeded_adapter, config, mock_embedder
    ):
        ret = Retriever(seeded_adapter, config, embedder=mock_embedder)
        ctx = await ret.retrieve("machine learning concepts")

        names = {n.name for n in ctx.nodes}
        # Should find ML-related nodes, not cooking / geography
        assert "Machine Learning" in names or "Deep Learning" in names
        assert "Cooking" not in names

    async def test_retrieve_irrelevant_query(
        self, seeded_adapter, config, mock_embedder
    ):
        ret = Retriever(seeded_adapter, config, embedder=mock_embedder)
        ctx = await ret.retrieve("recipes for dinner")

        names = {n.name for n in ctx.nodes}
        assert "Cooking" in names or "Baking" in names


# ------------------------------------------------------------------
# CHECKPOINT — 5 relevant nodes, serialised < 800 tokens
# ------------------------------------------------------------------

class TestCheckpoint:
    async def test_5_nodes_under_800_tokens(
        self, seeded_adapter, config, mock_embedder
    ):
        """
        Seed 20 science nodes, retrieve with top_k=5,
        verify exactly ≤5 returned and serialised < 800 tokens.
        """
        # Add more ML nodes so there are enough to pick 5
        extra = [
            "Regularization", "Dropout", "CNN", "RNN", "LSTM",
            "Transformer", "Attention", "BERT", "GPT", "Adam Optimizer",
            "SVM", "PCA",
        ]
        for i, name in enumerate(extra):
            vec = _basis(DIM, 0, 0.7 - i * 0.02)
            await seeded_adapter.create_node(
                GraphNode(
                    name=name,
                    node_type="concepto",
                    namespace="science",
                    embedding=_normalise(vec),
                    confidence=0.85,
                    temperature=0.5,
                )
            )

        ret = Retriever(seeded_adapter, config, embedder=mock_embedder)
        ctx = await ret.retrieve("machine learning concepts")

        assert 1 <= len(ctx.nodes) <= 5
        assert _estimate_tokens(ctx.serialized_context) <= 800

        # All returned nodes should have ids recorded
        assert len(ctx.nodes_used) == len(ctx.nodes)

    async def test_serialization_format(
        self, seeded_adapter, config, mock_embedder
    ):
        ret = Retriever(seeded_adapter, config, embedder=mock_embedder)
        ctx = await ret.retrieve("machine learning concepts")

        lines = ctx.serialized_context.strip().split("\n")
        for line in lines:
            assert line.startswith("CLAIM #")
