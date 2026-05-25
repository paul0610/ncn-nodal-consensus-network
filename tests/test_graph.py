"""Sprint 1 checkpoint — graph layer tests.

CHECKPOINT: insert 50 nodes, run queries, validate.
"""

from __future__ import annotations

import pytest

from core.config_loader import Config, GraphConfig, KuzuConfig
from core.models import Claim, ClaimStatus, GraphNode, GraphRelation
from graph.factory import create_graph_adapter
from graph.kuzu_adapter import KuzuAdapter
from graph.reader import GraphReader
from graph.writer import SingleWriter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    return Config(
        graph=GraphConfig(
            adapter="kuzu",
            kuzu=KuzuConfig(path=str(tmp_path / "test_graph")),
        )
    )


@pytest.fixture
async def adapter(config):
    a = KuzuAdapter(config)
    await a.initialize()
    yield a
    await a.close()


@pytest.fixture
async def writer(adapter, config):
    return SingleWriter(adapter, config)


@pytest.fixture
async def reader(adapter, config):
    return GraphReader(adapter, config)


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

class TestConfigLoader:
    def test_default_config(self):
        cfg = Config()
        assert cfg.graph.adapter == "kuzu"
        assert cfg.system.name == "NCN"

    def test_graph_defaults(self):
        cfg = Config()
        assert cfg.graph.temperature_boost == 0.10
        assert cfg.graph.confidence_boost == 0.05
        assert "concepto" in cfg.graph.node_types


# ------------------------------------------------------------------
# KuzuAdapter — basic CRUD
# ------------------------------------------------------------------

class TestKuzuCRUD:
    async def test_create_and_get_node(self, adapter):
        node = GraphNode(name="Einstein", node_type="persona")
        nid = await adapter.create_node(node)
        assert nid == node.node_id

        fetched = await adapter.get_node(nid)
        assert fetched is not None
        assert fetched.name == "Einstein"
        assert fetched.node_type == "persona"

    async def test_find_node_by_name(self, adapter):
        await adapter.create_node(
            GraphNode(name="Python", node_type="concepto")
        )
        found = await adapter.find_node_by_name("Python")
        assert found is not None
        assert found.name == "Python"

    async def test_find_node_by_name_missing(self, adapter):
        found = await adapter.find_node_by_name("NonExistent")
        assert found is None

    async def test_update_node(self, adapter):
        node = GraphNode(name="Temp", node_type="concepto", confidence=0.5)
        nid = await adapter.create_node(node)

        await adapter.update_node(nid, {"confidence": 0.9})
        updated = await adapter.get_node(nid)
        assert updated is not None
        assert updated.confidence == pytest.approx(0.9)

    async def test_create_relation(self, adapter):
        n1 = GraphNode(name="Einstein", node_type="persona")
        n2 = GraphNode(name="Relatividad", node_type="concepto")
        await adapter.create_node(n1)
        await adapter.create_node(n2)

        rel = GraphRelation(
            source_node_id=n1.node_id,
            target_node_id=n2.node_id,
            predicate="desarrolló",
        )
        await adapter.create_relation(rel)

        stats = await adapter.get_stats()
        assert stats.total_relations == 1


# ------------------------------------------------------------------
# KuzuAdapter — queries
# ------------------------------------------------------------------

class TestKuzuQueries:
    async def test_find_by_type(self, adapter):
        for i in range(5):
            await adapter.create_node(
                GraphNode(name=f"Person_{i}", node_type="persona")
            )
        for i in range(3):
            await adapter.create_node(
                GraphNode(name=f"Concept_{i}", node_type="concepto")
            )

        personas = await adapter.find_nodes_by_type("persona")
        assert len(personas) == 5

    async def test_find_by_namespace(self, adapter):
        for i in range(4):
            await adapter.create_node(
                GraphNode(
                    name=f"ML_{i}", node_type="concepto", namespace="ml"
                )
            )
        await adapter.create_node(
            GraphNode(name="Other", node_type="concepto", namespace="general")
        )

        ml = await adapter.find_nodes_by_namespace("ml")
        assert len(ml) == 4

    async def test_get_neighbors(self, adapter):
        a = GraphNode(name="A", node_type="concepto")
        b = GraphNode(name="B", node_type="concepto")
        c = GraphNode(name="C", node_type="concepto")
        for n in (a, b, c):
            await adapter.create_node(n)

        await adapter.create_relation(
            GraphRelation(
                source_node_id=a.node_id,
                target_node_id=b.node_id,
                predicate="related",
            )
        )
        await adapter.create_relation(
            GraphRelation(
                source_node_id=b.node_id,
                target_node_id=c.node_id,
                predicate="related",
            )
        )

        # 1-hop from A → should find B
        neighbors_1 = await adapter.get_neighbors(a.node_id, hops=1)
        assert len(neighbors_1) == 1
        assert neighbors_1[0].name == "B"

        # 2-hop from A → should find B and C
        neighbors_2 = await adapter.get_neighbors(a.node_id, hops=2)
        names = {n.name for n in neighbors_2}
        assert names == {"B", "C"}

    async def test_get_all_nodes(self, adapter):
        for i in range(10):
            await adapter.create_node(
                GraphNode(name=f"N_{i}", node_type="concepto")
            )
        all_nodes = await adapter.get_all_nodes()
        assert len(all_nodes) == 10

        limited = await adapter.get_all_nodes(limit=3)
        assert len(limited) == 3

    async def test_semantic_search(self, adapter):
        for i in range(5):
            vec = [0.0] * 10
            vec[i] = 1.0
            await adapter.create_node(
                GraphNode(
                    name=f"Vec_{i}",
                    node_type="concepto",
                    embedding=vec,
                )
            )

        query_vec = [0.0] * 10
        query_vec[2] = 1.0
        results = await adapter.semantic_search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0].name == "Vec_2"


# ------------------------------------------------------------------
# KuzuAdapter — stats
# ------------------------------------------------------------------

class TestKuzuStats:
    async def test_empty_stats(self, adapter):
        stats = await adapter.get_stats()
        assert stats.total_nodes == 0
        assert stats.total_relations == 0

    async def test_stats_after_inserts(self, adapter):
        n1 = GraphNode(name="A", node_type="persona", namespace="test")
        n2 = GraphNode(name="B", node_type="concepto", namespace="test")
        await adapter.create_node(n1)
        await adapter.create_node(n2)
        await adapter.create_relation(
            GraphRelation(
                source_node_id=n1.node_id,
                target_node_id=n2.node_id,
                predicate="knows",
            )
        )

        stats = await adapter.get_stats()
        assert stats.total_nodes == 2
        assert stats.total_relations == 1
        assert "test" in stats.namespaces
        assert "persona" in stats.node_types
        assert "concepto" in stats.node_types


# ------------------------------------------------------------------
# Insert 50 nodes — CHECKPOINT
# ------------------------------------------------------------------

ML_CONCEPTS = [
    "Machine Learning", "Deep Learning", "Neural Network", "Python",
    "TensorFlow", "PyTorch", "NLP", "Computer Vision",
    "Reinforcement Learning", "Transfer Learning", "BERT", "GPT",
    "Transformer", "Attention Mechanism", "CNN", "RNN", "LSTM", "GAN",
    "Autoencoder", "Random Forest", "Decision Tree", "SVM", "K-Means",
    "PCA", "Gradient Descent", "Backpropagation", "Overfitting",
    "Underfitting", "Regularization", "Dropout", "Batch Normalization",
    "Adam Optimizer", "Learning Rate", "Epoch", "Mini-batch",
    "Cross-validation", "Confusion Matrix", "Precision", "Recall",
    "F1 Score", "ROC Curve", "AUC", "Bias", "Variance",
    "Feature Engineering", "Data Augmentation", "Hyperparameter",
    "Grid Search", "Ensemble", "Bagging",
]


class TestInsert50Nodes:
    async def test_insert_50_and_query(self, adapter):
        ids: list[str] = []
        for concept in ML_CONCEPTS:
            node = GraphNode(
                name=concept, node_type="concepto", namespace="ml"
            )
            nid = await adapter.create_node(node)
            ids.append(nid)

        assert len(ids) == 50

        stats = await adapter.get_stats()
        assert stats.total_nodes == 50

        # Verify a few specific lookups
        dl = await adapter.find_node_by_name("Deep Learning")
        assert dl is not None
        assert dl.node_type == "concepto"

        ml_nodes = await adapter.find_nodes_by_namespace("ml", limit=100)
        assert len(ml_nodes) == 50

    async def test_insert_50_with_relations(self, adapter):
        """Insert 50 nodes, link them in a chain, verify traversal."""
        ids: list[str] = []
        for concept in ML_CONCEPTS:
            node = GraphNode(
                name=concept, node_type="concepto", namespace="ml"
            )
            ids.append(await adapter.create_node(node))

        # Chain: 0→1→2→…→49
        for i in range(len(ids) - 1):
            await adapter.create_relation(
                GraphRelation(
                    source_node_id=ids[i],
                    target_node_id=ids[i + 1],
                    predicate="related_to",
                )
            )

        stats = await adapter.get_stats()
        assert stats.total_nodes == 50
        assert stats.total_relations == 49

        # Neighbors of the first node (1-hop) → just the second node
        neighbors = await adapter.get_neighbors(ids[0], hops=1)
        assert len(neighbors) == 1
        assert neighbors[0].node_id == ids[1]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

class TestFactory:
    async def test_create_kuzu(self, config):
        adapter = await create_graph_adapter(config)
        assert isinstance(adapter, KuzuAdapter)
        stats = await adapter.get_stats()
        assert stats.total_nodes == 0
        await adapter.close()

    async def test_unknown_adapter(self, tmp_path):
        cfg = Config(
            graph=GraphConfig(
                adapter="unknown",
                kuzu=KuzuConfig(path=str(tmp_path / "g")),
            )
        )
        with pytest.raises(Exception, match="not recognised"):
            await create_graph_adapter(cfg)


# ------------------------------------------------------------------
# SingleWriter
# ------------------------------------------------------------------

class TestSingleWriter:
    async def test_write_claims(self, writer, adapter):
        claims = [
            Claim(
                subject="Einstein",
                predicate="developed",
                object="Relativity",
                confidence=0.9,
                source_model="test",
            ),
            Claim(
                subject="Newton",
                predicate="discovered",
                object="Gravity",
                confidence=0.85,
                source_model="test",
            ),
        ]

        node_ids = await writer.write_verified_claims(claims)
        assert len(node_ids) >= 4  # 4 unique entities

        einstein = await adapter.find_node_by_name("Einstein")
        assert einstein is not None
        assert einstein.node_type == "concepto"

        stats = await adapter.get_stats()
        assert stats.total_nodes == 4
        assert stats.total_relations == 2

    async def test_upsert_reinforces(self, writer, adapter):
        """Writing the same entity twice should boost confidence."""
        c1 = Claim(
            subject="Python",
            predicate="is",
            object="Language",
            confidence=0.8,
            source_model="m1",
        )
        c2 = Claim(
            subject="Python",
            predicate="is_used_for",
            object="AI",
            confidence=0.7,
            source_model="m2",
        )

        await writer.write_verified_claims([c1])
        py1 = await adapter.find_node_by_name("Python")
        assert py1 is not None
        original_conf = py1.confidence

        await writer.write_verified_claims([c2])
        py2 = await adapter.find_node_by_name("Python")
        assert py2 is not None
        assert py2.confidence > original_conf

    async def test_temperature_update(self, writer, adapter):
        node = GraphNode(
            name="Hot", node_type="concepto", temperature=0.5
        )
        nid = await adapter.create_node(node)

        await writer.update_temperatures([nid])
        updated = await adapter.get_node(nid)
        assert updated is not None
        assert updated.temperature > 0.5

    async def test_cool_down(self, writer, adapter):
        node = GraphNode(
            name="Cooling", node_type="concepto", temperature=0.8
        )
        await adapter.create_node(node)

        await writer.cool_down_all_nodes()
        cooled = await adapter.find_node_by_name("Cooling")
        assert cooled is not None
        assert cooled.temperature < 0.8


# ------------------------------------------------------------------
# GraphReader
# ------------------------------------------------------------------

class TestGraphReader:
    async def test_reader_delegates(self, reader, adapter):
        await adapter.create_node(
            GraphNode(name="Test", node_type="concepto")
        )

        found = await reader.find_by_name("Test")
        assert found is not None

        stats = await reader.get_stats()
        assert stats.total_nodes == 1

    async def test_reader_find_by_type(self, reader, adapter):
        for i in range(3):
            await adapter.create_node(
                GraphNode(name=f"E_{i}", node_type="evento")
            )

        events = await reader.find_by_type("evento")
        assert len(events) == 3
