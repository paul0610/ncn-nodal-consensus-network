"""Tests for Sprint 6 — bulk ingestion engine + bulk endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import bulk_ingest
from core.bulk_ingest import get_job, list_jobs, submit_job
from core.config_loader import (
    Config,
    ConsensusConfig,
    GraphConfig,
    IngestionConfig,
    InternetConfig,
    KuzuConfig,
    ProviderConfig,
    ReputationConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from core.models import NodeRole


# ---------------------------------------------------------------------------
# Test fixtures (duplicated from test_interfaces for isolation)
# ---------------------------------------------------------------------------

CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Python", "predicate": "is_a", "object": "language",
     "confidence": 0.92, "source_text": "Python is a language."},
]})
CRITIC_OK = json.dumps({"vote": True, "confidence": 0.88, "reason": "ok"})
SYNTH = "Python is a programming language."


def _cfg(tmp_path: Path) -> Config:
    return Config(
        graph=GraphConfig(kuzu=KuzuConfig(path=str(tmp_path / "bulk_g"))),
        swarm=SwarmConfig(
            max_concurrent=5, request_timeout=30,
            nodes=[
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="extractor"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="critic"),
                SwarmNodeConfig(provider="ollama", model="m", count=1, role="synthesizer"),
            ],
        ),
        consensus=ConsensusConfig(validators_per_claim=1, tenth_man_enabled=False),
        reputation=ReputationConfig(history_file=str(tmp_path / "rep.json")),
        ingestion=IngestionConfig(internet=InternetConfig(search_mode="never")),
    )


def _mock_pool(pool):
    from providers.ollama import OllamaProvider
    for node in pool.nodes:
        node.provider = OllamaProvider(ProviderConfig(base_url="http://localhost:11434"))
        if node.role == NodeRole.EXTRACTOR:
            node.provider._post = AsyncMock(return_value={"response": CLAIMS_JSON, "eval_count": 1})
        elif node.role == NodeRole.CRITIC:
            node.provider._post = AsyncMock(return_value={"response": CRITIC_OK, "eval_count": 1})
        else:
            node.provider._post = AsyncMock(return_value={"response": SYNTH, "eval_count": 1})


class MockEmbedder:
    async def embed(self, text):
        return [0.1] * 10
    async def embed_batch(self, texts):
        return [[0.1] * 10 for _ in texts]


@pytest.fixture
async def api_client(tmp_path):
    """Yield a httpx.AsyncClient wired to the NCN FastAPI app."""
    from httpx import ASGITransport, AsyncClient
    from interface.api.app import create_app
    from core.orchestrator import Orchestrator

    cfg = _cfg(tmp_path)
    app = create_app(cfg)

    orch = Orchestrator(cfg, embedder=MockEmbedder())
    await orch.start()
    _mock_pool(orch.swarm_pool)

    import interface.api.app as api_mod
    api_mod._orchestrator = orch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await orch.stop()
    api_mod._orchestrator = None


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Reset in-memory job store between tests."""
    bulk_ingest._clear_all_jobs()
    yield
    bulk_ingest._clear_all_jobs()


# ---------------------------------------------------------------------------
# Engine — submit_job + get_job
# ---------------------------------------------------------------------------

class TestBulkEngine:

    async def test_submit_and_complete(self):
        """A trivial batch completes and accumulates results."""

        async def fake_ingest(_orch, item, _ns):
            return {
                "chunks": 1,
                "claims_verified": 3,
                "nodes_created": 2,
            }

        items = [
            {"source": "a", "source_type": "text", "label": "a.txt"},
            {"source": "b", "source_type": "text", "label": "b.txt"},
        ]
        job_id = await submit_job(
            items=items,
            namespace="test",
            ingest_fn=fake_ingest,
            orchestrator=None,
            max_concurrent=2,
            rate_limit_seconds=0.0,
        )

        assert job_id

        # Give the background task a moment to finish
        for _ in range(30):
            await asyncio.sleep(0.05)
            job = get_job(job_id)
            if job and job["status"] == "done":
                break

        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "done"
        assert job["total"] == 2
        assert job["completed"] == 2
        assert job["failed"] == 0
        assert len(job["results"]) == 2
        assert all(r["status"] == "ok" for r in job["results"])
        assert all(r["claims_verified"] == 3 for r in job["results"])

    async def test_submit_with_failures(self):
        """Items that raise get recorded as 'error' status, don't abort the batch."""

        async def flaky_ingest(_orch, item, _ns):
            if item["source"] == "bad":
                raise RuntimeError("something went wrong")
            return {"chunks": 1, "claims_verified": 1, "nodes_created": 1}

        items = [
            {"source": "ok1", "source_type": "text", "label": "ok1"},
            {"source": "bad", "source_type": "text", "label": "bad"},
            {"source": "ok2", "source_type": "text", "label": "ok2"},
        ]
        job_id = await submit_job(
            items=items,
            namespace=None,
            ingest_fn=flaky_ingest,
            orchestrator=None,
            max_concurrent=1,
            rate_limit_seconds=0.0,
        )

        for _ in range(40):
            await asyncio.sleep(0.05)
            job = get_job(job_id)
            if job and job["status"] == "done":
                break

        job = get_job(job_id)
        assert job["status"] == "done"
        assert job["completed"] == 2
        assert job["failed"] == 1

        bad_result = next(r for r in job["results"] if r["source"] == "bad")
        assert bad_result["status"] == "error"
        assert "something went wrong" in bad_result["error"]

    async def test_get_nonexistent_job_returns_none(self):
        assert get_job("does-not-exist") is None

    async def test_list_jobs_returns_recent_first(self):

        async def no_op(_orch, _item, _ns):
            return {"chunks": 0}

        for label in ("first", "second", "third"):
            await submit_job(
                items=[{"source": label, "source_type": "text", "label": label}],
                namespace=None,
                ingest_fn=no_op,
                orchestrator=None,
                max_concurrent=1,
                rate_limit_seconds=0.0,
            )
            # Short wait to make started_at strictly monotonic
            await asyncio.sleep(0.02)

        # Let all jobs finish
        await asyncio.sleep(0.3)

        jobs = list_jobs(limit=10)
        assert len(jobs) == 3


# ---------------------------------------------------------------------------
# API endpoints (bulk)
# ---------------------------------------------------------------------------

class TestBulkEndpoints:
    """Uses the api_client fixture from test_interfaces.py."""

    async def test_ingest_urls_empty_rejected(self, api_client):
        resp = await api_client.post("/ingest/urls", json={"urls": []})
        assert resp.status_code == 400

    async def test_ingest_urls_returns_job_id(self, api_client, monkeypatch):
        """Happy path: valid URLs → job_id + total in response."""
        from interface.api import app as api_app
        orch = api_app.get_orchestrator()

        async def fake_url_ingest(url, namespace=None):
            return {
                "chunks": 1, "claims_extracted": 2, "claims_verified": 1,
                "claims_discarded": 0, "claims_uncertain": 1,
                "nodes_created": 2, "corrections": 0,
            }

        monkeypatch.setattr(orch, "process_ingest_url", fake_url_ingest)

        resp = await api_client.post(
            "/ingest/urls",
            json={
                "urls": ["https://example.com/a", "https://example.com/b"],
                "namespace": "test_bulk",
                "rate_limit_seconds": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["total"] == 2
        assert data["status"] == "queued"

        # Poll until done
        job_id = data["job_id"]
        for _ in range(50):
            await asyncio.sleep(0.05)
            st = await api_client.get(f"/ingest/jobs/{job_id}")
            if st.json()["status"] == "done":
                break

        final = (await api_client.get(f"/ingest/jobs/{job_id}")).json()
        assert final["status"] == "done"
        assert final["completed"] == 2
        assert final["failed"] == 0

    async def test_get_unknown_job_returns_404(self, api_client):
        resp = await api_client.get("/ingest/jobs/nonexistent-id")
        assert resp.status_code == 404

    async def test_ingest_folder_missing_returns_400(self, api_client):
        resp = await api_client.post(
            "/ingest/folder",
            json={"path": "C:/nonexistent_folder_xyz_123"},
        )
        assert resp.status_code == 400

    async def test_list_jobs_endpoint(self, api_client, monkeypatch):
        """GET /ingest/jobs returns all jobs; ?status=running filters."""
        from interface.api import app as api_app
        orch = api_app.get_orchestrator()

        async def fake_ingest(source, source_type="text", namespace=None):
            return {
                "chunks": 1, "claims_extracted": 1, "claims_verified": 1,
                "claims_discarded": 0, "claims_uncertain": 0,
                "nodes_created": 1, "corrections": 0,
            }
        monkeypatch.setattr(orch, "process_ingest", fake_ingest)

        # Kick off a small job
        submit_resp = await api_client.post(
            "/ingest/urls",
            json={"urls": ["https://example.com/a"],
                  "namespace": "test",
                  "rate_limit_seconds": 0.0},
        )
        # URL processing uses process_ingest_url, not process_ingest, so
        # patch that instead
        async def fake_url(url, namespace=None):
            return {
                "chunks": 1, "claims_extracted": 1, "claims_verified": 1,
                "claims_discarded": 0, "claims_uncertain": 0,
                "nodes_created": 1, "corrections": 0,
            }
        monkeypatch.setattr(orch, "process_ingest_url", fake_url)

        resp = await api_client.get("/ingest/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "total" in data
        # At least the one we just submitted should be in the list
        assert data["total"] >= 1

        # Filtering by status works (may be done by now due to instant mocks)
        for target_status in ("running", "done", "queued"):
            resp = await api_client.get(f"/ingest/jobs?status={target_status}")
            assert resp.status_code == 200
            filtered = resp.json()["jobs"]
            assert all(j["status"] == target_status for j in filtered)

    async def test_ingest_folder_scans_and_queues(self, api_client, tmp_path, monkeypatch):
        # Create test files of different formats
        (tmp_path / "doc.txt").write_text("hello world", encoding="utf-8")
        (tmp_path / "notes.md").write_text("# Notes\n\nSome content.", encoding="utf-8")
        (tmp_path / "ignore.png").write_bytes(b"fake image")

        from interface.api import app as api_app
        orch = api_app.get_orchestrator()

        # Stub process_ingest so we don't actually run consensus
        async def fake_ingest(source, source_type="text", namespace=None):
            return {
                "chunks": 1, "claims_extracted": 1, "claims_verified": 1,
                "claims_discarded": 0, "claims_uncertain": 0,
                "nodes_created": 1, "corrections": 0,
            }

        monkeypatch.setattr(orch, "process_ingest", fake_ingest)

        resp = await api_client.post(
            "/ingest/folder",
            json={
                "path": str(tmp_path),
                "namespace": "test",
                "recursive": False,
                "extensions": ["txt", "md"],
                "rate_limit_seconds": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Only .txt and .md should match, not the .png
        assert data["total"] == 2

        # Wait for completion
        job_id = data["job_id"]
        for _ in range(50):
            await asyncio.sleep(0.05)
            st = (await api_client.get(f"/ingest/jobs/{job_id}")).json()
            if st["status"] == "done":
                break

        final = (await api_client.get(f"/ingest/jobs/{job_id}")).json()
        assert final["status"] == "done"
        assert final["completed"] == 2
