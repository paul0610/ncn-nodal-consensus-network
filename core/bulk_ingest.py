"""Bulk ingestion engine — queued batch processing for URLs, files, and folders.

Design:
  - Clients submit a batch of *items* (URLs or local file paths) and get a
    ``job_id`` immediately. Ingestion runs as a background asyncio task.
  - Each item is processed concurrently up to ``max_concurrent`` at a time,
    with a ``rate_limit_seconds`` pause between submissions per worker.
  - Callers poll ``get_job(job_id)`` to track progress without blocking.

Storage is in-memory (``_JOBS`` dict). Fine for single-process deployments
(the typical NCN use case). If you need persistence or multi-process, plug
Redis/SQLite behind the same ``JobStore`` protocol.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine
from uuid import uuid4

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Each item is a dict with at least:
#   "source"        — the payload (URL or file path)
#   "source_type"   — "url" | "pdf" | "docx" | ... | "text"
#   "label"         — (optional) friendly name shown in UI
Item = dict[str, Any]

# The callback receives (orchestrator, item, namespace) and returns a dict
# with the ingestion result (chunks, claims_verified, ...).
IngestFn = Callable[[Any, Item, str | None], Coroutine[Any, Any, dict]]


# ---------------------------------------------------------------------------
# Job storage (in-memory, single-process)
# ---------------------------------------------------------------------------

_JOBS: dict[str, dict] = {}


def _make_empty_job(job_id: str, total: int) -> dict:
    return {
        "job_id": job_id,
        "status": "queued",  # queued | running | done | error
        "total": total,
        "completed": 0,
        "failed": 0,
        "current": None,
        "results": [],
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


def get_job(job_id: str) -> dict | None:
    """Return the current state of a job, or None if unknown."""
    return _JOBS.get(job_id)


def list_jobs(limit: int = 50) -> list[dict]:
    """List recent jobs (newest first). For admin/debug UIs."""
    items = sorted(
        _JOBS.values(),
        key=lambda j: j.get("started_at") or "",
        reverse=True,
    )
    return items[:limit]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def submit_job(
    items: list[Item],
    namespace: str | None,
    ingest_fn: IngestFn,
    orchestrator,
    max_concurrent: int = 2,
    rate_limit_seconds: float = 1.0,
) -> str:
    """Queue a batch of items for ingestion and return a ``job_id``.

    The job runs as a background ``asyncio.Task``. Callers can poll its
    state via ``get_job(job_id)`` at any time.

    Parameters
    ----------
    items:
        List of ingestion items. Each dict MUST have ``source`` and
        ``source_type``; may have ``label`` for UI display.
    namespace:
        Graph namespace where new claims are written. ``None`` means use
        the orchestrator's default.
    ingest_fn:
        Async callback called once per item. Returns a result dict merged
        into the job's per-item results.
    orchestrator:
        Passed through to ``ingest_fn`` — usually ``get_orchestrator()``.
    max_concurrent:
        Number of items processed in parallel. Keep low for local LLMs so
        the swarm doesn't saturate.
    rate_limit_seconds:
        Pause after each item completes (per worker), to be polite to
        external servers when scraping URLs.
    """
    job_id = str(uuid4())
    _JOBS[job_id] = _make_empty_job(job_id, len(items))

    # Kick off the background task (it registers itself as "running")
    asyncio.create_task(
        _run_job(
            job_id=job_id,
            items=items,
            namespace=namespace,
            ingest_fn=ingest_fn,
            orchestrator=orchestrator,
            max_concurrent=max_concurrent,
            rate_limit_seconds=rate_limit_seconds,
        )
    )
    logger.info(
        f"Bulk job {job_id}: {len(items)} items queued "
        f"(max_concurrent={max_concurrent}, rate_limit={rate_limit_seconds}s)"
    )
    return job_id


async def _run_job(
    job_id: str,
    items: list[Item],
    namespace: str | None,
    ingest_fn: IngestFn,
    orchestrator,
    max_concurrent: int,
    rate_limit_seconds: float,
) -> None:
    """Background worker — processes items with concurrency + rate limiting."""
    _JOBS[job_id]["status"] = "running"
    _JOBS[job_id]["started_at"] = datetime.now(UTC).isoformat()

    semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def process_one(item: Item):
        async with semaphore:
            label = item.get("label") or item.get("source", "")[:80]
            _JOBS[job_id]["current"] = label
            try:
                result = await ingest_fn(orchestrator, item, namespace)
                _JOBS[job_id]["results"].append({
                    "source": label,
                    "source_type": item.get("source_type"),
                    "status": "ok",
                    **result,
                })
                _JOBS[job_id]["completed"] += 1
            except Exception as exc:
                logger.error(f"Bulk ingest failed for {label}: {exc}")
                _JOBS[job_id]["results"].append({
                    "source": label,
                    "source_type": item.get("source_type"),
                    "status": "error",
                    "error": str(exc),
                })
                _JOBS[job_id]["failed"] += 1
            if rate_limit_seconds > 0:
                await asyncio.sleep(rate_limit_seconds)

    try:
        await asyncio.gather(
            *[process_one(item) for item in items],
            return_exceptions=True,
        )
        _JOBS[job_id]["status"] = "done"
    except Exception as exc:
        logger.exception(f"Bulk job {job_id} crashed")
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(exc)
    finally:
        _JOBS[job_id]["current"] = None
        _JOBS[job_id]["finished_at"] = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Default ingest callback — dispatches by source_type
# ---------------------------------------------------------------------------

async def default_ingest_fn(orchestrator, item: Item, namespace: str | None) -> dict:
    """Default item dispatcher:
      - ``source_type == 'url'`` → ``orchestrator.process_ingest_url()``
      - anything else           → ``orchestrator.process_ingest(source, type)``
    """
    source_type = item.get("source_type") or "text"
    source = item["source"]
    if source_type == "url":
        return await orchestrator.process_ingest_url(source, namespace=namespace)
    return await orchestrator.process_ingest(
        source, source_type=source_type, namespace=namespace
    )


# ---------------------------------------------------------------------------
# Test helpers (only used by the test suite)
# ---------------------------------------------------------------------------

def _clear_all_jobs() -> None:
    """Reset the in-memory job store. Called by tests between runs."""
    _JOBS.clear()
